import functools
import math
from typing import Sequence, Union

import jax
import jax.numpy as jnp
import flax.linen as nn


def cosine_ff_schedule(num_layers, d_ff=256, start_ratio=1.5, end_ratio=0.5,
                       multiple=64):
    """Per-layer FFN widths tapering wide -> narrow (arXiv:2606.23670).

    Capacity concentrates in the early layers. Widths are rounded to
    `multiple`, then corrected so the total equals `num_layers * d_ff`, so the
    schedule is parameter- and FLOP-neutral against a uniform net.

    It is NOT throughput-neutral: measured 4.3% slower datagen (70K -> 67K nps)
    at L=10, d_ff=256, because a 128-wide FFN GEMM is less efficient than a
    256-wide one and the 384-wide layers do not make it back. Equal FLOPs do
    not mean equal time when the reallocation narrows kernels.

    >>> cosine_ff_schedule(10, 256)
    (384, 384, 384, 320, 256, 256, 192, 128, 128, 128)
    """
    if num_layers < 2:
        return (d_ff,) * num_layers

    start, end = start_ratio * d_ff, end_ratio * d_ff
    exact = [end + (start - end) / 2 * (1 + math.cos(math.pi * l / (num_layers - 1)))
             for l in range(num_layers)]
    widths = [max(multiple, round(e / multiple) * multiple) for e in exact]

    # Nudge one `multiple` at a time, at whichever layer it least distorts the
    # curve, until the parameter budget matches. Unreachable when
    # num_layers * d_ff is not itself a multiple; bounded so it cannot spin.
    target = num_layers * d_ff
    for _ in range(4 * num_layers):
        if sum(widths) == target:
            break
        step = multiple if sum(widths) < target else -multiple
        candidates = [i for i in range(num_layers) if widths[i] + step >= multiple]
        if not candidates:
            break
        best = min(candidates,
                   key=lambda i: abs(widths[i] + step - exact[i])
                   - abs(widths[i] - exact[i]))
        widths[best] += step
    return tuple(widths)


# --- 1. Model Definitions (The Blueprint) ---

class Attention(nn.Module):
    d_model: int = 256
    num_heads: int = 8
    # QK and V head dims, decoupled from each other and from d_model.
    # None => d_model // num_heads (unchanged). Decoupling V matters when
    # changing num_heads, which would otherwise silently rescale it.
    d_qk_head: int = 64
    d_v_head: int = None
    # Compute dtype. Weights always stay float32 (see param_dtype below), so
    # bfloat16 here is mixed precision with fp32 master weights, not bf16
    # training. Softmax is forced back to fp32 regardless.
    dtype: jnp.dtype = jnp.float32

    # --- Dynamic attention bias (scaled-down Smolgen, lczero.org 2024/02) ---
    # A content-dependent 64x64 map added to the logits, on top of the static
    # spatial_bias. Unlike spatial_bias it depends on the position, so it lets
    # attention react to board state (constrain distant squares in closed
    # positions, open them up otherwise) rather than only to geometry.
    #
    # dyn_bias_code = 0 disables it entirely (the default; nothing changes).
    # Shared across heads within a layer, and NOT shared across layers: Leela
    # shares its decoder globally only because a full per-head 64x64 decode
    # would be ~84M parameters over all (layer, head) pairs. One map per layer
    # is 262K, which fits, so each layer gets its own.
    #
    # dyn_bias_rank = 0 decodes the full 64x64 map (4096 outputs).
    # dyn_bias_rank = r decodes U, V in R^{64 x r} and uses U V^T instead,
    # which is 4096/(2r) times narrower. Note a rank-1 map of the form
    # u_i + v_j would be pointless: anything constant along the key axis
    # cancels in the softmax, so only the v_j half would survive.
    dyn_bias_code: int = 0
    dyn_bias_compress: int = 8
    dyn_bias_rank: int = 0

    def _dynamic_bias(self, x, b, seq):
        """(b, seq, d_model) -> (b, seq, seq), shared across heads."""
        dense = functools.partial(nn.Dense, dtype=self.dtype,
                                  param_dtype=jnp.float32)
        # Squeeze each square to a few channels before flattening, or the
        # first dense would be (seq * d_model) wide.
        c = dense(self.dyn_bias_compress, name="dyn_compress")(x)
        c = c.reshape((b, seq * self.dyn_bias_compress))
        c = dense(self.dyn_bias_code, name="dyn_code")(c)
        # Without this the whole branch collapses to one linear map of x.
        c = nn.silu(nn.LayerNorm(dtype=self.dtype, param_dtype=jnp.float32,
                                 name="dyn_norm")(c))

        if self.dyn_bias_rank > 0:
            r = self.dyn_bias_rank
            u = dense(seq * r, name="dyn_u")(c).reshape((b, seq, r))
            # V starts at zero so the branch is a no-op at init and the net
            # begins identical to the baseline. Zeroing *both* factors would
            # trap them: dU = V^T(dL/dM) = 0 and dV = U^T(dL/dM) = 0.
            #
            # Note the zero decode also blocks gradient to compress/code/norm/u
            # for exactly one step, until V becomes nonzero. Verified to
            # unfreeze immediately: front-end |grad| is 0 at step 0 and ~5e-3
            # from step 1.
            v = dense(seq * r, name="dyn_v",
                      kernel_init=nn.initializers.zeros)(c).reshape((b, seq, r))
            # matmul rather than einsum. einsum exports fine, but ORT's
            # optimizer handles a plain MatMul better — it is what the fusion
            # patterns (FusedMatMul and friends) match on, and it is the same
            # (b,s,r) x (b,r,s) shape the policy head already uses. Do not
            # "simplify" this back to einsum.
            m = jnp.matmul(u, jnp.swapaxes(v, -1, -2))
        else:
            m = dense(seq * seq, name="dyn_decode",
                      kernel_init=nn.initializers.zeros)(c).reshape(
                          (b, seq, seq))

        # Retrievable with apply(..., capture_intermediates=True). The point of
        # this branch is that the map varies with the position; without a hook
        # there is no way to confirm it does rather than collapsing to a second
        # static bias, since the helper cannot be called on its own (Flax only
        # allows submodule creation inside the @compact method).
        self.sow("intermediates", "dyn_bias", m)
        return m

    @nn.compact
    def __call__(self, x):
        b, seq_len, _ = x.shape
        default_head = self.d_model // self.num_heads
        qk_head = self.d_qk_head if self.d_qk_head is not None else default_head
        v_head = self.d_v_head if self.d_v_head is not None else default_head
        d_qk = self.num_heads * qk_head
        d_v = self.num_heads * v_head
        dense = functools.partial(nn.Dense, dtype=self.dtype,
                                  param_dtype=jnp.float32)

        attn_input = nn.LayerNorm(dtype=self.dtype,
                                  param_dtype=jnp.float32)(x)

        q = dense(d_qk)(attn_input).reshape((b, seq_len, self.num_heads, qk_head)).transpose((0, 2, 1, 3))
        k = dense(d_qk)(attn_input).reshape((b, seq_len, self.num_heads, qk_head)).transpose((0, 2, 3, 1))
        v = dense(d_v)(attn_input).reshape((b, seq_len, self.num_heads, v_head)).transpose((0, 2, 1, 3))

        logits = jnp.matmul(q, k) / jnp.sqrt(qk_head)

        # Broadcastable Spatial Bias: (1, num_heads, seq_len, seq_len)
        spatial_bias = self.param(
            'spatial_bias',
            nn.initializers.normal(stddev=0.02),
            (1, self.num_heads, seq_len, seq_len)
        )
        logits = logits + spatial_bias.astype(logits.dtype)

        if self.dyn_bias_code > 0:
            # (b, seq, seq) -> broadcast over the head axis.
            logits = logits + self._dynamic_bias(attn_input, b, seq_len)[:, None]

        # Softmax in fp32 even in mixed precision: the exponentials and the
        # reduction are where low precision actually costs accuracy, and this
        # tensor is small relative to the matmuls it sits between.
        attn_weights = nn.softmax(logits.astype(jnp.float32),
                                  axis=-1).astype(self.dtype)
        attn_out = jnp.matmul(attn_weights, v)

        attn_out = attn_out.transpose((0, 2, 1, 3)).reshape((b, seq_len, d_v))
        attn_out = dense(self.d_model)(attn_out)

        return x + attn_out


class FeedForward(nn.Module):
    d_model: int = 256
    d_ff: int = 256
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        dense = functools.partial(nn.Dense, dtype=self.dtype,
                                  param_dtype=jnp.float32)
        ff_input = nn.LayerNorm(dtype=self.dtype, param_dtype=jnp.float32)(x)
        ff_out = dense(self.d_ff)(ff_input)
        ff_out = nn.silu(ff_out)
        ff_out = dense(self.d_model)(ff_out)
        return x + ff_out


class ShatranjBlock(nn.Module):
    d_model: int = 256
    num_heads: int = 8
    d_ff: int = 256
    d_qk_head: int = 64
    d_v_head: int = None
    dtype: jnp.dtype = jnp.float32
    dyn_bias_code: int = 0
    dyn_bias_compress: int = 8
    dyn_bias_rank: int = 0

    @nn.compact
    def __call__(self, x):
        # Keyword args deliberately: Attention has enough optional fields that
        # positional construction breaks silently when one is inserted.
        x = Attention(d_model=self.d_model, num_heads=self.num_heads,
                      d_qk_head=self.d_qk_head, d_v_head=self.d_v_head,
                      dtype=self.dtype, dyn_bias_code=self.dyn_bias_code,
                      dyn_bias_compress=self.dyn_bias_compress,
                      dyn_bias_rank=self.dyn_bias_rank)(x)
        x = FeedForward(self.d_model, self.d_ff, self.dtype)(x)
        return x

class ShatranjNet(nn.Module):
    num_layers: int = 10
    d_model: int = 256
    num_heads: int = 8
    # int = uniform width; tuple of length num_layers = per-layer widths
    # (see cosine_ff_schedule, kept for future reshaping experiments). Must be
    # a tuple, not a list — Flax hashes it.
    #
    # Uniform 256 after the cosine taper was tested and rejected: paired loss
    # +0.0002 +/- 0.0018 and a 400-game match at +6.9 +/- 15.8 both said no
    # effect, while datagen dropped 4.3%.
    d_ff: Union[int, Sequence[int]] = 256
    d_qk_head: int = 64
    d_v_head: int = None
    # jnp.bfloat16 runs the trunk in mixed precision: weights stay float32, only
    # activations and matmul inputs narrow. The model is bandwidth-bound (the
    # dominant GEMM sits at ~63 FLOP/byte against an L40S ridge point near 209),
    # so halving activation traffic is the main lever left on training speed.
    # Outputs are always returned as float32 so the loss is unaffected.
    dtype: jnp.dtype = jnp.float32
    # See Attention for what these mean. 0 = off, unchanged from before.
    dyn_bias_code: int = 0
    dyn_bias_compress: int = 8
    dyn_bias_rank: int = 0
    vocab_size: int = 13 * 64
    max_halfmoves: int = 140

    @nn.compact
    def __call__(self, board_tokens, halfmove_token):
        # No clipping guard here: int32 Clip has no CUDA kernel, so ORT put
        # it on the CPU provider and inserted MemcpyFromHost, which blocks
        # CUDA graph capture. The engine clamps halfmove instead (see
        # clamp_halfmove in src/inference.h); board tokens are bounded by
        # construction. Behaviour is unchanged either way, since JAX's
        # gather clamps out-of-range indices anyway.
        safe_board_tokens = board_tokens
        safe_halfmove_token = halfmove_token

        embed = functools.partial(nn.Embed, dtype=self.dtype,
                                  param_dtype=jnp.float32)
        dense = functools.partial(nn.Dense, dtype=self.dtype,
                                  param_dtype=jnp.float32)

        # Embeddings & Setup (using the safe tokens)
        x = embed(num_embeddings=self.vocab_size, features=self.d_model)(safe_board_tokens)

        g_emb = embed(num_embeddings=self.max_halfmoves, features=self.d_model)(safe_halfmove_token)
        x = x + jnp.expand_dims(g_emb, axis=-2)

        # Transformer Body
        widths = ((self.d_ff,) * self.num_layers
                  if isinstance(self.d_ff, int) else tuple(self.d_ff))
        if len(widths) != self.num_layers:
            raise ValueError(f"d_ff has {len(widths)} entries, "
                             f"num_layers is {self.num_layers}")
        for width in widths:
            x = ShatranjBlock(d_model=self.d_model, num_heads=self.num_heads,
                              d_ff=width, d_qk_head=self.d_qk_head,
                              d_v_head=self.d_v_head, dtype=self.dtype,
                              dyn_bias_code=self.dyn_bias_code,
                              dyn_bias_compress=self.dyn_bias_compress,
                              dyn_bias_rank=self.dyn_bias_rank)(x)

        x = nn.LayerNorm(dtype=self.dtype, param_dtype=jnp.float32)(x)

        # Value Head
        v = dense(16)(x)
        v = nn.relu(v)
        v = v.reshape((v.shape[0], -1))
        v = dense(32)(v)
        v = nn.relu(v)
        value_logits = dense(3)(v)

        # Policy Head (Dot-Product)
        p_from = dense(64)(x)
        p_to = dense(64)(x)
        p_to_transposed = jnp.swapaxes(p_to, -1, -2)
        policy_logits = jnp.matmul(p_from, p_to_transposed) / 8.0

        # Always hand fp32 to the loss / exporter, whatever the trunk ran in.
        return (policy_logits.astype(jnp.float32),
                value_logits.astype(jnp.float32))


# --- 2. The Testing Script ---

def main():
    print("Initializing JAX test...")

    # A. Setup JAX Random Number Generator (PRNG)
    # JAX requires explicit random keys for everything to guarantee reproducibility.
    key = jax.random.PRNGKey(42)
    key_board, key_halfmove, key_init = jax.random.split(key, 3)

    # B. Generate Dummy Data
    batch_size = 8
    # Random integers between 0 and 831 for the 64 squares
    dummy_boards = jax.random.randint(key_board, (batch_size, 64), minval=0, maxval=831)
    # Random integers between 0 and 100 for the halfmove clock
    dummy_halfmoves = jax.random.randint(key_halfmove, (batch_size,), minval=0, maxval=139)

    print(f"Generated Batch Size: {batch_size}")
    print(f"Dummy Boards Shape: {dummy_boards.shape}")
    print(f"Dummy Halfmoves Shape: {dummy_halfmoves.shape}")
    print("-" * 30)

    # C. Instantiate the model blueprint
    model = ShatranjNet()

    # D. Initialize the weights
    # We pass the dummy data to init() so Flax can trace the tensor shapes and build the matrices
    print("Initializing Model Weights...")
    variables = model.init(key_init, dummy_boards, dummy_halfmoves)

    # E. Run the Forward Pass!
    # We use model.apply() and pass the variables dictionary along with our data
    policy_logits, value_logits = model.apply(variables, dummy_boards, dummy_halfmoves)

    # F. Verify Outputs
    print("Forward Pass Complete!")
    print(f"Policy Output Shape: {policy_logits.shape}  <-- Expected: (8, 64, 64)")
    print(f"Value Output Shape:  {value_logits.shape}       <-- Expected: (8, 3)")
    print("-" * 30)

    # G. Count Total Parameters
    # jax.tree_util.tree_leaves flattens the nested dictionary of weights into a simple list of arrays
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(variables))
    print(f"Total Parameter Count: {param_count:,}")

if __name__ == "__main__":
    main()