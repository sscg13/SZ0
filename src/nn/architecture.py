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
    `multiple`, then corrected so the total equals `num_layers * d_ff` — the
    schedule is parameter-, FLOP- and traffic-neutral against a uniform net.

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

    @nn.compact
    def __call__(self, x):
        x = Attention(self.d_model, self.num_heads, self.d_qk_head,
                      self.d_v_head, self.dtype)(x)
        x = FeedForward(self.d_model, self.d_ff, self.dtype)(x)
        return x

class ShatranjNet(nn.Module):
    num_layers: int = 10
    d_model: int = 256
    num_heads: int = 8
    # int = uniform width; tuple of length num_layers = per-layer widths
    # (see cosine_ff_schedule). Must be a tuple, not a list — Flax hashes it.
    # Tapered wide -> narrow: (384, 384, 384, 320, 256, 256, 192, 128, 128, 128).
    # Sums to 10*256, so parameters, FLOPs and memory traffic are unchanged
    # against uniform d_ff=256 — this is purely a reallocation of capacity
    # toward the early layers. The 10 must match num_layers; the block loop
    # raises if it does not.
    d_ff: Union[int, Sequence[int]] = cosine_ff_schedule(10, 256)
    d_qk_head: int = 64
    d_v_head: int = None
    # jnp.bfloat16 runs the trunk in mixed precision: weights stay float32, only
    # activations and matmul inputs narrow. The model is bandwidth-bound (the
    # dominant GEMM sits at ~63 FLOP/byte against an L40S ridge point near 209),
    # so halving activation traffic is the main lever left on training speed.
    # Outputs are always returned as float32 so the loss is unaffected.
    dtype: jnp.dtype = jnp.float32
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
            x = ShatranjBlock(self.d_model, self.num_heads, width,
                              self.d_qk_head, self.d_v_head, self.dtype)(x)

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