import jax
import jax.numpy as jnp
import flax.linen as nn

# --- 1. Model Definitions (The Blueprint) ---

class Attention(nn.Module):
    d_model: int = 256
    num_heads: int = 8
    # QK and V head dims, decoupled from each other and from d_model.
    # None => d_model // num_heads (unchanged). Decoupling V matters when
    # changing num_heads, which would otherwise silently rescale it.
    d_qk_head: int = None
    d_v_head: int = None

    @nn.compact
    def __call__(self, x):
        b, seq_len, _ = x.shape
        default_head = self.d_model // self.num_heads
        qk_head = self.d_qk_head if self.d_qk_head is not None else default_head
        v_head = self.d_v_head if self.d_v_head is not None else default_head
        d_qk = self.num_heads * qk_head
        d_v = self.num_heads * v_head

        attn_input = nn.LayerNorm()(x)

        q = nn.Dense(d_qk)(attn_input).reshape((b, seq_len, self.num_heads, qk_head)).transpose((0, 2, 1, 3))
        k = nn.Dense(d_qk)(attn_input).reshape((b, seq_len, self.num_heads, qk_head)).transpose((0, 2, 3, 1))
        v = nn.Dense(d_v)(attn_input).reshape((b, seq_len, self.num_heads, v_head)).transpose((0, 2, 1, 3))

        logits = jnp.matmul(q, k) / jnp.sqrt(qk_head)

        # Broadcastable Spatial Bias: (1, num_heads, seq_len, seq_len)
        spatial_bias = self.param(
            'spatial_bias',
            nn.initializers.normal(stddev=0.02),
            (1, self.num_heads, seq_len, seq_len)
        )
        logits = logits + spatial_bias

        attn_weights = nn.softmax(logits, axis=-1)
        attn_out = jnp.matmul(attn_weights, v)

        attn_out = attn_out.transpose((0, 2, 1, 3)).reshape((b, seq_len, d_v))
        attn_out = nn.Dense(self.d_model)(attn_out)

        return x + attn_out


class FeedForward(nn.Module):
    d_model: int = 256
    d_ff: int = 256

    @nn.compact
    def __call__(self, x):
        ff_input = nn.LayerNorm()(x)
        ff_out = nn.Dense(self.d_ff)(ff_input)
        ff_out = nn.silu(ff_out)
        ff_out = nn.Dense(self.d_model)(ff_out)
        return x + ff_out


class ShatranjBlock(nn.Module):
    d_model: int = 256
    num_heads: int = 8
    d_ff: int = 256
    d_qk_head: int = None
    d_v_head: int = None

    @nn.compact
    def __call__(self, x):
        x = Attention(self.d_model, self.num_heads, self.d_qk_head, self.d_v_head)(x)
        x = FeedForward(self.d_model, self.d_ff)(x)
        return x

class ShatranjNet(nn.Module):
    num_layers: int = 10
    d_model: int = 256
    num_heads: int = 8
    d_ff: int = 256
    d_qk_head: int = None
    d_v_head: int = None
    vocab_size: int = 13 * 64
    max_halfmoves: int = 140

    @nn.compact
    def __call__(self, board_tokens, halfmove_token):
        # Guard just in case
        safe_board_tokens = jnp.clip(board_tokens, 0, self.vocab_size - 1)
        safe_halfmove_token = jnp.clip(halfmove_token, 0, self.max_halfmoves - 1)

        # Embeddings & Setup (using the safe tokens)
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model)(safe_board_tokens)

        g_emb = nn.Embed(num_embeddings=self.max_halfmoves, features=self.d_model)(safe_halfmove_token) 
        x = x + jnp.expand_dims(g_emb, axis=-2) 

        # Transformer Body
        for _ in range(self.num_layers):
            x = ShatranjBlock(self.d_model, self.num_heads, self.d_ff,
                              self.d_qk_head, self.d_v_head)(x)
            
        x = nn.LayerNorm()(x)

        # Value Head
        v = nn.Dense(16)(x)
        v = nn.relu(v) 
        v = v.reshape((v.shape[0], -1)) 
        v = nn.Dense(32)(v)
        v = nn.relu(v)
        value_logits = nn.Dense(3)(v)

        # Policy Head (Dot-Product)
        p_from = nn.Dense(64)(x) 
        p_to = nn.Dense(64)(x)   
        p_to_transposed = jnp.swapaxes(p_to, -1, -2)
        policy_logits = jnp.matmul(p_from, p_to_transposed) / 8.0

        return policy_logits, value_logits


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