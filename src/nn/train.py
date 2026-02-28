import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import time
import orbax.checkpoint as ocp
import os

from dataloader import load_sparse_dataset
from dataloader import SparseInMemoryDataLoader
from architecture import ShatranjNet

def compute_loss(params, apply_fn, batch):
    # --- HYPERPARAMETERS (Tune these!) ---
    VALUE_WEIGHT = 4  # Scales up Value Loss to compete with Policy Loss
    Q_WEIGHT = 1      # How much to care about MCTS Q vs Actual Game Outcome (WDL)
    # -------------------------------------

    halfmoves = batch['halfmoves'].squeeze(-1).astype(jnp.int32)
    policy_logits, value_logits = apply_fn({'params': params}, batch['boards'], halfmoves)
    
    # 1. Masked Policy Loss (Cross Entropy)
    batch_size = policy_logits.shape[0]
    flat_policy_logits = policy_logits.reshape((batch_size, 4096))
    sample_policy_loss = optax.softmax_cross_entropy(flat_policy_logits, batch['target_pi'])
    policy_mask = (batch['target_pi'].sum(axis=-1) > 0.5).astype(jnp.float32)
    valid_positions = jnp.maximum(policy_mask.sum(), 1.0)
    policy_loss = (sample_policy_loss * policy_mask).sum() / valid_positions

    # 2. Value Loss A: Ground Truth WDL (Cross Entropy)
    z = batch['target_z'].squeeze(-1)
    wdl_indices = jnp.where(z > 0.5, 0, jnp.where(z > -0.5, 1, 2))
    wdl_targets = jax.nn.one_hot(wdl_indices, 3) 
    wdl_loss = optax.softmax_cross_entropy(value_logits, wdl_targets).mean()

    # 3. Value Loss B: MCTS Q-Value (Mean Squared Error)
    # Convert value logits to probabilities to calculate Expected Value
    wdl_probs = jax.nn.softmax(value_logits, axis=-1)
    
    # E[v] = P(Win) - P(Loss)
    expected_value = wdl_probs[:, 0] - wdl_probs[:, 2]
    q_target = batch['target_q'].squeeze(-1)
    
    q_mse_loss = jnp.square(expected_value - q_target).mean()

    # 4. Combine and Balance
    # Blend the WDL cross-entropy with the Q MSE
    combined_value_loss = wdl_loss + (Q_WEIGHT * q_mse_loss)
    
    # Scale up the value loss so it doesn't get drowned out by the policy loss
    total_loss = policy_loss + (VALUE_WEIGHT * combined_value_loss)
    
    return total_loss, (policy_loss, combined_value_loss, wdl_loss, q_mse_loss)

# --- 2. The JIT-Compiled Training Step ---
@jax.jit
def train_step(state, batch):
    """
    Calculates gradients and updates the weights. @jax.jit makes this lightning fast.
    """
    # jax.value_and_grad computes both the loss and the gradients of the params
    # has_aux=True tells JAX that compute_loss returns a tuple, and to only differentiate the first item
    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    
    (loss, (p_loss, v_loss, wdl_loss, q_loss)), grads = grad_fn(state.params, state.apply_fn, batch)
    
    # Apply gradients using the optimizer to get the new state
    state = state.apply_gradients(grads=grads)
    
    return state, loss, p_loss, v_loss, wdl_loss, q_loss

# --- 3. The Main Training Engine ---
def train():
    print("Initializing ShatranjNet Training...")
    
    # Setup PRNG Keys
    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    
    # Instantiate Model & Optimizer
    model = ShatranjNet()
    # Learning rate warmups/decays are added here later. We use a static LR for now.
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=1e-4, weight_decay=1e-4)
    )
    
    # Create Dummy Batch to initialize shapes
    dummy_boards = jnp.zeros((1, 64), dtype=jnp.int32)
    dummy_halfmoves = jnp.zeros((1,), dtype=jnp.int32)
    variables = model.init(init_key, dummy_boards, dummy_halfmoves)
    
    # Create the TrainState
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer,
    )
    
    print(f"Model parameters: {sum(x.size for x in jax.tree_util.tree_leaves(state.params)):,}")
    print("Starting Training Loop...\n")
    
    # Load files
    data_files = ["run0.data", "run1.data", "run2.data", "run3.data"]
    full_dataset = load_sparse_dataset(data_files)
    dataloader = SparseInMemoryDataLoader(dataset_dict=full_dataset, batch_size=128)
    
    epochs = 10
    step = 0
    start_time = time.time()
    
    for epoch in range(epochs):
        
        for batch in dataloader.get_batches():
            if jnp.isnan(batch['target_pi']).any() or jnp.isnan(batch['target_z']).any():
                print(f"CRITICAL: NaN found in data at step {step}!")
                break

            state, loss, p_loss, v_loss, wdl_loss, q_loss = train_step(state, batch)
            loss.block_until_ready()

            step += 1
            
            if step % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Step {step:04d} | Total: {loss:.4f} "
                      f"[Pol: {p_loss:.4f} | Val: {v_loss:.4f} (WDL: {wdl_loss:.4f}, Q: {q_loss:.4f})] "
                      f"| Time: {elapsed:.2f}s")
                start_time = time.time()

        print(f"Saving checkpoint for epoch {epoch}...")
        checkpoint_manager.save(
            step=epoch,
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(state)
            )
        )

        checkpoint_manager.wait_until_finished()

    print("\nTraining complete!")
    return state

# Create an absolute path for your checkpoints
ckpt_dir = os.path.abspath("./sz0_weights")
# Configure the manager (e.g., only keep the 5 most recent checkpoints)
options = ocp.CheckpointManagerOptions(max_to_keep=5, create=True)
# Initialize the manager using the modern API
checkpoint_manager = ocp.CheckpointManager(
    ckpt_dir, 
    options=options, 
    item_names=('state',)
)

if __name__ == "__main__":
    final_state = train()