import os

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_gemm=false "
    "--xla_gpu_enable_cublaslt=true "
    "--xla_gpu_cublas_fallback=true "
    "--xla_gpu_enable_command_buffer="
)
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

import argparse
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import time
import math
import orbax.checkpoint as ocp
import glob

from dataloader import load_sparse_dataset
from dataloader import SparseInMemoryDataLoader
from architecture import ShatranjNet

def compute_loss(params, apply_fn, batch):
    # --- HYPERPARAMETERS (Tune these!) ---
    VALUE_WEIGHT = 1  # Scales up Value Loss to compete with Policy Loss
    Q_WEIGHT = 48      # How much to care about MCTS Q vs Actual Game Outcome (WDL)
    # -------------------------------------

    halfmoves = batch['halfmoves'].squeeze(-1).astype(jnp.int32)
    policy_logits, value_logits = apply_fn({'params': params}, batch['boards'], halfmoves)
    
    # 1. Masked Policy Loss (Cross Entropy)
    batch_size = policy_logits.shape[0]
    flat_policy_logits = policy_logits.reshape((batch_size, 4096))
    MIN_LOGIT = -1e9
    masked_logits = jnp.where(batch['legal_mask'], flat_policy_logits, MIN_LOGIT)
    sample_policy_loss = optax.softmax_cross_entropy(masked_logits, batch['target_pi'])
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
    #combined_value_loss = q_mse_loss
    
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
def train(checkpoint_manager, visit_temperature=1.0):
    print("Initializing ShatranjNet Training...")
    print(f"Visit temperature: {visit_temperature}")
    
    # 1. Standard Initialization
    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    model = ShatranjNet()
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=1e-4, weight_decay=1e-4)
    )
    
    dummy_boards = jnp.zeros((1, 64), dtype=jnp.int32)
    dummy_halfmoves = jnp.zeros((1,), dtype=jnp.int32)
    variables = model.init(init_key, dummy_boards, dummy_halfmoves)
    
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer,
    )
    
    print(f"Model parameters: {sum(x.size for x in jax.tree_util.tree_leaves(state.params)):,}")
    
    latest_step = checkpoint_manager.latest_step()
    if latest_step is not None:
        print("Weight sum BEFORE load:", jnp.sum(jax.tree_util.tree_leaves(state.params)[0]))
    
        print(f"Resuming from global step {latest_step}...")
        restored = checkpoint_manager.restore(
            latest_step,
            args=ocp.args.Composite(state=ocp.args.StandardRestore(state))
        )
        state = restored['state']
        global_offset = latest_step
        
        print("Weight sum AFTER load:", jnp.sum(jax.tree_util.tree_leaves(state.params)[0]))
    else:
        print("No checkpoint found. Starting fresh.")
        global_offset = 0
    
    print("Starting Training Loop...\n")
    
    # Load files
    data_files = glob.glob("*.data")
    full_dataset = load_sparse_dataset(data_files)
    dataloader = SparseInMemoryDataLoader(
        dataset_dict=full_dataset,
        batch_size=284,
        visit_temperature=visit_temperature
    )
    
    # --- DIAGNOSTIC CHECK ---
    # Get one single batch to test the network BEFORE the optimizer touches it
    test_iterator = dataloader.get_batches()
    test_batch = next(test_iterator)
    
    # Check the data integrity
    print("--- DATA INTEGRITY CHECK ---")
    print(f"Boards shape: {test_batch['boards'].shape}, Sum: {test_batch['boards'].sum()}")
    print(f"Target PI shape: {test_batch['target_pi'].shape}, Sum: {test_batch['target_pi'].sum()}")
    print(f"Target Z shape: {test_batch['target_z'].shape}, Sum: {test_batch['target_z'].sum()}")
    
    # Check the true checkpoint loss
    init_loss, (init_p, init_v, init_wdl, init_q) = compute_loss(state.params, state.apply_fn, test_batch)
    print("\n--- PRE-TRAIN LOSS CHECK ---")
    print(f"True Checkpoint Loss: {init_loss:.4f} [Pol: {init_p:.4f} | Val: {init_v:.4f} (WDL: {init_wdl:.4f}, Q: {init_q:.4f})]")
    print("----------------------------\n")
    
    # Reset dataloader so we don't skip this batch
    dataloader = SparseInMemoryDataLoader(
        dataset_dict=full_dataset,
        batch_size=284,
        visit_temperature=visit_temperature
    )
    
    batches = 300000
    local_step = 0
    start_time = time.time()
        
    for batch in dataloader.get_batches():
        if jnp.isnan(batch['target_pi']).any() or jnp.isnan(batch['target_z']).any():
            print(f"CRITICAL: NaN found in data at step {local_step}!")
            break

        state, loss, p_loss, v_loss, wdl_loss, q_loss = train_step(state, batch)
        loss.block_until_ready()

        local_step += 1
        
        if local_step % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Step {local_step:04d} | Total: {loss:.4f} "
                  f"[Pol: {p_loss:.4f} | Val: {v_loss:.4f} (WDL: {wdl_loss:.4f}, Q: {q_loss:.4f})] "
                  f"| Time: {elapsed:.2f}s")
            start_time = time.time()
        
        if local_step == batches:
            break

    print(f"Saving checkpoint for epoch {global_offset + 1}...")
    checkpoint_manager.save(
        step=global_offset + 1,
        args=ocp.args.Composite(
            state=ocp.args.StandardSave(state)
        )
    )

    checkpoint_manager.wait_until_finished()

    print("\nTraining complete!")
    return state

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Shatranj Zer0.")
    # Positional argument that defaults to your current hardcoded path
    parser.add_argument(
        "run_dir", 
        nargs="?", 
        default="./sz0_run2", 
        help="Path to the checkpoint directory for this training run (default: ./sz0_test_value)"
    )
    parser.add_argument(
        "--visit-temperature",
        type=float,
        default=1.0,
        help="Temperature applied to MCTS visit policy targets before training (default: 1.0)"
    )
    args = parser.parse_args()

    if not math.isfinite(args.visit_temperature) or args.visit_temperature < 0:
        parser.error("--visit-temperature must be finite and non-negative")

    # Create an absolute path for your checkpoints using the argument
    ckpt_dir = os.path.abspath(args.run_dir)
    print(f"Initializing checkpoint manager at: {ckpt_dir}")
    
    options = ocp.CheckpointManagerOptions(max_to_keep=None, create=True)
    checkpoint_manager = ocp.CheckpointManager(
        ckpt_dir, 
        options=options, 
        item_names=('state',)
    )
    
    final_state = train(checkpoint_manager, visit_temperature=args.visit_temperature)
