import numpy as np
import onnxruntime as ort
import jax
import jax.numpy as jnp
import os

from architecture import ShatranjNet 
import orbax.checkpoint as ocp
from flax.training import train_state
import optax

def verify_models(seed, checkpoint_base_dir, step_to_load, onnx_path="sz0_small_epoch9.onnx"):
    print("1. Loading JAX Model...")
    model = ShatranjNet()
    dummy_board = jnp.zeros((1, 64), dtype=jnp.int32)
    dummy_halfmove = jnp.zeros((1,), dtype=jnp.int32)
    
    variables = model.init(jax.random.PRNGKey(0), dummy_board, dummy_halfmove)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=1e-4))
    abstract_state = train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=optimizer)
    
    checkpoint_manager = ocp.CheckpointManager(checkpoint_base_dir)
    restored_dict = checkpoint_manager.restore(step_to_load, args=ocp.args.Composite(state=ocp.args.StandardRestore(abstract_state)))
    loaded_params = restored_dict['state'].params

    # --- Generate random test inputs ---
    np.random.seed(seed)
    test_board = np.random.randint(0, 13, size=(1, 64), dtype=np.int32)
    test_halfmove = np.random.randint(0, 50, size=(1,), dtype=np.int32)

    print("2. Running JAX Inference...")
    jax_policy, jax_value = model.apply({'params': loaded_params}, test_board, test_halfmove)
    jax_policy_np = np.array(jax_policy)
    jax_value_np = np.array(jax_value)

    print("3. Loading ONNX Model...")
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # Dynamically grab input and output names so we don't have to guess
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    print(f"   ONNX Input names: {input_names}")
    print(f"   ONNX Output names: {output_names}")

    print("4. Running ONNX Inference...")
    # Map our numpy arrays to the ONNX input names
    ort_inputs = {
        input_names[0]: test_board,
        input_names[1]: test_halfmove
    }
    
    onnx_outputs = session.run(None, ort_inputs)
    
    # We need to figure out which ONNX output is the policy and which is the value
    # Usually, we can tell by the shape: Policy is likely (1, 64, 64) and Value is (1, 3)
    if onnx_outputs[0].shape == jax_policy_np.shape:
        onnx_policy_np, onnx_value_np = onnx_outputs[0], onnx_outputs[1]
    else:
        onnx_policy_np, onnx_value_np = onnx_outputs[1], onnx_outputs[0]

    print("5. Comparing Results...")
    policy_diff = np.max(np.abs(jax_policy_np - onnx_policy_np))
    value_diff = np.max(np.abs(jax_value_np - onnx_value_np))

    print(f"   Max Policy Difference: {policy_diff}")
    print(f"   Max Value Difference:  {value_diff}")

    print("Do the models pick the same top move?", np.argmax(jax_policy_np) == np.argmax(onnx_policy_np))

if __name__ == "__main__":
    verify_models(2349, os.path.abspath("./sz0_small_run1"), 9, "sz0_small_epoch9.onnx")