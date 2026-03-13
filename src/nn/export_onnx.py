import os
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from flax.training import train_state
import optax
from jax2onnx import to_onnx
from onnxruntime.transformers.optimizer import optimize_model

from architecture import ShatranjNet 


def export_jax_to_onnx(checkpoint_base_dir, step_to_load, output_onnx_path, batch_size, convert_fp16):
    print("1. Loading Orbax Checkpoint...")
    model = ShatranjNet()
    
    dummy_board = jnp.zeros((1, 64), dtype=jnp.int32)
    dummy_halfmove = jnp.zeros((1,), dtype=jnp.int32)
    variables = model.init(jax.random.PRNGKey(0), dummy_board, dummy_halfmove)
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=1e-4, weight_decay=1e-4)
    )
    
    abstract_state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer
    )
    
    checkpoint_manager = ocp.CheckpointManager(checkpoint_base_dir)
    
    restored_dict = checkpoint_manager.restore(
        step_to_load, 
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(abstract_state)
        )
    )
    
    loaded_params = restored_dict['state'].params

    print("2. Wrapping JAX function...")
    def forward_pass(board, halfmove):
        return model.apply({'params': loaded_params}, board, halfmove)

    input_signatures = [
        jax.ShapeDtypeStruct(shape=(batch_size, 64), dtype=jnp.int32),
        jax.ShapeDtypeStruct(shape=(batch_size,), dtype=jnp.int32)
    ]
    print("3. Exporting to ONNX...")
    to_onnx(
        forward_pass,
        input_signatures, 
        return_mode="file",
        output_path=f"temp_{output_onnx_path}",
        opset=20
    )
    
    print("4. Optimizing...")
    optimized = optimize_model(
        f"temp_{output_onnx_path}",
        model_type='bert', # generic transformer
        num_heads=8,
        hidden_size=256
    )
    
    if (convert_fp16):
        optimized.convert_float_to_float16()
        
    optimized.save_model_to_file(output_onnx_path)
    print(f"Success! ONNX model saved to: {output_onnx_path}")

def get_latest_step(checkpoint_dir):
    steps = ocp.utils.checkpoint_steps(checkpoint_dir)
    if not steps:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    return max(steps)

if __name__ == "__main__":
    input_base_dir = os.path.abspath("./sz0_small_run1")
    
    try:
        latest_step = get_latest_step(input_base_dir)
        print(f"Detected latest step: {latest_step}")

        conversions = [
            {"batch": 1, "name": f"sz0_small_epoch{latest_step}.onnx", "fp16": False},
            {"batch": 284, "name": "sz0_small_batched.onnx", "fp16": True}
        ]

        for task in conversions:
            print(f"--- Starting conversion: Batch Size {task['batch']} ---")
            export_jax_to_onnx(
                input_base_dir, 
                latest_step, 
                task["name"], 
                task["batch"],
                task["fp16"]
            )
            print(f"Successfully exported to {task['name']}")

    except Exception as e:
        print(f"Error during conversion process: {e}")