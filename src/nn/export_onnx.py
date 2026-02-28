import os
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from flax.training import train_state
import optax
from jax2onnx import to_onnx

from architecture import ShatranjNet 

def export_jax_to_onnx(checkpoint_base_dir, step_to_load, output_onnx_path="shatranj_v1.onnx"):
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
        jax.ShapeDtypeStruct(shape=("b", 64), dtype=jnp.int32),
        jax.ShapeDtypeStruct(shape=("b",), dtype=jnp.int32)
    ]
    print("3. Exporting directly to ONNX...")
    to_onnx(
        forward_pass,
        input_signatures, 
        return_mode="file",
        output_path=output_onnx_path
    )
    
    print(f"Success! Universal ONNX model saved to: {output_onnx_path}")


if __name__ == "__main__":
    input_base_dir = os.path.abspath("./sz0_weights")
    step = 9 
    output_filename = "sz0_epoch9.onnx"
    
    print(f"Starting conversion for step {step} in {input_base_dir}...")
    export_jax_to_onnx(input_base_dir, step, output_filename)