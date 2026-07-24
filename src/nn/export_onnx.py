import argparse
import os
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import onnx

from flax.training import train_state
import optax
from jax2onnx import to_onnx
from onnxruntime.transformers.optimizer import optimize_model

from architecture import ShatranjNet
from make_dynamic_batch import make_dynamic


def export_jax_to_onnx(checkpoint_base_dir, step_to_load, output_onnx_path, batch_size, convert_fp16, dynamic=True):
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
        num_heads=model.num_heads,
        hidden_size=model.d_model
    )
    
    if (convert_fp16):
        optimized.convert_float_to_float16()
        
    optimized.save_model_to_file(output_onnx_path)

    print("5. Renaming outputs to stable names...")
    model = onnx.load(output_onnx_path)
    graph = model.graph
    # Outputs are in return order from ShatranjNet: (policy_logits, value_logits)
    stable_names = ["policy", "value"]
    for out_proto, new_name in zip(graph.output, stable_names):
        old_name = out_proto.name
        for node in graph.node:
            node.output[:] = [new_name if o == old_name else o for o in node.output]
        out_proto.name = new_name
    onnx.save(model, output_onnx_path)

    if dynamic:
        # Rewrite the (fixed) batch dimension to the symbolic name "batch",
        # which the engine pins to its actual batch size at session load via
        # ORT's free-dimension override. Export with batch_size=1 for this:
        # the Reshape-constant patching in make_dynamic_batch assumes a
        # leading literal 1, and batch-1 exports cannot hide the batch dim
        # inside fused constants (this path is verified bit-exact at batch 8
        # against the fixed export; re-verify after architecture changes).
        print("6. Converting to dynamic batch...")
        make_dynamic(output_onnx_path, output_onnx_path)

    print(f"Success! ONNX model saved to: {output_onnx_path}")

def get_latest_step(checkpoint_dir):
    steps = ocp.utils.checkpoint_steps(checkpoint_dir)
    if not steps:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    return max(steps)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Shatranj Zer0 JAX model to ONNX.")
    # Takes a positional argument, but defaults to your scratch folder if left blank
    parser.add_argument(
        "input_dir", 
        nargs="?", 
        default="./sz0_run2", 
        help="Path to the model directory (default: ./sz0_test_scratch)"
    )
    args = parser.parse_args()

    input_base_dir = os.path.abspath(args.input_dir)
    
    # Safely extract just the folder name (e.g., 'sz0_test_scratch' from '/path/to/sz0_test_scratch/')
    # rstrip ensures trailing slashes don't break the basename extraction
    dir_name = os.path.basename(input_base_dir.rstrip(os.sep))
    
    try:
        latest_step = get_latest_step(input_base_dir)
        print(f"Detected latest step: {latest_step} in {dir_name}")

        # One dynamic-batch file serves every batch size: the engine pins the
        # "batch" dim to searchbatchsize / datagenbatchsize / 1 at load (see
        # NNEvaluator fixed_batch). Must be exported at batch 1 — see the
        # make_dynamic note in export_jax_to_onnx.
        name = f"{dir_name}_epoch{latest_step}.onnx"
        export_jax_to_onnx(
            input_base_dir, latest_step, name,
            batch_size=1, convert_fp16=True
        )
        print(f"Successfully exported to {name}")

    except Exception as e:
        print(f"Error during conversion process: {e}")