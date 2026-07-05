"""Convert a fixed-batch SZ0 ONNX export to a dynamic-batch model.

Renames the leading dimension of all graph inputs/outputs to the symbolic
name "batch" (which the engine pins at load time via ORT's free-dimension
override — see NNEvaluator's fixed_batch parameter), and patches Reshape
shape constants that hard-code the export batch size.

Usage: python make_dynamic_batch.py input.onnx output.onnx

Verify the result against the source model before use, e.g. batch-8 outputs
of the dynamic model vs one-at-a-time outputs of the original (they matched
bit-exactly for the epoch-27 fp32 export with graph optimization disabled).
"""

import argparse

import numpy as np
import onnx
import onnx.numpy_helper as nh


def make_dynamic(input_path: str, output_path: str) -> None:
    model = onnx.load(input_path)
    graph = model.graph

    for vi in list(graph.input) + list(graph.output):
        dim = vi.type.tensor_type.shape.dim[0]
        dim.ClearField("dim_value")
        dim.dim_param = "batch"
        print(f"dynamic batch dim: {vi.name}")

    # Reshape targets like {1, 64, 256} bake in the export batch size; a
    # leading 1 becomes -1 (inferred) unless the shape already uses -1.
    initializers = {i.name: i for i in graph.initializer}
    patched = 0
    for node in graph.node:
        if node.op_type != "Reshape":
            continue
        shape_name = node.input[1]
        init = initializers.get(shape_name)
        if init is None:
            continue
        arr = nh.to_array(init).copy()
        if arr.ndim == 1 and arr.size > 1 and arr[0] == 1 and (arr == -1).sum() == 0:
            arr[0] = -1
            init.CopyFrom(nh.from_array(arr.astype(np.int64), shape_name))
            patched += 1
    print(f"patched {patched} Reshape shape constants")

    # The fixed-batch export leaves internal value_info entries with the old
    # literal batch size; ORT then warns about output shape mismatches on
    # every inference. Drop them — ORT re-infers shapes at session load.
    stale = len(graph.value_info)
    del graph.value_info[:]
    print(f"stripped {stale} stale value_info entries")

    onnx.save(model, output_path)
    print(f"saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="fixed-batch .onnx model")
    parser.add_argument("output", help="path for the dynamic-batch .onnx model")
    args = parser.parse_args()
    make_dynamic(args.input, args.output)
