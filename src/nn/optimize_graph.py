"""Offline ONNX rewrites that ORT does NOT apply on the CUDA provider.

Measured on the specialized graph (see inspect_graph.py): the CPU provider
folds the attention scale into FusedMatMul and fuses softmax, but the CUDA
provider does neither — it leaves the 1/sqrt(head_dim) Div as its own pass over
the scores tensor, and runs softmax as ReduceMax/Sub/Exp/ReduceSum/Div. On a
bandwidth-bound net the scores tensor is the largest thing in the block
(batch 284, 8 heads, seq 64 => 18.6 MB in fp16), so each avoidable pass over it
is worth real throughput.

  fold-attn-scale  `(q @ k) / sqrt(head_dim)` -> scale folded into the Q
                   projection Gemm's alpha/beta. Algebraically identical,
                   removes one full read+write of the scores tensor per block.

Usage:
    python src/nn/optimize_graph.py in.onnx out.onnx --verify

--verify runs both graphs on random inputs under onnxruntime (CPU) and reports
the max relative output difference; expect ~1e-5 from fp16 rounding order.
"""

import argparse
import collections

import numpy as np
import onnx
from onnx import numpy_helper


def node_by_output(graph):
    return {o: n for n in graph.node for o in n.output}


def consumer_count(graph):
    counts = collections.Counter()
    for n in graph.node:
        for i in n.input:
            counts[i] += 1
    for o in graph.output:
        counts[o.name] += 1
    return counts


def _trace_to_gemm(graph, producers, name, depth=6):
    """Walk back through shape-only ops to the Gemm that produced `name`."""
    for _ in range(depth):
        node = producers.get(name)
        if node is None:
            return None
        if node.op_type == "Gemm":
            return node
        if node.op_type not in ("Reshape", "Transpose"):
            return None
        name = node.input[0]
    return None


def fold_attention_scale(graph):
    """Fold `MatMul -> Div(const)` into the Q-side Gemm's alpha/beta."""
    inits = {i.name: i for i in graph.initializer}
    folded = 0

    for node in list(graph.node):
        if node.op_type != "Div" or len(node.input) != 2:
            continue
        divisor = inits.get(node.input[1])
        if divisor is None:
            continue
        value = numpy_helper.to_array(divisor)
        if value.size != 1:
            continue
        scale = 1.0 / float(value.reshape(-1)[0])

        producers = node_by_output(graph)
        matmul = producers.get(node.input[0])
        if matmul is None or matmul.op_type != "MatMul":
            continue
        if consumer_count(graph)[matmul.output[0]] != 1:
            continue
        gemm = _trace_to_gemm(graph, producers, matmul.input[0])
        if gemm is None:
            continue

        # Gemm: Y = alpha*(A@B) + beta*C. Scaling both scales Y exactly.
        attrs = {a.name: a for a in gemm.attribute}
        alpha = attrs["alpha"].f if "alpha" in attrs else 1.0
        beta = attrs["beta"].f if "beta" in attrs else 1.0
        for name, new in (("alpha", alpha * scale), ("beta", beta * scale)):
            if name in attrs:
                attrs[name].f = new
            else:
                gemm.attribute.append(onnx.helper.make_attribute(name, new))

        for other in graph.node:
            other.input[:] = [matmul.output[0] if i == node.output[0] else i
                              for i in other.input]
        for out in graph.output:
            if out.name == node.output[0]:
                matmul.output[0] = out.name
        graph.node.remove(node)
        folded += 1

    return folded


def strip_unused_initializers(graph):
    used = {i for n in graph.node for i in n.input}
    dropped = 0
    for init in list(graph.initializer):
        if init.name not in used:
            graph.initializer.remove(init)
            dropped += 1
    return dropped


def verify(path_a, path_b, batch=4, trials=3):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sa = ort.InferenceSession(path_a, opts, providers=["CPUExecutionProvider"])
    sb = ort.InferenceSession(path_b, opts, providers=["CPUExecutionProvider"])

    dim = sa.get_inputs()[0].shape[0]
    if isinstance(dim, int):
        batch = dim

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(trials):
        feed = {
            sa.get_inputs()[0].name:
                rng.integers(0, 13 * 64, size=(batch, 64), dtype=np.int32),
            sa.get_inputs()[1].name:
                rng.integers(0, 140, size=(batch,), dtype=np.int32),
        }
        for x, y in zip(sa.run(None, feed), sb.run(None, feed)):
            denom = max(float(np.abs(x).max()), 1e-6)
            worst = max(worst, float(np.abs(x - y).max()) / denom)
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    model = onnx.load(args.input)
    before = collections.Counter(n.op_type for n in model.graph.node)

    folded = fold_attention_scale(model.graph)
    dropped = strip_unused_initializers(model.graph)

    onnx.checker.check_model(model)
    onnx.save(model, args.output)

    after = collections.Counter(n.op_type for n in model.graph.node)
    print(f"{sum(before.values())} -> {sum(after.values())} nodes "
          f"({folded} scale folds, {dropped} initializers dropped)")

    if args.verify:
        print(f"max relative output difference: "
              f"{verify(args.input, args.output):.2e}")


if __name__ == "__main__":
    main()
