"""Show what ORT actually executes, after it specializes a model to a batch size.

Counting nodes in the exported .onnx is misleading: the dynamic model keeps a
lot of Reshape/Transpose scaffolding that ORT constant-folds away at session
load once the symbolic "batch" dim is pinned (the engine does this via
AddFreeDimensionOverrideByName — see NNEvaluator's fixed_batch). Offline graph
rewrites that only tidy up what ORT would fold anyway buy nothing at runtime.

This dumps the post-optimization graph for a given batch size and prints the
op histogram, so optimization effort can target what survives specialization.

Usage:
    python src/nn/inspect_graph.py model.onnx                 # batch 32
    python src/nn/inspect_graph.py model.onnx --batch 284
    python src/nn/inspect_graph.py a.onnx --vs b.onnx         # compare two
    python src/nn/inspect_graph.py model.onnx --provider CUDAExecutionProvider

Note the CPU and CUDA providers do not enable the same fusion set — run with
--provider CUDAExecutionProvider on the GPU box before concluding that a fusion
does not fire in production.
"""

import argparse
import collections
import os
import tempfile

import onnx
import onnxruntime as ort


def specialized_histogram(path, batch, provider, keep=None):
    """Load under ORT with `batch` pinned; return the optimized op histogram."""
    available = ort.get_available_providers()
    if provider not in available:
        # ORT falls back to CPU silently, which silently invalidates the whole
        # comparison (e.g. fp16 Softmax has no CPU kernel and gets decomposed,
        # but runs fused on CUDA). Refuse rather than report a CPU graph
        # labelled as CUDA.
        raise SystemExit(
            f"{provider} is not available in this onnxruntime build.\n"
            f"  available: {', '.join(available)}\n"
            f"  for CUDA numbers install onnxruntime-gpu (the C++ engine's "
            f"ORT build is separate from this Python one).")

    out = keep or os.path.join(tempfile.mkdtemp(), "specialized.onnx")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.add_free_dimension_override_by_name("batch", batch)
    opts.optimized_model_filepath = out
    opts.log_severity_level = 3  # suppress the hardware-specific-model warning
    ort.InferenceSession(path, opts, providers=[provider])
    graph = onnx.load(out).graph
    return collections.Counter(n.op_type for n in graph.node), out


# Ops that only move or reinterpret data — no arithmetic.
MOVEMENT_OPS = {"Reshape", "Transpose", "Squeeze", "Unsqueeze", "Identity",
                "Concat", "Split", "Slice", "Gather", "Cast"}


# Fusions worth knowing about, and what each one collapses.
KNOWN_FUSIONS = {
    "Attention": "whole attention block",
    "MultiHeadAttention": "whole attention block",
    "BiasSoftmax": "bias add + softmax",
    "FusedMatMul": "matmul + scalar scale",
    "SkipLayerNormalization": "residual + layernorm",
    "QuickGelu": "activation",
    "FusedGemm": "gemm + activation",
}


def report(hist, label):
    total = sum(hist.values())
    movement = sum(c for op, c in hist.items() if op in MOVEMENT_OPS)
    print(f"{label}: {total} nodes, {movement} data-movement "
          f"({100 * movement / total:.0f}%)")
    for op, what in KNOWN_FUSIONS.items():
        if hist.get(op):
            print(f"  fused: {op} x{hist[op]} ({what})")
    whole = any(hist.get(op) for op in ("Attention", "MultiHeadAttention"))
    if not whole:
        print("  NOT fused: whole attention block — scores tensor is "
              "materialised in HBM")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--vs", default=None, help="second model to compare against")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--provider", default="CPUExecutionProvider")
    ap.add_argument("--raw", action="store_true",
                    help="histogram the file as exported, without loading it "
                         "into ORT — shows what the export produced vs what "
                         "each provider then does to it")
    ap.add_argument("--save", default=None,
                    help="write the specialized graph here (for Netron)")
    args = ap.parse_args()

    if args.raw:
        hist = collections.Counter(
            n.op_type for n in onnx.load(args.model).graph.node)
        print(f"=== {args.model} as exported (no ORT) ===")
        for op, count in sorted(hist.items(), key=lambda kv: -kv[1]):
            print(f"  {op:26s} {count:4d}")
        report(hist, "total")
        return

    hist, saved = specialized_histogram(args.model, args.batch, args.provider,
                                        args.save)
    if args.vs is None:
        print(f"=== {args.model} @ batch {args.batch} ({args.provider}) ===")
        for op, count in sorted(hist.items(), key=lambda kv: -kv[1]):
            print(f"  {op:26s} {count:4d}")
        report(hist, "total")
        if args.save:
            print(f"specialized graph written to {saved}")
        return

    other, _ = specialized_histogram(args.vs, args.batch, args.provider)
    print(f"=== batch {args.batch} ({args.provider}) ===")
    print(f"{'op':28s} {'A':>6s} {'B':>6s}")
    for op in sorted(set(hist) | set(other)):
        a, b = hist[op], other[op]
        flag = "   <-- differs" if a != b else ""
        print(f"  {op:26s} {a:6d} {b:6d}{flag}")
    print(f"  {'TOTAL':26s} {sum(hist.values()):6d} {sum(other.values()):6d}")
    report(hist, f"A {args.model}")
    report(other, f"B {args.vs}")


if __name__ == "__main__":
    main()
