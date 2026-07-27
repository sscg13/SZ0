"""Evaluate a checkpoint's loss on the current dataset, without training.

Training logs are not a good way to compare nets: each net's log comes from its
own random batches (single-batch noise is ~0.12 on total loss), and from
whatever data window it happened to be training on. This runs any checkpoint
over the *same* fixed sample of the *current* data, so numbers from different
runs are directly comparable — and it needs no training log, so it still works
for a run whose logs are gone.

Usage:
    python src/nn/validation.py --run-dir sz0_baseline
    python src/nn/validation.py --run-dir sz0_run3 --blocks 6 --step 28
    python src/nn/validation.py --run-dir sz0_10blocks --batches 2000

`--blocks` must match the checkpoint's architecture (architecture.py currently
defaults to 10; run3 nets are 6). Use the same --seed and --batches across
checkpoints so they see identical batches.
"""

import os

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_gemm=false "
    "--xla_gpu_enable_cublaslt=true "
    "--xla_gpu_cublas_fallback=true "
    "--xla_gpu_enable_command_buffer="
)

import argparse
import glob
import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.training import train_state

from architecture import ShatranjNet
from dataloader import SparseInMemoryDataLoader, load_sparse_dataset
from train import compute_loss


def parse_d_ff(text):
    """'256' -> 256; '384,384,320,...' -> tuple of per-layer widths."""
    if text is None:
        return None
    parts = [p for p in text.replace(" ", "").split(",") if p]
    return int(parts[0]) if len(parts) == 1 else tuple(int(p) for p in parts)


def summarise(name, values):
    """Mean and 95% CI of the mean, over per-batch losses."""
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    se = sd / math.sqrt(n)
    return f"{name}: {mean:.4f} +/- {1.96 * se:.4f} (sd {sd:.4f})"


def load_net(run_dir, blocks, step, d_ff=None, d_qk_head=None):
    """Restore a checkpoint into a TrainState. Returns (state, label)."""
    overrides = {}
    if blocks is not None:
        overrides["num_layers"] = blocks
    if d_ff is not None:
        overrides["d_ff"] = d_ff
    if d_qk_head is not None:
        overrides["d_qk_head"] = d_qk_head
    model = ShatranjNet(**overrides)
    variables = model.init(
        jax.random.PRNGKey(42),
        jnp.zeros((1, 64), dtype=jnp.int32),
        jnp.zeros((1,), dtype=jnp.int32),
    )
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=1e-4, weight_decay=1e-4),
        ),
    )
    nparams = sum(x.size for x in jax.tree_util.tree_leaves(state.params))

    manager = ocp.CheckpointManager(
        os.path.abspath(run_dir),
        options=ocp.CheckpointManagerOptions(create=False),
        item_names=("state",),
    )
    step = step if step is not None else manager.latest_step()
    if step is None:
        raise SystemExit(f"no checkpoints found in {run_dir}")
    state = manager.restore(
        step, args=ocp.args.Composite(state=ocp.args.StandardRestore(state))
    )["state"]
    ff = (model.d_ff if isinstance(model.d_ff, int)
          else ",".join(str(w) for w in model.d_ff))
    qk = model.d_qk_head if model.d_qk_head is not None else (
        model.d_model // model.num_heads)
    label = (f"{run_dir} step {step} | {model.num_layers} blocks, "
             f"d_ff {ff}, d_qk_head {qk} | {nparams:,} params")
    return state, label


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True,
                    help="orbax checkpoint directory")
    ap.add_argument("--step", type=int, default=None,
                    help="checkpoint step to load (default: latest)")
    ap.add_argument("--blocks", type=int, default=None,
                    help="num_layers of the checkpoint (default: "
                         "architecture.py's current default)")
    ap.add_argument("--d-ff", default=None,
                    help="d_ff of the checkpoint: one int, or a comma-"
                         "separated per-layer list (default: architecture.py's "
                         "current default)")
    ap.add_argument("--d-qk-head", type=int, default=None,
                    help="QK head dim of the checkpoint (default: "
                         "architecture.py's current default)")
    ap.add_argument("--vs", default=None, metavar="RUN_DIR",
                    help="second checkpoint; reports the PAIRED per-batch "
                         "delta (this minus --vs), which cancels batch "
                         "difficulty and is far tighter than comparing two "
                         "separate runs' means")
    ap.add_argument("--vs-step", type=int, default=None)
    ap.add_argument("--vs-blocks", type=int, default=None)
    ap.add_argument("--vs-d-ff", default=None)
    ap.add_argument("--vs-d-qk-head", type=int, default=None)
    ap.add_argument("--batches", type=int, default=1000,
                    help="batches of 284 to average over (default 1000)")
    ap.add_argument("--seed", type=int, default=0,
                    help="batch-sampling seed; keep fixed across checkpoints")
    ap.add_argument("--visit-temperature", type=float, default=1.0)
    args = ap.parse_args()

    state_a, label_a = load_net(args.run_dir, args.blocks, args.step,
                                parse_d_ff(args.d_ff), args.d_qk_head)
    print(label_a)
    eval_a = jax.jit(
        lambda params, batch: compute_loss(params, state_a.apply_fn, batch)
    )

    state_b = eval_b = None
    if args.vs is not None:
        state_b, label_b = load_net(args.vs, args.vs_blocks, args.vs_step,
                                    parse_d_ff(args.vs_d_ff), args.vs_d_qk_head)
        print(f"  vs  {label_b}")
        eval_b = jax.jit(
            lambda params, batch: compute_loss(params, state_b.apply_fn, batch)
        )

    dataset = load_sparse_dataset(glob.glob("*.data"))
    loader = SparseInMemoryDataLoader(
        dataset_dict=dataset, batch_size=284,
        visit_temperature=args.visit_temperature,
    )

    names = ("Total ", "Policy", "Value ", "WDL   ", "Q     ")
    series_a = [[] for _ in names]
    deltas = [[] for _ in names]

    # Seed numpy so get_batches' permutation is identical across invocations.
    np.random.seed(args.seed)
    for i, batch in enumerate(loader.get_batches()):
        if i >= args.batches:
            break
        total, rest = eval_a(state_a.params, batch)
        va = (float(total),) + tuple(float(x) for x in rest)
        for series, v in zip(series_a, va):
            series.append(v)
        if eval_b is not None:
            total_b, rest_b = eval_b(state_b.params, batch)
            vb = (float(total_b),) + tuple(float(x) for x in rest_b)
            for series, a, b in zip(deltas, va, vb):
                series.append(a - b)

    n = len(series_a[0])
    print(f"averaged over {n} batches ({n * 284:,} positions), "
          f"seed {args.seed}")
    for name, series in zip(names, series_a):
        print("  " + summarise(name, series))

    if eval_b is not None:
        print("\npaired delta (this minus --vs; negative = this is better):")
        for name, series in zip(names, deltas):
            print("  " + summarise(name, series))


if __name__ == "__main__":
    main()
