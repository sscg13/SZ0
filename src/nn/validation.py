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


def summarise(name, values):
    """Mean and 95% CI of the mean, over per-batch losses."""
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    se = sd / math.sqrt(n)
    return f"{name}: {mean:.4f} +/- {1.96 * se:.4f} (sd {sd:.4f})"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True,
                    help="orbax checkpoint directory")
    ap.add_argument("--step", type=int, default=None,
                    help="checkpoint step to load (default: latest)")
    ap.add_argument("--blocks", type=int, default=None,
                    help="num_layers of the checkpoint (default: "
                         "architecture.py's current default)")
    ap.add_argument("--batches", type=int, default=1000,
                    help="batches of 284 to average over (default 1000)")
    ap.add_argument("--seed", type=int, default=0,
                    help="batch-sampling seed; keep fixed across checkpoints")
    ap.add_argument("--visit-temperature", type=float, default=1.0)
    args = ap.parse_args()

    model = (ShatranjNet() if args.blocks is None
             else ShatranjNet(num_layers=args.blocks))
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
        os.path.abspath(args.run_dir),
        options=ocp.CheckpointManagerOptions(create=False),
        item_names=("state",),
    )
    step = args.step if args.step is not None else manager.latest_step()
    if step is None:
        raise SystemExit(f"no checkpoints found in {args.run_dir}")
    state = manager.restore(
        step, args=ocp.args.Composite(state=ocp.args.StandardRestore(state))
    )["state"]
    print(f"{args.run_dir} step {step} | {model.num_layers} blocks | "
          f"{nparams:,} params")

    dataset = load_sparse_dataset(glob.glob("*.data"))
    loader = SparseInMemoryDataLoader(
        dataset_dict=dataset, batch_size=284,
        visit_temperature=args.visit_temperature,
    )

    eval_loss = jax.jit(
        lambda params, batch: compute_loss(params, state.apply_fn, batch)
    )

    # Seed numpy so get_batches' permutation is identical across checkpoints.
    np.random.seed(args.seed)
    totals, pols, vals, wdls, qs = [], [], [], [], []
    for i, batch in enumerate(loader.get_batches()):
        if i >= args.batches:
            break
        total, (pol, val, wdl, q) = eval_loss(state.params, batch)
        totals.append(float(total))
        pols.append(float(pol))
        vals.append(float(val))
        wdls.append(float(wdl))
        qs.append(float(q))

    print(f"averaged over {len(totals)} batches "
          f"({len(totals) * 284:,} positions), seed {args.seed}")
    for name, series in (("Total ", totals), ("Policy", pols),
                         ("Value ", vals), ("WDL   ", wdls), ("Q     ", qs)):
        print("  " + summarise(name, series))


if __name__ == "__main__":
    main()
