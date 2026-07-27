"""Where a training step's time actually goes.

Four configurations, all measured in one process against one architecture, so
the numbers are a within-run A/B rather than a comparison against some earlier
run. That matters: it stays valid while `d_ff` / `d_qk_head` / block count are
being changed underneath it, and it does not need a baseline from a net you no
longer have.

    py -3 src/nn/benchmark_train.py            # 100 steps, default arch
    py -3 src/nn/benchmark_train.py --steps 200 --d-ff 384,384,384,320,256,256,192,128,128,128

Read it as:

    dataloader   host-only cost of building batches, no GPU at all
    compute      GPU-only cost, one cached device-resident batch replayed
    current      the real loop as train.py runs it today
    pipelined    the same loop with the fixes (no per-step sync, prefetch)

If `compute` is close to `current`, the step is genuinely GPU-bound and the
pipeline work buys nothing. If `dataloader` is a large share and `pipelined`
lands near `compute`, the gap was serialization.
"""

import os

# Must precede the first jax import; matches train.py so the comparison holds.
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_gpu_enable_triton_gemm=false "
    "--xla_gpu_enable_cublaslt=true "
    "--xla_gpu_cublas_fallback=true "
    "--xla_gpu_enable_command_buffer=",
)

import argparse
import functools
import glob
import time

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from architecture import ShatranjNet
from dataloader import SparseInMemoryDataLoader, load_sparse_dataset, prefetch
from train import compute_loss


def make_step(donate):
    def step(state, batch):
        grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
        (loss, _aux), grads = grad_fn(state.params, state.apply_fn, batch)
        return state.apply_gradients(grads=grads), loss

    # donate_argnums lets XLA write the updated params and optimizer moments
    # into the buffers they came from instead of allocating a fresh set each
    # step. Safe because the caller always rebinds `state` to the result.
    return jax.jit(step, donate_argnums=(0,) if donate else ())


def parse_d_ff(text):
    if text is None:
        return None
    if "," in text:
        return tuple(int(x) for x in text.split(","))
    return int(text)


def build_state(args):
    model_kwargs = {}
    if args.blocks is not None:
        model_kwargs["num_layers"] = args.blocks
    d_ff = parse_d_ff(args.d_ff)
    if d_ff is not None:
        model_kwargs["d_ff"] = d_ff
    if args.d_qk_head is not None:
        model_kwargs["d_qk_head"] = args.d_qk_head

    model = ShatranjNet(**model_kwargs)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=1e-4, weight_decay=1e-4),
    )
    variables = model.init(
        jax.random.PRNGKey(42),
        jnp.zeros((1, 64), dtype=jnp.int32),
        jnp.zeros((1,), dtype=jnp.int32),
    )
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=variables["params"], tx=optimizer
    )
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    return state, model, n_params


def bench_dataloader(loader, steps):
    """Host cost alone: build batches and throw them away."""
    it = loader.get_batches()
    next(it)  # first batch pays for the epoch permutation
    t0 = time.perf_counter()
    n = 0
    for _ in it:
        n += 1
        if n >= steps:
            break
    return time.perf_counter() - t0, n


def bench_compute(state, batch, steps, donate):
    """GPU cost alone: replay one device-resident batch, no host work."""
    step = make_step(donate)
    device_batch = jax.device_put(batch)
    state, loss = step(state, device_batch)  # compile
    loss.block_until_ready()

    t0 = time.perf_counter()
    for _ in range(steps):
        state, loss = step(state, device_batch)
    loss.block_until_ready()
    return time.perf_counter() - t0


def bench_loop(state, loader, steps, donate, block_every, use_prefetch):
    """The real thing: dataloader plus GPU, with or without the fixes."""
    step = make_step(donate)
    source = loader.get_batches()
    if use_prefetch:
        source = prefetch(source, depth=3)

    first = next(source)
    state, loss = step(state, first)  # compile
    loss.block_until_ready()

    t0 = time.perf_counter()
    n = 0
    for batch in source:
        state, loss = step(state, batch)
        n += 1
        if block_every and n % block_every == 0:
            loss.block_until_ready()
        if n >= steps:
            break
    loss.block_until_ready()
    return time.perf_counter() - t0, n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=284)
    ap.add_argument("--blocks", type=int, default=None)
    ap.add_argument("--d-ff", type=str, default=None,
                    help="int, or comma-separated per-layer widths")
    ap.add_argument("--d-qk-head", type=int, default=None)
    ap.add_argument("--visit-temperature", type=float, default=1.0)
    ap.add_argument("--data-glob", type=str, default="*.data")
    args = ap.parse_args()

    files = sorted(glob.glob(args.data_glob))
    if not files:
        raise SystemExit(f"no data files matched {args.data_glob!r}")
    print(f"Data files: {len(files)}")

    state, _model, n_params = build_state(args)
    print(f"Parameters: {n_params:,}")
    print(f"Device: {jax.devices()[0]}")

    dataset = load_sparse_dataset(files)

    def new_loader():
        return SparseInMemoryDataLoader(
            dataset_dict=dataset,
            batch_size=args.batch_size,
            visit_temperature=args.visit_temperature,
        )

    steps = args.steps
    positions = steps * args.batch_size
    results = {}

    print(f"\nTiming {steps} steps x batch {args.batch_size} "
          f"= {positions:,} positions per configuration...\n")

    t, n = bench_dataloader(new_loader(), steps)
    results["dataloader (host only)"] = t * steps / max(n, 1)

    sample = next(new_loader().get_batches())
    results["compute (cached batch)"] = bench_compute(state, sample, steps, donate=False)
    results["compute + donate"] = bench_compute(state, sample, steps, donate=True)

    t, n = bench_loop(state, new_loader(), steps, donate=False,
                      block_every=1, use_prefetch=False)
    results["current (sync every step)"] = t * steps / max(n, 1)

    t, n = bench_loop(state, new_loader(), steps, donate=True,
                      block_every=100, use_prefetch=True)
    results["pipelined (donate+prefetch)"] = t * steps / max(n, 1)

    width = max(len(k) for k in results)
    print(f"{'configuration':<{width}}   {'sec':>7}  {'pos/s':>9}")
    print("-" * (width + 21))
    for name, secs in results.items():
        print(f"{name:<{width}}   {secs:7.2f}  {positions / secs:9,.0f}")

    compute = results["compute + donate"]
    current = results["current (sync every step)"]
    pipelined = results["pipelined (donate+prefetch)"]
    host = results["dataloader (host only)"]

    print()
    print(f"Host work is {100 * host / current:.0f}% of the current step time, "
          f"and {'fully' if host < compute else 'NOT'} hideable behind "
          f"{compute:.2f}s of compute.")
    print(f"Pipelining recovers {current - pipelined:+.2f}s "
          f"({100 * (current / pipelined - 1):+.0f}% throughput); "
          f"{pipelined - compute:.2f}s still sits above the compute floor.")
    if pipelined <= compute * 1.05:
        print("=> At the compute floor. Further gains need the model or the "
              "optimizer, not the input pipeline.")
    elif host > compute:
        print("=> Host-bound even when overlapped: the dataloader itself is "
              "the ceiling. Vectorize the per-sample Python loop next.")
    else:
        print("=> Gap remains above the floor with host work hidden — look at "
              "H2D transfer of the 6 MB of dense targets per batch.")


if __name__ == "__main__":
    main()
