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


def _step(state, batch):
    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (loss, _aux), grads = grad_fn(state.params, state.apply_fn, batch)
    return state.apply_gradients(grads=grads), loss


# Cached so the four configurations share two compilations instead of forcing
# four. donate_argnums lets XLA write the updated params and optimizer moments
# into the buffers they came from rather than allocating a fresh set each step;
# the donated `state` is destroyed by the call, which is why every benchmark
# below gets its own.
@functools.lru_cache(maxsize=None)
def make_step(donate):
    return jax.jit(_step, donate_argnums=(0,) if donate else ())


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
    model_kwargs["dtype"] = {"f32": jnp.float32,
                             "bf16": jnp.bfloat16}[args.dtype]

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


def bench_dataloader(loader, steps, warmup):
    """Host cost alone: build batches and throw them away.

    The warmup matters more here than anywhere else: the first `next()` pays for
    a full permutation of the dataset (tens of millions of elements), and the
    first batches also fault in pages of the sample arrays.
    """
    it = loader.get_batches()
    for _ in range(warmup):
        next(it)
    t0 = time.perf_counter()
    n = 0
    for _ in it:
        n += 1
        if n >= steps:
            break
    return (time.perf_counter() - t0) / max(n, 1)


def bench_compute(state, batch, steps, donate, warmup):
    """GPU cost alone: replay one device-resident batch, no host work."""
    step = make_step(donate)
    device_batch = jax.device_put(batch)
    for _ in range(warmup):
        state, loss = step(state, device_batch)
    loss.block_until_ready()

    t0 = time.perf_counter()
    for _ in range(steps):
        state, loss = step(state, device_batch)
    loss.block_until_ready()
    return (time.perf_counter() - t0) / steps


def bench_loop(state, loader, steps, donate, block_every, use_prefetch, warmup):
    """The real thing: dataloader plus GPU, with or without the fixes."""
    step = make_step(donate)
    source = loader.get_batches()
    if use_prefetch:
        source = prefetch(source, depth=3)

    for _ in range(warmup):
        state, loss = step(state, next(source))
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
    return (time.perf_counter() - t0) / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10,
                    help="steps discarded before timing each configuration; "
                         "covers JIT compilation, the dataset permutation, and "
                         "first-touch page faults (default 10)")
    ap.add_argument("--batch-size", type=int, default=284)
    ap.add_argument("--blocks", type=int, default=None)
    ap.add_argument("--d-ff", type=str, default=None,
                    help="int, or comma-separated per-layer widths")
    ap.add_argument("--d-qk-head", type=int, default=None)
    ap.add_argument("--dtype", choices=("f32", "bf16"), default="f32",
                    help="trunk compute dtype; weights stay f32 either way "
                         "(default f32, i.e. current behaviour)")
    ap.add_argument("--visit-temperature", type=float, default=1.0)
    ap.add_argument("--data-glob", type=str, default="*.data")
    args = ap.parse_args()

    files = sorted(glob.glob(args.data_glob))
    if not files:
        raise SystemExit(f"no data files matched {args.data_glob!r}")
    print(f"Data files: {len(files)}")

    _probe, _model, n_params = build_state(args)
    del _probe
    print(f"Parameters: {n_params:,}")
    print(f"Device: {jax.devices()[0]}")

    dataset = load_sparse_dataset(files)

    def new_loader():
        return SparseInMemoryDataLoader(
            dataset_dict=dataset,
            batch_size=args.batch_size,
            visit_temperature=args.visit_temperature,
        )

    # A donated state is destroyed by the call that consumes it, so every
    # configuration needs its own. The PRNG key is fixed, so all four start from
    # identical weights.
    def fresh_state():
        return build_state(args)[0]

    steps, warmup = args.steps, args.warmup
    per_step_positions = args.batch_size
    results = {}

    print(f"\n{steps} timed steps x batch {args.batch_size} per configuration "
          f"({warmup} warmup steps discarded)...\n")

    results["dataloader (host only)"] = bench_dataloader(
        new_loader(), steps, warmup)

    sample = next(new_loader().get_batches())
    results["compute (cached batch)"] = bench_compute(
        fresh_state(), sample, steps, donate=False, warmup=warmup)
    results["compute + donate"] = bench_compute(
        fresh_state(), sample, steps, donate=True, warmup=warmup)

    results["current (sync every step)"] = bench_loop(
        fresh_state(), new_loader(), steps, donate=False,
        block_every=1, use_prefetch=False, warmup=warmup)

    # Isolates the sync from the thread. JAX dispatches asynchronously, so
    # dropping the per-step block may already let the host build the next batch
    # while the GPU runs the current one — in which case the prefetch thread is
    # redundant and should not be carried.
    results["no per-step sync only"] = bench_loop(
        fresh_state(), new_loader(), steps, donate=False,
        block_every=100, use_prefetch=False, warmup=warmup)

    results["+ prefetch thread"] = bench_loop(
        fresh_state(), new_loader(), steps, donate=False,
        block_every=100, use_prefetch=True, warmup=warmup)

    width = max(len(k) for k in results)
    print(f"{'configuration':<{width}}   {'ms/step':>8}  {'pos/s':>9}  "
          f"{'sec/100':>8}")
    print("-" * (width + 32))
    for name, secs in results.items():
        print(f"{name:<{width}}   {1000 * secs:8.1f}  "
              f"{per_step_positions / secs:9,.0f}  {100 * secs:8.2f}")

    compute = min(results["compute (cached batch)"], results["compute + donate"])
    current = results["current (sync every step)"]
    nosync = results["no per-step sync only"]
    pipelined = min(nosync, results["+ prefetch thread"])
    host = results["dataloader (host only)"]

    print()
    print(f"Host work is {100 * host / current:.0f}% of the current step, and "
          f"{'fully' if host < compute else 'NOT'} hideable behind "
          f"{1000 * compute:.1f} ms of compute.")
    print(f"Pipelining: {1000 * current:.1f} -> {1000 * pipelined:.1f} ms/step "
          f"({100 * (current / pipelined - 1):+.0f}% throughput); "
          f"{1000 * (pipelined - compute):.1f} ms still above the compute "
          f"floor.")
    thread_gain = nosync - results["+ prefetch thread"]
    if thread_gain < 0.02 * nosync:
        print(f"The prefetch thread adds {1000 * thread_gain:+.1f} ms — inside "
              f"noise. Async dispatch already overlaps the dataloader; drop the "
              f"thread.")
    else:
        print(f"The prefetch thread is worth {1000 * thread_gain:.1f} ms/step "
              f"beyond removing the sync; keep it.")

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
