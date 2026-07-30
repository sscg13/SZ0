# Experiment log

Grouped by core area (search / data generation / NN architecture), from newest to oldest. 

Testing environment notes:
- **Local device (CPU):** ~250 nps per worker thread.
- **GPU server (L40S):** batch 32, fp16, ~40K nps. 
- Elo figures are joint round-robin fits unless stated; games average ~300
  ply, draw rates 50–60%.

Tests typically use either fixed-node or time control (6 sec, 0.1-0.5 sec increment).
Older tests have been using double back row randomization, newer tests use an unbalanced opening book.

Matches with access to raw cutechess logs are rescored with
[testing/pentanomial.py](testing/pentanomial.py), as games are played in pairs.

---

# Search

## Particle MCTS

Open follow-ups:

- more GPU tests (possibly higher time control)
- investigate whether most of the gain is attributed to Gumbel alone

### Batch size 64

Increased nps from 40k at batch size 32 to 50k, but was neutral in 100 games H2H.

### Collision merging (ParticleMerge) — negative result

Fold colliding rollouts into the owning in-flight evaluation (summed weight,
multiplicity capped at 4 per eval) instead of discard-and-retry.
Branch `particle-merge`, UCI check `ParticleMerge`, off by default.

Setup: 6 sec + 0.1 sec increment, batch 32, 200 games head-to-head vs base greedy.

| Engine | Elo |
|---|---|
| merge | −10.4 ± 32.3 (LOS 26.4%) |

Merging engages rarely (startpos probe: 351 merges / 50K collisions / 169K
rollouts) because visit-matching selection already counts in-flight virtual
visits and steers around pending leaves. Retained virtual visits do cut
collision spin ~33% (75K → 50K), but that saves worker CPU, not GPU
throughput, so Elo is neutral. 

### GPU time-control particle vs PUCT

Check that fixed-node results for particle scale to real GPU speeds.

Setup: 6 sec + 0.1 sec increment, 200 games per engine round robin.

| Engine | Elo | ± | Score | Draws |
|---|---|---|---|---|
| greedy (deterministic Gumbel) | +33 | 32 | 54.8% | 54.5% |
| eta100 (η=1.0 sampled) | +12 | 31 | 51.7% | 58.5% |
| cpuct-100 (PUCT, c=1.0) | −45 | 32 | 43.5% | 57.0% |

Ordering matches all local 400 node tests, but margins compress ~5×. 

Greedy adopted as default (`ParticleSearch=true, ParticleGreedy=true`) following this test.

### corrected particle greedy rule + CPuct tweak

Check fix for real Gumbel MuZero deterministic rule (argmax[π(a) − N(a)/(1+ΣN)]).
Also check if default CPuct is too exploration-heavy.

Setup: 400 nodes/move, 120 games per engine round robin.

| Engine | Elo |
|---|---|
| greedy (visit-matching rule) | +187 |
| eta100 (η=1.0 sampled) | +83 |
| base PUCT c=1.0 | −58 |
| base PUCT c=2.0 (old default) | −219 |

Greedy using corrected visit-matching rule is now best. 
Lower CPuct matters at least in short searches. 

### batched ladder: particle vs virtual losses

Batched version of sequential ladder 1 (below), with greedy removed.

Setup: batch 8, 400 nodes/move, 120 games per engine round robin.

| Engine | Elo |
|---|---|
| eta100 (η=1.0, weights ≡ 1) | +137 |
| particle η=1.5 | +67 |
| particle η=2.0 | −20 |
| base PUCT + virtual visits | −195 |

As in sequential ladder 1, many of these results are confounded by improved sampling.

### sequential ladder 1: first particle MCTS test

Setup: 400 nodes/move, 120 games per engine round robin.
η controls the importance weights (the UCI spin value is 100× the float η).
`greedy` here = plain argmax of the improved-policy logits (later discovered to be the wrong rule).

| Engine | Elo |
|---|---|
| eta100 | +144 |
| particle η=1.5 | +137 |
| eta200 | +83 |
| base PUCT (c=2.0) | −127 |
| greedy (plain argmax — degenerate) | −269 |

Many of these results are confounded by the concurrent change of improved sampling. 

---

# Data generation

Open follow-ups:

- combine the new (Gumbel greedy) selection with search contempt
- less nodes per datagen? most direct way to get faster datagen

## Search Contempt

- Replaced temperature-based move selection as the source of self-play diversity.
- Gained roughly 100 elo in the span of a few nets, but maybe confounded by standard self-play gains

---

# NN architecture / training

## Dynamic batch ONNX export (infrastructure)

Naive dynamic exports are up to ~10× slower on CUDA (reshape overhead).
The fix loads the dynamic export and optimizes it for constant batch size at runtime.

## What ORT actually executes (infrastructure)

Set `SZ0_DUMP_OPTIMIZED=<path>` to have the engine write the post-fusion graph,
then histogram it with `src/nn/inspect_graph.py --raw`. **Only measure this way**
— the pip `onnxruntime` is a different build (1.27, CPU-only) from the linked
C++ library (1.24, CUDA), and fusion differs by both provider and version. A
CPU-provider reading is actively misleading: fp16 Softmax has no CPU kernel, so
CPU decomposes it into ReduceMax/Sub/Exp/ReduceSum/Div and inserts ~130 Casts,
none of which happens on CUDA.

Production graph (10 blk, qk64, batch 32, CUDA, ORT 1.24): **322 nodes, 60%
data movement**. Already fused: `BiasSoftmax` ×10 (spatial bias + softmax),
`FusedMatMul` ×11 (attention scale folded into the matmul),
`SkipLayerNormalization` ×20, `QuickGelu` ×10. Offline rewrites for any of
these are redundant — tried and reverted.

Left on the table: whole-attention fusion never fires (no `MultiHeadAttention`,
so the scores tensor still round-trips through HBM), and 146 Reshape + 40
Transpose survive.

### CUDA graph capture — +10–15% search, adopted behind a flag

`SZ0_CUDA_GRAPH=1` (requires a pinned batch). ORT records the ~320 kernel
launches once and replays them as a single launch. Measured **+10–15% search
nps**, ≈ +5 Elo at TC — below the noise floor, so adopt on the nps measurement,
not on games.

Two blockers had to be cleared first:

- **`jnp.clip` in the model made capture impossible.** int32 `Clip` has no CUDA
  kernel, so ORT put both guards on the CPU provider and inserted
  `MemcpyFromHost`; capture requires *every* node on the CUDA EP. Removed from
  `architecture.py`, halfmove clamped in C++ instead (`clamp_halfmove`). Only
  findable via `SZ0_ORT_VERBOSE=1`, which logs node placement — the error
  message itself names no op.
- **Exit aborted** (`cudaGraphExecDestroy`: "driver shutting down"). The session
  global outlived CUDA's atexit handler; `nn.reset()` at the end of `uci()`
  fixes it. Pre-existing latent bug — `cudaStreamDestroy` was already failing on
  every GPU exit, ORT just logged instead of throwing.

Predicted 2.2× from a `c = 0.59 ms` fixed-cost fit; got 10–15%, so launch
overhead was only ~0.1–0.15 ms of it. Fitting a fixed cost says nothing about
what the cost *is* — see below for where it actually went.

### Where a batched inference call spends its time (`SZ0_TIME_IO=1`)

Per-phase timing of the captured-graph path, datagen batch 284, steady state
over 29K batches (very stable, ±2%):

| Phase | µs | Share |
|---|---|---|
| `run` (GPU) | 3550 | 72.5% |
| `d2h` (4.65 MB) | 370 | 7.6% |
| `copy` staging → temporary | 350 | 7.1% |
| `scatter` temporary → `shared_results` | 240 | 4.9% |
| `stage` malloc + zero-fill 4.65 MB | 245 | 5.0% |
| `results` zero-fill 4.66 MB | 127 | 2.6% |
| `h2d` (73 KB) | 20 | 0.4% |

**The transfer was never the problem.** 4.65 MB in 370 µs is 12.6 GB/s, already
respectable for pageable memory. The cost was the three redundant host passes
around it (`results` + `stage` + `copy` = 722 µs, ~2× the transfer): two
zero-fills of buffers that are overwritten immediately, and a copy into a
temporary that exists only to be scattered out of.

Fix: one persistent pinned landing buffer allocated at setup, each caller
scattering directly out of it.

### Result: datagen 47K → 62K nps (+32%)

| Phase | before | after |
|---|---|---|
| `run` | 3550 | 3670 |
| `d2h` | 370 | 190 |
| `copy` | 350 | 0 |
| `scatter` | 240 | 230 |
| `results` + `stage` | 372 | 0 |
| host-side share | 27.5% | 10.7% |

Pinning delivered exactly as predicted (12.6 → 24.5 GB/s). The overall win did
not: predicted +15%, got +32%.

**The instrumentation timed allocation but not deallocation**, and the gap is
the difference. Measured total fell 4900 → 4115 µs (−785), but the full batch
cycle fell 6043 → 4581 µs (−1462). The missing 677 µs is destruction of the
three multi-MB vectors, which happened after the last timestamp: at those sizes
glibc uses `mmap`/`munmap`, so every batch returned pages to the kernel and
re-faulted them on the next allocation. Zero-fill was the visible half of that
cost; free plus page faults was the rest. Lesson: instrument the whole cycle,
not just the phases you suspect.

`run` rose 3.4% and drifts upward within a run (3596 → 3685) — same graph, same
batch, so most likely clock throttle from sustaining 32% more work. Unconfirmed.

`run` = 1684 MB / 3670 µs = **459 GB/s, 53% of the L40S's 864** — well above the
41% below, which was computed against the whole batch cycle rather than the GPU
phase. Memory bound is confirmed more strongly than that number suggested, and
it caps `run` headroom at ~1.9×.

### Parallelised the result scatter

`infer_packed` no longer scatters results into per-game structs — it exposes the
batch via `batch_policy()`/`batch_value()` and each worker reads its own row
during the parallel expand phase. Thread-0 serial time 4109 → ~3900 µs;
host-side share of the inference phase 10.7% → ~6%. Also removed a dead
zero-filled 4.65 MB vector from the non-graph path, and made that path hold its
ORT output tensors in a member so it needs no copy either. Both paths are now
zero-copy.

### Careful: nps is only comparable at equal nodes/move

The 62K figure was measured at **100** nodes/move; at 300 the same build reads
60K. Nodes/move changes tree depth, which changes `select()` path length and
`select_best_puct` scans — all in the parallel phase, which the `[io]` timer
does not cover. Reconstructed from nps: unmeasured (tree) work is ~466 µs/batch
at 100 nodes vs ~833 µs at 300. **Always state nodes/move alongside a datagen
nps number.**

`SZ0_TIME_IO=1` now also prints a `[dg]` line breaking the datagen loop into
select / infer / expand plus barrier wait, measured on thread 0. That covers the
phases the `[io]` timer structurally cannot.

Two gotchas when reading it:

- **`nps` counts NN evals, not iterations**, so `datagenbatchsize / cycle`
  overstates it. See batch fill below — the `[dg]` line reports fill and implied
  nps directly.
- **Set `SZ0_TIME_IO_EVERY` to a multiple of nodes/move.** The arena clears
  every root move, so tree depth sweeps shallow→deep with period exactly
  `nodecount`, and all games stay in lockstep (one rollout each per iteration).
  Reporting every 1000 at 300 nodes/move aliases that cycle and makes `select`
  swing 60–464 µs between windows. 3000 gives ten whole periods.

### Batch fill decay — the reason datagen nps falls during a run

Measured over one run at 300 nodes/move, 3000-iteration windows:

| | fill | implied nps | cycle |
|---|---|---|---|
| early | 284/284 | 64K | 4414 |
| mid | 266/284 | 58K | 4607 |
| late | 228/284 | 50K | 4526 |

Fill and nps track exactly (284/228 = 1.25 vs 64/50 = 1.28) while cycle time
barely moves — **the decay is the batch emptying, not anything getting slower.**

Cause: `select()` finishes a rollout itself when the leaf is terminal (bare
king, 70-move, repetition, stalemate), returning false, and the batch row was
left empty. Terminal leaves get commoner as games reach endgames, and shatranj's
bare-king and 70-move rules make them very common there. The captured graph runs
all 284 rows whether occupied or not, so by late run ~20% of every inference was
computing padding.

Fixed by retrying instead of yielding the slot: on a terminal leaf, roll out
again for that game until a slot is filled or the budget is exhausted (bounded —
the only false return increments `rollouts_completed`). Expected ~+20% at steady
state, no extra memory. Search semantics unchanged; RNG interleaving across
games differs, so data is statistically equivalent but not bit-identical.

**This also invalidates comparing nps readings taken at different points in a
run** — early readings are optimistic. Compare at equal fill.

### Arena sizing

Peak occupancy measured at **11211 / 65536 nodes** (300 nodes/move), i.e. ~37
children per expansion, with zero overflows. `datagenarenasize` cut to 32768
(2.9× headroom, ~800 nodes/move): 1536 → 768 KB per game, ~218 MB freed across
284 games. `expand()` had a silent failure mode — a full arena skips adding
children but still backprops, leaving a leaf that is re-selected forever — now
counted and warned about rather than degrading quality invisibly.

### Pipelined datagen (two game pools)

Inference was ~85% of the cycle with 7 of 8 workers parked at a barrier. It
cannot be overlapped within one pool (select for rollout N+1 depends on expand
of N), so there are now two pools: while the GPU evaluates pool p, the workers
do tree work for pool 1-p. Inference moved to its own thread; the middle barrier
became a per-pool submitted/completed handshake.

Measured **70K nps from 62K (+13%)**, at `gpu-bound 97%` — tree work is fully
hidden and the GPU is the only thing left on the critical path.

Verified correct: end-of-run peak arena 11020/32768, statistically identical to
the 11211 measured before the change, so trees are growing normally. (The check
mattered because a broken `expand` would have produced a fast-looking loop, a
perfect 284/284 fill, and worthless data.) `select` fell ~290 → ~50 µs and
`expand` ~180 → ~28 µs across this change. **This is a rate effect, not less
work:** identical peak arena means identical tree sizes, so `select` walks the
same trees; and all four CPU columns scaled by the same factor, which real work
reductions never do. Likely mechanism — the old design parked 7 workers on a
`std::barrier` (which spins before blocking) for ~85% of wall time, burning
power and ping-ponging one cache line while the GPU ran; the new design sleeps
them on a condition variable for 97%, leaving turbo headroom and an uncontended
memory subsystem for the short work burst.

**Cost: the end-of-run tail doubles.** 568 games are now in flight and all are
discarded partway when the position target is hit — ~29000 positions of wasted
compute against ~14500 before. Irrelevant on a 2M-position run (1.5%),
dominant on a 30K one, where it can outweigh the throughput gain entirely.
Datagen games average ~102 ply (294 games / 30064 positions), so the tail is
roughly 568 × 51 positions regardless of run length. Fix if short runs matter:
stop resetting completed games once the target is in reach and let the
in-flight ones drain.

Correctness measures, since a race here would silently corrupt training data
rather than crash:

- Everything a batch touches is per-pool (games, packed inputs, both index
  maps, the evaluator result slot). Only the output file and counters are
  shared.
- `NNEvaluator` has two pinned landing buffers. With one, a Run would overwrite
  results still being read by workers. Device buffers stay single — only one
  Run is ever in flight.
- Workers compare `completed` against a per-worker consumed count rather than
  trusting a wakeup, so a coalesced or spurious notify cannot read as fresh
  data.
- The batch→game index map is round-tripped and `abort()`s on mismatch before
  any result is consumed. A mis-scatter is the failure mode that would poison
  data invisibly.
- Single loop exit, read by all workers after a barrier, so all eight leave on
  the same iteration. No `arrive_and_drop` anywhere.
- Inference-thread exceptions fail both pools and release any blocked worker.

`[dg]` now reports `gpu_wait` — time workers spent blocked on results. High is
healthy (GPU-bound). If it approaches zero, tree work has become the bottleneck
and a third pool would buy nothing.

Also fixed: `total_nodes_evaluated` was `int`, overflowing at ~2.1B nodes.

Remaining after this: anything inside the ONNX graph (a few %); overlapping the
h2d/d2h copies needs async streams and a second capture — larger change,
smallest prize.

## sz0_run4

Fresh trains from scratch on the accumulated window, not iteratively refined.
127.5M positions (recent sliding window, older deleted); 1.8M steps × batch 284
≈ 4 epochs; ~10 h at 6 blocks, ~13.5 h at 10. `baseline` is the 6-block fresh
train.

**nps caveat:** every datagen figure below (95K, 63K, 59K, 55K, 47K) predates
the inference-path fixes and carries ~28% host-side overhead. They are
comparable to each other but *not* to anything measured after — the same qk64
net that read 47K reads 62K now. Architecture cost ratios between them still
hold; absolute numbers do not.

Open follow-ups:

- figure out how to close the ~67 Elo data/iterative gap (below) — 10× the
  noise floor, unlike any architecture tweak so far, will likely need to wait for
  run4 to make more headway on this
- increase data window and/or train longer (confounded — see below)
- reserve a small slice of datagen for representative validation
- datagen throughput budget — 60K at 300 nodes/move vs 95K originally (was 47K
  before the inference-path fixes, but that was measured at a different
  nodes/move — see the nps caveat below); decide explicitly rather than letting
  adoptions erode it further
- double-buffer datagen: two game pools alternating, so tree work hides under
  the next batch's GPU work. Ceiling ~73K nps, biggest remaining lever
- MoE: not optimistic (Leela failed; MoE adds params + traffic, the wrong
  trade for a memory-bound, data-limited net). Head specialization IS proven —
  NNUE output buckets are hard-routed head-MoE (hand-selected by piece count).
  For a data-limited net prefer a hand-designed router over a learned gate.
  Caveat: a deep trunk may already learn phase-conditioning implicitly, so less
  headroom than shallow NNUE. 
- per-layer `d_ff` schedule (wider in EARLY layers) — fixed-budget reallocation
  so ~neutral throughput, but the effect is sub-floor: needs multi-seed or a
  large reallocation to detect
- widen `d_model` — raises intensity but also traffic, 
  will be pretty costly for datagen, unless nodes can decrease
- confirm whether the 3.4% `run` rise is GPU clock throttle (`nvidia-smi -q -d
  CLOCK` during datagen)

### The baseline anchor is ~67 Elo below the iterative net

100 games, paired openings, fresh 6-block vs `run3_epoch28`:
−66.8 ± 36.4 pentanomial, LOS 0.01%.

Likely insufficient data — loss seems to saturate. 
See below, but validation indicates probably no major overfitting
Need more testing later, probably ~150-200M positions to match.

| Net | Current window | Old slice |
|---|---|---|
| baseline (6 blk) | 2.3993 | 2.2066 |
| 10 blocks | 2.3789 | 2.1951 |
| run3_epoch28 | 2.3236 | 2.1269 |

#### Step count alone can account for the gap (2026-07-30)

The dyn-bias run produced validation on all six 300k-step checkpoints, which is
the loss-vs-log(step) fit that was an open follow-up here. Dropping checkpoint 1
as the embedding transient:

| step | 600k | 900k | 1200k | 1500k | 1800k |
|---|---|---|---|---|---|
| total | 2.4259 | 2.4003 | 2.3763 | 2.3627 | 2.3538 |

A clean log law, **slope −0.0675 nats per e-fold of steps, no saturation.**
Extrapolating 1.8M → `run3_epoch28`'s 8.4M cumulative steps gives **−0.104
nats**, against a remaining gap of only 0.054 (doubleqk final 2.3779 vs run3
2.3236). Step count over-covers the gap on its own.

This is *not* the LR-schedule claim, which was already ruled out — run3 used the
same constant 1e-4, it just took 4.7× more steps to get there. It also
contradicts the earlier "loss saturates" reading above, which was a linear fit
inside the last 100k steps and badly under-powered next to six checkpoints
spanning 6× in steps.

Two caveats it cannot resolve by itself: the fit is on the dyn-bias run, which
has 11.5% more parameters and so more room to keep fitting; and 8.4M steps over
a 127.5M window is ~18.7 epochs, so on *training-window* loss "more steps" and
"fits the window better" are the same curve. Held-out validation now works, so
scoring these same six checkpoints out of sample settles it: if the slope
survives, it is undertraining and an end-of-run LR anneal is the cheap fix; if
it flattens, it is data and the anneal buys nothing.

Cheapest probe either way, far short of doubling a 13-hour run: cosine-decay the
LR over the last 200–300k steps.

### 6 → 10 transformer blocks — adopted

200 games head-to-head, paired openings, batch 32 fp16:

| Test | Score | Trinomial | Pentanomial | LOS |
|---|---|---|---|---|
| 6 sec + 0.1 sec | 0.5275 | +19.1 ± 31.2 | +19.1 ± 25.7 | 92.9% |
| 5000 nodes/move | 0.5475 | +33.1 ± 34.4 | +33.1 ± 24.8 | 99.6% |

Fixed node isolates evaluation quality, TC nets it against the speed loss; the
~14 Elo gap is the cost of 25% fewer nodes.

| Regime | Batch | nps 6 → 10 | Slowdown | Estimated overhead |
|---|---|---|---|---|
| Search | 32 | 40K → 30K | 1.33× | ~50% |
| Datagen | 284 | 95K → 59K | 1.61× | ~8% |

Adopted despite TC falling short of significance: datagen is fixed-node, so the
relevant figure is +33 against 38% fewer positions per hour. Caveats: one
training seed.

### Measurement floor: run-to-run variance ≈ 0.0065 nats ≈ ~10 Elo

"d_ff 512" checkpoint was mislabeled and actually `d_ff 256`.
Two architecturally identical 10-block nets (separate training runs, same step
count) differ by 0.0065 (roughly 10 Elo) in total validation loss. Future runs
will focus more on datagen throughput as the decider.

The one real `d_ff 512` net (also `qk 16`, so confounded) scores +0.0063 vs a `d_ff 256` net,
within the noise floor above, no quality signal. Datagen: `d_ff 512 + qk 16` runs 55K nps vs 59K
for `d_ff 256 + qk 32`, ~7% slower despite qk 16 helping, so `d_ff 512` alone
costs more than that. Default remains `d_ff 256`. 
`d_qk_head` kept as a parameter (default = `d_model // num_heads`).

### qk 32 → 16 — strongly negative, rejected

EfficientViT-style halving of the QK head dim (V untouched). 400 games, paired
openings, 6s+0.1, both vs the `d_ff 256 + qk 32` net:

| Test net | Pentanomial | LOS | Pairs |
|---|---|---|---|
| `d_ff 256` + `qk 16` (single variable) | −37.5 ± 16.7 | ~0% | `[3, 63, 109, 24, 1]` |
| `d_ff 512` + `qk 16` (confounded) | −5.2 ± 17.6 | 28% | `[1, 51, 103, 43, 2]` |

The EfficientViT result does not transfer. Likely because the spatial bias
already covers static geometry, leaving QK to carry *all* content-dependent
piece-piece relations — the sharp distinctions that decide games.

Loss predicted this. Paired validation on the single-variable pair gives
+0.0149 ± 0.0016 — 2.3× the noise floor, i.e. flagged as clearly bad before any
games. Loss is a usable screen between identically-trained fresh nets.

`d_ff 512` may partly compensate for lost QK capacity, but not enough it seems. 

### qk 32 → 64 — adopted

400 games, paired, 6s+0.1, vs `d_ff256 + qk32`: +20.9 ± 17.7 pentanomial
(+20.9 ± 23.8 trinomial), LOS 99.0%, pairs `[0, 43, 90, 67, 0]`. Paired loss
−0.0079 ± 0.0017. TC understates quality since qk64 is slower (~+27 Elo if
search loses ~12% nps). Datagen 59K → 47K (−20%).

Well into diminishing returns, but still marginally more elo / datagen cost 
compared to going from 6 blocks to 10 blocks.

Loss underestimated the change. −0.0079 is 1.2× the noise floor ("ambiguous") yet
the match was significant. Probably nat to elo is not always constant per change.
Datagen cost, now decomposable:

| Config | nps | vs baseline |
|---|---|---|
| `d_ff256 + qk16` | 63K | +6.8% |
| `d_ff256 + qk32` (baseline) | 59K | — |
| `d_ff512 + qk16` | 55K | −6.8% |
| `d_ff512 + qk32` (inferred) | ~52K | −12% |

Fitting `T = c + b·d_qk_head`, QK is only ~13% of datagen time at qk32;
extrapolating up gives qk48 ≈ 55.5K, qk64 ≈ 52K.

### Cosine FFN taper — no signal, costs throughput, rejected

Per-layer `d_ff` tapered wide→narrow (arXiv:2606.23670), `(384, 384, 384, 320,
256, 256, 192, 128, 128, 128)`, summing to 10×256 so parameters and FLOPs match
uniform `d_ff=256` exactly.

400 games paired, 6s+0.1, vs `d_ff256 + qk64`: **+6.9 ± 15.8 pentanomial**
(+6.9 ± 22.1 trinomial), LOS 80.6%, pairs `[0, 39, 114, 47, 0]`. Inside the
~10 Elo floor and the interval includes zero. **Zero WW and zero LL pairs in
200** — no opening either net could win from both sides, i.e. the two nets play
nearly identical chess.

Paired validation agrees: **Total +0.0002 ± 0.0018** — 0.03× the noise floor,
the flattest null the method can produce. Datagen **70K → 67K (−4.3%)**.
Reverted to uniform `d_ff=256`; `cosine_ff_schedule` kept for future reshaping.

**Third calibration point for loss screening, and the cleanest one.** Loss and
match now agree at all three: qk16 loss +0.0149 (2.3× floor) / match −37.5,
both clearly bad; qk64 loss −0.0079 (1.2× floor) / match +20.9, loss
understated; cosine taper loss +0.0002 / match +6.9 ± 15.8, both null. Loss has
not produced a false verdict — it under-calls on QK-dense axes but never
contradicts.

**Equal FLOPs is not equal time.** The reallocation is arithmetically neutral
but a 128-wide FFN GEMM is less efficient than a 256-wide one, and the 384-wide
layers do not make it back. Worth remembering for any future fixed-budget
reshaping: narrowing kernels costs even at constant FLOPs.

### Attention logit algebra — two ideas ruled out on paper

Expanding `q_i·k_j` with the projection biases (`q_i = W_Q x_i + b_Q`,
`k_j = W_K x_j + b_K`) gives four terms, and softmax over `j` discards anything
constant along `j`:

| term | depends on | survives softmax? |
|---|---|---|
| `(W_Q x_i)·(W_K x_j)` | i, j | yes — the real attention |
| `(W_Q x_i)·b_K` | i only | no |
| `b_Q·(W_K x_j)` | j only | yes — per-key content bias |
| `b_Q·b_K` | neither | no |

**Rejected: `QKᵀ + W1 X + (X W2)ᵀ + B` with W1/W2 shared across heads.** The
`W1 X` half is a per-query bias, which cancels identically — zero effect, zero
gradient. The `(X W2)ᵀ` half is a per-key linear functional of content, which
term 3 already provides: `b_Q·(W_K x_j) = (W_Kᵀ b_Q)·x_j`, and since
`d_qk 512 > d_model 256` that functional is unconstrained. Sharing across heads
makes the proposal strictly weaker than what every head has today.

**`b_K` is inert** — it appears only in the two cancelling terms, and `k` is used
nowhere else. Verified numerically (weights identical to 3e-16 even with a 10×
bias; `b_Q` control moves them 0.11). Confirmed in the trained net: `b_K` RMS
~2e-3 against `b_Q`'s ~1.5e-1, **50× smaller, uniformly across all 10 blocks**.
Not exactly zero despite a zero init, because softmax shift-invariance is exact
in ℝ but not in floating point — `b_K` picks up a rounding-noise gradient, and
Adam normalises by gradient magnitude, so even pure noise random-walks the
parameter at order the learning rate. Holds only because there is
no RoPE and no QK-norm — a position-dependent or nonlinear transform between
projection and dot product breaks the argument, which is why it is not general
MHA folklore. Left in place: removing it would break `StandardRestore` for every
existing checkpoint, for 0.09% of parameters. Revisit at run4 start.

### Dynamic attention bias (scaled-down Smolgen) — rank 4 trained, null match

Content-dependent 64×64 map added to the logits alongside the static
`spatial_bias`, so attention can react to the position rather than only to
geometry ([Leela](https://lczero.org/blog/2024/02/transformer-progress/)).
Shared across heads within a layer; **not** shared across layers — Leela shares
its decoder globally only because a per-head full decode would be ~84M
parameters over all (layer, head) pairs, which low rank or head-sharing avoids.

`architecture.py`: `dyn_bias_code` (0 = off, the default), `dyn_bias_rank`
(0 = full 64×64 decode, r > 0 = rank-r `U Vᵀ`), `dyn_bias_compress`.

| variant | params | vs base | FLOPs | traffic | pos/param |
|---|---|---|---|---|---|
| off | 5.92M | — | — | — | 21 |
| rank 4, code 64 | 6.60M | +11.5% | +0.6% | +1.4% | 19 |
| rank 8, code 64 | 6.94M | +17.2% | +0.8% | +1.4% | 18 |
| full, code 64 | 8.93M | +50.9% | +1.2% | +1.4% | 14 |
| full, code 256 (Leela's) | 17.79M | +200% | — | — | 7 |

The FLOPs and traffic columns are the *predicted* cost of the new ops in
isolation. **They were wrong by 15× — see the throughput section below.** Kept
here as a record of the mistake, not as guidance.

Verified: parameter counts, exact no-op at init (final decode zero-initialised),
content-dependence (map varies 0.97 across boards in one batch after 30 steps,
via a `sow` hook readable with `capture_intermediates=True`), gradients reaching
every stage, and `to_onnx` + `make_dynamic` handling the new ops and reshapes
(zero unpatched batch reshapes vs baseline). **Not verified: `optimize_model` +
fp16**, which needs torch — export a random net on the server before training.

Gotcha worth keeping: the zero-init decode blocks gradient to the whole front
end (compress/code/norm/u) at step 0, since a zero kernel has zero Jacobian. It
unfreezes at step 1 (front-end ‖grad‖ 0 → 5e-3). Zeroing *both* rank factors
would trap them permanently — only `V` is zeroed.

#### Result: rank 4 is a real but small quality gain, at an unaffordable price

400 games paired 6s+0.1 vs `sz0_doubleqk`: **+0.9 ± 17.5 pentanomial**, pairs
`[0, 52, 95, 53, 0]`, LOS 53.9% — null. Paired validation loss said −0.0242,
3.7× the 0.0065 seed floor, which looked like a clear win. Both are correct;
the total is just the wrong statistic.

**`Q_WEIGHT = 48` (`train.py:31`) means `total = policy + wdl + 48·q_mse`.**
Splitting the final checkpoint's delta:

| component | trained-on | held out | share of total |
|---|---|---|---|
| policy | −0.0057 | −0.0035 | 24% |
| wdl | −0.0042 | −0.0020 | 17% |
| 48 × q_mse | −0.0144 | −0.0096 | **59%** |
| total | −0.0242 | −0.0151 | |

The Q term moved 0.0041 → 0.0038 MSE — RMS error on expected value 0.064 →
0.062 — and the 48× weight presents that as 0.014 nats. On the components the
engine consumes, policy and wdl, the change is under the noise floor.

**Do not quote total paired loss for any change that shifts the policy/value
mix. Always split it.** The earlier calibration points (qk16, qk64, cosine
taper) were never split and should be re-read before the ~1600–2500 Elo/nat
range is trusted.

Held-out validation clears the memorisation worry: **62% of the gain survives
out of sample** (−0.0151 of −0.0242), in line with the 10-block net's 56%. The
gain is real, just small. The engine was also searching ~8% fewer nodes at the
time of the match, worth about −4 Elo, so quality-per-node is plausibly ~+5.

#### Throughput: −16% datagen, −24% search. The cost is node count.

| | baseline | rank 4 | |
|---|---|---|---|
| datagen, batch 284, CUDA graph | 70.0K | 58.5K | −16.4% |
| search, batch 32, no CUDA graph | 27.5K | 20.5K | −25.5% |
| search, batch 32, CUDA graph | 34.0K | 26.0K | −23.5% |

Predicted +1%. Two mechanisms, neither of which is the new arithmetic.

**1. The extra bias add un-fuses `BiasSoftmax`.** Confirmed by
`SZ0_DUMP_OPTIMIZED=dynbias_opt.onnx ./SZ0` + `inspect_graph.py --raw`:
`BiasSoftmax` went **10 → 0**, replaced by `Softmax ×10` and raw `Add`s; the
graph went 322 → 462 nodes. The scores tensor is the largest in the net
(8·64·64 = 64 KB/position/block) and it now round-trips through memory two
extra times per block.

Reordering the adds cannot be replaced by *summing* the two biases first:
`(1,h,64,64) + (b,1,64,64) → (b,h,64,64)`, a tensor the same size as the
scores, so you write a full-size tensor to avoid reading one. Per block per
position, in 64 KB units:

| | matmul | bias adds | softmax | total |
|---|---|---|---|---|
| baseline | w1 | *(fused)* | r1+w1 | 192 KB |
| current | w1 | r1+w1, r1+w1 | r1+w1 | 456 KB |
| pre-summed biases | w1 | w1 | r2+w1 | 320 KB |
| reordered | w1 | r1+w1 | r1+w1 | 320 KB |

`BiasSoftmax` needs a bias that broadcasts on the outer or inner dims;
`spatial_bias` is `(1,h,64,64)` and fuses, the dynamic map is `(b,1,64,64)` and
broadcasts on the **head** axis, which is neither. So the fix is to put the
dynamic add *first*, leaving the fusable constant adjacent to the softmax —
one line at `architecture.py:159-163`, no change to the function computed.

**2. Per-kernel overhead on a 43% larger graph — the bigger half.** Fitting the
two graph-on measurements to `cost = a + b·batch`:

```
+290 µs = a +  32b   =>   a = 225 µs per batch (fixed)
+799 µs = a + 284b        b = 2.02 µs per position
```

Both terms are independently confirmable: **225 µs / 140 nodes = 1.61 µs per
node**, textbook captured-graph per-kernel overhead; and **2.02 µs/pos ×
864 GB/s = 1.75 MB/pos**, against the 2.64 MB/pos the round-trip model predicts
with roughly a third L2-served (the scores tensor is 18.6 MB at batch 284,
2.1 MB at batch 32, both inside the L40S's 48 MB L2).

So it is ~⅓ bandwidth and ~⅔ per-kernel overhead. At batch 32 the fixed term
alone is 24% of the entire forward pass.

**CUDA graph capture does not fix this.** It removes CPU-side launch overhead
(−25.5% → −23.5%, about 0.76 µs/node), but a captured graph still executes 462
kernels in sequence and each retains ~1.6 µs of GPU-side scheduling regardless
of how little work it does.

**Consequence: the QK-concat idea is withdrawn.** `S + U Vᵀ` is algebraically
the score contribution of `r` extra QK dims, so concatenating `U` onto q and
`Vᵀ` onto k folds the whole thing into the existing `FusedMatMul` — an elegant
bandwidth fix. But it removes one `Add` while adding `Expand`+`Concat` ×2 per
block, i.e. *more* nodes, and node count is the dominant term.

**Verdict: rejected as implemented.** The reorder is free and worth doing
(predicted datagen 58.5K → ~64.5K, search 26K → ~27.6K) but still leaves −8%
and −19% for ~+5 Elo, against depth's 0.87 Elo per % of nps. A per-block
Smolgen branch is ~14 tiny kernels amortised over a `d_model=256` block that
takes ~94 µs at batch 32; Leela's design assumes a much larger net. The only
variants with a path are one branch shared across all 10 blocks (140 → ~24
nodes, ≈ −7%) or applying it to a subset of blocks — both of which would also
shrink the ~+5 Elo.

#### Generalisable: predict throughput from the graph, not from arithmetic

The most useful throughput lesson of the project so far. Two rules, both
learned by being wrong here:

1. **On a fused graph, the cost of a new op is dominated by which fusion it
   breaks, not by its own FLOPs or bytes.** The +1% estimate counted the new
   ops' traffic correctly and was irrelevant.
2. **On a small model, cost is dominated by node count, not by work.** At
   `d_model=256` every kernel is tiny, so ~1.6 µs of per-kernel overhead
   swamps the arithmetic. CUDA graph capture reduces this but does not remove
   it.

Always dump the optimised graph and diff the fused-op counts before predicting
the cost of an architecture change.

### Policy head width 64 → 256 — implemented, not yet trained

`architecture.py` field `d_policy_head` (default 64, unchanged). The policy head
is a single bilinear form:

```
policy[i,j] = (x_i W_from) · (x_j W_to) = x_iᵀ A x_j,   A = W_from W_toᵀ
```

`A` is 256×256 and `d_policy_head` caps its rank, so 64 is a 4× restriction and
256 removes it. Same algebra as `d_qk_head`, where this net proved very
rank-sensitive (32→16 cost −37.5 Elo, 32→64 gained +20.9). The trunk carries 80
bilinear forms of rank 64 (10 blocks × 8 heads); the entire 4096-way policy
readout is one form of rank 64.

**Pre-screen on `sz0_doubleqk_epoch6` (`scratchpad/policy_rank.py`)** — singular
spectra of the trained projections, against a random-matrix baseline, since a
flat spectrum only means something relative to one:

| | observed | random 256×64 |
|---|---|---|
| `W_from` participation ratio | 57.8 / 64 | 59.9 |
| `W_from` σ_max/σ_min | 4.1 | 2.88 |
| `A` participation ratio | 44.7 / 64 | 55.8 |
| `A` σ_max/σ_min | 11.8 | 4.57 |

`A` is clearly structured, so the head has learned something real. But neither
projection has dead directions — `W_from`'s smallest singular value is 0.98
against a largest of 4.06, barely distinguishable from random. Weight decay 1e-4
over 1.8M steps would have collapsed surplus output dimensions and did not.

**Asymmetric conclusion: rules out "64 is already enough", does not prove 256
helps** — a rank-64 truncation of a higher-rank optimum looks much the same.
Caveat: for any single position the 64 tokens span at most a 64-dim subspace, so
the cap never binds within one board; it binds across boards, because a rank-64
`A` must reuse the same subspace for all of them.

Cost, by the two rules above: **zero new nodes, zero fusions broken** — same two
`Gemm`s and one `FusedMatMul`, just wider. +98,688 params (+1.7%), +14.2
MFLOP/position, +96 KB/position of traffic. Predicted 2–4%, and this is the case
where that arithmetic should hold, because both mechanisms that broke the
dyn-bias prediction are structurally absent. Confirm node count is still 322 in
the dumped graph before trusting it.

Screen the result on the **policy** component, not the total — this change is
policy-targeted, so the expected signature is the mirror image of the dyn bias
(policy moves, value barely does).

### Kimi Attention Residuals — assessed, not run

`h_l = Σ_{i<l} α_{i→l} v_i`, attention over depth with a learned per-layer
pseudo-query ([arXiv:2603.15031](https://arxiv.org/pdf/2603.15031)). Parameters
negligible, but each layer reads all previous layer outputs: `Σ l × 32 KB`
= 1.44 MB/position against the current 5.93, so **~+24% memory traffic** on a
bandwidth-bound net ≈ datagen 70K → ~56K.

Its purpose is fixing depth pathologies (PreNorm dilution, unbounded hidden-state
growth) measured at 48B/1.4T tokens. **Measured on the exported nets: neither
pathology exists here.** Numpy forward pass over the ONNX weights (verified
against onnxruntime to 4e-3), reporting per-sublayer `‖Δ‖/‖x‖`:

| | 10 blk (doubleqk) | 6 blk (run3_e27) |
|---|---|---|
| update ratio, blocks 1–2 | 0.181 | 0.138 |
| update ratio, last 2 blocks | 0.229 | 0.257 |
| late / early | 1.27 | 1.86 |
| monotonically declining | no | no |

The trace is U-shaped, not decaying: 0.27 → 0.083 through blocks 1–5, back to
0.28 at block 9. Residual norm flat across the middle (88 → 102 over six blocks),
ending at 144 — 1.6× over nine blocks. Later blocks are the *most* active
proportionally. **Not run:** ~24% traffic for a fix to a problem this depth does
not have. Block 0 must be excluded from any such summary — the embedding arrives
at `‖x‖` 3.3 and leaves at 88, a transient that makes any early-vs-late statistic
report dilution regardless of the rest of the net.

Untested third signature: gradients concentrating in late layers, which needs a
backward pass and cannot come from an ONNX file.

The Delta variant ([arXiv:2605.18855](https://arxiv.org/pdf/2605.18855)) is a
different group, preprint, 220M–7.6B, and reports 20% throughput / 26% memory
overhead — worse for a throughput-bound project.

### GPU utilisation: memory bound, not compute bound

The model is far too small to saturate the L40S. Estimated from the shapes in
`architecture.py` (worth confirming with `nsys`/`ncu`):

| Datagen | MFLOP/pos | Compute | Traffic (fused) | Bandwidth |
|---|---|---|---|---|
| 6 blk, 95K nps | 327 | 17.2% of 181 TFLOPS | 3.56 MB/pos | 338 GB/s = 39% of 864 |
| 10 blk, 59K nps | 545 | 17.8% | 5.93 MB/pos | 350 GB/s = 41% |

Bandwidth barely moves between the two depths while compute stays ~17%, so the
1.61× depth scaling above is not evidence of being compute bound. Arithmetic
intensity 46–92 FLOP/byte against an equilibrium of 209.

These are computed against the *whole* datagen batch cycle, so they understate
utilisation during the GPU phase itself — measured separately as 55% of peak
bandwidth (see `SZ0_TIME_IO` above). Roughly 28% of the cycle was host-side
marshalling, now largely removed.

Depth costs close to full price (1.61× for +67% FLOPs). Width is *partially*
discounted but **not** free: `d_ff` 256→512 (+31% FLOPs) measured ≥~7% slower
datagen (see above; confounded with qk16 which helps, so likely ~10–15% alone)
— well under the ~24% a compute-bound net would pay, but real.

Will have more info as I gradually move to larger models.

### Training throughput

Batch 284 over 100 steps: 6 blocks ~2.0 s (14.2K pos/s), 10 blocks ~2.7 s
(10.5K pos/s). 1.35× for 1.67× the FLOPs — overhead-bound rather than compute-bound.

## sz0_run3

### Post-attention bias ablations / experiments

Minor tweaks against the then-current baseline
(8 heads, QKV bias, QK^T + B spatial attention bias, GELU activation in FFN).

| Variant | Result | Verdict |
|---|---|---|
| no QKV bias | much worse quality | keep bias |
| 2:1 attention:FFN blocks (8 attn, 4 FFN) | similar quality, 15% slower | rejected |
| SwiGLU | similar quality, 10% slower | rejected |
| fewer blocks, 2× FFN width | 20% faster, much worse quality | rejected |
| 4 attention heads (from 8) | slightly faster, worse quality | rejected |
| Swish (SiLU) | neutral | adopted (being a slightly simpler activation) |
| 2-hidden-layer FFN | similar quality, 15% slower | rejected |

### Spatial Attention Bias

Unlike LLMs, context is fixed and absolute instead of relative, so adding an attention bias per head is feasible.
QK^T + B per head significantly increased quality for negligible speed cost.
This is a simplified version of Geometric Attention Bias in Leela Chess Zero, which further improves by replacing B with a small MLP.

## sz0_run2

### Piece-Square tokens

Expanding the token vocabulary from only 13 to 13 * 64 (a distinct token for each square) improved both quality and speed (by removing the need for a bias term in the embedding)

### Individual token info in value head

Global pooling in the value head was replaced by per-token local compression, which better utilized available info.
