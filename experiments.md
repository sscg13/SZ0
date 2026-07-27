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
scattering directly out of it. Removes all three, and pinning should roughly
halve the transfer. Host-side ~1350 → ~450 µs, i.e. ~15% of the full 6040 µs
datagen batch cycle (the ~1140 µs this timer does not see is tree work).

Also measured `run` = 1684 MB / 3550 µs = **474 GB/s, 55% of the L40S's 864** —
higher than the 41% below, which was computed against the whole batch cycle
rather than the GPU phase. Memory bound is confirmed more strongly than that
number suggested, and it caps `run` headroom at ~1.8×.

## sz0_run4

Fresh trains from scratch on the accumulated window, not iteratively refined.
127.5M positions (recent sliding window, older deleted); 1.8M steps × batch 284
≈ 4 epochs; ~10 h at 6 blocks, ~13.5 h at 10. `baseline` is the 6-block fresh
train.

Open follow-ups:

- figure out how to close the ~67 Elo data/iterative gap (below) — 10× the
  noise floor, unlike any architecture tweak so far, will likely need to wait for
  run4 to make more headway on this
- increase data window and/or train longer (confounded — see below)
- reserve a small slice of datagen for representative validation
- datagen throughput budget — now 47K vs 95K originally; decide explicitly
  rather than letting adoptions erode it further
- MoE: not optimistic (Leela failed; MoE adds params + traffic, the wrong
  trade for a memory-bound, data-limited net). Head specialization IS proven —
  NNUE output buckets are hard-routed head-MoE (hand-selected by piece count).
  For a data-limited net prefer a hand-designed router over a learned gate.
  Caveat: a deep trunk may already learn phase-conditioning implicitly, so less
  headroom than shallow NNUE. 
- per-layer `d_ff` schedule (wider in deeper layers) — fixed-budget reallocation
  so ~neutral throughput, but the effect is sub-floor: needs multi-seed or a
  large reallocation to detect
- widen `d_model` — raises intensity but also traffic, 
  will be pretty costly for datagen, unless nodes can decrease
- double-buffer datagen so host marshalling overlaps the next batch's GPU work
  (the remaining ~450 µs/batch is serial with `run` today)

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
