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

This log is condensed: closed work is kept as results, rules, and the numbers
behind them, not as narrative. The full blow-by-blow — every intermediate
reading, wrong prediction, and retraction — is in git, reachable with
`git log -p --follow experiments.md` from commit `0646647` back.

---

# Standing rules

Distilled from the results below — what to check before designing an experiment,
not conclusions about any one change.

**Measurement**

- Score every paired-opening match with
  [testing/pentanomial.py](testing/pentanomial.py). Cutechess's trinomial bars
  charge for book variance the pairing already cancels; on real matches this cut
  the standard error 17–28%, enough to flip a result from non-significant to
  significant.
- **Noise floor ≈ 0.0065 nats ≈ ~10 Elo.** Two architecturally identical nets,
  separately trained, differ by that much. Single-seed changes below ~2× the
  floor are unmeasurable by loss *or* by match — either clear it or run multiple
  seeds. Paired `--vs` validation cancels batch difficulty but **not** seed
  variance, so its tight CI is falsely precise for architecture comparisons.
- **Always split the loss by component.** `Q_WEIGHT = 48` (`train.py:31`) makes
  `total = policy + wdl + 48·q_mse`, so a total can be dominated by a tiny MSE
  move. Never quote a total for a change that shifts the policy/value mix.
- Screen with `validation.py --vs` first (free with the train); spend a 400-game
  match only when loss is ambiguous or the Elo number itself is wanted. Not all
  nats convert equally — QK-dense axes run ~2500–3400 Elo/nat against depth's
  ~1600.
- Validate on **held-out** data. Roughly 55–60% of a capacity gain survives out
  of sample; the rest is fitting the training window.
- **Pin the eval set.** `validation.py` globs `*.data`, so stray datagen output
  silently changes it — the same `--vs` baseline read 2.3779 in one session and
  2.3658 in another. Paired deltas within one invocation survive this; absolute
  losses across sessions do not, which matters for any curve fitted over
  checkpoints scored at different times.
- **nps is comparable only at equal nodes/move**, and only at equal batch fill —
  fill decays through a run as games reach endgames.
- **Verify a checkpoint's actual config immediately after training** (dump the
  orbax tree or Netron the ONNX). A mislabeled `d_ff 512` net propagated through
  a match, a loss test, a datagen reading and an adopted default before it was
  caught.

**Throughput**

- **Predict cost from the dumped graph, not from arithmetic.**
  `SZ0_DUMP_OPTIMIZED=<path> ./SZ0`, then `inspect_graph.py --raw`, and diff the
  fused-op counts. Three consecutive predictions made from FLOPs and bytes alone
  were wrong, one of them by 15×.
- On a **fused** graph, the cost of a new op is dominated by *which fusion it
  breaks*, not by its own FLOPs or bytes.
- On a **small** model, cost is dominated by *node count*, not work — at
  `d_model = 256` every kernel is tiny, so ~1.6 µs of per-kernel overhead swamps
  the arithmetic. CUDA graph capture reduces this but does not remove it: it
  eliminates CPU-side launch overhead, not GPU-side per-kernel scheduling.
- **Equal FLOPs is not equal time.** Narrow GEMMs are less efficient, so
  fixed-budget reallocation costs rather than being free.
- The net is **memory bound** (~53% of the L40S's 864 GB/s during the GPU
  phase), so depth costs full price and width is partially discounted.
- **Never measure the graph with the pip `onnxruntime`** — a different build and
  provider from the engine's linked library, and fusion differs by both.

**Rules that cost something to learn**

Every rule above came from being wrong about something. The list is short and
worth keeping, because a rule without its scar is easy to talk yourself out of.

- Predicted **+1%** throughput for the dynamic attention bias; measured
  **−16.4%**. Counted the new ops' bytes correctly and ignored that inserting an
  op into a fused pattern *un-fuses* it.
- Predicted search would absorb that better than datagen; it was hit **worse**.
  "Batch 32 has ~50% fixed overhead so it absorbs slowdown" is true of FLOPs;
  that change added *nodes*, which land directly on the fixed overhead itself.
- Predicted CUDA graph capture would fix it (−3 to −6%); got **−23.5%**. Capture
  removes CPU-side launch overhead, not GPU-side per-kernel scheduling.
- Predicted **2.2×** from CUDA graph capture, fitted from a `c = 0.59 ms` fixed
  cost; got 10–15%. Fitting a fixed cost tells you its size, never its
  composition.
- Predicted **+15%** from the inference-path fix; got **+32%**. The timer
  measured allocation but not *de*allocation — 677 µs of freeing multi-MB vectors
  happened after the last timestamp.
- First dilution statistic was a **false positive**: averaging blocks 0–1 is
  dominated by the embedding transient (`‖x‖` 3.3 → 88). Exclude block 0 from any
  early-vs-late statistic.
- A **mislabeled checkpoint** ("d_ff 512" was actually 256) propagated through a
  match, a loss test, a datagen reading, an adopted default, and a strategic
  thread before it was caught.
- `policy_rank.py` was a *correct* pre-screen that still failed to predict the
  null, because a negative finding ("no surplus capacity") is far weaker evidence
  than a positive one. Treat a clean diagnostic as permission to run the
  experiment, not as a prediction of its outcome.

**Architecture**

- **The token count (64) is the ceiling for every per-token bilinear rank cap.**
  `d_qk_head` and `d_policy_head` both cap the rank of a 256×256 form evaluated
  as `x_iᵀ A x_j`. For a single position `x` has rank ≤ 64, so rank ≥ 64 leaves
  that position's 64×64 map completely unconstrained; going higher only lets
  *different* boards have more independent maps, which measured as nothing.
  Predicts all three results: qk 32→16 (far below) −37.5 Elo, qk 32→64 (reaches
  it) +20.9, policy 64→256 (already at it) null. **Do not raise any bilinear
  rank cap above 64** — that closes `qk 128` on capacity grounds, which is
  stronger than the cost grounds it was parked on.
- **A widening is a superset in function space**, so it cannot be worse at the
  optimum. A widened net scoring worse on *training-window* loss is an
  optimizer/seed outcome, never a capacity finding.
- Softmax over `j` discards anything constant along `j`, which kills every
  per-query additive term — see the logit algebra under run4.

---

# Search

## Particle MCTS

Adopted as the default (`ParticleSearch=true, ParticleGreedy=true`) with the
corrected Gumbel-MuZero deterministic rule, `argmax[π(a) − N(a)/(1+ΣN)]`.

| Setup | Engine | Elo |
|---|---|---|
| 6s+0.1, batch 32, 200 games RR | greedy (deterministic Gumbel) | **+33 ± 32** |
| | eta100 (η=1.0 sampled) | +12 ± 31 |
| | cpuct-100 (PUCT, c=1.0) | −45 ± 32 |
| 400 nodes/move, 120 games RR | greedy (visit-matching rule) | **+187** |
| | eta100 | +83 |
| | PUCT c=1.0 | −58 |
| | PUCT c=2.0 (old default) | −219 |

Ordering is identical at fixed node count and at TC, but **margins compress ~5×
on GPU** (+187 → +33) — the fixed-node figure measures the selection rule, the TC
figure measures what survives real speed. Lower CPuct matters in short searches.

Earlier ladders (sequential and batched, 400 nodes, 120 games RR) put particle
η=1.0–1.5 at +137 to +144 over PUCT at −127 to −195, but those are **confounded
by a concurrent change to improved sampling** and by the then-broken greedy rule
(plain argmax of the improved-policy logits, which is degenerate and scored
−269). Kept only as history; the two tables above are the trustworthy ones.

**Negative result — collision merging (`ParticleMerge`, branch
`particle-merge`, off by default).** Folding colliding rollouts into the owning
in-flight evaluation (summed weight, multiplicity capped at 4) instead of
discard-and-retry: **−10.4 ± 32.3**, LOS 26.4%, 200 games at 6s+0.1. It engages
rarely — 351 merges per 50K collisions per 169K rollouts from a startpos probe —
because visit-matching selection already counts in-flight virtual visits and
steers around pending leaves. Retained virtual visits do cut collision spin ~33%
(75K → 50K), but that saves worker CPU, not GPU throughput, so Elo is neutral.

**Batch size 32 → 64:** 40K → 50K nps, neutral in 200 games H2H.

Open follow-ups: more GPU tests at higher TC; isolate how much of the gain is
Gumbel alone.

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

`SZ0_DUMP_OPTIMIZED=<path> ./SZ0` writes the post-fusion graph;
`src/nn/inspect_graph.py --raw` histograms it. This is the only valid way to
measure it — see the standing rules for why a pip-`onnxruntime` reading is
actively misleading.

**Production graph (10 blk, qk64, batch 32, CUDA, ORT 1.24): 322 nodes, 60% data
movement.** Already fused, so offline rewrites for these are redundant (tried and
reverted twice): `BiasSoftmax` ×10 (spatial bias + softmax), `FusedMatMul` ×11
(attention scale folded in), `SkipLayerNormalization` ×20, `QuickGelu` ×10.

Left on the table: whole-attention fusion never fires (no `MultiHeadAttention`,
so the scores tensor round-trips through HBM); 146 Reshape + 40 Transpose
survive.

### Datagen throughput: 47K → 70K nps

| change | nps | note |
|---|---|---|
| baseline (10 blk, qk64) | 47K | |
| + remove redundant host passes, pinned D2H | 62K | +32% |
| + parallel scatter, batch backfill | — | folded into the above |
| + two-pool pipelining | **70K** | +13%, `gpu-bound 97%` |

**CUDA graph capture** (`SZ0_CUDA_GRAPH=1`, needs a pinned batch): +10–15% search
nps ≈ +5 Elo, adopted on the nps measurement since the Elo is under the floor.
Two blockers first: (a) `jnp.clip` in the model made capture impossible — int32
`Clip` has no CUDA kernel, so ORT placed it on CPU and inserted `MemcpyFromHost`,
and capture needs *every* node on the CUDA EP. Removed from `architecture.py`,
halfmove clamped in C++ (`clamp_halfmove`); only findable via
`SZ0_ORT_VERBOSE=1`, which logs node placement, as the error names no op.
(b) Exit aborted (`cudaGraphExecDestroy`: "driver shutting down") — the session
global outlived CUDA's atexit handler, fixed by `nn.reset()` at the end of
`uci()`. Pre-existing latent bug; `cudaStreamDestroy` had been failing silently
on every GPU exit.

**The real win was host-side, not the transfer.** `SZ0_TIME_IO=1` showed 4.65 MB
of D2H in 370 µs = 12.6 GB/s, already fine — while three redundant host passes
around it cost 722 µs, roughly 2× the transfer: two zero-fills of buffers
overwritten immediately, plus a copy into a temporary that existed only to be
scattered out of. Fix: one persistent pinned landing buffer allocated at setup,
callers scattering directly from it. Host-side share 27.5% → 10.7%, then → ~6%
once the scatter was parallelised (each worker reads its own row via
`batch_policy()`/`batch_value()` during the expand phase). Both paths are now
zero-copy.

Post-fix breakdown at batch 284: `run` 3670 µs (GPU), `d2h` 190, `scatter` 230,
`h2d` 19. **`run` = 1684 MB / 3670 µs = 459 GB/s = 53% of the L40S's 864**, which
caps `run` headroom at ~1.9×. `run` also drifts up ~3.4% within a run on an
identical graph and batch — probably clock throttle, unconfirmed.

**Batch fill decay** was costing ~20% late in every run. At 300 nodes/move:
early fill 284/284 → 64K implied nps, late 228/284 → 50K, while cycle time barely
moved (4414 → 4526 µs) — the batch was emptying, nothing was getting slower.
Cause: `select()` finishes terminal-leaf rollouts itself and returns false,
leaving the row empty, and terminal leaves get common in endgames (shatranj's
bare-king and 70-move rules). The captured graph runs all 284 rows regardless, so
~20% of every late inference was computing padding. Fixed by retrying instead of
yielding the slot. Search semantics unchanged; RNG interleaving differs, so data
is statistically equivalent but not bit-identical.

**Arena sizing:** peak occupancy 11211 / 65536 nodes at 300 nodes/move (~37
children per expansion), zero overflows ⇒ `datagenarenasize` cut to 32768 (2.9×
headroom, covers ~800 nodes/move), freeing ~218 MB across 284 games — which paid
for the second pool. `expand()` had a silent failure mode: a full arena skips
children but still backprops, so the leaf is re-selected forever and quality
degrades invisibly. Now counted and warned.

### Pipelined datagen (two game pools)

Inference was ~85% of the cycle with 7 of 8 workers parked at a barrier, and it
cannot be overlapped within one pool (select for rollout N+1 depends on expand of
N). Two pools alternate: while the GPU evaluates pool p, workers do tree work for
pool 1−p. Inference moved to its own thread; the middle barrier became a per-pool
submitted/completed handshake.

**70K nps from 62K, `gpu-bound 97%`.** Verified by end-of-run peak arena
11020/32768, statistically identical to the 11211 before the change — the
decisive check, since a broken `expand` would have shown a fast loop, a perfect
284/284 fill, and worthless data.

`select` fell ~290 → ~50 µs and `expand` ~180 → ~28 µs, but **that is a rate
effect, not less work**: identical peak arena means identical trees, and all four
CPU columns scaled by the same factor, which real work reductions never do. The
old design parked 7 workers on a `std::barrier` (which spins before blocking) for
~85% of wall time, burning power and ping-ponging a cache line; the new one
sleeps them on a condition variable for 97%.

Cost: the end-of-run tail doubles — 568 games in flight, all discarded partway
when the target is hit, ~29000 wasted positions against ~14500. Fine at the
production 1.5M-position chunk size (1.9%), dominant on a 30K run.

Correctness measures, since a race here corrupts training data rather than
crashing: everything a batch touches is per-pool (games, packed inputs, both
index maps, the evaluator result slot); `NNEvaluator` holds **two** pinned
landing buffers, or a Run would overwrite results workers are still reading
(device buffers stay single — only one Run is ever in flight); workers compare
`completed` against a per-worker consumed count rather than trusting a wakeup;
the batch→game index map is round-tripped with `abort()` on mismatch before any
result is consumed; a single loop exit read after a barrier so all eight workers
leave on the same iteration; inference-thread exceptions fail both pools and
release blocked workers. `[dg]` reports `gpu_wait` — high is healthy; near zero
means tree work became the bottleneck and a third pool would buy nothing.

Also fixed along the way: `total_nodes_evaluated` was `int` and overflowed at
~2.1B nodes; static load imbalance gave thread 7 thirty-nine games to everyone
else's thirty-five.

Remaining: anything inside the ONNX graph (a few %); overlapping the h2d/d2h
copies needs async streams and a second capture — larger change, smallest prize.

## sz0_run4

Fresh trains from scratch on the accumulated window, not iteratively refined.
127.5M positions (recent sliding window, older deleted), batch 284.

**Adopted architecture: 10 blocks, `d_model 256`, 8 heads, `d_qk_head 64`,
`d_ff 256`, `d_policy_head 64`, `dyn_bias` off.** Two changes from the 6-block
baseline, both found early; everything tried after `qk 64` came back null or
negative.

**nps caveat:** datagen figures from the architecture experiments (95K, 63K, 59K,
55K, 47K) predate the inference-path fixes and carry ~28% host-side overhead.
They are comparable to each other but not to anything measured after — the same
qk64 net that read 47K reads 62K afterwards, and 70K once pipelined.

### Results

400 games paired 6s+0.1 unless noted; loss is paired `--vs` on the same fixed
sample. Noise floor ~0.0065 nats / ~10 Elo, so anything inside that is a null.

| Change | Loss | Match (pentanomial) | Datagen | Verdict |
|---|---|---|---|---|
| 6 → 10 blocks | −0.0204 | +33.1 ± 24.8 @ 5000 nodes | −38% | **adopted** |
| `d_qk_head` 32 → 64 | −0.0079 | +20.9 ± 17.7, LOS 99% | −20% | **adopted** |
| `d_qk_head` 32 → 16 | +0.0149 | −37.5 ± 16.7, LOS ~0% | +7% | rejected |
| `d_ff` 256 → 512 | +0.0063 (confounded) | — | ~−10% | rejected |
| Cosine `d_ff` taper | +0.0002 | +6.9 ± 15.8 | −4.3% | rejected |
| Dynamic attention bias, rank 4 | −0.0151 held out | +0.9 ± 17.5 | −16.4% | rejected, revisit at larger `d_model` |
| `d_policy_head` 64 → 256 | +0.0099 | −9.6 ± 18.0 | ~−3% | rejected (rank cap never binding) |
| Attention residuals (Kimi, trunk) | — | — | ~−24% predicted | not run, premise measured false |
| `QKᵀ + W1X + (XW2)ᵀ` | — | — | — | rejected on paper (per-query term cancels) |

The two adopted changes were decided on the **fixed-node** match, not TC: datagen
is fixed-node, so data *quality* is the relevant axis, traded against
positions/hour. 6 → 10 blocks read +33.1 ± 24.8 at 5000 nodes but only
+19.1 ± 25.7 at 6s+0.1 — the ~14 Elo gap is the cost of 25% fewer nodes.

Open follow-ups:

- datagen throughput budget — 70K at 300 nodes/move vs 95K originally; decide
  explicitly rather than letting adoptions erode it further
- widen `d_model` — the main capacity lever left, and it gates the dynamic
  attention bias. Raises arithmetic intensity, which is the only way off the
  ~53%-of-bandwidth ceiling, but also raises traffic
- MoE: not optimistic (Leela failed; MoE adds params + traffic, the wrong trade
  for a memory-bound net). Head specialisation IS proven — NNUE output buckets
  are hard-routed head-MoE, hand-selected by piece count — so prefer a
  hand-designed router over a learned gate. Caveat: a deep trunk may already
  learn phase-conditioning implicitly
- `num_heads` 8 → 16, and removing the inert `b_K`: see Ideas for run5
- confirm whether the 3.4% `run` rise is GPU clock throttle

### The ~67 Elo gap: resolved, it was optimisation not data

The original anchor: fresh 6-block vs `run3_epoch28`, 100 paired games,
**−66.8 ± 36.4 pentanomial**, LOS 0.01%. Loss on two eval sets, current window /
older slice: baseline 2.3993 / 2.2066, 10 blocks 2.3789 / 2.1951, run3_epoch28
2.3236 / 2.1269. The lead was real rather than an eval artifact (−0.0757
symmetric vs −0.0797 asymmetric), and the fresh nets essentially do not overfit —
the baseline↔run3 gap moves only 0.004 between held-out and trained-on data.

The original reading was "insufficient data, need ~150–200M positions". **That
was wrong.** Six 300k-step checkpoints of one run, dropping #1 as the embedding
transient:

| step | 600k | 900k | 1200k | 1500k | 1800k |
|---|---|---|---|---|---|
| total | 2.4259 | 2.4003 | 2.3763 | 2.3627 | 2.3538 |

A clean log law, **slope −0.0675 nats per e-fold of steps, no saturation** —
overturning an earlier "loss saturates" claim that came from a linear fit inside
the last 100k steps, badly under-powered against six checkpoints spanning 6× in
steps. This is not the LR-schedule claim, which was separately ruled out: run3
used the same constant 1e-4, it just took 4.7× more steps.

**Two pre-registered predictions, both validated:**

| | predicted | observed |
|---|---|---|
| parity with run3 | 4.0M steps | **3.6M** |
| total at 5.4M | −0.026 | **−0.0225 ± 0.0027** |

The first extrapolated 2× beyond the fitted range and missed by about one noise
floor; the second by half. The log-law is usable for planning.

Matches: **+4.3 ± 17.0** at 3.6M, **+7.8 ± 17.8** at 5.4M — statistically
indistinguishable from each other.

**But the totals hide two large opposite-signed components**, the case the
standing rule exists for:

| | 3.6M | 5.4M | per e-fold |
|---|---|---|---|
| policy (run4 worse) | +0.0196 | +0.0166 | 0.0074 |
| value (run4 better) | −0.0184 | −0.0391 | 0.051 |
| total | +0.0012 | −0.0225 | 0.0585 |

**The policy gap closes at 0.0074 nats per e-fold, so the remaining 0.0166 needs
~51M steps** — 14× more training, ~16 days of GPU. Technically steps, practically
structural. Value improves 7× faster and was 87% of the last increment, so
further pretraining buys almost purely value nats at a poor cross-lineage
exchange rate: −0.0225 total (3.5× the floor) is worth +7.8 Elo where the
within-lineage rate predicts ~+36. **Pretraining therefore stopped at 5.4M on the
evidence, not as a compromise.**

**The policy deficit is a recipe artifact, not an architecture one.**
`run3_epoch28` *is* a distillation pass — epoch27 plus one pass over this window,
trained to match search output, which is exactly what buys a lower policy loss.
The policy targets in the data are also MCTS visits generated by run3-lineage
search, so run3's prior correlates with the labels through the process that made
them. Once run4 generates its own datagen the advantage **transfers rather than
having to be overcome**: what 51M pretraining steps could not fix, the first RL
iteration fixes by construction. run4 enters RL with a **−0.039 value advantage,
6× the noise floor** — the largest clean margin measured in this project, on the
component that guides search and therefore shapes the data.

At the boundary: run a distillation pass on run4 (run3 got one, run4 has not),
and re-measure against run3 after the first RL iteration watching whether policy
crosses. If the lineage explanation is right, that delta should move far faster
than 0.0074/e-fold once the data is run4's own.

Minor confound checked: `VALUE_WEIGHT`/`Q_WEIGHT` flipped from 48/1 to 1/48 on
2026-03-25, so run3's early epochs used a different objective and its last ~15
ran under the current one. Probably washed out.

### Durable findings

**The token-count ceiling.** `d_qk_head` and `d_policy_head` both cap the rank of
a 256×256 form evaluated as `x_iᵀ A x_j`. For a single position `x` is (64, 256)
so it has rank ≤ 64, and any `A` of rank ≥ 64 leaves that position's 64×64 map
completely unconstrained; higher rank only lets *different* boards have more
independent maps, which measured as nothing. Predicts all three results at once:
qk 32→16 (far below) −37.5 Elo, qk 32→64 (reaches it) +20.9, policy 64→256
(already at it) null. `qk 128` is closed on capacity, not merely on its
super-linear cost.

Corollary that stopped a misreading: a widening is a **superset in function
space**, so it cannot be worse at the optimum. `d_policy_head=256` scoring +0.0099
worse on *training-window* loss is provably an optimizer/seed outcome — and 65%
of that regression landed in the value head, which the change does not touch.

**Attention logit algebra.** Expanding `q_i·k_j` with the projection biases gives
four terms, and softmax over `j` discards anything constant along `j`:

| term | depends on | survives? |
|---|---|---|
| `(W_Q x_i)·(W_K x_j)` | i, j | yes — the real attention |
| `(W_Q x_i)·b_K` | i only | no |
| `b_Q·(W_K x_j)` | j only | yes — a per-key content bias |
| `b_Q·b_K` | neither | no |

This killed `QKᵀ + W1 X + (X W2)ᵀ + B` on paper: `W1 X` is a per-query bias that
cancels identically (zero effect *and* zero gradient), and `(X W2)ᵀ` is term 3,
which every head already has unconstrained since `d_qk 512 > d_model 256` —
sharing it across heads makes the proposal strictly weaker than the status quo.

It also makes **`b_K` provably inert** here: it appears only in the two
cancelling terms and `k` is used nowhere else. Verified numerically (weights
identical to 3e-16 under a 10× bias, against a `b_Q` control that moves them
0.11) and in the trained net (`b_K` RMS ~2e-3 vs `b_Q` ~1.5e-1, 50× smaller,
uniform across all 10 blocks). Not exactly zero despite a zero init, because
softmax shift-invariance is exact in ℝ but not in floating point and Adam
normalises by gradient magnitude, so even rounding noise random-walks the
parameter. **Not general MHA folklore** — RoPE or QK-norm breaks the argument;
SZ0 has neither.

**`Q_WEIGHT = 48`.** `total = policy + wdl + 48·q_mse`, so a total can be
dominated by a tiny MSE move. The dyn-bias result made this concrete: of its
−0.0242, 59% was `48 × (−0.0003)` on a term that moved 0.0041 → 0.0038 MSE
(RMS eval error 0.064 → 0.062), while policy moved −0.0057 and wdl −0.0042, both
under the floor. The match agreed with the components, not the total.

**Dyn-bias throughput decomposition**, the measurement behind the two throughput
rules. Predicted +1%, measured −16.4% datagen and −23.5% search. Fitting
`cost = a + b·batch` to two CUDA-graph-on points (+290 µs at batch 32, +799 µs at
284) gives **a = 225 µs/batch, b = 2.02 µs/position**, and both confirm
independently: 225 µs / 140 added nodes = 1.61 µs per node, and 2.02 µs/pos ×
864 GB/s = 1.75 MB/pos against the 2.64 MB/pos round-trip model with ~⅓
L2-served. So ~⅓ bandwidth (the extra bias add un-fuses `BiasSoftmax`, 10 → 0,
graph 322 → 462 nodes) and ~⅔ per-kernel overhead. CUDA graph capture barely
helped (−25.5% → −23.5%): it removes CPU-side launch overhead, not GPU-side
per-kernel scheduling.

**Loss screening works**, with four calibration points and no false verdicts —
qk16 +0.0149 / −37.5, qk64 −0.0079 / +20.9, cosine taper +0.0002 / +6.9, dyn bias
per-component null / +0.9. It under-calls on QK-dense axes (~2500–3400 Elo/nat
against depth's ~1600) but has never contradicted a match *when read per
component*. It does **not** convert across lineages: 0.0757 nats ↔ 67 Elo is
~885 Elo/nat, and the 5.4M point is worse still.

## Ideas for run5

Staging area, nothing tested. Ordered cheapest-and-most-untested first. Every
one of these is subject to the standing rules — in particular the ~10 Elo floor,
so anything predicted sub-floor needs multiple seeds or should not be run.

### Policy head: add the missing nonlinearity

The policy head is **the only purely linear readout in the net** — the trunk's
final LayerNorm feeds `p_from`/`p_to` directly. The value head has two
activations, every block has SiLU.

Lc0's attention policy head reportedly has an extra projection + nonlinearity
before the QK mapping, and uses `d_policy = d_model`, which looks like it
contradicts the token-count ceiling. It probably doesn't: **their single number
does two jobs**, and the ceiling constrains only one.

```
t      = σ(W_p x)      # d_policy sets the width feeding the NONLINEARITY
p_from = t W_from      # d_qk sets the RANK of the bilinear form
```

`t` is still (64, d) per position, so rank ≥ 64 still saturates the map — the
ceiling holds. But `d_policy = d_model` means *no compression before the
nonlinear step*, an entirely different claim. run4 moved the rank with no
nonlinearity present, i.e. tested the one variable that could not help.

Supporting evidence, retroactively: the pre-screen measured `A`'s participation
ratio at **44.7 of 64 available** — the form was not using the rank it already
had. That reading was ambiguous at the time (a truncated higher-rank optimum
looks similar) but the null makes "not rank-starved" the supported one.

Test, decoupling the two roles:

```python
t = nn.silu(dense(self.d_model)(x))       # new: 256 -> 256, nonlinear
p_from = dense(self.d_policy_head)(t)     # keep 64
p_to   = dense(self.d_policy_head)(t)
```

Cost is the cheap regime: **one** extra Dense plus an activation, not per block —
~3 nodes on 322, +65k params, +8.4 MFLOP/position, one extra 32 KB/position
tensor. Contrast the dyn bias's 140 nodes.

Humility note: Lc0 has run far more compute than this project, so if they landed
on `d_policy = d_model` there may be a reason beyond this reconstruction.

### Value head: SiLU, and the 16/32 bottleneck

Current head is `Dense(16)` per token → ReLU → flatten to 1024 → `Dense(32)` →
ReLU → `Dense(3)`. The 16/32 widths date from run2's replacement of global
pooling with per-token local compression.

Two separable questions:

- **ReLU → SiLU.** run3 tested exactly this swap in the FFN and got *neutral*,
  so the prior is low. But the mechanism that would make it matter here is
  width-dependent: a dead unit costs 1/16 or 1/32 of a layer rather than 1/256,
  and SiLU has no dead-unit failure mode. So the FFN null does not transfer
  cleanly. **Cheap diagnostic first:** measure the fraction of the 16 and 32
  units that are zero across a real validation batch (`capture_intermediates`,
  a few lines on top of `validation.py`). If nothing is dead, skip the
  experiment; if a meaningful fraction is, the mechanism is real. Same shape as
  the `policy_rank.py` screen — and note that screen's lesson: it was correct
  but asymmetric, so a null diagnostic is weaker evidence than a positive one.
- **The widths themselves.** The entire value estimate passes through **32
  numbers** — by far the narrowest point in the net, against `d_model = 256` and
  `d_ff = 256` everywhere else. Widening is nearly free: the whole head is ~37k
  params (0.6%), and by the node-count rule adds zero nodes. This may be the
  larger lever of the two, and the dyn-bias result is weak evidence for it —
  that change's gain was 77% value-side, which says value capacity was the
  binding axis.

### Depth attention at the readout (Leela dev's WDA/PDA variant)

Reported by a Leela developer as working well; no numbers seen. Adapts Kimi's
attention residuals ([arXiv:2603.15031](https://arxiv.org/pdf/2603.15031)) to the
*heads* rather than the trunk. Two components:

1. **Per-square depth attention** — each square queries its own representation
   across all 10 layer outputs ("which layer knows most about this square?") and
   pools accordingly.
2. **Full-width global channel** — the final residual `[64×256]` goes to WDL
   directly, bypassing the value and policy embedding funnel entirely.

**The run4 rejection of Kimi attention residuals does not cover this.** That test
measured *dilution* — whether late layers contribute less — and found it false at
L=10 (late/early 1.27, non-monotone). This variant's premise is different: the
final residual is a lossy summary, and what was salient at L4 may have been
overwritten by L10. A layer can be highly active and still discard what an
earlier layer knew.

Cost is also a different class, because depth attention happens **once** rather
than per layer:

| | trunk variant (rejected) | head variant |
|---|---|---|
| reads of prior layer outputs | `Σ l × 32 KB` = 1.44 MB/pos | `10 × 32 KB` = 320 KB/pos |
| traffic | +24% | **+5.4%** |
| new nodes | ~10 per block × 10 | **~10, once** |

Linear in depth instead of quadratic, and ~10 nodes once is the cheap regime by
the node-count rule.

**Caution: the claim bundles two changes and does not attribute.** Component 2 is
the value-head bottleneck item above, arrived at independently — so if the
bottleneck is what binds, most of the reported gain could be the bypass with the
depth mechanism contributing little. **Test the full-width channel first**; only
try depth attention if that fails to capture it.

**Screen the premise before training:** does settling depth actually vary by
square? For each square, find the depth after which its residual stops changing
materially. If all 64 settle together there is no variance for per-square depth
attention to exploit and a single global depth pooling would do. Needs a real
validation batch, so it is a server-side script — pair it with the value-head
dead-unit check.

Transfer caveat with a track record: three ideas imported from larger nets have
now failed at `d_model = 256` — EfficientViT's QK halving (−37.5 Elo), Smolgen
(real gain, unaffordable), and the Kimi trunk premise (measured false). Leela
runs 15–40 blocks at `d_model` 768–1024, where there is far more room for
different squares to peak at different depths than there is at L=10.

### Dynamic attention bias — revisit at larger `d_model`

Rejected in run4 for cost, not quality: real gain (~+5 Elo, 62% held-out
survival) at −16% datagen, because ~14 tiny kernels per block do not amortise at
`d_model = 256`. Reconsider once `d_model` grows. If revisited, start from the
cheap variants — one branch shared across all 10 blocks (140 → ~24 nodes) or a
subset of blocks — and apply the add-reordering fix so `BiasSoftmax` still fuses
(put the dynamic add *first*, leaving the broadcastable constant adjacent to the
softmax; one line at `architecture.py:159-163`, function unchanged).

The implementation still lives in `architecture.py` behind `dyn_bias_code`
(0 = off), `dyn_bias_rank` (0 = full 64×64 decode, r > 0 = rank-r `U Vᵀ`),
`dyn_bias_compress`. Costs measured: off 5.92M params, rank4+code64 6.60M
(+11.5%), rank8 6.94M (+17.2%), full+code64 8.93M (+50.9%), full+code256
(Leela's dim) 17.79M (+200%).

**Implementation trap worth keeping:** the zero-init decode blocks gradient to
the entire front end at step 0, since a zero kernel has zero Jacobian. It
unfreezes at step 1 (front-end `‖grad‖` 0 → 5e-3), so it is harmless — but
zeroing *both* rank factors would trap them permanently. Only `V` is zeroed.
Verified at implementation: parameter counts, exact no-op at init,
content-dependence (the map varies 0.97 across boards in one batch after 30
steps, via a `sow` hook read with `capture_intermediates=True`), gradients
reaching every stage, and `to_onnx` + `make_dynamic` handling the new ops.

### Widen `d_model`

The main capacity lever left, and the one that gates the item above. Raises
arithmetic intensity, which is the only way to move off the ~53%-of-bandwidth
ceiling — but also raises traffic, so it is costly for datagen unless
nodes/move can drop. Note the run4 evidence that depth costs full FLOPs price
while width is partially discounted.

### `num_heads` 8 → 16

**Decouple `d_v_head` first** or it is three variables at once: the scores
tensor doubles (largest tensor in the net), the spatial bias doubles (more
expressive, and the bias was one of run3's biggest wins), and V's head dim
silently halves to 16 — and a 32→16 head-dim cut on QK cost 37.5 Elo.
Counter-evidence in its favour: run3 found 4 heads worse than 8, so
more-heads-smaller-dim beat the reverse at constant `d_qk`.

### Remove `b_K`

Provably inert (see the logit algebra under run4), 0.09% of parameters. Left in
place only because removing it breaks `StandardRestore` for every existing
checkpoint, so it has to happen at a run boundary.

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
