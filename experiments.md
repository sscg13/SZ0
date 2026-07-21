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

## Search Contempt

- Replaced temperature-based move selection as the source of self-play diversity.
- Gained roughly 100 elo in the span of a few nets, but maybe confounded by standard self-play gains

---

# NN architecture / training

## Dynamic batch ONNX export (infrastructure)

Naive dynamic exports are up to ~10× slower on CUDA (reshape overhead).
The fix loads the dynamic export and optimizes it for constant batch size at runtime.

## sz0_run4

Fresh trains from scratch on the accumulated window, not iteratively refined.
127.5M positions (recent sliding window, older deleted); 1.8M steps × batch 284
≈ 4 epochs; ~10 h at 6 blocks, ~13.5 h at 10. `baseline` is the 6-block fresh
train.

Open follow-ups:

- increase data window and/or train longer (confounded — see below)
- reserve a small slice of datagen for representative validation
- `d_ff` 256 → 384 (Leela's 1.5× ratio) or 512 (2×)
- widen `d_model` — raise arithmetic intensity
- more optimizations (fusing? Cuda graph?)

### The baseline anchor is ~67 Elo below the iterative net

100 games, paired openings, fresh 6-block vs `run3_epoch28`:
**−66.8 ± 36.4** pentanomial (−66.8 ± 49.2 trinomial), LOS 0.01%.

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
| 5000 nodes/move | 0.5475 | +33.1 ± 34.4 | **+33.1 ± 24.8** | 99.6% |

Fixed node isolates evaluation quality, TC nets it against the speed loss; the
~14 Elo gap is the cost of 25% fewer nodes.

| Regime | Batch | nps 6 → 10 | Slowdown | Estimated overhead |
|---|---|---|---|---|
| Search | 32 | 40K → 30K | 1.33× | ~50% |
| Datagen | 284 | 95K → 59K | 1.61× | ~8% |

Adopted despite TC falling short of significance: datagen is fixed-node, so the
relevant figure is +33 against 38% fewer positions per hour. Caveats: one
training seed.

### GPU utilisation: memory bound, not compute bound

The model is far too small to saturate the L40S. Estimated from the shapes in
`architecture.py` (worth confirming with `nsys`/`ncu`):

| Datagen | MFLOP/pos | Compute | Traffic (fused) | Bandwidth |
|---|---|---|---|---|
| 6 blk, 95K nps | 327 | 17.2% of 181 TFLOPS | 3.56 MB/pos | 338 GB/s = 39% of 864 |
| 10 blk, 59K nps | 545 | 17.8% | 5.93 MB/pos | 350 GB/s = 41% |

Bandwidth barely moves between the two depths while compute stays ~17%, so the
1.61× depth scaling above is not evidence of being compute bound. Arithmetic
intensity 46–92 FLOP/byte against a ridge point of 209, set by `d_model`.

So prediction is **depth costs full price, width is discounted**: `d_ff` 384 is +15% FLOPs
and less in traffic, `d_ff` 512 +31%, against +67% for 6→10 blocks. 

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
