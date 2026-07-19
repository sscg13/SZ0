# Experiment log

Grouped by core area (search / data generation / NN architecture), from newest to oldest. 

Testing environment notes:
- **Local device (CPU):** ~250 nps per worker thread.
- **GPU server (L40S):** batch 32, fp16, ~40K nps. 
- Elo figures are joint round-robin fits unless stated; games average ~300
  ply, draw rates 50–60%.

Tests typically use either fixed-node or time control (6 sec, 0.1-0.5 sec increment).
Older tests have been using double back row randomization, newer tests use an unbalanced opening book.

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
