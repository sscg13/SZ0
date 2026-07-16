# SZ0 — Shatranj Zer0

A UCI engine for **shatranj** (the predecessor of chess).
Uses some modern improvements to the original AlphaZero style tree search.
The vision transformer architecture for the policy-value integrated neural network is inspired by Leela Chess Zero.
Training is conducted via JAX / Flax, inference via ONNX.

## Estimated Strength

2500 (CPU, 1 thread, ~250 nps), 3200 (GPU, ~40k nps).
Selfplay and training have been done intermittently over a few months on a single L40S.
Total estimated compute usage is roughly equivalent to 2 hours of AlphaZero.

## Building

Requires clang++ (C++23) and ONNX Runtime headers + libs under `onnx/`
(`onnx/include`, `onnx/lib`).

```sh
make GPU=no  EXE=SZ0            # CPU inference
make GPU=no  BATCH=yes EXE=SZ0  # CPU inference (GPU emulation)
make GPU=yes EXE=SZ0            # CUDA execution provider (needs CUDNN_DIR, CUDA-enabled onnxruntime)
```

`searchbatchsize` (`src/consts.h`) currently controls GPU batch size. (TODO: runtime batch sizing)
With a dynamic-batch model export the engine pins the model's symbolic batch
dimension to this value at load, so one `.onnx` file serves any batch size.

## Running

Use a UCI compatible GUI like cutechess (variant `shatranj`). Winboard with a custom adapter may also be okay.
SZ0 autodiscovers the newest `.onnx` in the working directory; set `WeightsFile` to override.

| Option | Default | Notes |
|---|---|---|
| `Threads` | 1 | search worker threads (1–16), used in CPU inference |
| `Hash` | 72 | Maximum tree size in MB, set higher to enable longer searches |
| `WeightsFile` | `<autodiscover>` | path to the `.onnx` network (no spaces) |
| `ParticleSearch` | true | Gumbel improved-policy selection instead of PUCT + virtual losses |
| `ParticleGreedy` | true | deterministic visit-matching argmax (Gumbel MuZero non-root rule); `false` = sample with importance weights |
| `ParticleEta` | 150 | proposal temperature ×100 for sampled mode (ignored when greedy) |
| `CPuct` | 200 | PUCT exploration constant ×100 (only used when `ParticleSearch=false`) |
| `SearchContemptNodeLimit` | 0 | node count used for opponent model for search contempt; >0 forces particle off |

Scores are reported as `cp = 182·atanh(q)`, calibrated so +1.00 pawn ≈ 75%
expected score following modern engine conventions.

## Search

Default selection follows the deterministic Gumbel MuZero rule: children are
scored by the improved policy `π ∝ exp(log prior + β·q)` with
`β = 0.5·(50 + max child visits)`, and the argmax of `π(a) − N(a)/(1+ΣN)`
tracks it with visits. This replaces PUCT + virtual losses both sequentially
and in the batched pipeline (leaf collisions are discarded and retried with
one sampled selection). The sampled variant (`ParticleGreedy=false`)
implements particle MCTS (arXiv:2605.08982) with importance-weighted
fractional visits. Match results behind the defaults are in
[experiments.md](experiments.md).

## Data generation

```sh
./SZ0 datagen <position_count> <nodes_per_move> <output_prefix>
```

`datagenbatchsize` (`src/consts.h`) controls the number of concurrent self-play games.
The default setting is optimized for throughput on an L40S GPU.
Finished games write to `<output_prefix>.data` as binary `TrainingPosition` records (root Q, move
count, halfmove clock, outcome, 64 board tokens, visit-derived policy
target).
Datagen uses search contempt (`contempt_nscl` in consts.h), not the
particle defaults.

## Training / export (`src/nn/`)

- `architecture.py` — ShatranjNet (transformer-style, policy + WDL value
  heads; inputs are 64 piece-square tokens + halfmove count).
- `train.py`, `dataloader.py`, `readdata.py` — training loop and `.data`
  readers (JAX/Flax, Orbax checkpoints).
- `export_onnx.py` — exports a checkpoint to ONNX: fp32 batch-1, fp16
  batch-32 (search), fp16 batch-284 (datagen), and an fp16 dynamic-batch
  file (via `make_dynamic_batch.py`) that the engine pins at load.
- `compare_onnx_jax.py` — checks export fidelity.
