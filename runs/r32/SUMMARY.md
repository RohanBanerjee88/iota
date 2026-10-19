# iota run `r32`

- **updated:** 2026-10-19 01:06 UTC (last stage: `eval 8,9,10`)
- **code:** `36f3d6f` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | – | – | – | – | – | – | not trained yet |
| gated_linear | 9500 | 0.627 | 0.255 | 1.000 | 0.392 | 0.00075 | ok |
| hybrid | – | – | – | – | – | – | not trained yet |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `*_sweep_history.json` (and `logs/train.log`); `*_sweep.json` stops at the best checkpoint.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| gated_linear | 0.234 | 0.052 | 1.000 | 1.000 | pass |

n=500 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| n_bindings | true tokens | gated_linear |
|---:|---:|---|
| 2 | 256 | 0.932 [0.92–0.94] (0.86) |
| 4 | 256 | 0.828 [0.82–0.84] (0.44) |
| 8 | 256 | 0.676 [0.67–0.69] (0.02) |
| 16 | 256 | 0.507 [0.50–0.51] (0.00) |
| 32 | 289 | 0.216 [0.21–0.22] (0.00) |
| 64 | 386 | 0.110 [0.11–0.12] (0.00) |
| 128 | 773 | 0.065 [0.06–0.07] (0.00) |

## 3. Pass 2 — length generalisation

_`pass2_length.csv` not synced yet._

## 4. Pass 3 — state_track control

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| seq_len | true tokens | gated_linear |
|---:|---:|---|
| 128 | 136 | 1.000 [1.00–1.00] (1.00) |
| 256 | 264 | 1.000 [1.00–1.00] (1.00) |
| 512 | 520 | 1.000 [1.00–1.00] (1.00) |
| 1024 | 1032 | 1.000 [1.00–1.00] (1.00) |
| 2048 | 2056 | 1.000 [1.00–1.00] (1.00) |
| 4096 | 4104 | 1.000 [1.00–1.00] (1.00) |
| 8192 | 8200 | 1.000 [1.00–1.00] (1.00) |

## 4f. Decay gates (Pass 8): what γ does on real prompts

Mean γ ± std over tokens × heads, by token role. `retention` = mean log10 of the share of a fact value's write still in the state at the first question. A frozen gate has std ≈ 0; selective forgetting shows as γ(distractor) < γ(fact value).

| model | bindings | layer | fact value | distractor | query | answer | retention (log10) | gate bias / ‖W‖ |
|---|---:|---|---|---|---|---|---:|---|
| gated_linear | 8 | backbone.blocks.0.mixer | 0.385 ± 0.167 | 0.385 ± 0.121 | 0.454 ± 0.235 | 0.383 ± 0.172 | -101.776 | -0.01 / 0.700 |
| gated_linear | 8 | backbone.blocks.1.mixer | 0.453 ± 0.203 | 0.263 ± 0.073 | 0.404 ± 0.252 | 0.489 ± 0.224 | -134.77 | -0.03 / 0.974 |
| gated_linear | 8 | backbone.blocks.2.mixer | 0.597 ± 0.267 | 0.472 ± 0.087 | 0.778 ± 0.257 | 0.800 ± 0.218 | -74.888 | +0.03 / 1.026 |
| gated_linear | 8 | backbone.blocks.3.mixer | 0.726 ± 0.236 | 0.598 ± 0.237 | 0.879 ± 0.173 | 0.902 ± 0.155 | -59.159 | +0.17 / 2.462 |
| gated_linear | 8 | backbone.blocks.4.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.998 ± 0.006 | 1.000 ± 0.000 | -0.002 | +0.13 / 2.355 |
| gated_linear | 32 | backbone.blocks.0.mixer | 0.385 ± 0.167 | 0.387 ± 0.123 | 0.458 ± 0.235 | 0.384 ± 0.172 | -83.764 | -0.01 / 0.700 |
| gated_linear | 32 | backbone.blocks.1.mixer | 0.453 ± 0.205 | 0.266 ± 0.083 | 0.412 ± 0.252 | 0.490 ± 0.224 | -103.671 | -0.03 / 0.974 |
| gated_linear | 32 | backbone.blocks.2.mixer | 0.596 ± 0.266 | 0.476 ± 0.099 | 0.782 ± 0.258 | 0.800 ± 0.218 | -56.667 | +0.03 / 1.026 |
| gated_linear | 32 | backbone.blocks.3.mixer | 0.722 ± 0.236 | 0.596 ± 0.240 | 0.878 ± 0.176 | 0.901 ± 0.156 | -53.312 | +0.17 / 2.462 |
| gated_linear | 32 | backbone.blocks.4.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.001 | 1.000 ± 0.000 | -0.002 | +0.13 / 2.355 |

## 4g. Gate-clamp intervention (Pass 9): which layers' forgetting does recall need?

Per-query recall with one GLA layer's gate replaced by a constant: `keep` = σ(6) ≈ 0.9975 (no forgetting), `mean` = the head's own average γ (same amount of forgetting, no selectivity). Δ vs the unclamped model on the same prompts, paired 95% CI.

**gated_linear**

| condition | 8 facts | 32 facts | 64 facts |
|---|---|---|---|
| none | 0.687 | 0.220 | 0.111 |
| keep_all | 0.011 (-0.676 [-0.69, -0.66]) | 0.011 (-0.209 [-0.22, -0.20]) | 0.013 (-0.099 [-0.11, -0.09]) |
| keep_L0 | 0.008 (-0.679 [-0.69, -0.67]) | 0.010 (-0.210 [-0.22, -0.20]) | 0.011 (-0.100 [-0.11, -0.09]) |
| mean_L0 | 0.688 (+0.000 [-0.01, +0.01]) | 0.220 (-0.000 [-0.00, +0.00]) | 0.110 (-0.001 [-0.00, +0.00]) |
| keep_L1 | 0.635 (-0.052 [-0.07, -0.04]) | 0.192 (-0.028 [-0.04, -0.02]) | 0.095 (-0.016 [-0.02, -0.01]) |
| mean_L1 | 0.684 (-0.004 [-0.01, +0.00]) | 0.216 (-0.004 [-0.01, -0.00]) | 0.112 (+0.001 [-0.00, +0.00]) |
| keep_L2 | 0.652 (-0.035 [-0.04, -0.03]) | 0.206 (-0.014 [-0.02, -0.01]) | 0.109 (-0.003 [-0.01, +0.00]) |
| mean_L2 | 0.668 (-0.019 [-0.03, -0.01]) | 0.214 (-0.006 [-0.01, +0.00]) | 0.110 (-0.001 [-0.00, +0.00]) |
| keep_L3 | 0.641 (-0.046 [-0.06, -0.03]) | 0.200 (-0.021 [-0.03, -0.01]) | 0.100 (-0.011 [-0.02, -0.01]) |
| mean_L3 | 0.661 (-0.026 [-0.04, -0.02]) | 0.214 (-0.006 [-0.01, +0.00]) | 0.109 (-0.002 [-0.01, +0.00]) |
| keep_L4 | 0.672 (-0.015 [-0.02, -0.01]) | 0.219 (-0.001 [-0.01, +0.01]) | 0.105 (-0.006 [-0.01, +0.00]) |
| mean_L4 | 0.679 (-0.008 [-0.02, -0.00]) | 0.219 (-0.001 [-0.01, +0.00]) | 0.111 (-0.000 [-0.00, +0.00]) |

## 4h. Gate replay (Pass 10): does the timing of forgetting matter?

One layer's recorded gates over the context span are replayed unchanged, shuffled in time (same values, same product), replaced by their geometric mean (same product, no timing) or their arithmetic mean (forgets less). Per-query recall, Δ vs baseline [paired 95% CI].

**gated_linear**

| facts | layer | baseline | replay | shuffle (range) | geomean | arith |
|---:|---|---:|---|---|---|---|
| 8 | L0 | 0.678 | 0.678 (+0.000 [0.00, 0.00]) | 0.675 (0.67–0.68) (-0.003 [-0.01, 0.00]) | 0.676 (-0.002 [-0.01, 0.00]) | 0.676 (-0.002 [-0.01, 0.00]) |
| 8 | L1 | 0.678 | 0.678 (+0.000 [0.00, 0.00]) | 0.676 (0.67–0.68) (-0.002 [-0.01, 0.00]) | 0.677 (-0.001 [-0.00, 0.00]) | 0.677 (-0.000 [-0.00, 0.00]) |
| 8 | L2 | 0.678 | 0.678 (+0.000 [0.00, 0.00]) | 0.674 (0.67–0.67) (-0.004 [-0.01, -0.00]) | 0.674 (-0.003 [-0.01, 0.00]) | 0.674 (-0.004 [-0.01, 0.00]) |
| 8 | L3 | 0.678 | 0.678 (+0.000 [0.00, 0.00]) | 0.673 (0.67–0.67) (-0.004 [-0.01, -0.00]) | 0.674 (-0.003 [-0.01, 0.00]) | 0.674 (-0.003 [-0.01, 0.00]) |
| 8 | L4 | 0.678 | 0.678 (+0.000 [0.00, 0.00]) | 0.653 (0.65–0.65) (-0.025 [-0.03, -0.02]) | 0.660 (-0.018 [-0.03, -0.01]) | 0.671 (-0.007 [-0.01, 0.00]) |
| 32 | L0 | 0.219 | 0.219 (+0.000 [0.00, 0.00]) | 0.218 (0.22–0.22) (-0.001 [-0.00, 0.00]) | 0.220 (+0.000 [-0.00, 0.00]) | 0.220 (+0.000 [-0.00, 0.00]) |
| 32 | L1 | 0.219 | 0.219 (+0.000 [0.00, 0.00]) | 0.218 (0.22–0.22) (-0.001 [-0.00, 0.00]) | 0.218 (-0.001 [-0.00, 0.00]) | 0.220 (+0.000 [-0.00, 0.00]) |
| 32 | L2 | 0.219 | 0.219 (+0.000 [0.00, 0.00]) | 0.219 (0.22–0.22) (-0.000 [-0.00, 0.00]) | 0.221 (+0.001 [-0.00, 0.00]) | 0.219 (+0.000 [-0.00, 0.00]) |
| 32 | L3 | 0.219 | 0.219 (+0.000 [0.00, 0.00]) | 0.219 (0.22–0.22) (-0.000 [-0.00, 0.00]) | 0.218 (-0.001 [-0.00, 0.00]) | 0.217 (-0.002 [-0.01, 0.00]) |
| 32 | L4 | 0.219 | 0.219 (+0.000 [0.00, 0.00]) | 0.216 (0.21–0.22) (-0.004 [-0.01, 0.00]) | 0.215 (-0.004 [-0.01, 0.00]) | 0.217 (-0.002 [-0.01, 0.00]) |

## 5. Prefill cost (one forward pass, batch 1; mostly kernel quality)

_`cost_profile.csv` not synced yet._

## 5b. Decode cost (one token at a time, batch 1)

_`decode_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
