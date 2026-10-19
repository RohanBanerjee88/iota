# iota run `r24`

- **updated:** 2026-10-19 06:40 UTC (last stage: `eval 8,9,10`)
- **code:** `36f3d6f` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | – | – | – | – | – | – | not trained yet |
| gated_linear | 14500 | 0.945 | 0.889 | 1.000 | 0.698 | 0.003 | ok |
| hybrid | – | – | – | – | – | – | not trained yet |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `*_sweep_history.json` (and `logs/train.log`); `*_sweep.json` stops at the best checkpoint.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| gated_linear | 0.870 | 0.508 | 1.000 | 1.000 | pass |

n=500 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| n_bindings | true tokens | gated_linear |
|---:|---:|---|
| 2 | 256 | 1.000 [1.00–1.00] (1.00) |
| 4 | 256 | 0.996 [0.99–1.00] (0.98) |
| 8 | 256 | 0.993 [0.99–0.99] (0.95) |
| 16 | 256 | 0.978 [0.98–0.98] (0.71) |
| 32 | 289 | 0.914 [0.91–0.92] (0.23) |
| 64 | 386 | 0.597 [0.59–0.60] (0.00) |
| 128 | 773 | 0.306 [0.30–0.31] (0.00) |

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
| 2048 | 2056 | 0.999 [1.00–1.00] (1.00) |
| 4096 | 4104 | 0.309 [0.28–0.34] (0.31) |
| 8192 | 8200 | 0.137 [0.12–0.16] (0.14) |

## 4f. Decay gates (Pass 8): what γ does on real prompts

Mean γ ± std over tokens × heads, by token role. `retention` = mean log10 of the share of a fact value's write still in the state at the first question. A frozen gate has std ≈ 0; selective forgetting shows as γ(distractor) < γ(fact value).

| model | bindings | layer | fact value | distractor | query | answer | retention (log10) | gate bias / ‖W‖ |
|---|---:|---|---|---|---|---|---:|---|
| gated_linear | 8 | backbone.blocks.0.mixer | 0.327 ± 0.362 | 0.497 ± 0.359 | 0.195 ± 0.243 | 0.339 ± 0.366 | -157.351 | -0.05 / 2.276 |
| gated_linear | 8 | backbone.blocks.1.mixer | 0.885 ± 0.305 | 0.994 ± 0.064 | 0.998 ± 0.009 | 0.997 ± 0.023 | -12.975 | +0.19 / 4.015 |
| gated_linear | 8 | backbone.blocks.2.mixer | 0.533 ± 0.355 | 0.793 ± 0.199 | 0.846 ± 0.286 | 0.920 ± 0.154 | -37.759 | +0.11 / 3.035 |
| gated_linear | 8 | backbone.blocks.3.mixer | 0.671 ± 0.290 | 0.765 ± 0.083 | 0.993 ± 0.022 | 0.998 ± 0.006 | -31.672 | +0.23 / 2.696 |
| gated_linear | 8 | backbone.blocks.4.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.001 | -0.002 | +0.29 / 2.746 |
| gated_linear | 32 | backbone.blocks.0.mixer | 0.327 ± 0.362 | 0.498 ± 0.360 | 0.198 ± 0.246 | 0.339 ± 0.367 | -150.055 | -0.05 / 2.276 |
| gated_linear | 32 | backbone.blocks.1.mixer | 0.886 ± 0.304 | 0.989 ± 0.095 | 0.998 ± 0.010 | 0.997 ± 0.023 | -47.655 | +0.19 / 4.015 |
| gated_linear | 32 | backbone.blocks.2.mixer | 0.542 ± 0.355 | 0.813 ± 0.185 | 0.848 ± 0.287 | 0.923 ± 0.155 | -62.387 | +0.11 / 3.035 |
| gated_linear | 32 | backbone.blocks.3.mixer | 0.695 ± 0.282 | 0.844 ± 0.097 | 0.997 ± 0.010 | 0.998 ± 0.005 | -34.179 | +0.23 / 2.696 |
| gated_linear | 32 | backbone.blocks.4.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.003 | +0.29 / 2.746 |

## 4g. Gate-clamp intervention (Pass 9): which layers' forgetting does recall need?

Per-query recall with one GLA layer's gate replaced by a constant: `keep` = σ(6) ≈ 0.9975 (no forgetting), `mean` = the head's own average γ (same amount of forgetting, no selectivity). Δ vs the unclamped model on the same prompts, paired 95% CI.

**gated_linear**

| condition | 8 facts | 32 facts | 64 facts |
|---|---|---|---|
| none | 0.994 | 0.913 | 0.591 |
| keep_all | 0.013 (-0.982 [-0.99, -0.98]) | 0.011 (-0.902 [-0.91, -0.90]) | 0.010 (-0.580 [-0.59, -0.57]) |
| keep_L0 | 0.013 (-0.982 [-0.99, -0.98]) | 0.011 (-0.901 [-0.91, -0.89]) | 0.010 (-0.580 [-0.59, -0.57]) |
| mean_L0 | 0.143 (-0.851 [-0.86, -0.84]) | 0.324 (-0.589 [-0.60, -0.58]) | 0.217 (-0.373 [-0.39, -0.36]) |
| keep_L1 | 0.465 (-0.530 [-0.54, -0.52]) | 0.325 (-0.588 [-0.60, -0.58]) | 0.206 (-0.385 [-0.40, -0.37]) |
| mean_L1 | 0.515 (-0.479 [-0.49, -0.46]) | 0.470 (-0.443 [-0.45, -0.43]) | 0.342 (-0.248 [-0.26, -0.24]) |
| keep_L2 | 0.864 (-0.130 [-0.14, -0.12]) | 0.576 (-0.337 [-0.35, -0.33]) | 0.302 (-0.288 [-0.30, -0.28]) |
| mean_L2 | 0.953 (-0.042 [-0.05, -0.04]) | 0.808 (-0.105 [-0.11, -0.10]) | 0.511 (-0.079 [-0.09, -0.07]) |
| keep_L3 | 0.986 (-0.008 [-0.01, -0.01]) | 0.870 (-0.043 [-0.05, -0.04]) | 0.595 (+0.004 [-0.01, +0.01]) |
| mean_L3 | 0.976 (-0.019 [-0.02, -0.01]) | 0.819 (-0.094 [-0.10, -0.09]) | 0.520 (-0.070 [-0.08, -0.06]) |
| keep_L4 | 0.994 (-0.001 [-0.00, +0.00]) | 0.881 (-0.032 [-0.04, -0.03]) | 0.512 (-0.078 [-0.09, -0.07]) |
| mean_L4 | 0.994 (+0.000 [+0.00, +0.00]) | 0.913 (-0.000 [-0.00, +0.00]) | 0.590 (-0.000 [-0.00, +0.00]) |

## 4h. Gate replay (Pass 10): does the timing of forgetting matter?

One layer's recorded gates over the context span are replayed unchanged, shuffled in time (same values, same product), replaced by their geometric mean (same product, no timing) or their arithmetic mean (forgets less). Per-query recall, Δ vs baseline [paired 95% CI].

**gated_linear**

| facts | layer | baseline | replay | shuffle (range) | geomean | arith |
|---:|---|---:|---|---|---|---|
| 8 | L0 | 0.993 | 0.993 (+0.000 [0.00, 0.00]) | 0.246 (0.24–0.25) (-0.747 [-0.76, -0.74]) | 0.367 (-0.625 [-0.64, -0.61]) | 0.248 (-0.745 [-0.76, -0.73]) |
| 8 | L1 | 0.993 | 0.993 (+0.000 [0.00, 0.00]) | 0.596 (0.59–0.60) (-0.397 [-0.41, -0.38]) | 0.790 (-0.203 [-0.22, -0.19]) | 0.523 (-0.469 [-0.48, -0.46]) |
| 8 | L2 | 0.993 | 0.993 (+0.000 [0.00, 0.00]) | 0.943 (0.94–0.94) (-0.050 [-0.06, -0.04]) | 0.974 (-0.018 [-0.02, -0.01]) | 0.944 (-0.049 [-0.06, -0.04]) |
| 8 | L3 | 0.993 | 0.993 (+0.000 [0.00, 0.00]) | 0.992 (0.99–0.99) (-0.001 [-0.00, 0.00]) | 0.992 (-0.001 [-0.00, 0.00]) | 0.992 (-0.001 [-0.00, 0.00]) |
| 8 | L4 | 0.993 | 0.993 (+0.000 [0.00, 0.00]) | 0.993 (0.99–0.99) (+0.000 [0.00, 0.00]) | 0.993 (+0.000 [0.00, 0.00]) | 0.993 (+0.000 [0.00, 0.00]) |
| 32 | L0 | 0.916 | 0.916 (+0.000 [0.00, 0.00]) | 0.228 (0.22–0.23) (-0.689 [-0.70, -0.68]) | 0.358 (-0.558 [-0.57, -0.55]) | 0.409 (-0.507 [-0.52, -0.50]) |
| 32 | L1 | 0.916 | 0.916 (+0.000 [0.00, 0.00]) | 0.599 (0.60–0.60) (-0.317 [-0.33, -0.31]) | 0.811 (-0.105 [-0.11, -0.10]) | 0.544 (-0.372 [-0.38, -0.36]) |
| 32 | L2 | 0.916 | 0.916 (+0.000 [0.00, 0.00]) | 0.801 (0.80–0.80) (-0.115 [-0.12, -0.11]) | 0.887 (-0.029 [-0.03, -0.02]) | 0.840 (-0.076 [-0.08, -0.07]) |
| 32 | L3 | 0.916 | 0.916 (+0.000 [0.00, 0.00]) | 0.906 (0.90–0.91) (-0.011 [-0.01, -0.01]) | 0.908 (-0.009 [-0.01, -0.01]) | 0.908 (-0.009 [-0.01, -0.01]) |
| 32 | L4 | 0.916 | 0.916 (+0.000 [0.00, 0.00]) | 0.916 (0.92–0.92) (-0.000 [-0.00, 0.00]) | 0.916 (+0.000 [-0.00, 0.00]) | 0.916 (+0.000 [-0.00, 0.00]) |

## 5. Prefill cost (one forward pass, batch 1; mostly kernel quality)

_`cost_profile.csv` not synced yet._

## 5b. Decode cost (one token at a time, batch 1)

_`decode_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
