# iota run `r36`

- **updated:** 2026-10-20 21:59 UTC (last stage: `eval 8,9,10`)
- **code:** `4d49fdb` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | – | – | – | – | – | – | not trained yet |
| gated_linear | 14500 | 0.944 | 0.887 | 1.000 | 0.676 | 0.00075 | ok |
| hybrid | – | – | – | – | – | – | not trained yet |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `*_sweep_history.json` (and `logs/train.log`); `*_sweep.json` stops at the best checkpoint.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| gated_linear | 0.877 | 0.534 | 1.000 | 1.000 | pass |

n=500 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| n_bindings | true tokens | gated_linear |
|---:|---:|---|
| 2 | 256 | 0.999 [1.00–1.00] (1.00) |
| 4 | 256 | 0.998 [1.00–1.00] (0.99) |
| 8 | 256 | 0.991 [0.99–0.99] (0.94) |
| 16 | 256 | 0.974 [0.97–0.98] (0.68) |
| 32 | 289 | 0.914 [0.91–0.92] (0.24) |
| 64 | 386 | 0.642 [0.63–0.65] (0.00) |
| 128 | 773 | 0.358 [0.35–0.37] (0.00) |

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
| gated_linear | 8 | backbone.blocks.0.mixer | 0.329 ± 0.147 | 0.299 ± 0.118 | 0.301 ± 0.176 | 0.329 ± 0.148 | -128.049 | -0.03 / 0.703 |
| gated_linear | 8 | backbone.blocks.1.mixer | 0.530 ± 0.169 | 0.418 ± 0.151 | 0.464 ± 0.251 | 0.522 ± 0.175 | -94.833 | -0.02 / 0.924 |
| gated_linear | 8 | backbone.blocks.2.mixer | 0.619 ± 0.418 | 0.761 ± 0.117 | 0.951 ± 0.085 | 0.930 ± 0.133 | -31.829 | +0.06 / 1.475 |
| gated_linear | 8 | backbone.blocks.3.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.999 ± 0.003 | 1.000 ± 0.000 | -0.008 | +0.10 / 2.156 |
| gated_linear | 8 | backbone.blocks.4.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.004 | 1.000 ± 0.000 | -0.009 | +0.13 / 1.908 |
| gated_linear | 32 | backbone.blocks.0.mixer | 0.328 ± 0.147 | 0.300 ± 0.119 | 0.302 ± 0.175 | 0.328 ± 0.148 | -105.521 | -0.03 / 0.703 |
| gated_linear | 32 | backbone.blocks.1.mixer | 0.529 ± 0.170 | 0.418 ± 0.152 | 0.465 ± 0.253 | 0.523 ± 0.175 | -81.786 | -0.02 / 0.924 |
| gated_linear | 32 | backbone.blocks.2.mixer | 0.619 ± 0.418 | 0.760 ± 0.127 | 0.954 ± 0.082 | 0.929 ± 0.133 | -37.537 | +0.06 / 1.475 |
| gated_linear | 32 | backbone.blocks.3.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.999 ± 0.003 | 1.000 ± 0.000 | -0.005 | +0.10 / 2.156 |
| gated_linear | 32 | backbone.blocks.4.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.002 | 1.000 ± 0.000 | -0.005 | +0.13 / 1.908 |

## 4g. Gate-clamp intervention (Pass 9): which layers' forgetting does recall need?

Per-query recall with one GLA layer's gate replaced by a constant: `keep` = σ(6) ≈ 0.9975 (no forgetting), `mean` = the head's own average γ (same amount of forgetting, no selectivity). Δ vs the unclamped model on the same prompts, paired 95% CI.

**gated_linear**

| condition | 8 facts | 32 facts | 64 facts |
|---|---|---|---|
| none | 0.993 | 0.915 | 0.652 |
| keep_all | 0.011 (-0.983 [-0.99, -0.98]) | 0.009 (-0.906 [-0.91, -0.90]) | 0.010 (-0.642 [-0.65, -0.63]) |
| keep_L0 | 0.013 (-0.980 [-0.98, -0.98]) | 0.013 (-0.902 [-0.91, -0.89]) | 0.010 (-0.642 [-0.65, -0.63]) |
| mean_L0 | 0.964 (-0.030 [-0.04, -0.02]) | 0.845 (-0.070 [-0.08, -0.06]) | 0.594 (-0.058 [-0.07, -0.05]) |
| keep_L1 | 0.133 (-0.861 [-0.87, -0.85]) | 0.058 (-0.857 [-0.86, -0.85]) | 0.030 (-0.621 [-0.63, -0.61]) |
| mean_L1 | 0.982 (-0.011 [-0.01, -0.01]) | 0.890 (-0.025 [-0.03, -0.02]) | 0.609 (-0.043 [-0.05, -0.04]) |
| keep_L2 | 0.667 (-0.326 [-0.34, -0.31]) | 0.446 (-0.469 [-0.48, -0.46]) | 0.261 (-0.391 [-0.40, -0.38]) |
| mean_L2 | 0.902 (-0.091 [-0.10, -0.08]) | 0.805 (-0.110 [-0.12, -0.10]) | 0.601 (-0.051 [-0.06, -0.04]) |
| keep_L3 | 0.991 (-0.002 [-0.00, -0.00]) | 0.897 (-0.018 [-0.02, -0.01]) | 0.619 (-0.033 [-0.04, -0.03]) |
| mean_L3 | 0.993 (-0.001 [-0.00, +0.00]) | 0.914 (-0.001 [-0.00, +0.00]) | 0.653 (+0.001 [-0.00, +0.00]) |
| keep_L4 | 0.994 (+0.001 [-0.00, +0.00]) | 0.860 (-0.055 [-0.06, -0.05]) | 0.476 (-0.176 [-0.19, -0.17]) |
| mean_L4 | 0.994 (+0.001 [-0.00, +0.00]) | 0.864 (-0.051 [-0.06, -0.04]) | 0.523 (-0.129 [-0.14, -0.12]) |

## 4h. Gate replay (Pass 10): does the timing of forgetting matter?

One layer's recorded gates over the context span are replayed unchanged, shuffled in time (same values, same product), replaced by their geometric mean (same product, no timing) or their arithmetic mean (forgets less). Per-query recall, Δ vs baseline [paired 95% CI].

**gated_linear**

| facts | layer | baseline | replay | shuffle (range) | geomean | arith |
|---:|---|---:|---|---|---|---|
| 8 | L0 | 0.992 | 0.992 (+0.000 [0.00, 0.00]) | 0.958 (0.96–0.96) (-0.034 [-0.04, -0.03]) | 0.968 (-0.024 [-0.03, -0.02]) | 0.962 (-0.030 [-0.04, -0.02]) |
| 8 | L1 | 0.992 | 0.992 (+0.000 [0.00, 0.00]) | 0.972 (0.97–0.97) (-0.020 [-0.02, -0.02]) | 0.977 (-0.015 [-0.02, -0.01]) | 0.976 (-0.016 [-0.02, -0.01]) |
| 8 | L2 | 0.992 | 0.992 (+0.000 [0.00, 0.00]) | 0.922 (0.92–0.92) (-0.069 [-0.08, -0.06]) | 0.949 (-0.043 [-0.05, -0.04]) | 0.922 (-0.069 [-0.08, -0.06]) |
| 8 | L3 | 0.992 | 0.992 (+0.000 [0.00, 0.00]) | 0.991 (0.99–0.99) (-0.000 [-0.00, 0.00]) | 0.992 (-0.000 [-0.00, 0.00]) | 0.992 (-0.000 [-0.00, 0.00]) |
| 8 | L4 | 0.992 | 0.992 (+0.000 [0.00, 0.00]) | 0.941 (0.94–0.95) (-0.051 [-0.06, -0.05]) | 0.974 (-0.018 [-0.02, -0.01]) | 0.990 (-0.002 [-0.00, 0.00]) |
| 32 | L0 | 0.910 | 0.910 (+0.000 [0.00, 0.00]) | 0.782 (0.78–0.79) (-0.128 [-0.14, -0.12]) | 0.840 (-0.070 [-0.08, -0.06]) | 0.802 (-0.108 [-0.12, -0.10]) |
| 32 | L1 | 0.910 | 0.910 (+0.000 [0.00, 0.00]) | 0.832 (0.83–0.83) (-0.079 [-0.08, -0.07]) | 0.864 (-0.046 [-0.05, -0.04]) | 0.857 (-0.053 [-0.06, -0.05]) |
| 32 | L2 | 0.910 | 0.910 (+0.000 [0.00, 0.00]) | 0.815 (0.81–0.82) (-0.095 [-0.10, -0.09]) | 0.886 (-0.024 [-0.03, -0.02]) | 0.843 (-0.068 [-0.07, -0.06]) |
| 32 | L3 | 0.910 | 0.910 (+0.000 [0.00, 0.00]) | 0.901 (0.90–0.90) (-0.009 [-0.01, -0.01]) | 0.905 (-0.005 [-0.01, -0.00]) | 0.906 (-0.005 [-0.01, -0.00]) |
| 32 | L4 | 0.910 | 0.910 (+0.000 [0.00, 0.00]) | 0.593 (0.58–0.60) (-0.317 [-0.33, -0.31]) | 0.623 (-0.287 [-0.30, -0.27]) | 0.828 (-0.083 [-0.09, -0.08]) |

## 5. Prefill cost (one forward pass, batch 1; mostly kernel quality)

_`cost_profile.csv` not synced yet._

## 5b. Decode cost (one token at a time, batch 1)

_`decode_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
