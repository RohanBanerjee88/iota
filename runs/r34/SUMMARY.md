# iota run `r34`

- **updated:** 2026-10-19 23:12 UTC (last stage: `eval 8,9,10`)
- **code:** `09602bd` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | – | – | – | – | – | – | not trained yet |
| gated_linear | 14000 | 0.932 | 0.864 | 1.000 | 0.684 | 0.00075 | ok |
| hybrid | – | – | – | – | – | – | not trained yet |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `*_sweep_history.json` (and `logs/train.log`); `*_sweep.json` stops at the best checkpoint.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| gated_linear | 0.850 | 0.480 | 1.000 | 1.000 | pass |

n=500 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| n_bindings | true tokens | gated_linear |
|---:|---:|---|
| 2 | 256 | 0.999 [1.00–1.00] (1.00) |
| 4 | 256 | 0.997 [1.00–1.00] (0.99) |
| 8 | 256 | 0.993 [0.99–0.99] (0.94) |
| 16 | 256 | 0.973 [0.97–0.98] (0.67) |
| 32 | 289 | 0.897 [0.89–0.90] (0.19) |
| 64 | 386 | 0.589 [0.58–0.60] (0.00) |
| 128 | 773 | 0.266 [0.26–0.27] (0.00) |

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
| 4096 | 4104 | 0.998 [0.99–1.00] (1.00) |
| 8192 | 8200 | 0.979 [0.97–0.99] (0.98) |

## 4f. Decay gates (Pass 8): what γ does on real prompts

Mean γ ± std over tokens × heads, by token role. `retention` = mean log10 of the share of a fact value's write still in the state at the first question. A frozen gate has std ≈ 0; selective forgetting shows as γ(distractor) < γ(fact value).

| model | bindings | layer | fact value | distractor | query | answer | retention (log10) | gate bias / ‖W‖ |
|---|---:|---|---|---|---|---|---:|---|
| gated_linear | 8 | backbone.blocks.0.mixer | 0.272 ± 0.138 | 0.130 ± 0.050 | 0.244 ± 0.161 | 0.276 ± 0.139 | -207.311 | -0.05 / 0.710 |
| gated_linear | 8 | backbone.blocks.1.mixer | 0.998 ± 0.012 | 0.965 ± 0.064 | 1.000 ± 0.002 | 0.999 ± 0.003 | -3.587 | +0.06 / 2.058 |
| gated_linear | 8 | backbone.blocks.2.mixer | 0.975 ± 0.030 | 0.909 ± 0.051 | 0.995 ± 0.007 | 0.989 ± 0.012 | -8.981 | +0.15 / 1.742 |
| gated_linear | 8 | backbone.blocks.3.mixer | 0.932 ± 0.093 | 0.749 ± 0.162 | 0.940 ± 0.087 | 0.955 ± 0.056 | -29.675 | +0.14 / 1.777 |
| gated_linear | 8 | backbone.blocks.4.mixer | 0.792 ± 0.340 | 0.783 ± 0.352 | 0.843 ± 0.262 | 0.809 ± 0.326 | -44.258 | +0.07 / 1.380 |
| gated_linear | 32 | backbone.blocks.0.mixer | 0.273 ± 0.138 | 0.131 ± 0.051 | 0.245 ± 0.161 | 0.277 ± 0.139 | -155.969 | -0.05 / 0.710 |
| gated_linear | 32 | backbone.blocks.1.mixer | 0.998 ± 0.013 | 0.964 ± 0.066 | 1.000 ± 0.002 | 0.999 ± 0.003 | -2.145 | +0.06 / 2.058 |
| gated_linear | 32 | backbone.blocks.2.mixer | 0.978 ± 0.027 | 0.926 ± 0.041 | 0.997 ± 0.004 | 0.992 ± 0.008 | -4.131 | +0.15 / 1.742 |
| gated_linear | 32 | backbone.blocks.3.mixer | 0.939 ± 0.081 | 0.772 ± 0.145 | 0.945 ± 0.080 | 0.959 ± 0.054 | -16.163 | +0.14 / 1.777 |
| gated_linear | 32 | backbone.blocks.4.mixer | 0.799 ± 0.334 | 0.792 ± 0.341 | 0.851 ± 0.252 | 0.817 ± 0.312 | -31.625 | +0.07 / 1.380 |

## 4g. Gate-clamp intervention (Pass 9): which layers' forgetting does recall need?

Per-query recall with one GLA layer's gate replaced by a constant: `keep` = σ(6) ≈ 0.9975 (no forgetting), `mean` = the head's own average γ (same amount of forgetting, no selectivity). Δ vs the unclamped model on the same prompts, paired 95% CI.

**gated_linear**

| condition | 8 facts | 32 facts | 64 facts |
|---|---|---|---|
| none | 0.991 | 0.896 | 0.593 |
| keep_all | 0.010 (-0.981 [-0.99, -0.98]) | 0.009 (-0.887 [-0.89, -0.88]) | 0.013 (-0.580 [-0.59, -0.57]) |
| keep_L0 | 0.009 (-0.982 [-0.99, -0.98]) | 0.009 (-0.887 [-0.89, -0.88]) | 0.013 (-0.580 [-0.59, -0.57]) |
| mean_L0 | 0.921 (-0.070 [-0.08, -0.06]) | 0.782 (-0.114 [-0.12, -0.11]) | 0.519 (-0.073 [-0.08, -0.06]) |
| keep_L1 | 0.988 (-0.003 [-0.01, -0.00]) | 0.863 (-0.033 [-0.04, -0.03]) | 0.544 (-0.049 [-0.06, -0.04]) |
| mean_L1 | 0.989 (-0.002 [-0.00, +0.00]) | 0.873 (-0.024 [-0.03, -0.02]) | 0.581 (-0.013 [-0.02, -0.01]) |
| keep_L2 | 0.991 (+0.001 [-0.00, +0.00]) | 0.895 (-0.001 [-0.00, +0.00]) | 0.601 (+0.008 [+0.00, +0.01]) |
| mean_L2 | 0.991 (+0.000 [-0.00, +0.00]) | 0.895 (-0.001 [-0.00, +0.00]) | 0.591 (-0.003 [-0.00, -0.00]) |
| keep_L3 | 0.991 (+0.000 [-0.00, +0.00]) | 0.896 (-0.000 [-0.00, +0.00]) | 0.593 (-0.001 [-0.00, +0.00]) |
| mean_L3 | 0.991 (+0.000 [-0.00, +0.00]) | 0.896 (-0.001 [-0.00, +0.00]) | 0.593 (+0.000 [-0.00, +0.00]) |
| keep_L4 | 0.991 (+0.000 [-0.00, +0.00]) | 0.892 (-0.004 [-0.01, -0.00]) | 0.588 (-0.005 [-0.01, -0.00]) |
| mean_L4 | 0.991 (+0.000 [-0.00, +0.00]) | 0.895 (-0.001 [-0.00, +0.00]) | 0.593 (-0.001 [-0.00, +0.00]) |

## 4h. Gate replay (Pass 10): does the timing of forgetting matter?

One layer's recorded gates over the context span are replayed unchanged, shuffled in time (same values, same product), replaced by their geometric mean (same product, no timing) or their arithmetic mean (forgets less). Per-query recall, Δ vs baseline [paired 95% CI].

**gated_linear**

| facts | layer | baseline | replay | shuffle (range) | geomean | arith |
|---:|---|---:|---|---|---|---|
| 8 | L0 | 0.992 | 0.992 (+0.000 [0.00, 0.00]) | 0.916 (0.91–0.92) (-0.076 [-0.08, -0.07]) | 0.924 (-0.068 [-0.08, -0.06]) | 0.938 (-0.054 [-0.06, -0.05]) |
| 8 | L1 | 0.992 | 0.992 (+0.000 [0.00, 0.00]) | 0.992 (0.99–0.99) (-0.000 [-0.00, 0.00]) | 0.991 (-0.001 [-0.00, 0.00]) | 0.991 (-0.001 [-0.00, 0.00]) |
| 8 | L2 | 0.992 | 0.992 (+0.000 [0.00, 0.00]) | 0.992 (0.99–0.99) (-0.000 [-0.00, 0.00]) | 0.992 (-0.000 [-0.00, 0.00]) | 0.992 (-0.000 [-0.00, 0.00]) |
| 8 | L3 | 0.992 | 0.992 (+0.000 [0.00, 0.00]) | 0.992 (0.99–0.99) (+0.000 [0.00, 0.00]) | 0.992 (+0.000 [0.00, 0.00]) | 0.992 (+0.000 [0.00, 0.00]) |
| 8 | L4 | 0.992 | 0.992 (+0.000 [0.00, 0.00]) | 0.992 (0.99–0.99) (-0.000 [-0.00, 0.00]) | 0.992 (-0.001 [-0.00, 0.00]) | 0.992 (-0.001 [-0.00, 0.00]) |
| 32 | L0 | 0.899 | 0.899 (+0.000 [0.00, 0.00]) | 0.702 (0.70–0.70) (-0.197 [-0.20, -0.19]) | 0.751 (-0.148 [-0.16, -0.14]) | 0.793 (-0.106 [-0.11, -0.10]) |
| 32 | L1 | 0.899 | 0.899 (+0.000 [0.00, 0.00]) | 0.896 (0.90–0.90) (-0.002 [-0.00, -0.00]) | 0.897 (-0.002 [-0.00, 0.00]) | 0.897 (-0.002 [-0.00, 0.00]) |
| 32 | L2 | 0.899 | 0.899 (+0.000 [0.00, 0.00]) | 0.898 (0.90–0.90) (-0.000 [-0.00, 0.00]) | 0.899 (-0.000 [-0.00, 0.00]) | 0.898 (-0.000 [-0.00, 0.00]) |
| 32 | L3 | 0.899 | 0.899 (+0.000 [0.00, 0.00]) | 0.899 (0.90–0.90) (-0.000 [-0.00, 0.00]) | 0.899 (-0.000 [-0.00, 0.00]) | 0.899 (-0.000 [-0.00, 0.00]) |
| 32 | L4 | 0.899 | 0.899 (+0.000 [0.00, 0.00]) | 0.898 (0.90–0.90) (-0.001 [-0.00, 0.00]) | 0.897 (-0.001 [-0.00, -0.00]) | 0.897 (-0.001 [-0.00, 0.00]) |

## 5. Prefill cost (one forward pass, batch 1; mostly kernel quality)

_`cost_profile.csv` not synced yet._

## 5b. Decode cost (one token at a time, batch 1)

_`decode_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
