# iota run `r30`

- **updated:** 2026-10-17 23:47 UTC (last stage: `eval 8,9,10`)
- **code:** `1e87232` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | – | – | – | – | – | – | not trained yet |
| gated_linear | 14500 | 0.931 | 0.861 | 1.000 | 0.656 | 0.00075 | ok |
| hybrid | – | – | – | – | – | – | not trained yet |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `*_sweep_history.json` (and `logs/train.log`); `*_sweep.json` stops at the best checkpoint.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| gated_linear | 0.844 | 0.470 | 1.000 | 1.000 | pass |

n=500 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| n_bindings | true tokens | gated_linear |
|---:|---:|---|
| 2 | 256 | 1.000 [1.00–1.00] (1.00) |
| 4 | 256 | 0.998 [1.00–1.00] (0.99) |
| 8 | 256 | 0.993 [0.99–0.99] (0.95) |
| 16 | 256 | 0.968 [0.97–0.97] (0.64) |
| 32 | 289 | 0.879 [0.87–0.88] (0.15) |
| 64 | 386 | 0.617 [0.61–0.62] (0.00) |
| 128 | 773 | 0.339 [0.33–0.35] (0.00) |

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
| 4096 | 4104 | 0.964 [0.95–0.97] (0.96) |
| 8192 | 8200 | 0.858 [0.84–0.88] (0.86) |

## 4f. Decay gates (Pass 8): what γ does on real prompts

Mean γ ± std over tokens × heads, by token role. `retention` = mean log10 of the share of a fact value's write still in the state at the first question. A frozen gate has std ≈ 0; selective forgetting shows as γ(distractor) < γ(fact value).

| model | bindings | layer | fact value | distractor | query | answer | retention (log10) | gate bias / ‖W‖ |
|---|---:|---|---|---|---|---|---:|---|
| gated_linear | 8 | backbone.blocks.0.mixer | 0.366 ± 0.125 | 0.158 ± 0.058 | 0.239 ± 0.158 | 0.363 ± 0.128 | -185.098 | -0.02 / 0.653 |
| gated_linear | 8 | backbone.blocks.1.mixer | 0.558 ± 0.171 | 0.443 ± 0.100 | 0.659 ± 0.164 | 0.587 ± 0.172 | -83.72 | +0.01 / 0.900 |
| gated_linear | 8 | backbone.blocks.2.mixer | 0.548 ± 0.317 | 0.443 ± 0.303 | 0.431 ± 0.334 | 0.551 ± 0.319 | -110.516 | -0.01 / 1.054 |
| gated_linear | 8 | backbone.blocks.3.mixer | 0.999 ± 0.004 | 0.957 ± 0.078 | 1.000 ± 0.002 | 1.000 ± 0.001 | -4.365 | +0.13 / 2.331 |
| gated_linear | 8 | backbone.blocks.4.mixer | 1.000 ± 0.000 | 1.000 ± 0.001 | 0.998 ± 0.010 | 1.000 ± 0.000 | -0.046 | +0.09 / 1.758 |
| gated_linear | 32 | backbone.blocks.0.mixer | 0.365 ± 0.125 | 0.159 ± 0.060 | 0.241 ± 0.157 | 0.363 ± 0.129 | -133.419 | -0.02 / 0.653 |
| gated_linear | 32 | backbone.blocks.1.mixer | 0.557 ± 0.172 | 0.442 ± 0.101 | 0.661 ± 0.165 | 0.587 ± 0.172 | -68.11 | +0.01 / 0.900 |
| gated_linear | 32 | backbone.blocks.2.mixer | 0.547 ± 0.317 | 0.443 ± 0.302 | 0.432 ± 0.334 | 0.551 ± 0.319 | -91.658 | -0.01 / 1.054 |
| gated_linear | 32 | backbone.blocks.3.mixer | 0.999 ± 0.003 | 0.956 ± 0.082 | 1.000 ± 0.001 | 1.000 ± 0.001 | -2.102 | +0.13 / 2.331 |
| gated_linear | 32 | backbone.blocks.4.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.999 ± 0.005 | 1.000 ± 0.000 | -0.015 | +0.09 / 1.758 |

## 4g. Gate-clamp intervention (Pass 9): which layers' forgetting does recall need?

Per-query recall with one GLA layer's gate replaced by a constant: `keep` = σ(6) ≈ 0.9975 (no forgetting), `mean` = the head's own average γ (same amount of forgetting, no selectivity). Δ vs the unclamped model on the same prompts, paired 95% CI.

**gated_linear**

| condition | 8 facts | 32 facts | 64 facts |
|---|---|---|---|
| none | 0.990 | 0.884 | 0.624 |
| keep_all | 0.011 (-0.979 [-0.98, -0.97]) | 0.011 (-0.873 [-0.88, -0.87]) | 0.009 (-0.614 [-0.62, -0.60]) |
| keep_L0 | 0.011 (-0.979 [-0.98, -0.97]) | 0.011 (-0.872 [-0.88, -0.86]) | 0.009 (-0.615 [-0.62, -0.60]) |
| mean_L0 | 0.883 (-0.106 [-0.12, -0.10]) | 0.801 (-0.083 [-0.09, -0.08]) | 0.552 (-0.071 [-0.08, -0.06]) |
| keep_L1 | 0.565 (-0.425 [-0.44, -0.41]) | 0.407 (-0.476 [-0.49, -0.46]) | 0.281 (-0.342 [-0.35, -0.33]) |
| mean_L1 | 0.989 (-0.001 [-0.00, +0.00]) | 0.877 (-0.007 [-0.01, -0.00]) | 0.616 (-0.007 [-0.01, -0.00]) |
| keep_L2 | 0.362 (-0.627 [-0.64, -0.61]) | 0.227 (-0.657 [-0.67, -0.65]) | 0.139 (-0.484 [-0.50, -0.47]) |
| mean_L2 | 0.978 (-0.011 [-0.02, -0.01]) | 0.847 (-0.036 [-0.04, -0.03]) | 0.582 (-0.041 [-0.05, -0.03]) |
| keep_L3 | 0.984 (-0.006 [-0.01, -0.00]) | 0.718 (-0.166 [-0.18, -0.16]) | 0.391 (-0.233 [-0.24, -0.22]) |
| mean_L3 | 0.965 (-0.025 [-0.03, -0.02]) | 0.731 (-0.153 [-0.16, -0.14]) | 0.430 (-0.193 [-0.20, -0.18]) |
| keep_L4 | 0.990 (+0.000 [-0.00, +0.00]) | 0.880 (-0.004 [-0.01, -0.00]) | 0.608 (-0.016 [-0.02, -0.01]) |
| mean_L4 | 0.990 (+0.000 [+0.00, +0.00]) | 0.884 (+0.000 [-0.00, +0.00]) | 0.624 (+0.001 [-0.00, +0.00]) |

## 4h. Gate replay (Pass 10): does the timing of forgetting matter?

One layer's recorded gates over the context span are replayed unchanged, shuffled in time (same values, same product), replaced by their geometric mean (same product, no timing) or their arithmetic mean (forgets less). Per-query recall, Δ vs baseline [paired 95% CI].

**gated_linear**

| facts | layer | baseline | replay | shuffle (range) | geomean | arith |
|---:|---|---:|---|---|---|---|
| 8 | L0 | 0.988 | 0.988 (+0.000 [0.00, 0.00]) | 0.824 (0.82–0.83) (-0.164 [-0.17, -0.15]) | 0.826 (-0.163 [-0.17, -0.15]) | 0.852 (-0.136 [-0.15, -0.13]) |
| 8 | L1 | 0.988 | 0.988 (+0.000 [0.00, 0.00]) | 0.988 (0.99–0.99) (+0.000 [-0.00, 0.00]) | 0.988 (-0.000 [-0.00, 0.00]) | 0.988 (-0.000 [-0.00, 0.00]) |
| 8 | L2 | 0.988 | 0.988 (+0.000 [0.00, 0.00]) | 0.965 (0.96–0.97) (-0.024 [-0.03, -0.02]) | 0.972 (-0.016 [-0.02, -0.01]) | 0.977 (-0.012 [-0.02, -0.01]) |
| 8 | L3 | 0.988 | 0.988 (+0.000 [0.00, 0.00]) | 0.534 (0.53–0.55) (-0.454 [-0.47, -0.44]) | 0.621 (-0.367 [-0.38, -0.35]) | 0.980 (-0.008 [-0.01, -0.00]) |
| 8 | L4 | 0.988 | 0.988 (+0.000 [0.00, 0.00]) | 0.989 (0.99–0.99) (+0.000 [0.00, 0.00]) | 0.989 (+0.000 [0.00, 0.00]) | 0.989 (+0.000 [0.00, 0.00]) |
| 32 | L0 | 0.878 | 0.878 (+0.000 [0.00, 0.00]) | 0.716 (0.71–0.72) (-0.162 [-0.17, -0.15]) | 0.783 (-0.094 [-0.10, -0.09]) | 0.825 (-0.052 [-0.06, -0.05]) |
| 32 | L1 | 0.878 | 0.878 (+0.000 [0.00, 0.00]) | 0.876 (0.88–0.88) (-0.001 [-0.00, 0.00]) | 0.877 (-0.001 [-0.00, 0.00]) | 0.877 (-0.001 [-0.00, 0.00]) |
| 32 | L2 | 0.878 | 0.878 (+0.000 [0.00, 0.00]) | 0.797 (0.80–0.80) (-0.081 [-0.09, -0.07]) | 0.831 (-0.046 [-0.05, -0.04]) | 0.847 (-0.031 [-0.04, -0.03]) |
| 32 | L3 | 0.878 | 0.878 (+0.000 [0.00, 0.00]) | 0.342 (0.34–0.35) (-0.535 [-0.55, -0.53]) | 0.375 (-0.503 [-0.51, -0.49]) | 0.650 (-0.228 [-0.24, -0.22]) |
| 32 | L4 | 0.878 | 0.878 (+0.000 [0.00, 0.00]) | 0.878 (0.88–0.88) (+0.000 [-0.00, 0.00]) | 0.878 (+0.000 [-0.00, 0.00]) | 0.878 (+0.000 [-0.00, 0.00]) |

## 5. Prefill cost (one forward pass, batch 1; mostly kernel quality)

_`cost_profile.csv` not synced yet._

## 5b. Decode cost (one token at a time, batch 1)

_`decode_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
