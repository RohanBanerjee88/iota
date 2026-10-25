# iota run `r37`

- **updated:** 2026-10-25 17:12 UTC (last stage: `eval 8,9,10`)
- **code:** `45e8d0d` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | – | – | – | – | – | – | not trained yet |
| gated_linear | 15000 | 0.964 | 0.928 | 1.000 | 0.770 | 0.0015 | ok |
| hybrid | – | – | – | – | – | – | not trained yet |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `*_sweep_history.json` (and `logs/train.log`); `*_sweep.json` stops at the best checkpoint.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| gated_linear | 0.920 | 0.630 | 1.000 | 1.000 | pass |

n=500 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| n_bindings | true tokens | gated_linear |
|---:|---:|---|
| 2 | 256 | 1.000 [1.00–1.00] (1.00) |
| 4 | 256 | 0.998 [1.00–1.00] (0.99) |
| 8 | 256 | 0.996 [0.99–1.00] (0.97) |
| 16 | 256 | 0.989 [0.99–0.99] (0.84) |
| 32 | 289 | 0.955 [0.95–0.96] (0.48) |
| 64 | 386 | 0.802 [0.80–0.81] (0.03) |
| 128 | 773 | 0.441 [0.43–0.45] (0.00) |

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
| 4096 | 4104 | 0.988 [0.98–0.99] (0.99) |
| 8192 | 8200 | 0.942 [0.93–0.96] (0.94) |

## 4f. Decay gates (Pass 8): what γ does on real prompts

Mean γ ± std over tokens × heads, by token role. `retention` = mean log10 of the share of a fact value's write still in the state at the first question. A frozen gate has std ≈ 0; selective forgetting shows as γ(distractor) < γ(fact value).

| model | bindings | layer | fact value | distractor | query | answer | retention (log10) | gate bias / ‖W‖ |
|---|---:|---|---|---|---|---|---:|---|
| gated_linear | 8 | backbone.blocks.0.mixer | 0.627 ± 0.241 | 0.381 ± 0.217 | 0.380 ± 0.329 | 0.641 ± 0.238 | -117.581 | +1.70 / 1.498 |
| gated_linear | 8 | backbone.blocks.1.mixer | 0.965 ± 0.081 | 0.992 ± 0.035 | 0.995 ± 0.017 | 0.984 ± 0.041 | -2.281 | +1.96 / 2.546 |
| gated_linear | 8 | backbone.blocks.2.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.0 | +2.05 / 2.811 |
| gated_linear | 8 | backbone.blocks.3.mixer | 0.999 ± 0.001 | 1.000 ± 0.000 | 0.999 ± 0.001 | 0.999 ± 0.002 | -0.021 | +1.98 / 2.226 |
| gated_linear | 8 | backbone.blocks.4.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.002 | +2.09 / 2.532 |
| gated_linear | 32 | backbone.blocks.0.mixer | 0.627 ± 0.240 | 0.381 ± 0.217 | 0.380 ± 0.330 | 0.642 ± 0.237 | -102.296 | +1.70 / 1.498 |
| gated_linear | 32 | backbone.blocks.1.mixer | 0.965 ± 0.082 | 0.990 ± 0.050 | 0.994 ± 0.018 | 0.984 ± 0.041 | -6.436 | +1.96 / 2.546 |
| gated_linear | 32 | backbone.blocks.2.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.0 | +2.05 / 2.811 |
| gated_linear | 32 | backbone.blocks.3.mixer | 0.999 ± 0.002 | 1.000 ± 0.000 | 0.999 ± 0.001 | 0.999 ± 0.002 | -0.063 | +1.98 / 2.226 |
| gated_linear | 32 | backbone.blocks.4.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.005 | +2.09 / 2.532 |

## 4g. Gate-clamp intervention (Pass 9): which layers' forgetting does recall need?

Per-query recall with one GLA layer's gate replaced by a constant: `keep` = σ(6) ≈ 0.9975 (no forgetting), `mean` = the head's own average γ (same amount of forgetting, no selectivity). Δ vs the unclamped model on the same prompts, paired 95% CI.

**gated_linear**

| condition | 8 facts | 32 facts | 64 facts |
|---|---|---|---|
| none | 0.997 | 0.956 | 0.801 |
| keep_all | 0.009 (-0.987 [-0.99, -0.98]) | 0.004 (-0.952 [-0.96, -0.95]) | 0.007 (-0.794 [-0.80, -0.78]) |
| keep_L0 | 0.011 (-0.986 [-0.99, -0.98]) | 0.005 (-0.951 [-0.96, -0.95]) | 0.006 (-0.795 [-0.80, -0.79]) |
| mean_L0 | 0.805 (-0.192 [-0.20, -0.18]) | 0.598 (-0.358 [-0.37, -0.35]) | 0.402 (-0.399 [-0.41, -0.39]) |
| keep_L1 | 0.992 (-0.005 [-0.01, -0.00]) | 0.909 (-0.047 [-0.05, -0.04]) | 0.706 (-0.094 [-0.10, -0.09]) |
| mean_L1 | 0.993 (-0.004 [-0.01, -0.00]) | 0.932 (-0.024 [-0.03, -0.02]) | 0.765 (-0.036 [-0.04, -0.03]) |
| keep_L2 | 0.989 (-0.008 [-0.01, -0.00]) | 0.756 (-0.201 [-0.21, -0.19]) | 0.426 (-0.374 [-0.39, -0.36]) |
| mean_L2 | 0.981 (-0.015 [-0.02, -0.01]) | 0.748 (-0.208 [-0.22, -0.20]) | 0.449 (-0.352 [-0.36, -0.34]) |
| keep_L3 | 0.997 (+0.000 [+0.00, +0.00]) | 0.957 (+0.000 [-0.00, +0.00]) | 0.799 (-0.002 [-0.00, +0.00]) |
| mean_L3 | 0.997 (+0.000 [+0.00, +0.00]) | 0.957 (+0.001 [-0.00, +0.00]) | 0.801 (+0.000 [-0.00, +0.00]) |
| keep_L4 | 0.997 (-0.000 [-0.00, +0.00]) | 0.955 (-0.002 [-0.00, +0.00]) | 0.788 (-0.013 [-0.02, -0.01]) |
| mean_L4 | 0.997 (+0.000 [+0.00, +0.00]) | 0.956 (+0.000 [+0.00, +0.00]) | 0.801 (+0.000 [+0.00, +0.00]) |

## 4h. Gate replay (Pass 10): does the timing of forgetting matter?

One layer's recorded gates over the context span are replayed unchanged, shuffled in time (same values, same product), replaced by their geometric mean (same product, no timing) or their arithmetic mean (forgets less). Per-query recall, Δ vs baseline [paired 95% CI].

**gated_linear**

| facts | layer | baseline | replay | shuffle (range) | geomean | arith |
|---:|---|---:|---|---|---|---|
| 8 | L0 | 0.997 | 0.997 (+0.000 [0.00, 0.00]) | 0.722 (0.71–0.73) (-0.274 [-0.28, -0.26]) | 0.815 (-0.181 [-0.19, -0.17]) | 0.797 (-0.200 [-0.21, -0.19]) |
| 8 | L1 | 0.997 | 0.997 (+0.000 [0.00, 0.00]) | 0.994 (0.99–0.99) (-0.003 [-0.00, -0.00]) | 0.995 (-0.002 [-0.00, -0.00]) | 0.994 (-0.002 [-0.00, -0.00]) |
| 8 | L2 | 0.997 | 0.997 (+0.000 [0.00, 0.00]) | 0.216 (0.22–0.22) (-0.780 [-0.79, -0.77]) | 0.228 (-0.769 [-0.78, -0.76]) | 0.969 (-0.028 [-0.03, -0.02]) |
| 8 | L3 | 0.997 | 0.997 (+0.000 [0.00, 0.00]) | 0.997 (1.00–1.00) (+0.000 [0.00, 0.00]) | 0.997 (+0.000 [0.00, 0.00]) | 0.997 (+0.000 [0.00, 0.00]) |
| 8 | L4 | 0.997 | 0.997 (+0.000 [0.00, 0.00]) | 0.997 (1.00–1.00) (+0.000 [0.00, 0.00]) | 0.997 (+0.000 [0.00, 0.00]) | 0.997 (+0.000 [0.00, 0.00]) |
| 32 | L0 | 0.958 | 0.958 (+0.000 [0.00, 0.00]) | 0.423 (0.42–0.43) (-0.535 [-0.54, -0.53]) | 0.641 (-0.317 [-0.33, -0.31]) | 0.572 (-0.386 [-0.40, -0.38]) |
| 32 | L1 | 0.958 | 0.958 (+0.000 [0.00, 0.00]) | 0.926 (0.92–0.93) (-0.032 [-0.04, -0.03]) | 0.945 (-0.013 [-0.02, -0.01]) | 0.939 (-0.019 [-0.02, -0.01]) |
| 32 | L2 | 0.958 | 0.958 (+0.000 [0.00, 0.00]) | 0.158 (0.15–0.16) (-0.800 [-0.81, -0.79]) | 0.218 (-0.740 [-0.75, -0.73]) | 0.700 (-0.257 [-0.27, -0.25]) |
| 32 | L3 | 0.958 | 0.958 (+0.000 [0.00, 0.00]) | 0.958 (0.96–0.96) (-0.000 [-0.00, 0.00]) | 0.958 (+0.000 [-0.00, 0.00]) | 0.958 (+0.001 [-0.00, 0.00]) |
| 32 | L4 | 0.958 | 0.958 (+0.000 [0.00, 0.00]) | 0.958 (0.96–0.96) (+0.000 [0.00, 0.00]) | 0.958 (+0.000 [0.00, 0.00]) | 0.958 (+0.000 [0.00, 0.00]) |

## 5. Prefill cost (one forward pass, batch 1; mostly kernel quality)

_`cost_profile.csv` not synced yet._

## 5b. Decode cost (one token at a time, batch 1)

_`decode_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
