# iota run `r16`

- **updated:** 2026-10-14 22:10 UTC (last stage: `eval 8,9,10`)
- **code:** `0df1f56` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | – | – | – | – | – | – | not trained yet |
| gated_linear | 15000 | 0.819 | 0.638 | 1.000 | 0.482 | 0.0015 | ok |
| hybrid | – | – | – | – | – | – | not trained yet |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `*_sweep_history.json` (and `logs/train.log`); `*_sweep.json` stops at the best checkpoint.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| gated_linear | 0.621 | 0.218 | 1.000 | 1.000 | pass |

n=500 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| n_bindings | true tokens | gated_linear |
|---:|---:|---|
| 2 | 256 | 0.994 [0.99–1.00] (0.99) |
| 4 | 256 | 0.984 [0.98–0.99] (0.94) |
| 8 | 256 | 0.955 [0.95–0.96] (0.69) |
| 16 | 256 | 0.871 [0.87–0.88] (0.10) |
| 32 | 289 | 0.662 [0.66–0.67] (0.00) |
| 64 | 386 | 0.311 [0.30–0.32] (0.00) |
| 128 | 773 | 0.097 [0.09–0.10] (0.00) |

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
| 4096 | 4104 | 0.977 [0.97–0.99] (0.98) |
| 8192 | 8200 | 0.882 [0.86–0.90] (0.88) |

## 4f. Decay gates (Pass 8): what γ does on real prompts

Mean γ ± std over tokens × heads, by token role. `retention` = mean log10 of the share of a fact value's write still in the state at the first question. A frozen gate has std ≈ 0; selective forgetting shows as γ(distractor) < γ(fact value).

| model | bindings | layer | fact value | distractor | query | answer | retention (log10) | gate bias / ‖W‖ |
|---|---:|---|---|---|---|---|---:|---|
| gated_linear | 8 | backbone.blocks.0.mixer | 0.125 ± 0.082 | 0.426 ± 0.253 | 0.346 ± 0.305 | 0.130 ± 0.086 | -108.062 | -0.07 / 1.582 |
| gated_linear | 8 | backbone.blocks.1.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.001 | -0.0 | +1.94 / 3.656 |
| gated_linear | 8 | backbone.blocks.2.mixer | 0.999 ± 0.003 | 1.000 ± 0.001 | 0.996 ± 0.016 | 0.999 ± 0.001 | -0.054 | +2.17 / 3.523 |
| gated_linear | 8 | backbone.blocks.3.mixer | 0.979 ± 0.051 | 0.942 ± 0.105 | 0.994 ± 0.016 | 0.992 ± 0.021 | -6.196 | +1.96 / 2.313 |
| gated_linear | 8 | backbone.blocks.4.mixer | 0.974 ± 0.059 | 0.877 ± 0.148 | 0.998 ± 0.010 | 0.999 ± 0.002 | -13.595 | +2.16 / 3.364 |
| gated_linear | 32 | backbone.blocks.0.mixer | 0.124 ± 0.082 | 0.428 ± 0.254 | 0.348 ± 0.306 | 0.130 ± 0.088 | -107.058 | -0.07 / 1.582 |
| gated_linear | 32 | backbone.blocks.1.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.001 | -0.0 | +1.94 / 3.656 |
| gated_linear | 32 | backbone.blocks.2.mixer | 0.999 ± 0.002 | 1.000 ± 0.001 | 0.997 ± 0.011 | 1.000 ± 0.001 | -0.086 | +2.17 / 3.523 |
| gated_linear | 32 | backbone.blocks.3.mixer | 0.989 ± 0.031 | 0.963 ± 0.068 | 0.997 ± 0.007 | 0.996 ± 0.009 | -1.993 | +1.96 / 2.313 |
| gated_linear | 32 | backbone.blocks.4.mixer | 0.983 ± 0.042 | 0.921 ± 0.094 | 0.999 ± 0.005 | 0.999 ± 0.001 | -4.232 | +2.16 / 3.364 |

## 4g. Gate-clamp intervention (Pass 9): which layers' forgetting does recall need?

Per-query recall with one GLA layer's gate replaced by a constant: `keep` = σ(6) ≈ 0.9975 (no forgetting), `mean` = the head's own average γ (same amount of forgetting, no selectivity). Δ vs the unclamped model on the same prompts, paired 95% CI.

**gated_linear**

| condition | 8 facts | 32 facts | 64 facts |
|---|---|---|---|
| none | 0.949 | 0.665 | 0.320 |
| keep_all | 0.017 (-0.931 [-0.94, -0.92]) | 0.011 (-0.653 [-0.66, -0.64]) | 0.011 (-0.309 [-0.32, -0.30]) |
| keep_L0 | 0.018 (-0.931 [-0.94, -0.92]) | 0.011 (-0.653 [-0.66, -0.64]) | 0.010 (-0.310 [-0.32, -0.30]) |
| mean_L0 | 0.431 (-0.517 [-0.53, -0.50]) | 0.218 (-0.447 [-0.46, -0.44]) | 0.100 (-0.220 [-0.23, -0.21]) |
| keep_L1 | 0.950 (+0.002 [-0.00, +0.00]) | 0.598 (-0.067 [-0.07, -0.06]) | 0.227 (-0.093 [-0.10, -0.08]) |
| mean_L1 | 0.949 (+0.000 [+0.00, +0.00]) | 0.665 (+0.000 [-0.00, +0.00]) | 0.320 (+0.000 [-0.00, +0.00]) |
| keep_L2 | 0.943 (-0.006 [-0.01, -0.00]) | 0.651 (-0.014 [-0.02, -0.01]) | 0.308 (-0.013 [-0.02, -0.01]) |
| mean_L2 | 0.950 (+0.001 [-0.00, +0.00]) | 0.668 (+0.003 [+0.00, +0.01]) | 0.322 (+0.002 [+0.00, +0.00]) |
| keep_L3 | 0.949 (-0.000 [-0.00, +0.00]) | 0.662 (-0.003 [-0.01, +0.00]) | 0.324 (+0.004 [+0.00, +0.01]) |
| mean_L3 | 0.943 (-0.005 [-0.01, -0.00]) | 0.664 (-0.001 [-0.01, +0.00]) | 0.319 (-0.001 [-0.00, +0.00]) |
| keep_L4 | 0.927 (-0.022 [-0.03, -0.02]) | 0.649 (-0.016 [-0.02, -0.01]) | 0.307 (-0.013 [-0.02, -0.01]) |
| mean_L4 | 0.863 (-0.086 [-0.10, -0.08]) | 0.614 (-0.051 [-0.06, -0.04]) | 0.307 (-0.013 [-0.02, -0.01]) |

## 4h. Gate replay (Pass 10): does the timing of forgetting matter?

One layer's recorded gates over the context span are replayed unchanged, shuffled in time (same values, same product), replaced by their geometric mean (same product, no timing) or their arithmetic mean (forgets less). Per-query recall, Δ vs baseline [paired 95% CI].

**gated_linear**

| facts | layer | baseline | replay | shuffle (range) | geomean | arith |
|---:|---|---:|---|---|---|---|
| 8 | L0 | 0.956 | 0.956 (+0.000 [0.00, 0.00]) | 0.496 (0.49–0.50) (-0.460 [-0.47, -0.45]) | 0.654 (-0.301 [-0.31, -0.29]) | 0.532 (-0.424 [-0.44, -0.41]) |
| 8 | L1 | 0.956 | 0.956 (+0.000 [0.00, 0.00]) | 0.956 (0.96–0.96) (+0.000 [0.00, 0.00]) | 0.956 (+0.000 [0.00, 0.00]) | 0.956 (+0.000 [0.00, 0.00]) |
| 8 | L2 | 0.956 | 0.956 (+0.000 [0.00, 0.00]) | 0.957 (0.96–0.96) (+0.001 [0.00, 0.00]) | 0.957 (+0.001 [0.00, 0.00]) | 0.957 (+0.001 [0.00, 0.00]) |
| 8 | L3 | 0.956 | 0.956 (+0.000 [0.00, 0.00]) | 0.957 (0.96–0.96) (+0.001 [-0.00, 0.00]) | 0.957 (+0.001 [-0.00, 0.00]) | 0.956 (+0.001 [-0.00, 0.00]) |
| 8 | L4 | 0.956 | 0.956 (+0.000 [0.00, 0.00]) | 0.955 (0.95–0.96) (-0.000 [-0.00, 0.00]) | 0.955 (-0.000 [-0.00, 0.00]) | 0.955 (-0.001 [-0.00, 0.00]) |
| 32 | L0 | 0.667 | 0.667 (+0.000 [0.00, 0.00]) | 0.222 (0.22–0.23) (-0.445 [-0.46, -0.43]) | 0.307 (-0.360 [-0.37, -0.35]) | 0.246 (-0.421 [-0.43, -0.41]) |
| 32 | L1 | 0.667 | 0.667 (+0.000 [0.00, 0.00]) | 0.667 (0.67–0.67) (-0.000 [-0.00, 0.00]) | 0.667 (-0.000 [-0.00, 0.00]) | 0.667 (-0.000 [-0.00, 0.00]) |
| 32 | L2 | 0.667 | 0.667 (+0.000 [0.00, 0.00]) | 0.666 (0.67–0.67) (-0.001 [-0.00, 0.00]) | 0.666 (-0.001 [-0.00, 0.00]) | 0.666 (-0.001 [-0.00, 0.00]) |
| 32 | L3 | 0.667 | 0.667 (+0.000 [0.00, 0.00]) | 0.665 (0.66–0.67) (-0.002 [-0.00, 0.00]) | 0.665 (-0.002 [-0.00, 0.00]) | 0.665 (-0.002 [-0.00, 0.00]) |
| 32 | L4 | 0.667 | 0.667 (+0.000 [0.00, 0.00]) | 0.662 (0.66–0.66) (-0.005 [-0.01, -0.00]) | 0.662 (-0.005 [-0.01, -0.00]) | 0.662 (-0.005 [-0.01, -0.00]) |

## 5. Prefill cost (one forward pass, batch 1; mostly kernel quality)

_`cost_profile.csv` not synced yet._

## 5b. Decode cost (one token at a time, batch 1)

_`decode_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
