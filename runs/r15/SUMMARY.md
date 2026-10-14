# iota run `r15`

- **updated:** 2026-10-14 01:09 UTC (last stage: `eval 8,9,10`)
- **code:** `04ff3c6` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | – | – | – | – | – | – | not trained yet |
| gated_linear | 15000 | 0.754 | 0.509 | 1.000 | 0.462 | 0.0015 | ok |
| hybrid | – | – | – | – | – | – | not trained yet |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `*_sweep_history.json` (and `logs/train.log`); `*_sweep.json` stops at the best checkpoint.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| gated_linear | 0.500 | 0.184 | 1.000 | 1.000 | pass |

n=500 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| n_bindings | true tokens | gated_linear |
|---:|---:|---|
| 2 | 256 | 0.994 [0.99–1.00] (0.99) |
| 4 | 256 | 0.975 [0.97–0.98] (0.90) |
| 8 | 256 | 0.924 [0.92–0.93] (0.54) |
| 16 | 256 | 0.807 [0.80–0.81] (0.02) |
| 32 | 289 | 0.522 [0.51–0.53] (0.00) |
| 64 | 386 | 0.192 [0.19–0.20] (0.00) |
| 128 | 773 | 0.080 [0.08–0.08] (0.00) |

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
| gated_linear | 8 | backbone.blocks.0.mixer | 0.500 ± 0.289 | 0.481 ± 0.259 | 0.437 ± 0.344 | 0.513 ± 0.284 | -99.02 | +1.71 / 1.428 |
| gated_linear | 8 | backbone.blocks.1.mixer | 0.756 ± 0.271 | 0.445 ± 0.121 | 0.806 ± 0.307 | 0.796 ± 0.286 | -83.537 | +0.05 / 1.669 |
| gated_linear | 8 | backbone.blocks.2.mixer | 0.696 ± 0.301 | 0.450 ± 0.169 | 0.900 ± 0.177 | 0.842 ± 0.292 | -85.579 | +0.10 / 1.888 |
| gated_linear | 8 | backbone.blocks.3.mixer | 0.974 ± 0.082 | 0.910 ± 0.160 | 1.000 ± 0.002 | 0.999 ± 0.003 | -10.886 | +0.20 / 2.807 |
| gated_linear | 8 | backbone.blocks.4.mixer | 0.778 ± 0.360 | 0.749 ± 0.407 | 0.774 ± 0.390 | 0.781 ± 0.379 | -77.07 | +0.12 / 2.114 |
| gated_linear | 32 | backbone.blocks.0.mixer | 0.499 ± 0.290 | 0.482 ± 0.260 | 0.441 ± 0.344 | 0.512 ± 0.284 | -86.477 | +1.71 / 1.428 |
| gated_linear | 32 | backbone.blocks.1.mixer | 0.758 ± 0.269 | 0.449 ± 0.134 | 0.806 ± 0.311 | 0.796 ± 0.286 | -65.044 | +0.05 / 1.669 |
| gated_linear | 32 | backbone.blocks.2.mixer | 0.699 ± 0.302 | 0.452 ± 0.176 | 0.900 ± 0.177 | 0.843 ± 0.292 | -66.082 | +0.10 / 1.888 |
| gated_linear | 32 | backbone.blocks.3.mixer | 0.974 ± 0.081 | 0.908 ± 0.166 | 1.000 ± 0.002 | 0.999 ± 0.003 | -7.27 | +0.20 / 2.807 |
| gated_linear | 32 | backbone.blocks.4.mixer | 0.776 ± 0.367 | 0.750 ± 0.405 | 0.774 ± 0.390 | 0.775 ± 0.389 | -56.385 | +0.12 / 2.114 |

## 4g. Gate-clamp intervention (Pass 9): which layers' forgetting does recall need?

Per-query recall with one GLA layer's gate replaced by a constant: `keep` = σ(6) ≈ 0.9975 (no forgetting), `mean` = the head's own average γ (same amount of forgetting, no selectivity). Δ vs the unclamped model on the same prompts, paired 95% CI.

**gated_linear**

| condition | 8 facts | 32 facts | 64 facts |
|---|---|---|---|
| none | 0.928 | 0.527 | 0.202 |
| keep_all | 0.018 (-0.910 [-0.92, -0.90]) | 0.012 (-0.516 [-0.53, -0.51]) | 0.012 (-0.190 [-0.20, -0.18]) |
| keep_L0 | 0.023 (-0.905 [-0.92, -0.90]) | 0.010 (-0.517 [-0.53, -0.51]) | 0.011 (-0.191 [-0.20, -0.18]) |
| mean_L0 | 0.645 (-0.283 [-0.30, -0.27]) | 0.294 (-0.233 [-0.24, -0.22]) | 0.128 (-0.074 [-0.08, -0.07]) |
| keep_L1 | 0.588 (-0.340 [-0.36, -0.32]) | 0.225 (-0.302 [-0.32, -0.29]) | 0.115 (-0.087 [-0.10, -0.08]) |
| mean_L1 | 0.772 (-0.156 [-0.17, -0.14]) | 0.415 (-0.112 [-0.12, -0.10]) | 0.190 (-0.012 [-0.02, -0.01]) |
| keep_L2 | 0.484 (-0.444 [-0.46, -0.42]) | 0.174 (-0.353 [-0.37, -0.34]) | 0.079 (-0.123 [-0.13, -0.11]) |
| mean_L2 | 0.726 (-0.203 [-0.22, -0.19]) | 0.360 (-0.168 [-0.18, -0.16]) | 0.181 (-0.021 [-0.03, -0.01]) |
| keep_L3 | 0.877 (-0.051 [-0.06, -0.04]) | 0.397 (-0.130 [-0.14, -0.12]) | 0.137 (-0.065 [-0.07, -0.06]) |
| mean_L3 | 0.876 (-0.052 [-0.06, -0.04]) | 0.479 (-0.048 [-0.06, -0.04]) | 0.194 (-0.008 [-0.02, -0.00]) |
| keep_L4 | 0.921 (-0.007 [-0.01, -0.00]) | 0.513 (-0.014 [-0.02, -0.01]) | 0.190 (-0.012 [-0.02, -0.01]) |
| mean_L4 | 0.928 (+0.000 [-0.00, +0.00]) | 0.525 (-0.003 [-0.01, +0.00]) | 0.202 (+0.000 [-0.00, +0.00]) |

## 4h. Gate replay (Pass 10): does the timing of forgetting matter?

One layer's recorded gates over the context span are replayed unchanged, shuffled in time (same values, same product), replaced by their geometric mean (same product, no timing) or their arithmetic mean (forgets less). Per-query recall, Δ vs baseline [paired 95% CI].

**gated_linear**

| facts | layer | baseline | replay | shuffle (range) | geomean | arith |
|---:|---|---:|---|---|---|---|
| 8 | L0 | 0.924 | 0.924 (+0.000 [0.00, 0.00]) | 0.589 (0.58–0.60) (-0.335 [-0.35, -0.32]) | 0.708 (-0.217 [-0.23, -0.20]) | 0.637 (-0.287 [-0.30, -0.27]) |
| 8 | L1 | 0.924 | 0.924 (+0.000 [0.00, 0.00]) | 0.892 (0.89–0.89) (-0.033 [-0.04, -0.03]) | 0.900 (-0.025 [-0.03, -0.02]) | 0.900 (-0.024 [-0.03, -0.02]) |
| 8 | L2 | 0.924 | 0.924 (+0.000 [0.00, 0.00]) | 0.837 (0.84–0.84) (-0.087 [-0.10, -0.08]) | 0.846 (-0.079 [-0.09, -0.07]) | 0.843 (-0.081 [-0.09, -0.07]) |
| 8 | L3 | 0.924 | 0.924 (+0.000 [0.00, 0.00]) | 0.901 (0.90–0.90) (-0.023 [-0.03, -0.02]) | 0.911 (-0.013 [-0.02, -0.01]) | 0.910 (-0.015 [-0.02, -0.01]) |
| 8 | L4 | 0.924 | 0.924 (+0.000 [0.00, 0.00]) | 0.925 (0.92–0.93) (+0.000 [-0.00, 0.00]) | 0.925 (+0.001 [-0.00, 0.00]) | 0.925 (+0.000 [-0.00, 0.00]) |
| 32 | L0 | 0.525 | 0.525 (+0.000 [0.00, 0.00]) | 0.222 (0.22–0.23) (-0.303 [-0.31, -0.29]) | 0.340 (-0.184 [-0.20, -0.17]) | 0.231 (-0.294 [-0.31, -0.28]) |
| 32 | L1 | 0.525 | 0.525 (+0.000 [0.00, 0.00]) | 0.478 (0.47–0.48) (-0.047 [-0.05, -0.04]) | 0.496 (-0.029 [-0.04, -0.02]) | 0.502 (-0.022 [-0.03, -0.02]) |
| 32 | L2 | 0.525 | 0.525 (+0.000 [0.00, 0.00]) | 0.422 (0.42–0.42) (-0.103 [-0.11, -0.09]) | 0.449 (-0.075 [-0.08, -0.07]) | 0.434 (-0.091 [-0.10, -0.08]) |
| 32 | L3 | 0.525 | 0.525 (+0.000 [0.00, 0.00]) | 0.465 (0.46–0.47) (-0.059 [-0.07, -0.05]) | 0.490 (-0.035 [-0.04, -0.03]) | 0.497 (-0.028 [-0.03, -0.02]) |
| 32 | L4 | 0.525 | 0.525 (+0.000 [0.00, 0.00]) | 0.522 (0.52–0.52) (-0.003 [-0.01, -0.00]) | 0.523 (-0.002 [-0.00, 0.00]) | 0.524 (-0.001 [-0.00, 0.00]) |

## 5. Prefill cost (one forward pass, batch 1; mostly kernel quality)

_`cost_profile.csv` not synced yet._

## 5b. Decode cost (one token at a time, batch 1)

_`decode_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
