# iota run `r31`

- **updated:** 2026-10-18 02:34 UTC (last stage: `eval 8,9,10`)
- **code:** `1e87232` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | – | – | – | – | – | – | not trained yet |
| gated_linear | 15000 | 0.850 | 0.700 | 1.000 | 0.532 | 0.00075 | ok |
| hybrid | – | – | – | – | – | – | not trained yet |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `*_sweep_history.json` (and `logs/train.log`); `*_sweep.json` stops at the best checkpoint.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| gated_linear | 0.672 | 0.270 | 1.000 | 1.000 | pass |

n=500 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| n_bindings | true tokens | gated_linear |
|---:|---:|---|
| 2 | 256 | 0.998 [0.99–1.00] (0.99) |
| 4 | 256 | 0.987 [0.98–0.99] (0.95) |
| 8 | 256 | 0.968 [0.96–0.97] (0.77) |
| 16 | 256 | 0.911 [0.91–0.92] (0.24) |
| 32 | 289 | 0.727 [0.72–0.73] (0.01) |
| 64 | 386 | 0.141 [0.14–0.15] (0.00) |
| 128 | 773 | 0.062 [0.06–0.07] (0.00) |

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
| 2048 | 2056 | 0.994 [0.99–1.00] (0.99) |
| 4096 | 4104 | 0.863 [0.84–0.88] (0.86) |
| 8192 | 8200 | 0.388 [0.36–0.42] (0.39) |

## 4f. Decay gates (Pass 8): what γ does on real prompts

Mean γ ± std over tokens × heads, by token role. `retention` = mean log10 of the share of a fact value's write still in the state at the first question. A frozen gate has std ≈ 0; selective forgetting shows as γ(distractor) < γ(fact value).

| model | bindings | layer | fact value | distractor | query | answer | retention (log10) | gate bias / ‖W‖ |
|---|---:|---|---|---|---|---|---:|---|
| gated_linear | 8 | backbone.blocks.0.mixer | 0.285 ± 0.123 | 0.454 ± 0.160 | 0.241 ± 0.157 | 0.285 ± 0.130 | -90.089 | -0.04 / 0.749 |
| gated_linear | 8 | backbone.blocks.1.mixer | 0.417 ± 0.202 | 0.442 ± 0.093 | 0.495 ± 0.141 | 0.417 ± 0.198 | -83.603 | -0.01 / 0.683 |
| gated_linear | 8 | backbone.blocks.2.mixer | 0.735 ± 0.301 | 0.696 ± 0.186 | 0.790 ± 0.231 | 0.792 ± 0.264 | -40.719 | +0.06 / 1.282 |
| gated_linear | 8 | backbone.blocks.3.mixer | 0.998 ± 0.006 | 0.927 ± 0.131 | 0.999 ± 0.006 | 1.000 ± 0.000 | -7.937 | +0.15 / 2.166 |
| gated_linear | 8 | backbone.blocks.4.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.002 | 1.000 ± 0.000 | -0.003 | +0.09 / 1.853 |
| gated_linear | 32 | backbone.blocks.0.mixer | 0.285 ± 0.122 | 0.454 ± 0.160 | 0.243 ± 0.159 | 0.285 ± 0.130 | -92.722 | -0.04 / 0.749 |
| gated_linear | 32 | backbone.blocks.1.mixer | 0.416 ± 0.202 | 0.444 ± 0.095 | 0.499 ± 0.142 | 0.417 ± 0.198 | -69.633 | -0.01 / 0.683 |
| gated_linear | 32 | backbone.blocks.2.mixer | 0.732 ± 0.303 | 0.696 ± 0.191 | 0.791 ± 0.233 | 0.791 ± 0.265 | -36.779 | +0.06 / 1.282 |
| gated_linear | 32 | backbone.blocks.3.mixer | 0.998 ± 0.006 | 0.926 ± 0.134 | 1.000 ± 0.004 | 1.000 ± 0.000 | -3.78 | +0.15 / 2.166 |
| gated_linear | 32 | backbone.blocks.4.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.001 | 1.000 ± 0.000 | -0.006 | +0.09 / 1.853 |

## 4g. Gate-clamp intervention (Pass 9): which layers' forgetting does recall need?

Per-query recall with one GLA layer's gate replaced by a constant: `keep` = σ(6) ≈ 0.9975 (no forgetting), `mean` = the head's own average γ (same amount of forgetting, no selectivity). Δ vs the unclamped model on the same prompts, paired 95% CI.

**gated_linear**

| condition | 8 facts | 32 facts | 64 facts |
|---|---|---|---|
| none | 0.974 | 0.732 | 0.148 |
| keep_all | 0.010 (-0.964 [-0.97, -0.96]) | 0.001 (-0.730 [-0.74, -0.72]) | 0.010 (-0.138 [-0.15, -0.13]) |
| keep_L0 | 0.010 (-0.964 [-0.97, -0.96]) | 0.010 (-0.722 [-0.73, -0.71]) | 0.011 (-0.138 [-0.15, -0.13]) |
| mean_L0 | 0.754 (-0.220 [-0.23, -0.21]) | 0.629 (-0.102 [-0.11, -0.09]) | 0.144 (-0.004 [-0.01, +0.00]) |
| keep_L1 | 0.823 (-0.150 [-0.16, -0.14]) | 0.431 (-0.301 [-0.31, -0.29]) | 0.111 (-0.037 [-0.04, -0.03]) |
| mean_L1 | 0.971 (-0.003 [-0.01, +0.00]) | 0.729 (-0.003 [-0.01, +0.00]) | 0.146 (-0.002 [-0.01, +0.00]) |
| keep_L2 | 0.462 (-0.512 [-0.52, -0.50]) | 0.165 (-0.566 [-0.58, -0.55]) | 0.062 (-0.087 [-0.10, -0.08]) |
| mean_L2 | 0.955 (-0.018 [-0.02, -0.01]) | 0.703 (-0.029 [-0.04, -0.02]) | 0.150 (+0.002 [-0.00, +0.01]) |
| keep_L3 | 0.953 (-0.021 [-0.03, -0.02]) | 0.625 (-0.106 [-0.12, -0.10]) | 0.130 (-0.018 [-0.03, -0.01]) |
| mean_L3 | 0.732 (-0.241 [-0.25, -0.23]) | 0.542 (-0.189 [-0.20, -0.18]) | 0.144 (-0.005 [-0.01, +0.00]) |
| keep_L4 | 0.965 (-0.008 [-0.01, -0.00]) | 0.720 (-0.012 [-0.02, -0.01]) | 0.146 (-0.003 [-0.01, +0.00]) |
| mean_L4 | 0.965 (-0.009 [-0.01, -0.01]) | 0.726 (-0.006 [-0.01, -0.00]) | 0.146 (-0.002 [-0.01, +0.00]) |

## 4h. Gate replay (Pass 10): does the timing of forgetting matter?

One layer's recorded gates over the context span are replayed unchanged, shuffled in time (same values, same product), replaced by their geometric mean (same product, no timing) or their arithmetic mean (forgets less). Per-query recall, Δ vs baseline [paired 95% CI].

**gated_linear**

| facts | layer | baseline | replay | shuffle (range) | geomean | arith |
|---:|---|---:|---|---|---|---|
| 8 | L0 | 0.964 | 0.964 (+0.000 [0.00, 0.00]) | 0.777 (0.77–0.78) (-0.187 [-0.20, -0.18]) | 0.861 (-0.104 [-0.11, -0.09]) | 0.798 (-0.166 [-0.18, -0.15]) |
| 8 | L1 | 0.964 | 0.964 (+0.000 [0.00, 0.00]) | 0.965 (0.96–0.96) (+0.001 [-0.00, 0.00]) | 0.965 (+0.001 [-0.00, 0.00]) | 0.966 (+0.001 [-0.00, 0.00]) |
| 8 | L2 | 0.964 | 0.964 (+0.000 [0.00, 0.00]) | 0.950 (0.95–0.95) (-0.014 [-0.02, -0.01]) | 0.955 (-0.010 [-0.01, -0.01]) | 0.955 (-0.010 [-0.01, -0.01]) |
| 8 | L3 | 0.964 | 0.964 (+0.000 [0.00, 0.00]) | 0.777 (0.77–0.78) (-0.187 [-0.20, -0.18]) | 0.901 (-0.063 [-0.07, -0.06]) | 0.961 (-0.003 [-0.01, 0.00]) |
| 8 | L4 | 0.964 | 0.964 (+0.000 [0.00, 0.00]) | 0.952 (0.95–0.95) (-0.012 [-0.02, -0.01]) | 0.958 (-0.006 [-0.01, -0.00]) | 0.954 (-0.011 [-0.01, -0.01]) |
| 32 | L0 | 0.727 | 0.727 (+0.000 [0.00, 0.00]) | 0.515 (0.51–0.52) (-0.212 [-0.22, -0.20]) | 0.636 (-0.090 [-0.10, -0.08]) | 0.546 (-0.181 [-0.19, -0.17]) |
| 32 | L1 | 0.727 | 0.727 (+0.000 [0.00, 0.00]) | 0.716 (0.71–0.72) (-0.011 [-0.02, -0.01]) | 0.722 (-0.005 [-0.01, -0.00]) | 0.722 (-0.005 [-0.01, -0.00]) |
| 32 | L2 | 0.727 | 0.727 (+0.000 [0.00, 0.00]) | 0.697 (0.70–0.70) (-0.029 [-0.03, -0.02]) | 0.716 (-0.011 [-0.02, -0.01]) | 0.717 (-0.010 [-0.02, -0.00]) |
| 32 | L3 | 0.727 | 0.727 (+0.000 [0.00, 0.00]) | 0.295 (0.29–0.30) (-0.432 [-0.44, -0.42]) | 0.467 (-0.260 [-0.27, -0.25]) | 0.604 (-0.123 [-0.13, -0.11]) |
| 32 | L4 | 0.727 | 0.727 (+0.000 [0.00, 0.00]) | 0.682 (0.68–0.69) (-0.044 [-0.05, -0.04]) | 0.716 (-0.011 [-0.02, -0.01]) | 0.724 (-0.003 [-0.01, 0.00]) |

## 5. Prefill cost (one forward pass, batch 1; mostly kernel quality)

_`cost_profile.csv` not synced yet._

## 5b. Decode cost (one token at a time, batch 1)

_`decode_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
