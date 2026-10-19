# iota run `r33`

- **updated:** 2026-10-19 20:17 UTC (last stage: `eval 8,9,10`)
- **code:** `09602bd` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | – | – | – | – | – | – | not trained yet |
| gated_linear | 14000 | 0.606 | 0.211 | 1.000 | 0.360 | 0.00075 | ok |
| hybrid | – | – | – | – | – | – | not trained yet |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `*_sweep_history.json` (and `logs/train.log`); `*_sweep.json` stops at the best checkpoint.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| gated_linear | 0.195 | 0.034 | 0.998 | 0.998 | pass |

n=500 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| n_bindings | true tokens | gated_linear |
|---:|---:|---|
| 2 | 256 | 0.814 [0.80–0.83] (0.63) |
| 4 | 256 | 0.673 [0.66–0.69] (0.16) |
| 8 | 256 | 0.512 [0.50–0.52] (0.00) |
| 16 | 256 | 0.369 [0.36–0.38] (0.00) |
| 32 | 289 | 0.174 [0.17–0.18] (0.00) |
| 64 | 386 | 0.052 [0.05–0.05] (0.00) |
| 128 | 773 | 0.025 [0.02–0.03] (0.00) |

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
| 4096 | 4104 | 0.999 [1.00–1.00] (1.00) |
| 8192 | 8200 | 1.000 [1.00–1.00] (1.00) |

## 4f. Decay gates (Pass 8): what γ does on real prompts

Mean γ ± std over tokens × heads, by token role. `retention` = mean log10 of the share of a fact value's write still in the state at the first question. A frozen gate has std ≈ 0; selective forgetting shows as γ(distractor) < γ(fact value).

| model | bindings | layer | fact value | distractor | query | answer | retention (log10) | gate bias / ‖W‖ |
|---|---:|---|---|---|---|---|---:|---|
| gated_linear | 8 | backbone.blocks.0.mixer | 0.735 ± 0.301 | 0.730 ± 0.198 | 0.745 ± 0.333 | 0.755 ± 0.287 | -46.076 | +1.86 / 1.292 |
| gated_linear | 8 | backbone.blocks.1.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.0 | +2.09 / 2.467 |
| gated_linear | 8 | backbone.blocks.2.mixer | 0.829 ± 0.314 | 0.772 ± 0.396 | 0.866 ± 0.252 | 0.865 ± 0.274 | -58.604 | +2.07 / 2.152 |
| gated_linear | 8 | backbone.blocks.3.mixer | 1.000 ± 0.002 | 1.000 ± 0.001 | 0.999 ± 0.003 | 0.999 ± 0.001 | -0.035 | +2.07 / 2.064 |
| gated_linear | 8 | backbone.blocks.4.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.003 | +2.21 / 2.698 |
| gated_linear | 32 | backbone.blocks.0.mixer | 0.733 ± 0.301 | 0.729 ± 0.199 | 0.743 ± 0.333 | 0.755 ± 0.287 | -58.778 | +1.86 / 1.292 |
| gated_linear | 32 | backbone.blocks.1.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.001 | +2.09 / 2.467 |
| gated_linear | 32 | backbone.blocks.2.mixer | 0.811 ± 0.340 | 0.767 ± 0.404 | 0.850 ± 0.274 | 0.846 ± 0.299 | -44.37 | +2.07 / 2.152 |
| gated_linear | 32 | backbone.blocks.3.mixer | 1.000 ± 0.001 | 1.000 ± 0.001 | 0.999 ± 0.002 | 0.999 ± 0.001 | -0.056 | +2.07 / 2.064 |
| gated_linear | 32 | backbone.blocks.4.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.006 | +2.21 / 2.698 |

## 4g. Gate-clamp intervention (Pass 9): which layers' forgetting does recall need?

Per-query recall with one GLA layer's gate replaced by a constant: `keep` = σ(6) ≈ 0.9975 (no forgetting), `mean` = the head's own average γ (same amount of forgetting, no selectivity). Δ vs the unclamped model on the same prompts, paired 95% CI.

**gated_linear**

| condition | 8 facts | 32 facts | 64 facts |
|---|---|---|---|
| none | 0.503 | 0.180 | 0.052 |
| keep_all | 0.018 (-0.485 [-0.50, -0.47]) | 0.014 (-0.166 [-0.17, -0.16]) | 0.006 (-0.047 [-0.05, -0.04]) |
| keep_L0 | 0.018 (-0.485 [-0.50, -0.47]) | 0.014 (-0.166 [-0.17, -0.16]) | 0.005 (-0.047 [-0.05, -0.04]) |
| mean_L0 | 0.120 (-0.383 [-0.40, -0.37]) | 0.042 (-0.138 [-0.15, -0.13]) | 0.031 (-0.022 [-0.03, -0.02]) |
| keep_L1 | 0.484 (-0.019 [-0.03, -0.01]) | 0.109 (-0.071 [-0.08, -0.06]) | 0.038 (-0.014 [-0.02, -0.01]) |
| mean_L1 | 0.499 (-0.004 [-0.01, +0.00]) | 0.178 (-0.002 [-0.01, +0.00]) | 0.050 (-0.003 [-0.00, -0.00]) |
| keep_L2 | 0.465 (-0.038 [-0.05, -0.03]) | 0.133 (-0.048 [-0.05, -0.04]) | 0.046 (-0.006 [-0.01, -0.00]) |
| mean_L2 | 0.504 (+0.001 [-0.00, +0.00]) | 0.180 (+0.000 [-0.00, +0.00]) | 0.052 (-0.001 [-0.00, +0.00]) |
| keep_L3 | 0.496 (-0.007 [-0.01, -0.00]) | 0.180 (+0.000 [-0.00, +0.00]) | 0.053 (+0.000 [-0.00, +0.00]) |
| mean_L3 | 0.501 (-0.002 [-0.00, -0.00]) | 0.180 (-0.000 [-0.00, +0.00]) | 0.052 (+0.000 [-0.00, +0.00]) |
| keep_L4 | 0.501 (-0.002 [-0.01, +0.00]) | 0.181 (+0.001 [-0.00, +0.00]) | 0.051 (-0.001 [-0.00, +0.00]) |
| mean_L4 | 0.502 (-0.001 [-0.00, +0.00]) | 0.180 (-0.001 [-0.00, +0.00]) | 0.052 (+0.000 [-0.00, +0.00]) |

## 4h. Gate replay (Pass 10): does the timing of forgetting matter?

One layer's recorded gates over the context span are replayed unchanged, shuffled in time (same values, same product), replaced by their geometric mean (same product, no timing) or their arithmetic mean (forgets less). Per-query recall, Δ vs baseline [paired 95% CI].

**gated_linear**

| facts | layer | baseline | replay | shuffle (range) | geomean | arith |
|---:|---|---:|---|---|---|---|
| 8 | L0 | 0.512 | 0.512 (+0.000 [0.00, 0.00]) | 0.177 (0.17–0.18) (-0.336 [-0.35, -0.32]) | 0.312 (-0.200 [-0.22, -0.18]) | 0.164 (-0.349 [-0.36, -0.33]) |
| 8 | L1 | 0.512 | 0.512 (+0.000 [0.00, 0.00]) | 0.508 (0.51–0.51) (-0.004 [-0.01, 0.00]) | 0.513 (+0.001 [-0.00, 0.01]) | 0.513 (+0.001 [-0.00, 0.01]) |
| 8 | L2 | 0.512 | 0.512 (+0.000 [0.00, 0.00]) | 0.512 (0.51–0.51) (+0.000 [-0.00, 0.00]) | 0.512 (+0.000 [-0.00, 0.00]) | 0.512 (+0.000 [-0.00, 0.00]) |
| 8 | L3 | 0.512 | 0.512 (+0.000 [0.00, 0.00]) | 0.513 (0.51–0.51) (+0.001 [-0.00, 0.00]) | 0.513 (+0.001 [-0.00, 0.00]) | 0.513 (+0.001 [-0.00, 0.00]) |
| 8 | L4 | 0.512 | 0.512 (+0.000 [0.00, 0.00]) | 0.512 (0.51–0.51) (-0.000 [-0.00, 0.00]) | 0.512 (-0.001 [-0.00, 0.00]) | 0.513 (+0.000 [-0.00, 0.00]) |
| 32 | L0 | 0.177 | 0.177 (+0.000 [0.00, 0.00]) | 0.061 (0.06–0.06) (-0.116 [-0.12, -0.11]) | 0.108 (-0.069 [-0.08, -0.06]) | 0.066 (-0.111 [-0.12, -0.10]) |
| 32 | L1 | 0.177 | 0.177 (+0.000 [0.00, 0.00]) | 0.171 (0.17–0.17) (-0.007 [-0.01, -0.00]) | 0.175 (-0.002 [-0.01, 0.00]) | 0.175 (-0.002 [-0.01, 0.00]) |
| 32 | L2 | 0.177 | 0.177 (+0.000 [0.00, 0.00]) | 0.176 (0.18–0.18) (-0.001 [-0.00, -0.00]) | 0.176 (-0.001 [-0.00, -0.00]) | 0.176 (-0.001 [-0.00, -0.00]) |
| 32 | L3 | 0.177 | 0.177 (+0.000 [0.00, 0.00]) | 0.177 (0.18–0.18) (+0.000 [-0.00, 0.00]) | 0.177 (+0.000 [-0.00, 0.00]) | 0.177 (+0.000 [-0.00, 0.00]) |
| 32 | L4 | 0.177 | 0.177 (+0.000 [0.00, 0.00]) | 0.177 (0.18–0.18) (+0.000 [-0.00, 0.00]) | 0.177 (-0.000 [-0.00, 0.00]) | 0.177 (-0.000 [-0.00, 0.00]) |

## 5. Prefill cost (one forward pass, batch 1; mostly kernel quality)

_`cost_profile.csv` not synced yet._

## 5b. Decode cost (one token at a time, batch 1)

_`decode_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
