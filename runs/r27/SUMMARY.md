# iota run `r27`

- **updated:** 2026-10-18 05:22 UTC (last stage: `eval 8,9,10`)
- **code:** `1e87232` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | – | – | – | – | – | – | not trained yet |
| gated_linear | 14500 | 0.873 | 0.745 | 1.000 | 0.528 | 0.00075 | ok |
| hybrid | – | – | – | – | – | – | not trained yet |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `*_sweep_history.json` (and `logs/train.log`); `*_sweep.json` stops at the best checkpoint.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| gated_linear | 0.705 | 0.298 | 1.000 | 1.000 | pass |

n=500 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| n_bindings | true tokens | gated_linear |
|---:|---:|---|
| 2 | 256 | 0.995 [0.99–1.00] (0.99) |
| 4 | 256 | 0.984 [0.98–0.99] (0.94) |
| 8 | 256 | 0.963 [0.96–0.97] (0.74) |
| 16 | 256 | 0.916 [0.91–0.92] (0.25) |
| 32 | 289 | 0.773 [0.77–0.78] (0.02) |
| 64 | 386 | 0.145 [0.14–0.15] (0.00) |
| 128 | 773 | 0.069 [0.07–0.07] (0.00) |

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
| 4096 | 4104 | 0.989 [0.98–0.99] (0.99) |
| 8192 | 8200 | 0.923 [0.91–0.94] (0.92) |

## 4f. Decay gates (Pass 8): what γ does on real prompts

Mean γ ± std over tokens × heads, by token role. `retention` = mean log10 of the share of a fact value's write still in the state at the first question. A frozen gate has std ≈ 0; selective forgetting shows as γ(distractor) < γ(fact value).

| model | bindings | layer | fact value | distractor | query | answer | retention (log10) | gate bias / ‖W‖ |
|---|---:|---|---|---|---|---|---:|---|
| gated_linear | 8 | backbone.blocks.0.mixer | 0.732 ± 0.262 | 0.832 ± 0.073 | 0.529 ± 0.373 | 0.758 ± 0.228 | -28.556 | +1.87 / 1.233 |
| gated_linear | 8 | backbone.blocks.1.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.0 | +2.12 / 2.881 |
| gated_linear | 8 | backbone.blocks.2.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.0 | +2.32 / 3.678 |
| gated_linear | 8 | backbone.blocks.3.mixer | 1.000 ± 0.001 | 1.000 ± 0.000 | 1.000 ± 0.001 | 0.999 ± 0.002 | -0.013 | +2.39 / 4.049 |
| gated_linear | 8 | backbone.blocks.4.mixer | 0.998 ± 0.005 | 0.999 ± 0.002 | 0.995 ± 0.015 | 0.998 ± 0.003 | -0.101 | +2.25 / 3.348 |
| gated_linear | 32 | backbone.blocks.0.mixer | 0.729 ± 0.265 | 0.831 ± 0.076 | 0.527 ± 0.375 | 0.758 ± 0.229 | -59.185 | +1.87 / 1.233 |
| gated_linear | 32 | backbone.blocks.1.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.0 | +2.12 / 2.881 |
| gated_linear | 32 | backbone.blocks.2.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.001 | +2.32 / 3.678 |
| gated_linear | 32 | backbone.blocks.3.mixer | 1.000 ± 0.001 | 1.000 ± 0.001 | 1.000 ± 0.001 | 0.999 ± 0.001 | -0.034 | +2.39 / 4.049 |
| gated_linear | 32 | backbone.blocks.4.mixer | 0.998 ± 0.005 | 0.999 ± 0.003 | 0.996 ± 0.012 | 0.999 ± 0.003 | -0.21 | +2.25 / 3.348 |

## 4g. Gate-clamp intervention (Pass 9): which layers' forgetting does recall need?

Per-query recall with one GLA layer's gate replaced by a constant: `keep` = σ(6) ≈ 0.9975 (no forgetting), `mean` = the head's own average γ (same amount of forgetting, no selectivity). Δ vs the unclamped model on the same prompts, paired 95% CI.

**gated_linear**

| condition | 8 facts | 32 facts | 64 facts |
|---|---|---|---|
| none | 0.964 | 0.770 | 0.144 |
| keep_all | 0.011 (-0.953 [-0.96, -0.95]) | 0.008 (-0.762 [-0.77, -0.75]) | 0.005 (-0.139 [-0.15, -0.13]) |
| keep_L0 | 0.011 (-0.953 [-0.96, -0.95]) | 0.007 (-0.763 [-0.77, -0.75]) | 0.006 (-0.138 [-0.15, -0.13]) |
| mean_L0 | 0.037 (-0.927 [-0.94, -0.92]) | 0.045 (-0.725 [-0.74, -0.72]) | 0.038 (-0.106 [-0.11, -0.10]) |
| keep_L1 | 0.816 (-0.148 [-0.16, -0.14]) | 0.590 (-0.180 [-0.19, -0.17]) | 0.127 (-0.017 [-0.02, -0.01]) |
| mean_L1 | 0.963 (-0.002 [-0.00, -0.00]) | 0.771 (+0.001 [-0.00, +0.00]) | 0.144 (+0.000 [-0.00, +0.00]) |
| keep_L2 | 0.965 (+0.001 [-0.00, +0.00]) | 0.771 (+0.001 [-0.00, +0.01]) | 0.144 (+0.000 [-0.00, +0.00]) |
| mean_L2 | 0.964 (+0.000 [+0.00, +0.00]) | 0.770 (+0.000 [+0.00, +0.00]) | 0.144 (+0.000 [+0.00, +0.00]) |
| keep_L3 | 0.964 (-0.000 [-0.00, +0.00]) | 0.769 (-0.001 [-0.00, +0.00]) | 0.145 (+0.000 [-0.00, +0.00]) |
| mean_L3 | 0.965 (+0.000 [+0.00, +0.00]) | 0.771 (+0.001 [-0.00, +0.00]) | 0.144 (+0.000 [+0.00, +0.00]) |
| keep_L4 | 0.961 (-0.004 [-0.01, -0.00]) | 0.769 (-0.001 [-0.01, +0.00]) | 0.144 (-0.001 [-0.00, +0.00]) |
| mean_L4 | 0.964 (-0.000 [-0.00, +0.00]) | 0.771 (+0.001 [-0.00, +0.00]) | 0.145 (+0.001 [+0.00, +0.00]) |

## 4h. Gate replay (Pass 10): does the timing of forgetting matter?

One layer's recorded gates over the context span are replayed unchanged, shuffled in time (same values, same product), replaced by their geometric mean (same product, no timing) or their arithmetic mean (forgets less). Per-query recall, Δ vs baseline [paired 95% CI].

**gated_linear**

| facts | layer | baseline | replay | shuffle (range) | geomean | arith |
|---:|---|---:|---|---|---|---|
| 8 | L0 | 0.960 | 0.960 (+0.000 [0.00, 0.00]) | 0.042 (0.04–0.04) (-0.919 [-0.93, -0.91]) | 0.078 (-0.882 [-0.89, -0.87]) | 0.038 (-0.922 [-0.93, -0.91]) |
| 8 | L1 | 0.960 | 0.960 (+0.000 [0.00, 0.00]) | 0.961 (0.96–0.96) (+0.001 [0.00, 0.00]) | 0.961 (+0.001 [0.00, 0.00]) | 0.961 (+0.001 [0.00, 0.00]) |
| 8 | L2 | 0.960 | 0.960 (+0.000 [0.00, 0.00]) | 0.960 (0.96–0.96) (+0.000 [0.00, 0.00]) | 0.960 (+0.000 [0.00, 0.00]) | 0.960 (+0.000 [0.00, 0.00]) |
| 8 | L3 | 0.960 | 0.960 (+0.000 [0.00, 0.00]) | 0.961 (0.96–0.96) (+0.001 [0.00, 0.00]) | 0.961 (+0.001 [0.00, 0.00]) | 0.961 (+0.001 [0.00, 0.00]) |
| 8 | L4 | 0.960 | 0.960 (+0.000 [0.00, 0.00]) | 0.960 (0.96–0.96) (+0.000 [-0.00, 0.00]) | 0.961 (+0.000 [-0.00, 0.00]) | 0.961 (+0.000 [-0.00, 0.00]) |
| 32 | L0 | 0.775 | 0.775 (+0.000 [0.00, 0.00]) | 0.053 (0.05–0.06) (-0.721 [-0.73, -0.71]) | 0.137 (-0.638 [-0.65, -0.63]) | 0.052 (-0.723 [-0.73, -0.71]) |
| 32 | L1 | 0.775 | 0.775 (+0.000 [0.00, 0.00]) | 0.774 (0.77–0.77) (-0.000 [-0.00, 0.00]) | 0.774 (-0.001 [-0.00, 0.00]) | 0.774 (-0.001 [-0.00, 0.00]) |
| 32 | L2 | 0.775 | 0.775 (+0.000 [0.00, 0.00]) | 0.775 (0.77–0.77) (+0.000 [0.00, 0.00]) | 0.775 (+0.000 [0.00, 0.00]) | 0.775 (+0.000 [0.00, 0.00]) |
| 32 | L3 | 0.775 | 0.775 (+0.000 [0.00, 0.00]) | 0.775 (0.78–0.78) (+0.000 [-0.00, 0.00]) | 0.775 (+0.001 [-0.00, 0.00]) | 0.775 (+0.000 [-0.00, 0.00]) |
| 32 | L4 | 0.775 | 0.775 (+0.000 [0.00, 0.00]) | 0.777 (0.78–0.78) (+0.002 [-0.00, 0.00]) | 0.777 (+0.002 [-0.00, 0.00]) | 0.777 (+0.002 [-0.00, 0.00]) |

## 5. Prefill cost (one forward pass, batch 1; mostly kernel quality)

_`cost_profile.csv` not synced yet._

## 5b. Decode cost (one token at a time, batch 1)

_`decode_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
