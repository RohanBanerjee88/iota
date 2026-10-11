# iota run `r13`

- **updated:** 2026-10-11 02:07 UTC (last stage: `eval 8,9`)
- **code:** `ccb5ac0` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | – | – | – | – | – | – | not trained yet |
| gated_linear | 15000 | 0.589 | 0.178 | 1.000 | 0.350 | 0.0015 | ok |
| hybrid | – | – | – | – | – | – | not trained yet |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `logs/train.log` and the `*_sweep.json` files.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| gated_linear | 0.161 | 0.022 | 1.000 | 1.000 | pass |

n=500 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| n_bindings | true tokens | gated_linear |
|---:|---:|---|
| 2 | 256 | 0.776 [0.76–0.79] (0.56) |
| 4 | 256 | 0.608 [0.59–0.62] (0.09) |
| 8 | 256 | 0.474 [0.47–0.48] (0.00) |
| 16 | 256 | 0.343 [0.34–0.35] (0.00) |
| 32 | 289 | 0.149 [0.14–0.15] (0.00) |
| 64 | 386 | 0.068 [0.06–0.07] (0.00) |
| 128 | 773 | 0.024 [0.02–0.03] (0.00) |

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
| 8192 | 8200 | 0.908 [0.89–0.93] (0.91) |

## 4f. Decay gates (Pass 8): what γ does on real prompts

Mean γ ± std over tokens × heads, by token role. `retention` = mean log10 of the share of a fact value's write still in the state at the first question. A frozen gate has std ≈ 0; selective forgetting shows as γ(distractor) < γ(fact value).

| model | bindings | layer | fact value | distractor | query | answer | retention (log10) | gate bias / ‖W‖ |
|---|---:|---|---|---|---|---|---:|---|
| gated_linear | 8 | backbone.blocks.0.mixer | 0.275 ± 0.300 | 0.794 ± 0.178 | 0.664 ± 0.210 | 0.289 ± 0.306 | -33.705 | +1.72 / 1.563 |
| gated_linear | 8 | backbone.blocks.1.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.001 | -0.0 | +1.98 / 2.963 |
| gated_linear | 8 | backbone.blocks.2.mixer | 0.998 ± 0.003 | 1.000 ± 0.001 | 0.998 ± 0.005 | 0.999 ± 0.003 | -0.049 | +2.06 / 2.506 |
| gated_linear | 8 | backbone.blocks.3.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.006 | +2.30 / 4.343 |
| gated_linear | 8 | backbone.blocks.4.mixer | 0.995 ± 0.005 | 0.991 ± 0.004 | 0.997 ± 0.003 | 0.999 ± 0.001 | -0.863 | +1.90 / 1.579 |
| gated_linear | 32 | backbone.blocks.0.mixer | 0.276 ± 0.299 | 0.794 ± 0.178 | 0.663 ± 0.212 | 0.290 ± 0.306 | -52.296 | +1.72 / 1.563 |
| gated_linear | 32 | backbone.blocks.1.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.001 | -0.0 | +1.98 / 2.963 |
| gated_linear | 32 | backbone.blocks.2.mixer | 0.999 ± 0.002 | 1.000 ± 0.001 | 0.999 ± 0.003 | 0.999 ± 0.002 | -0.092 | +2.06 / 2.506 |
| gated_linear | 32 | backbone.blocks.3.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.006 | +2.30 / 4.343 |
| gated_linear | 32 | backbone.blocks.4.mixer | 0.995 ± 0.006 | 0.989 ± 0.004 | 0.998 ± 0.002 | 0.998 ± 0.002 | -0.747 | +1.90 / 1.579 |

## 4g. Gate-clamp intervention (Pass 9): which layers' forgetting does recall need?

Per-query recall with one GLA layer's gate replaced by a constant: `keep` = σ(6) ≈ 0.9975 (no forgetting), `mean` = the head's own average γ (same amount of forgetting, no selectivity). Δ vs the unclamped model on the same prompts, paired 95% CI.

**gated_linear**

| condition | 8 facts | 32 facts | 64 facts |
|---|---|---|---|
| none | 0.468 | 0.156 | 0.073 |
| keep_all | 0.014 (-0.454 [-0.47, -0.44]) | 0.010 (-0.146 [-0.15, -0.14]) | 0.010 (-0.063 [-0.07, -0.06]) |
| keep_L0 | 0.015 (-0.453 [-0.47, -0.44]) | 0.011 (-0.145 [-0.15, -0.14]) | 0.011 (-0.062 [-0.07, -0.06]) |
| mean_L0 | 0.051 (-0.416 [-0.43, -0.40]) | 0.067 (-0.089 [-0.10, -0.08]) | 0.040 (-0.033 [-0.04, -0.03]) |
| keep_L1 | 0.431 (-0.036 [-0.05, -0.03]) | 0.121 (-0.035 [-0.04, -0.03]) | 0.040 (-0.033 [-0.04, -0.03]) |
| mean_L1 | 0.468 (+0.001 [-0.00, +0.00]) | 0.156 (+0.000 [-0.00, +0.00]) | 0.073 (+0.000 [-0.00, +0.00]) |
| keep_L2 | 0.468 (+0.001 [-0.00, +0.00]) | 0.154 (-0.002 [-0.00, +0.00]) | 0.072 (-0.000 [-0.00, +0.00]) |
| mean_L2 | 0.468 (-0.000 [-0.00, +0.00]) | 0.156 (+0.000 [-0.00, +0.00]) | 0.073 (+0.000 [-0.00, +0.00]) |
| keep_L3 | 0.457 (-0.011 [-0.02, -0.00]) | 0.156 (+0.000 [-0.00, +0.00]) | 0.073 (+0.001 [-0.00, +0.00]) |
| mean_L3 | 0.467 (-0.001 [-0.00, -0.00]) | 0.156 (+0.000 [-0.00, +0.00]) | 0.073 (+0.000 [+0.00, +0.00]) |
| keep_L4 | 0.463 (-0.005 [-0.01, +0.00]) | 0.151 (-0.004 [-0.01, -0.00]) | 0.073 (+0.000 [-0.00, +0.00]) |
| mean_L4 | 0.469 (+0.001 [-0.00, +0.00]) | 0.156 (+0.000 [-0.00, +0.00]) | 0.072 (-0.000 [-0.00, +0.00]) |

## 5. Prefill cost (one forward pass, batch 1; mostly kernel quality)

_`cost_profile.csv` not synced yet._

## 5b. Decode cost (one token at a time, batch 1)

_`decode_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
