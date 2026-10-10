# iota run `r11`

- **updated:** 2026-10-10 07:28 UTC (last stage: `eval 8,9`)
- **code:** `9150b4e` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | – | – | – | – | – | – | not trained yet |
| gated_linear | 14000 | 0.760 | 0.521 | 1.000 | 0.464 | 0.0015 | ok |
| hybrid | – | – | – | – | – | – | not trained yet |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `logs/train.log` and the `*_sweep.json` files.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| gated_linear | 0.484 | 0.142 | 1.000 | 1.000 | pass |

n=500 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| n_bindings | true tokens | gated_linear |
|---:|---:|---|
| 2 | 256 | 0.988 [0.98–0.99] (0.98) |
| 4 | 256 | 0.968 [0.96–0.97] (0.88) |
| 8 | 256 | 0.905 [0.90–0.91] (0.44) |
| 16 | 256 | 0.765 [0.76–0.77] (0.01) |
| 32 | 289 | 0.515 [0.51–0.52] (0.00) |
| 64 | 386 | 0.096 [0.09–0.10] (0.00) |
| 128 | 773 | 0.050 [0.05–0.05] (0.00) |

## 3. Pass 2 — length generalisation

_`pass2_length.csv` not synced yet._

## 4. Pass 3 — state_track control

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| seq_len | true tokens | gated_linear |
|---:|---:|---|
| 128 | 136 | 1.000 [1.00–1.00] (1.00) |
| 256 | 264 | 0.999 [1.00–1.00] (1.00) |
| 512 | 520 | 1.000 [1.00–1.00] (1.00) |
| 1024 | 1032 | 1.000 [1.00–1.00] (1.00) |
| 2048 | 2056 | 0.996 [0.99–1.00] (1.00) |
| 4096 | 4104 | 0.795 [0.77–0.82] (0.80) |
| 8192 | 8200 | 0.574 [0.54–0.60] (0.57) |

## 4f. Decay gates (Pass 8): what γ does on real prompts

Mean γ ± std over tokens × heads, by token role. `retention` = mean log10 of the share of a fact value's write still in the state at the first question. A frozen gate has std ≈ 0; selective forgetting shows as γ(distractor) < γ(fact value).

| model | bindings | layer | fact value | distractor | query | answer | retention (log10) | gate bias / ‖W‖ |
|---|---:|---|---|---|---|---|---:|---|
| gated_linear | 8 | backbone.blocks.0.mixer | 0.685 ± 0.335 | 0.644 ± 0.276 | 0.588 ± 0.414 | 0.711 ± 0.314 | -64.556 | +1.78 / 1.650 |
| gated_linear | 8 | backbone.blocks.1.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.0 | +2.01 / 4.020 |
| gated_linear | 8 | backbone.blocks.2.mixer | 1.000 ± 0.001 | 1.000 ± 0.001 | 0.997 ± 0.012 | 0.999 ± 0.001 | -0.023 | +2.12 / 3.362 |
| gated_linear | 8 | backbone.blocks.3.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.005 | +2.01 / 2.513 |
| gated_linear | 8 | backbone.blocks.4.mixer | 1.000 ± 0.001 | 1.000 ± 0.000 | 0.999 ± 0.016 | 1.000 ± 0.000 | -0.002 | +2.22 / 3.835 |
| gated_linear | 32 | backbone.blocks.0.mixer | 0.683 ± 0.336 | 0.645 ± 0.276 | 0.588 ± 0.416 | 0.712 ± 0.315 | -81.533 | +1.78 / 1.650 |
| gated_linear | 32 | backbone.blocks.1.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.0 | +2.01 / 4.020 |
| gated_linear | 32 | backbone.blocks.2.mixer | 1.000 ± 0.001 | 1.000 ± 0.001 | 0.998 ± 0.008 | 1.000 ± 0.001 | -0.044 | +2.12 / 3.362 |
| gated_linear | 32 | backbone.blocks.3.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.012 | +2.01 / 2.513 |
| gated_linear | 32 | backbone.blocks.4.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.999 ± 0.008 | 1.000 ± 0.000 | -0.008 | +2.22 / 3.835 |

## 4g. Gate-clamp intervention (Pass 9): which layers' forgetting does recall need?

Per-query recall with one GLA layer's gate replaced by a constant: `keep` = σ(6) ≈ 0.9975 (no forgetting), `mean` = the head's own average γ (same amount of forgetting, no selectivity). Δ vs the unclamped model on the same prompts, paired 95% CI.

**gated_linear**

| condition | 8 facts | 32 facts | 64 facts |
|---|---|---|---|
| none | 0.906 | 0.509 | 0.098 |
| keep_all | 0.010 (-0.896 [-0.91, -0.89]) | 0.010 (-0.498 [-0.51, -0.49]) | 0.011 (-0.087 [-0.09, -0.08]) |
| keep_L0 | 0.010 (-0.896 [-0.91, -0.89]) | 0.011 (-0.498 [-0.51, -0.49]) | 0.011 (-0.087 [-0.09, -0.08]) |
| mean_L0 | 0.116 (-0.790 [-0.80, -0.78]) | 0.059 (-0.450 [-0.46, -0.44]) | 0.033 (-0.066 [-0.07, -0.06]) |
| keep_L1 | 0.899 (-0.007 [-0.01, -0.00]) | 0.381 (-0.128 [-0.14, -0.12]) | 0.083 (-0.016 [-0.02, -0.01]) |
| mean_L1 | 0.901 (-0.004 [-0.01, -0.00]) | 0.501 (-0.007 [-0.01, -0.00]) | 0.096 (-0.002 [-0.00, -0.00]) |
| keep_L2 | 0.901 (-0.004 [-0.01, -0.00]) | 0.507 (-0.002 [-0.00, +0.00]) | 0.097 (-0.001 [-0.00, +0.00]) |
| mean_L2 | 0.906 (+0.000 [-0.00, +0.00]) | 0.509 (+0.001 [-0.00, +0.00]) | 0.098 (+0.000 [-0.00, +0.00]) |
| keep_L3 | 0.851 (-0.054 [-0.06, -0.05]) | 0.493 (-0.015 [-0.02, -0.01]) | 0.096 (-0.002 [-0.01, +0.00]) |
| mean_L3 | 0.862 (-0.044 [-0.05, -0.04]) | 0.498 (-0.010 [-0.02, -0.00]) | 0.094 (-0.004 [-0.01, -0.00]) |
| keep_L4 | 0.905 (-0.000 [-0.00, +0.00]) | 0.506 (-0.002 [-0.01, +0.00]) | 0.099 (+0.000 [-0.00, +0.00]) |
| mean_L4 | 0.905 (-0.001 [-0.00, +0.00]) | 0.509 (+0.001 [-0.00, +0.00]) | 0.099 (+0.001 [+0.00, +0.00]) |

## 5. Prefill cost (one forward pass, batch 1; mostly kernel quality)

_`cost_profile.csv` not synced yet._

## 5b. Decode cost (one token at a time, batch 1)

_`decode_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
