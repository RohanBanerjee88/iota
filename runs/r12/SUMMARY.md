# iota run `r12`

- **updated:** 2026-10-10 23:29 UTC (last stage: `eval 8,9`)
- **code:** `ccb5ac0` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | – | – | – | – | – | – | not trained yet |
| gated_linear | 15000 | 0.926 | 0.852 | 1.000 | 0.644 | 0.0015 | ok |
| hybrid | – | – | – | – | – | – | not trained yet |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `logs/train.log` and the `*_sweep.json` files.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| gated_linear | 0.846 | 0.470 | 1.000 | 1.000 | pass |

n=500 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| n_bindings | true tokens | gated_linear |
|---:|---:|---|
| 2 | 256 | 1.000 [1.00–1.00] (1.00) |
| 4 | 256 | 0.997 [0.99–1.00] (0.99) |
| 8 | 256 | 0.989 [0.99–0.99] (0.92) |
| 16 | 256 | 0.970 [0.97–0.97] (0.63) |
| 32 | 289 | 0.883 [0.88–0.89] (0.16) |
| 64 | 386 | 0.637 [0.63–0.64] (0.00) |
| 128 | 773 | 0.309 [0.30–0.32] (0.00) |

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
| 4096 | 4104 | 0.986 [0.98–0.99] (0.99) |
| 8192 | 8200 | 0.789 [0.76–0.81] (0.79) |

## 4f. Decay gates (Pass 8): what γ does on real prompts

Mean γ ± std over tokens × heads, by token role. `retention` = mean log10 of the share of a fact value's write still in the state at the first question. A frozen gate has std ≈ 0; selective forgetting shows as γ(distractor) < γ(fact value).

| model | bindings | layer | fact value | distractor | query | answer | retention (log10) | gate bias / ‖W‖ |
|---|---:|---|---|---|---|---|---:|---|
| gated_linear | 8 | backbone.blocks.0.mixer | 0.368 ± 0.203 | 0.112 ± 0.091 | 0.189 ± 0.191 | 0.371 ± 0.202 | -256.557 | -0.06 / 1.132 |
| gated_linear | 8 | backbone.blocks.1.mixer | 0.570 ± 0.231 | 0.482 ± 0.164 | 0.530 ± 0.212 | 0.597 ± 0.218 | -79.345 | -0.00 / 1.495 |
| gated_linear | 8 | backbone.blocks.2.mixer | 0.909 ± 0.193 | 0.991 ± 0.049 | 0.993 ± 0.009 | 0.986 ± 0.021 | -5.27 | +0.17 / 2.608 |
| gated_linear | 8 | backbone.blocks.3.mixer | 0.400 ± 0.340 | 0.546 ± 0.207 | 0.492 ± 0.252 | 0.642 ± 0.285 | -74.899 | -0.01 / 1.571 |
| gated_linear | 8 | backbone.blocks.4.mixer | 0.855 ± 0.285 | 0.829 ± 0.312 | 0.781 ± 0.383 | 0.901 ± 0.224 | -33.7 | +0.07 / 2.362 |
| gated_linear | 32 | backbone.blocks.0.mixer | 0.369 ± 0.204 | 0.113 ± 0.092 | 0.190 ± 0.191 | 0.372 ± 0.202 | -181.757 | -0.06 / 1.132 |
| gated_linear | 32 | backbone.blocks.1.mixer | 0.570 ± 0.232 | 0.482 ± 0.166 | 0.531 ± 0.213 | 0.597 ± 0.219 | -66.823 | -0.00 / 1.495 |
| gated_linear | 32 | backbone.blocks.2.mixer | 0.909 ± 0.193 | 0.988 ± 0.070 | 0.993 ± 0.010 | 0.985 ± 0.022 | -18.063 | +0.17 / 2.608 |
| gated_linear | 32 | backbone.blocks.3.mixer | 0.396 ± 0.340 | 0.560 ± 0.213 | 0.496 ± 0.256 | 0.657 ± 0.284 | -77.327 | -0.01 / 1.571 |
| gated_linear | 32 | backbone.blocks.4.mixer | 0.854 ± 0.286 | 0.822 ± 0.323 | 0.777 ± 0.388 | 0.896 ± 0.230 | -31.017 | +0.07 / 2.362 |

## 4g. Gate-clamp intervention (Pass 9): which layers' forgetting does recall need?

Per-query recall with one GLA layer's gate replaced by a constant: `keep` = σ(6) ≈ 0.9975 (no forgetting), `mean` = the head's own average γ (same amount of forgetting, no selectivity). Δ vs the unclamped model on the same prompts, paired 95% CI.

**gated_linear**

| condition | 8 facts | 32 facts | 64 facts |
|---|---|---|---|
| none | 0.989 | 0.882 | 0.639 |
| keep_all | 0.006 (-0.983 [-0.99, -0.98]) | 0.009 (-0.874 [-0.88, -0.87]) | 0.010 (-0.629 [-0.64, -0.62]) |
| keep_L0 | 0.000 (-0.989 [-0.99, -0.99]) | 0.000 (-0.882 [-0.89, -0.87]) | 0.001 (-0.638 [-0.65, -0.63]) |
| mean_L0 | 0.933 (-0.056 [-0.06, -0.05]) | 0.811 (-0.071 [-0.08, -0.06]) | 0.570 (-0.069 [-0.08, -0.06]) |
| keep_L1 | 0.801 (-0.188 [-0.20, -0.18]) | 0.580 (-0.303 [-0.31, -0.29]) | 0.361 (-0.278 [-0.29, -0.27]) |
| mean_L1 | 0.985 (-0.004 [-0.01, -0.00]) | 0.865 (-0.018 [-0.02, -0.01]) | 0.619 (-0.021 [-0.03, -0.01]) |
| keep_L2 | 0.960 (-0.029 [-0.03, -0.02]) | 0.760 (-0.122 [-0.13, -0.11]) | 0.467 (-0.172 [-0.18, -0.16]) |
| mean_L2 | 0.972 (-0.017 [-0.02, -0.01]) | 0.842 (-0.040 [-0.05, -0.03]) | 0.611 (-0.029 [-0.04, -0.02]) |
| keep_L3 | 0.255 (-0.734 [-0.75, -0.72]) | 0.093 (-0.789 [-0.80, -0.78]) | 0.060 (-0.580 [-0.59, -0.57]) |
| mean_L3 | 0.957 (-0.032 [-0.04, -0.03]) | 0.810 (-0.073 [-0.08, -0.07]) | 0.562 (-0.077 [-0.09, -0.07]) |
| keep_L4 | 0.937 (-0.052 [-0.06, -0.04]) | 0.717 (-0.165 [-0.17, -0.16]) | 0.445 (-0.195 [-0.21, -0.18]) |
| mean_L4 | 0.988 (-0.001 [-0.00, +0.00]) | 0.878 (-0.004 [-0.01, -0.00]) | 0.635 (-0.005 [-0.01, +0.00]) |

## 5. Prefill cost (one forward pass, batch 1; mostly kernel quality)

_`cost_profile.csv` not synced yet._

## 5b. Decode cost (one token at a time, batch 1)

_`decode_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
