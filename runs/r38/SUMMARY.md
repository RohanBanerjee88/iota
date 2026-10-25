# iota run `r38`

- **updated:** 2026-10-25 19:51 UTC (last stage: `eval 8,9,10`)
- **code:** `45e8d0d` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | – | – | – | – | – | – | not trained yet |
| gated_linear | 13000 | 0.639 | 0.277 | 1.000 | 0.382 | 0.0015 | ok |
| hybrid | – | – | – | – | – | – | not trained yet |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `*_sweep_history.json` (and `logs/train.log`); `*_sweep.json` stops at the best checkpoint.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| gated_linear | 0.260 | 0.066 | 1.000 | 1.000 | pass |

n=500 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| n_bindings | true tokens | gated_linear |
|---:|---:|---|
| 2 | 256 | 0.950 [0.94–0.96] (0.90) |
| 4 | 256 | 0.861 [0.85–0.87] (0.54) |
| 8 | 256 | 0.704 [0.69–0.71] (0.05) |
| 16 | 256 | 0.466 [0.46–0.47] (0.00) |
| 32 | 289 | 0.225 [0.22–0.23] (0.00) |
| 64 | 386 | 0.108 [0.10–0.11] (0.00) |
| 128 | 773 | 0.057 [0.05–0.06] (0.00) |

## 3. Pass 2 — length generalisation

_`pass2_length.csv` not synced yet._

## 4. Pass 3 — state_track control

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| seq_len | true tokens | gated_linear |
|---:|---:|---|
| 128 | 136 | 0.999 [1.00–1.00] (1.00) |
| 256 | 264 | 0.999 [1.00–1.00] (1.00) |
| 512 | 520 | 1.000 [1.00–1.00] (1.00) |
| 1024 | 1032 | 1.000 [1.00–1.00] (1.00) |
| 2048 | 2056 | 0.968 [0.96–0.98] (0.97) |
| 4096 | 4104 | 0.112 [0.09–0.13] (0.11) |
| 8192 | 8200 | 0.023 [0.01–0.03] (0.02) |

## 4f. Decay gates (Pass 8): what γ does on real prompts

Mean γ ± std over tokens × heads, by token role. `retention` = mean log10 of the share of a fact value's write still in the state at the first question. A frozen gate has std ≈ 0; selective forgetting shows as γ(distractor) < γ(fact value).

| model | bindings | layer | fact value | distractor | query | answer | retention (log10) | gate bias / ‖W‖ |
|---|---:|---|---|---|---|---|---:|---|
| gated_linear | 8 | backbone.blocks.0.mixer | 0.332 ± 0.276 | 0.140 ± 0.071 | 0.142 ± 0.126 | 0.331 ± 0.274 | -205.147 | -0.07 / 1.190 |
| gated_linear | 8 | backbone.blocks.1.mixer | 0.770 ± 0.272 | 0.981 ± 0.087 | 0.885 ± 0.231 | 0.881 ± 0.211 | -15.976 | -0.04 / 1.941 |
| gated_linear | 8 | backbone.blocks.2.mixer | 0.500 ± 0.257 | 0.620 ± 0.151 | 0.947 ± 0.053 | 0.958 ± 0.039 | -62.202 | +0.06 / 1.852 |
| gated_linear | 8 | backbone.blocks.3.mixer | 0.745 ± 0.370 | 0.731 ± 0.403 | 0.989 ± 0.054 | 0.993 ± 0.015 | -85.13 | +0.20 / 2.773 |
| gated_linear | 8 | backbone.blocks.4.mixer | 1.000 ± 0.001 | 0.999 ± 0.001 | 1.000 ± 0.000 | 0.999 ± 0.005 | -0.06 | +0.11 / 2.197 |
| gated_linear | 32 | backbone.blocks.0.mixer | 0.331 ± 0.276 | 0.142 ± 0.078 | 0.142 ± 0.127 | 0.331 ± 0.273 | -161.362 | -0.07 / 1.190 |
| gated_linear | 32 | backbone.blocks.1.mixer | 0.766 ± 0.273 | 0.970 ± 0.127 | 0.882 ± 0.233 | 0.880 ± 0.211 | -58.326 | -0.04 / 1.941 |
| gated_linear | 32 | backbone.blocks.2.mixer | 0.490 ± 0.258 | 0.603 ± 0.169 | 0.946 ± 0.055 | 0.957 ± 0.040 | -90.077 | +0.06 / 1.852 |
| gated_linear | 32 | backbone.blocks.3.mixer | 0.739 ± 0.374 | 0.722 ± 0.405 | 0.991 ± 0.042 | 0.993 ± 0.015 | -76.45 | +0.20 / 2.773 |
| gated_linear | 32 | backbone.blocks.4.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.999 ± 0.003 | -0.018 | +0.11 / 2.197 |

## 4g. Gate-clamp intervention (Pass 9): which layers' forgetting does recall need?

Per-query recall with one GLA layer's gate replaced by a constant: `keep` = σ(6) ≈ 0.9975 (no forgetting), `mean` = the head's own average γ (same amount of forgetting, no selectivity). Δ vs the unclamped model on the same prompts, paired 95% CI.

**gated_linear**

| condition | 8 facts | 32 facts | 64 facts |
|---|---|---|---|
| none | 0.701 | 0.232 | 0.112 |
| keep_all | 0.010 (-0.691 [-0.70, -0.68]) | 0.011 (-0.220 [-0.23, -0.21]) | 0.008 (-0.103 [-0.11, -0.10]) |
| keep_L0 | 0.010 (-0.691 [-0.70, -0.68]) | 0.009 (-0.223 [-0.23, -0.21]) | 0.009 (-0.103 [-0.11, -0.10]) |
| mean_L0 | 0.390 (-0.310 [-0.33, -0.29]) | 0.127 (-0.105 [-0.11, -0.10]) | 0.078 (-0.033 [-0.04, -0.03]) |
| keep_L1 | 0.049 (-0.652 [-0.67, -0.64]) | 0.021 (-0.211 [-0.22, -0.20]) | 0.015 (-0.097 [-0.10, -0.09]) |
| mean_L1 | 0.111 (-0.590 [-0.61, -0.57]) | 0.109 (-0.123 [-0.13, -0.11]) | 0.080 (-0.031 [-0.04, -0.02]) |
| keep_L2 | 0.456 (-0.245 [-0.26, -0.23]) | 0.136 (-0.096 [-0.11, -0.09]) | 0.059 (-0.052 [-0.06, -0.04]) |
| mean_L2 | 0.649 (-0.052 [-0.06, -0.04]) | 0.208 (-0.024 [-0.03, -0.02]) | 0.108 (-0.003 [-0.01, +0.00]) |
| keep_L3 | 0.662 (-0.038 [-0.05, -0.03]) | 0.209 (-0.023 [-0.03, -0.01]) | 0.097 (-0.015 [-0.02, -0.01]) |
| mean_L3 | 0.686 (-0.015 [-0.02, -0.01]) | 0.225 (-0.007 [-0.01, -0.00]) | 0.112 (+0.000 [-0.00, +0.01]) |
| keep_L4 | 0.698 (-0.003 [-0.01, +0.00]) | 0.230 (-0.002 [-0.01, +0.00]) | 0.109 (-0.003 [-0.01, +0.00]) |
| mean_L4 | 0.702 (+0.001 [-0.00, +0.00]) | 0.233 (+0.001 [-0.00, +0.00]) | 0.111 (-0.000 [-0.00, +0.00]) |

## 4h. Gate replay (Pass 10): does the timing of forgetting matter?

One layer's recorded gates over the context span are replayed unchanged, shuffled in time (same values, same product), replaced by their geometric mean (same product, no timing) or their arithmetic mean (forgets less). Per-query recall, Δ vs baseline [paired 95% CI].

**gated_linear**

| facts | layer | baseline | replay | shuffle (range) | geomean | arith |
|---:|---|---:|---|---|---|---|
| 8 | L0 | 0.699 | 0.699 (+0.000 [0.00, 0.00]) | 0.438 (0.44–0.44) (-0.261 [-0.28, -0.25]) | 0.463 (-0.236 [-0.25, -0.22]) | 0.482 (-0.217 [-0.23, -0.20]) |
| 8 | L1 | 0.699 | 0.699 (+0.000 [0.00, 0.00]) | 0.196 (0.19–0.20) (-0.503 [-0.52, -0.49]) | 0.371 (-0.328 [-0.34, -0.31]) | 0.129 (-0.570 [-0.59, -0.55]) |
| 8 | L2 | 0.699 | 0.699 (+0.000 [0.00, 0.00]) | 0.672 (0.67–0.67) (-0.026 [-0.04, -0.02]) | 0.684 (-0.015 [-0.02, -0.00]) | 0.674 (-0.025 [-0.04, -0.01]) |
| 8 | L3 | 0.699 | 0.699 (+0.000 [0.00, 0.00]) | 0.696 (0.69–0.70) (-0.003 [-0.01, 0.00]) | 0.696 (-0.003 [-0.01, 0.00]) | 0.698 (-0.002 [-0.01, 0.00]) |
| 8 | L4 | 0.699 | 0.699 (+0.000 [0.00, 0.00]) | 0.699 (0.70–0.70) (-0.000 [-0.00, 0.00]) | 0.699 (-0.001 [-0.00, 0.00]) | 0.699 (-0.000 [-0.00, 0.00]) |
| 32 | L0 | 0.226 | 0.226 (+0.000 [0.00, 0.00]) | 0.136 (0.13–0.14) (-0.090 [-0.10, -0.08]) | 0.147 (-0.079 [-0.09, -0.07]) | 0.164 (-0.062 [-0.07, -0.05]) |
| 32 | L1 | 0.226 | 0.226 (+0.000 [0.00, 0.00]) | 0.150 (0.15–0.16) (-0.077 [-0.08, -0.07]) | 0.204 (-0.022 [-0.03, -0.01]) | 0.148 (-0.078 [-0.09, -0.07]) |
| 32 | L2 | 0.226 | 0.226 (+0.000 [0.00, 0.00]) | 0.214 (0.21–0.21) (-0.013 [-0.02, -0.01]) | 0.221 (-0.006 [-0.01, 0.00]) | 0.223 (-0.004 [-0.01, 0.00]) |
| 32 | L3 | 0.226 | 0.226 (+0.000 [0.00, 0.00]) | 0.225 (0.22–0.23) (-0.002 [-0.00, 0.00]) | 0.226 (-0.001 [-0.00, 0.00]) | 0.226 (-0.000 [-0.00, 0.00]) |
| 32 | L4 | 0.226 | 0.226 (+0.000 [0.00, 0.00]) | 0.227 (0.23–0.23) (+0.001 [-0.00, 0.00]) | 0.227 (+0.001 [-0.00, 0.00]) | 0.227 (+0.001 [-0.00, 0.00]) |

## 5. Prefill cost (one forward pass, batch 1; mostly kernel quality)

_`cost_profile.csv` not synced yet._

## 5b. Decode cost (one token at a time, batch 1)

_`decode_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
