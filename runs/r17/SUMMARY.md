# iota run `r17`

- **updated:** 2026-10-15 00:57 UTC (last stage: `eval 8,9,10`)
- **code:** `0df1f56` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | – | – | – | – | – | – | not trained yet |
| gated_linear | 15000 | 0.665 | 0.329 | 1.000 | 0.398 | 0.0015 | ok |
| hybrid | – | – | – | – | – | – | not trained yet |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `*_sweep_history.json` (and `logs/train.log`); `*_sweep.json` stops at the best checkpoint.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| gated_linear | 0.317 | 0.096 | 1.000 | 1.000 | pass |

n=500 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| n_bindings | true tokens | gated_linear |
|---:|---:|---|
| 2 | 256 | 0.959 [0.95–0.97] (0.92) |
| 4 | 256 | 0.883 [0.87–0.89] (0.61) |
| 8 | 256 | 0.772 [0.76–0.78] (0.12) |
| 16 | 256 | 0.610 [0.60–0.62] (0.00) |
| 32 | 289 | 0.298 [0.29–0.31] (0.00) |
| 64 | 386 | 0.096 [0.09–0.10] (0.00) |
| 128 | 773 | 0.055 [0.05–0.06] (0.00) |

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
| gated_linear | 8 | backbone.blocks.0.mixer | 0.210 ± 0.206 | 0.741 ± 0.115 | 0.436 ± 0.373 | 0.219 ± 0.207 | -42.716 | +1.68 / 1.738 |
| gated_linear | 8 | backbone.blocks.1.mixer | 0.727 ± 0.370 | 0.495 ± 0.328 | 0.673 ± 0.358 | 0.746 ± 0.356 | -96.861 | +0.02 / 1.528 |
| gated_linear | 8 | backbone.blocks.2.mixer | 0.226 ± 0.184 | 0.626 ± 0.206 | 0.528 ± 0.269 | 0.269 ± 0.221 | -60.184 | -0.06 / 1.649 |
| gated_linear | 8 | backbone.blocks.3.mixer | 0.549 ± 0.308 | 0.470 ± 0.164 | 0.474 ± 0.359 | 0.669 ± 0.255 | -83.199 | +0.00 / 1.513 |
| gated_linear | 8 | backbone.blocks.4.mixer | 1.000 ± 0.004 | 0.948 ± 0.111 | 0.999 ± 0.004 | 0.999 ± 0.005 | -6.027 | +0.16 / 2.988 |
| gated_linear | 32 | backbone.blocks.0.mixer | 0.209 ± 0.206 | 0.739 ± 0.118 | 0.433 ± 0.373 | 0.218 ± 0.207 | -75.825 | +1.68 / 1.738 |
| gated_linear | 32 | backbone.blocks.1.mixer | 0.726 ± 0.371 | 0.499 ± 0.331 | 0.673 ± 0.357 | 0.745 ± 0.356 | -70.806 | +0.02 / 1.528 |
| gated_linear | 32 | backbone.blocks.2.mixer | 0.233 ± 0.187 | 0.622 ± 0.207 | 0.533 ± 0.270 | 0.269 ± 0.224 | -69.903 | -0.06 / 1.649 |
| gated_linear | 32 | backbone.blocks.3.mixer | 0.556 ± 0.306 | 0.465 ± 0.175 | 0.476 ± 0.361 | 0.668 ± 0.256 | -74.112 | +0.00 / 1.513 |
| gated_linear | 32 | backbone.blocks.4.mixer | 1.000 ± 0.002 | 0.942 ± 0.132 | 1.000 ± 0.003 | 0.999 ± 0.005 | -3.513 | +0.16 / 2.988 |

## 4g. Gate-clamp intervention (Pass 9): which layers' forgetting does recall need?

Per-query recall with one GLA layer's gate replaced by a constant: `keep` = σ(6) ≈ 0.9975 (no forgetting), `mean` = the head's own average γ (same amount of forgetting, no selectivity). Δ vs the unclamped model on the same prompts, paired 95% CI.

**gated_linear**

| condition | 8 facts | 32 facts | 64 facts |
|---|---|---|---|
| none | 0.779 | 0.296 | 0.095 |
| keep_all | 0.011 (-0.768 [-0.78, -0.76]) | 0.011 (-0.286 [-0.30, -0.28]) | 0.011 (-0.084 [-0.09, -0.08]) |
| keep_L0 | 0.014 (-0.765 [-0.78, -0.75]) | 0.010 (-0.286 [-0.30, -0.28]) | 0.010 (-0.085 [-0.09, -0.08]) |
| mean_L0 | 0.240 (-0.539 [-0.56, -0.52]) | 0.202 (-0.095 [-0.11, -0.08]) | 0.079 (-0.016 [-0.02, -0.01]) |
| keep_L1 | 0.618 (-0.162 [-0.18, -0.15]) | 0.225 (-0.071 [-0.08, -0.06]) | 0.095 (+0.000 [-0.00, +0.01]) |
| mean_L1 | 0.707 (-0.073 [-0.09, -0.06]) | 0.273 (-0.023 [-0.03, -0.01]) | 0.095 (+0.000 [-0.00, +0.00]) |
| keep_L2 | 0.636 (-0.143 [-0.16, -0.13]) | 0.191 (-0.106 [-0.12, -0.10]) | 0.086 (-0.009 [-0.02, -0.00]) |
| mean_L2 | 0.752 (-0.028 [-0.04, -0.02]) | 0.282 (-0.015 [-0.02, -0.01]) | 0.093 (-0.002 [-0.00, +0.00]) |
| keep_L3 | 0.353 (-0.426 [-0.44, -0.41]) | 0.123 (-0.173 [-0.18, -0.16]) | 0.086 (-0.009 [-0.02, -0.00]) |
| mean_L3 | 0.700 (-0.079 [-0.09, -0.07]) | 0.246 (-0.051 [-0.06, -0.04]) | 0.097 (+0.002 [-0.00, +0.01]) |
| keep_L4 | 0.621 (-0.158 [-0.17, -0.14]) | 0.215 (-0.082 [-0.09, -0.07]) | 0.094 (-0.001 [-0.01, +0.01]) |
| mean_L4 | 0.599 (-0.180 [-0.19, -0.17]) | 0.253 (-0.043 [-0.05, -0.03]) | 0.095 (+0.000 [-0.00, +0.00]) |

## 4h. Gate replay (Pass 10): does the timing of forgetting matter?

One layer's recorded gates over the context span are replayed unchanged, shuffled in time (same values, same product), replaced by their geometric mean (same product, no timing) or their arithmetic mean (forgets less). Per-query recall, Δ vs baseline [paired 95% CI].

**gated_linear**

| facts | layer | baseline | replay | shuffle (range) | geomean | arith |
|---:|---|---:|---|---|---|---|
| 8 | L0 | 0.775 | 0.775 (+0.000 [0.00, 0.00]) | 0.490 (0.49–0.49) (-0.284 [-0.30, -0.27]) | 0.679 (-0.095 [-0.11, -0.08]) | 0.491 (-0.284 [-0.30, -0.27]) |
| 8 | L1 | 0.775 | 0.775 (+0.000 [0.00, 0.00]) | 0.690 (0.69–0.69) (-0.085 [-0.10, -0.07]) | 0.695 (-0.080 [-0.09, -0.07]) | 0.697 (-0.078 [-0.09, -0.07]) |
| 8 | L2 | 0.775 | 0.775 (+0.000 [0.00, 0.00]) | 0.748 (0.75–0.75) (-0.027 [-0.03, -0.02]) | 0.750 (-0.025 [-0.03, -0.02]) | 0.746 (-0.029 [-0.04, -0.02]) |
| 8 | L3 | 0.775 | 0.775 (+0.000 [0.00, 0.00]) | 0.735 (0.73–0.74) (-0.040 [-0.05, -0.03]) | 0.741 (-0.034 [-0.04, -0.02]) | 0.737 (-0.038 [-0.05, -0.03]) |
| 8 | L4 | 0.775 | 0.775 (+0.000 [0.00, 0.00]) | 0.771 (0.77–0.77) (-0.004 [-0.01, -0.00]) | 0.776 (+0.001 [-0.00, 0.00]) | 0.776 (+0.001 [-0.00, 0.00]) |
| 32 | L0 | 0.301 | 0.301 (+0.000 [0.00, 0.00]) | 0.200 (0.20–0.20) (-0.102 [-0.11, -0.09]) | 0.275 (-0.027 [-0.03, -0.02]) | 0.223 (-0.079 [-0.09, -0.07]) |
| 32 | L1 | 0.301 | 0.301 (+0.000 [0.00, 0.00]) | 0.278 (0.28–0.28) (-0.023 [-0.03, -0.02]) | 0.280 (-0.021 [-0.03, -0.01]) | 0.281 (-0.020 [-0.03, -0.01]) |
| 32 | L2 | 0.301 | 0.301 (+0.000 [0.00, 0.00]) | 0.282 (0.28–0.28) (-0.019 [-0.02, -0.01]) | 0.293 (-0.009 [-0.01, -0.00]) | 0.284 (-0.017 [-0.02, -0.01]) |
| 32 | L3 | 0.301 | 0.301 (+0.000 [0.00, 0.00]) | 0.263 (0.26–0.27) (-0.039 [-0.04, -0.03]) | 0.274 (-0.028 [-0.03, -0.02]) | 0.262 (-0.039 [-0.05, -0.03]) |
| 32 | L4 | 0.301 | 0.301 (+0.000 [0.00, 0.00]) | 0.294 (0.29–0.29) (-0.007 [-0.01, -0.00]) | 0.296 (-0.005 [-0.01, -0.00]) | 0.296 (-0.005 [-0.01, 0.00]) |

## 5. Prefill cost (one forward pass, batch 1; mostly kernel quality)

_`cost_profile.csv` not synced yet._

## 5b. Decode cost (one token at a time, batch 1)

_`decode_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
