# iota run `r14`

- **updated:** 2026-10-13 22:30 UTC (last stage: `eval 8,9,10`)
- **code:** `04ff3c6` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | – | – | – | – | – | – | not trained yet |
| gated_linear | 14500 | 0.725 | 0.449 | 1.000 | 0.424 | 0.0015 | ok |
| hybrid | – | – | – | – | – | – | not trained yet |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `*_sweep_history.json` (and `logs/train.log`); `*_sweep.json` stops at the best checkpoint.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| gated_linear | 0.433 | 0.130 | 1.000 | 1.000 | pass |

n=500 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| n_bindings | true tokens | gated_linear |
|---:|---:|---|
| 2 | 256 | 0.983 [0.98–0.99] (0.97) |
| 4 | 256 | 0.946 [0.94–0.95] (0.80) |
| 8 | 256 | 0.872 [0.86–0.88] (0.34) |
| 16 | 256 | 0.739 [0.73–0.74] (0.00) |
| 32 | 289 | 0.442 [0.43–0.45] (0.00) |
| 64 | 386 | 0.219 [0.21–0.23] (0.00) |
| 128 | 773 | 0.089 [0.09–0.09] (0.00) |

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
| 2048 | 2056 | 0.999 [1.00–1.00] (1.00) |
| 4096 | 4104 | 0.970 [0.96–0.98] (0.97) |
| 8192 | 8200 | 0.618 [0.59–0.65] (0.62) |

## 4f. Decay gates (Pass 8): what γ does on real prompts

Mean γ ± std over tokens × heads, by token role. `retention` = mean log10 of the share of a fact value's write still in the state at the first question. A frozen gate has std ≈ 0; selective forgetting shows as γ(distractor) < γ(fact value).

| model | bindings | layer | fact value | distractor | query | answer | retention (log10) | gate bias / ‖W‖ |
|---|---:|---|---|---|---|---|---:|---|
| gated_linear | 8 | backbone.blocks.0.mixer | 0.334 ± 0.287 | 0.513 ± 0.149 | 0.532 ± 0.326 | 0.342 ± 0.288 | -75.572 | -0.04 / 1.463 |
| gated_linear | 8 | backbone.blocks.1.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.001 | +1.98 / 3.324 |
| gated_linear | 8 | backbone.blocks.2.mixer | 0.999 ± 0.001 | 0.999 ± 0.001 | 0.999 ± 0.005 | 1.000 ± 0.000 | -0.057 | +1.91 / 1.770 |
| gated_linear | 8 | backbone.blocks.3.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.001 | -0.001 | +1.95 / 1.980 |
| gated_linear | 8 | backbone.blocks.4.mixer | 0.994 ± 0.017 | 0.998 ± 0.004 | 1.000 ± 0.001 | 1.000 ± 0.000 | -0.244 | +2.09 / 3.008 |
| gated_linear | 32 | backbone.blocks.0.mixer | 0.333 ± 0.287 | 0.511 ± 0.151 | 0.531 ± 0.327 | 0.342 ± 0.288 | -78.8 | -0.04 / 1.463 |
| gated_linear | 32 | backbone.blocks.1.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.001 | +1.98 / 3.324 |
| gated_linear | 32 | backbone.blocks.2.mixer | 0.999 ± 0.001 | 1.000 ± 0.001 | 0.999 ± 0.005 | 1.000 ± 0.000 | -0.043 | +1.91 / 1.770 |
| gated_linear | 32 | backbone.blocks.3.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.003 | +1.95 / 1.980 |
| gated_linear | 32 | backbone.blocks.4.mixer | 0.997 ± 0.009 | 0.999 ± 0.003 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.123 | +2.09 / 3.008 |

## 4g. Gate-clamp intervention (Pass 9): which layers' forgetting does recall need?

Per-query recall with one GLA layer's gate replaced by a constant: `keep` = σ(6) ≈ 0.9975 (no forgetting), `mean` = the head's own average γ (same amount of forgetting, no selectivity). Δ vs the unclamped model on the same prompts, paired 95% CI.

**gated_linear**

| condition | 8 facts | 32 facts | 64 facts |
|---|---|---|---|
| none | 0.872 | 0.434 | 0.221 |
| keep_all | 0.013 (-0.859 [-0.87, -0.85]) | 0.014 (-0.420 [-0.43, -0.41]) | 0.012 (-0.209 [-0.22, -0.20]) |
| keep_L0 | 0.012 (-0.859 [-0.87, -0.85]) | 0.012 (-0.422 [-0.43, -0.41]) | 0.011 (-0.210 [-0.22, -0.20]) |
| mean_L0 | 0.495 (-0.377 [-0.39, -0.36]) | 0.189 (-0.245 [-0.26, -0.23]) | 0.083 (-0.138 [-0.15, -0.13]) |
| keep_L1 | 0.873 (+0.001 [-0.00, +0.01]) | 0.396 (-0.038 [-0.05, -0.03]) | 0.170 (-0.051 [-0.06, -0.04]) |
| mean_L1 | 0.873 (+0.002 [-0.00, +0.00]) | 0.433 (-0.001 [-0.00, +0.00]) | 0.220 (-0.002 [-0.01, +0.00]) |
| keep_L2 | 0.866 (-0.006 [-0.01, -0.00]) | 0.428 (-0.006 [-0.01, -0.00]) | 0.220 (-0.002 [-0.01, +0.00]) |
| mean_L2 | 0.872 (+0.000 [-0.00, +0.00]) | 0.433 (-0.001 [-0.00, +0.00]) | 0.222 (+0.001 [-0.00, +0.00]) |
| keep_L3 | 0.857 (-0.015 [-0.02, -0.01]) | 0.413 (-0.021 [-0.03, -0.01]) | 0.196 (-0.025 [-0.03, -0.02]) |
| mean_L3 | 0.859 (-0.012 [-0.02, -0.01]) | 0.423 (-0.011 [-0.02, -0.00]) | 0.208 (-0.013 [-0.02, -0.01]) |
| keep_L4 | 0.868 (-0.004 [-0.01, +0.00]) | 0.427 (-0.006 [-0.01, -0.00]) | 0.220 (-0.001 [-0.00, +0.00]) |
| mean_L4 | 0.870 (-0.002 [-0.00, +0.00]) | 0.433 (-0.000 [-0.00, +0.00]) | 0.220 (-0.001 [-0.00, +0.00]) |

## 4h. Gate replay (Pass 10): does the timing of forgetting matter?

One layer's recorded gates over the context span are replayed unchanged, shuffled in time (same values, same product), replaced by their geometric mean (same product, no timing) or their arithmetic mean (forgets less). Per-query recall, Δ vs baseline [paired 95% CI].

**gated_linear**

| facts | layer | baseline | replay | shuffle (range) | geomean | arith |
|---:|---|---:|---|---|---|---|
| 8 | L0 | 0.875 | 0.875 (+0.000 [0.00, 0.00]) | 0.482 (0.48–0.49) (-0.393 [-0.41, -0.38]) | 0.578 (-0.297 [-0.31, -0.28]) | 0.500 (-0.375 [-0.39, -0.36]) |
| 8 | L1 | 0.875 | 0.875 (+0.000 [0.00, 0.00]) | 0.873 (0.87–0.87) (-0.002 [-0.00, -0.00]) | 0.875 (-0.000 [-0.00, 0.00]) | 0.875 (-0.000 [-0.00, 0.00]) |
| 8 | L2 | 0.875 | 0.875 (+0.000 [0.00, 0.00]) | 0.876 (0.88–0.88) (+0.001 [0.00, 0.00]) | 0.876 (+0.001 [0.00, 0.00]) | 0.876 (+0.001 [0.00, 0.00]) |
| 8 | L3 | 0.875 | 0.875 (+0.000 [0.00, 0.00]) | 0.839 (0.84–0.84) (-0.036 [-0.04, -0.03]) | 0.845 (-0.030 [-0.04, -0.02]) | 0.860 (-0.015 [-0.02, -0.01]) |
| 8 | L4 | 0.875 | 0.875 (+0.000 [0.00, 0.00]) | 0.873 (0.87–0.87) (-0.002 [-0.00, 0.00]) | 0.872 (-0.003 [-0.01, -0.00]) | 0.872 (-0.003 [-0.01, -0.00]) |
| 32 | L0 | 0.440 | 0.440 (+0.000 [0.00, 0.00]) | 0.190 (0.19–0.19) (-0.250 [-0.26, -0.24]) | 0.260 (-0.179 [-0.19, -0.17]) | 0.186 (-0.253 [-0.26, -0.24]) |
| 32 | L1 | 0.440 | 0.440 (+0.000 [0.00, 0.00]) | 0.433 (0.43–0.43) (-0.007 [-0.01, -0.00]) | 0.434 (-0.005 [-0.01, -0.00]) | 0.436 (-0.004 [-0.01, 0.00]) |
| 32 | L2 | 0.440 | 0.440 (+0.000 [0.00, 0.00]) | 0.439 (0.44–0.44) (-0.000 [-0.00, 0.00]) | 0.440 (+0.000 [-0.00, 0.00]) | 0.440 (+0.000 [-0.00, 0.00]) |
| 32 | L3 | 0.440 | 0.440 (+0.000 [0.00, 0.00]) | 0.396 (0.40–0.40) (-0.043 [-0.05, -0.04]) | 0.402 (-0.037 [-0.05, -0.03]) | 0.423 (-0.017 [-0.02, -0.01]) |
| 32 | L4 | 0.440 | 0.440 (+0.000 [0.00, 0.00]) | 0.442 (0.44–0.44) (+0.002 [-0.00, 0.00]) | 0.442 (+0.002 [-0.00, 0.00]) | 0.442 (+0.002 [-0.00, 0.00]) |

## 5. Prefill cost (one forward pass, batch 1; mostly kernel quality)

_`cost_profile.csv` not synced yet._

## 5b. Decode cost (one token at a time, batch 1)

_`decode_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
