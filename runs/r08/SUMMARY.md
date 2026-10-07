# iota run `r08`

- **updated:** 2026-10-07 18:08 UTC (last stage: `eval 8,9`)
- **code:** `c3632aa` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | 14000 | 0.904 | 0.809 | 1.000 | 0.600 | 0.0015 | ok |
| gated_linear | 8000 | 0.597 | 0.194 | 1.000 | 0.362 | 0.0015 | ok |
| hybrid | 5000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.00075 | ok |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `logs/train.log` and the `*_sweep.json` files.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| hybrid | 0.999 | 0.994 | 1.000 | 1.000 | pass |
| transformer | 0.809 | 0.380 | 1.000 | 1.000 | pass |
| gated_linear | 0.164 | 0.034 | 1.000 | 1.000 | pass |

n=500 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| n_bindings | true tokens | transformer | gated_linear | hybrid |
|---:|---:|---|---|---|
| 2 | 256 | 0.997 [0.99–1.00] (0.99) | 0.851 [0.84–0.87] (0.70) | 1.000 [1.00–1.00] (1.00) |
| 4 | 256 | 0.991 [0.99–0.99] (0.96) | 0.705 [0.69–0.72] (0.20) | 1.000 [1.00–1.00] (1.00) |
| 8 | 256 | 0.975 [0.97–0.98] (0.82) | 0.531 [0.52–0.54] (0.00) | 1.000 [1.00–1.00] (1.00) |
| 16 | 256 | 0.945 [0.94–0.95] (0.43) | 0.360 [0.35–0.37] (0.00) | 1.000 [1.00–1.00] (0.99) |
| 32 | 289 | 0.842 [0.84–0.85] (0.05) | 0.143 [0.14–0.15] (0.00) | 1.000 [1.00–1.00] (0.99) |
| 64 | 386 | 0.620 [0.61–0.63] (0.00) | 0.079 [0.08–0.08] (0.00) | 0.988 [0.99–0.99] (0.81) |
| 128 | 773 | 0.347 [0.34–0.35] (0.00) | 0.049 [0.05–0.05] (0.00) | 0.986 [0.98–0.99] (0.80) |

## 3. Pass 2 — length generalisation

_`pass2_length.csv` not synced yet._

## 4. Pass 3 — state_track control

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| seq_len | true tokens | transformer | gated_linear | hybrid |
|---:|---:|---|---|---|
| 128 | 136 | 1.000 [1.00–1.00] (1.00) | 0.999 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) |
| 256 | 264 | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) |
| 512 | 520 | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) |
| 1024 | 1032 | 0.955 [0.94–0.97] (0.95) | 1.000 [1.00–1.00] (1.00) | 0.967 [0.96–0.98] (0.97) |
| 2048 | 2056 | 0.189 [0.17–0.21] (0.19) | 1.000 [1.00–1.00] (1.00) | 0.613 [0.58–0.64] (0.61) |
| 4096 | 4104 | 0.017 [0.01–0.03] (0.02) | 0.998 [0.99–1.00] (1.00) | 0.184 [0.16–0.21] (0.18) |
| 8192 | 8200 | 0.020 [0.01–0.03] (0.02) | 0.970 [0.96–0.98] (0.97) | 0.386 [0.36–0.42] (0.39) |

## 4b. Diagnostic — recall by key length (digits), same prompts as Pass 1

| n_bindings | key digits | transformer | gated_linear | hybrid |
|---:|---:|---:|---:|---:|
| 16 | 1 | 0.948 | 0.695 | 0.998 |
| 16 | 2 | 0.952 | 0.286 | 1.000 |
| 16 | 3 | 0.920 | 0.489 | 1.000 |
| 32 | 1 | 0.874 | 0.431 | 0.998 |
| 32 | 2 | 0.848 | 0.088 | 1.000 |
| 32 | 3 | 0.808 | 0.215 | 1.000 |
| 64 | 1 | 0.674 | 0.223 | 0.985 |
| 64 | 2 | 0.638 | 0.057 | 0.987 |
| 64 | 3 | 0.545 | 0.100 | 0.991 |

If one model's errors pile up on 3-digit keys, its drop with load is partly key resolution, not memory capacity.

## 4f. Decay gates (Pass 8): what γ does on real prompts

Mean γ ± std over tokens × heads, by token role. `retention` = mean log10 of the share of a fact value's write still in the state at the first question. A frozen gate has std ≈ 0; selective forgetting shows as γ(distractor) < γ(fact value).

| model | bindings | layer | fact value | distractor | query | answer | retention (log10) | gate bias / ‖W‖ |
|---|---:|---|---|---|---|---|---:|---|
| gated_linear | 8 | backbone.blocks.0.mixer | 0.483 ± 0.430 | 0.772 ± 0.168 | 0.663 ± 0.366 | 0.507 ± 0.425 | -39.665 | +1.79 / 1.829 |
| gated_linear | 8 | backbone.blocks.1.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.0 | +2.01 / 2.966 |
| gated_linear | 8 | backbone.blocks.2.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.999 ± 0.006 | -0.001 | +2.02 / 2.696 |
| gated_linear | 8 | backbone.blocks.3.mixer | 0.991 ± 0.028 | 0.982 ± 0.034 | 0.997 ± 0.011 | 0.999 ± 0.003 | -1.939 | +2.10 / 2.993 |
| gated_linear | 8 | backbone.blocks.4.mixer | 0.976 ± 0.056 | 0.987 ± 0.024 | 0.996 ± 0.007 | 0.997 ± 0.006 | -1.474 | +1.99 / 2.088 |
| gated_linear | 32 | backbone.blocks.0.mixer | 0.482 ± 0.430 | 0.773 ± 0.169 | 0.663 ± 0.367 | 0.507 ± 0.425 | -71.393 | +1.79 / 1.829 |
| gated_linear | 32 | backbone.blocks.1.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.0 | +2.01 / 2.966 |
| gated_linear | 32 | backbone.blocks.2.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.002 | -0.001 | +2.02 / 2.696 |
| gated_linear | 32 | backbone.blocks.3.mixer | 0.993 ± 0.022 | 0.989 ± 0.022 | 0.998 ± 0.006 | 0.999 ± 0.003 | -1.148 | +2.10 / 2.993 |
| gated_linear | 32 | backbone.blocks.4.mixer | 0.983 ± 0.043 | 0.991 ± 0.016 | 0.997 ± 0.005 | 0.997 ± 0.004 | -1.044 | +1.99 / 2.088 |
| hybrid | 8 | backbone.blocks.0.mixer | 0.458 ± 0.336 | 0.659 ± 0.181 | 0.247 ± 0.209 | 0.461 ± 0.336 | -56.461 | +1.83 / 1.113 |
| hybrid | 8 | backbone.blocks.1.mixer | 0.706 ± 0.362 | 0.761 ± 0.377 | 0.737 ± 0.341 | 0.748 ± 0.304 | -59.369 | +1.97 / 1.215 |
| hybrid | 8 | backbone.blocks.3.mixer | 0.909 ± 0.154 | 0.928 ± 0.120 | 0.926 ± 0.125 | 0.919 ± 0.136 | -8.542 | +2.03 / 1.365 |
| hybrid | 32 | backbone.blocks.0.mixer | 0.458 ± 0.336 | 0.656 ± 0.183 | 0.240 ± 0.207 | 0.461 ± 0.336 | -81.884 | +1.83 / 1.113 |
| hybrid | 32 | backbone.blocks.1.mixer | 0.705 ± 0.363 | 0.762 ± 0.376 | 0.738 ± 0.342 | 0.749 ± 0.303 | -52.243 | +1.97 / 1.215 |
| hybrid | 32 | backbone.blocks.3.mixer | 0.911 ± 0.152 | 0.931 ± 0.115 | 0.927 ± 0.123 | 0.919 ± 0.136 | -7.172 | +2.03 / 1.365 |

## 4g. Gate-clamp intervention (Pass 9): which layers' forgetting does recall need?

Per-query recall with one GLA layer's gate replaced by a constant: `keep` = σ(6) ≈ 0.9975 (no forgetting), `mean` = the head's own average γ (same amount of forgetting, no selectivity). Δ vs the unclamped model on the same prompts, paired 95% CI.

**gated_linear**

| condition | 8 facts | 32 facts | 64 facts |
|---|---|---|---|
| none | 0.523 | 0.144 | 0.080 |
| keep_all | 0.011 (-0.511 [-0.52, -0.50]) | 0.010 (-0.134 [-0.14, -0.13]) | 0.010 (-0.070 [-0.08, -0.06]) |
| keep_L0 | 0.013 (-0.510 [-0.52, -0.50]) | 0.010 (-0.134 [-0.14, -0.13]) | 0.009 (-0.070 [-0.08, -0.06]) |
| mean_L0 | 0.183 (-0.339 [-0.35, -0.32]) | 0.111 (-0.033 [-0.04, -0.02]) | 0.066 (-0.014 [-0.02, -0.01]) |
| keep_L1 | 0.517 (-0.005 [-0.01, +0.00]) | 0.145 (+0.001 [-0.00, +0.00]) | 0.080 (+0.000 [-0.00, +0.00]) |
| mean_L1 | 0.523 (+0.000 [+0.00, +0.00]) | 0.144 (+0.000 [+0.00, +0.00]) | 0.080 (+0.000 [+0.00, +0.00]) |
| keep_L2 | 0.513 (-0.009 [-0.02, +0.00]) | 0.139 (-0.005 [-0.01, -0.00]) | 0.078 (-0.001 [-0.01, +0.00]) |
| mean_L2 | 0.523 (+0.000 [-0.00, +0.00]) | 0.143 (-0.001 [-0.00, +0.00]) | 0.080 (+0.000 [-0.00, +0.00]) |
| keep_L3 | 0.518 (-0.005 [-0.01, -0.00]) | 0.142 (-0.002 [-0.01, +0.00]) | 0.078 (-0.002 [-0.00, +0.00]) |
| mean_L3 | 0.520 (-0.003 [-0.01, +0.00]) | 0.143 (-0.001 [-0.00, +0.00]) | 0.079 (-0.001 [-0.00, +0.00]) |
| keep_L4 | 0.518 (-0.005 [-0.01, +0.00]) | 0.143 (-0.001 [-0.00, +0.00]) | 0.079 (-0.001 [-0.00, +0.00]) |
| mean_L4 | 0.520 (-0.002 [-0.01, +0.00]) | 0.144 (+0.000 [-0.00, +0.00]) | 0.079 (-0.001 [-0.00, +0.00]) |

**hybrid**

| condition | 8 facts | 32 facts | 64 facts |
|---|---|---|---|
| none | 1.000 | 1.000 | 0.989 |
| keep_all | 0.015 (-0.985 [-0.99, -0.98]) | 0.012 (-0.987 [-0.99, -0.98]) | 0.013 (-0.976 [-0.98, -0.97]) |
| keep_L0 | 0.017 (-0.983 [-0.99, -0.98]) | 0.013 (-0.987 [-0.99, -0.98]) | 0.013 (-0.975 [-0.98, -0.97]) |
| mean_L0 | 0.996 (-0.004 [-0.01, -0.00]) | 0.999 (-0.001 [-0.00, -0.00]) | 0.987 (-0.001 [-0.00, -0.00]) |
| keep_L1 | 0.845 (-0.155 [-0.17, -0.14]) | 0.553 (-0.446 [-0.46, -0.44]) | 0.412 (-0.577 [-0.59, -0.57]) |
| mean_L1 | 1.000 (-0.000 [-0.00, +0.00]) | 0.998 (-0.002 [-0.00, -0.00]) | 0.986 (-0.003 [-0.00, -0.00]) |
| keep_L3 | 1.000 (+0.000 [+0.00, +0.00]) | 1.000 (+0.000 [+0.00, +0.00]) | 0.989 (+0.000 [+0.00, +0.00]) |
| mean_L3 | 1.000 (+0.000 [+0.00, +0.00]) | 1.000 (+0.000 [+0.00, +0.00]) | 0.989 (+0.000 [+0.00, +0.00]) |

## 5. Prefill cost (one forward pass, batch 1; mostly kernel quality)

_`cost_profile.csv` not synced yet._

## 5b. Decode cost (one token at a time, batch 1)

_`decode_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
