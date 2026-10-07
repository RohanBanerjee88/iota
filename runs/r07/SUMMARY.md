# iota run `r07`

- **updated:** 2026-10-07 04:36 UTC (last stage: `eval 9`)
- **code:** `ac9a40c` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | 14000 | 0.904 | 0.809 | 1.000 | 0.600 | 0.0015 | ok |
| gated_linear | 10500 | 0.692 | 0.385 | 1.000 | 0.412 | 0.0015 | ok |
| hybrid | 12500 | 0.689 | 0.377 | 1.000 | 0.412 | 0.00075 | ok |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `logs/train.log` and the `*_sweep.json` files.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| hybrid | 0.357 | 0.092 | 1.000 | 1.000 | pass |
| transformer | 0.809 | 0.380 | 1.000 | 1.000 | pass |
| gated_linear | 0.367 | 0.114 | 1.000 | 1.000 | pass |

n=500 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| n_bindings | true tokens | transformer | gated_linear | hybrid |
|---:|---:|---|---|---|
| 2 | 256 | 0.997 [0.99–1.00] (0.99) | 0.978 [0.97–0.98] (0.95) | 0.979 [0.97–0.98] (0.96) |
| 4 | 256 | 0.991 [0.99–0.99] (0.96) | 0.929 [0.92–0.94] (0.74) | 0.926 [0.92–0.93] (0.72) |
| 8 | 256 | 0.975 [0.97–0.98] (0.82) | 0.854 [0.85–0.86] (0.24) | 0.848 [0.84–0.86] (0.23) |
| 16 | 256 | 0.945 [0.94–0.95] (0.43) | 0.718 [0.71–0.72] (0.00) | 0.717 [0.71–0.72] (0.00) |
| 32 | 289 | 0.842 [0.84–0.85] (0.05) | 0.355 [0.35–0.36] (0.00) | 0.357 [0.35–0.36] (0.00) |
| 64 | 386 | 0.620 [0.61–0.63] (0.00) | 0.082 [0.08–0.09] (0.00) | 0.113 [0.11–0.12] (0.00) |
| 128 | 773 | 0.347 [0.34–0.35] (0.00) | 0.045 [0.04–0.05] (0.00) | 0.060 [0.06–0.06] (0.00) |

## 3. Pass 2 — length generalisation

_`pass2_length.csv` not synced yet._

## 4. Pass 3 — state_track control

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| seq_len | true tokens | transformer | gated_linear | hybrid |
|---:|---:|---|---|---|
| 128 | 136 | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) |
| 256 | 264 | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) |
| 512 | 520 | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) |
| 1024 | 1032 | 0.955 [0.94–0.97] (0.95) | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) |
| 2048 | 2056 | 0.189 [0.17–0.21] (0.19) | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) |
| 4096 | 4104 | 0.017 [0.01–0.03] (0.02) | 0.998 [0.99–1.00] (1.00) | 0.927 [0.91–0.94] (0.93) |
| 8192 | 8200 | 0.020 [0.01–0.03] (0.02) | 0.926 [0.91–0.94] (0.93) | 0.180 [0.16–0.20] (0.18) |

## 4b. Diagnostic — recall by key length (digits), same prompts as Pass 1

| n_bindings | key digits | transformer | gated_linear | hybrid |
|---:|---:|---:|---:|---:|
| 16 | 1 | 0.948 | 0.727 | 0.716 |
| 16 | 2 | 0.952 | 0.721 | 0.720 |
| 16 | 3 | 0.920 | 0.708 | 0.709 |
| 32 | 1 | 0.874 | 0.348 | 0.350 |
| 32 | 2 | 0.848 | 0.353 | 0.357 |
| 32 | 3 | 0.808 | 0.365 | 0.360 |
| 64 | 1 | 0.674 | 0.081 | 0.116 |
| 64 | 2 | 0.638 | 0.083 | 0.114 |
| 64 | 3 | 0.545 | 0.078 | 0.107 |

If one model's errors pile up on 3-digit keys, its drop with load is partly key resolution, not memory capacity.

## 4f. Decay gates (Pass 8): what γ does on real prompts

Mean γ ± std over tokens × heads, by token role. `retention` = mean log10 of the share of a fact value's write still in the state at the first question. A frozen gate has std ≈ 0; selective forgetting shows as γ(distractor) < γ(fact value).

| model | bindings | layer | fact value | distractor | query | answer | retention (log10) | gate bias / ‖W‖ |
|---|---:|---|---|---|---|---|---:|---|
| gated_linear | 8 | backbone.blocks.0.mixer | 1.000 ± 0.000 | 0.976 ± 0.050 | 1.000 ± 0.000 | 1.000 ± 0.000 | -2.394 | +5.82 / 4.343 |
| gated_linear | 8 | backbone.blocks.1.mixer | 0.997 ± 0.008 | 1.000 ± 0.001 | 0.998 ± 0.003 | 0.997 ± 0.010 | -0.049 | +5.64 / 2.442 |
| gated_linear | 8 | backbone.blocks.2.mixer | 1.000 ± 0.003 | 1.000 ± 0.003 | 0.999 ± 0.004 | 0.999 ± 0.002 | -0.028 | +5.74 / 3.373 |
| gated_linear | 8 | backbone.blocks.3.mixer | 0.992 ± 0.023 | 0.998 ± 0.008 | 0.991 ± 0.023 | 0.993 ± 0.014 | -0.264 | +5.75 / 3.057 |
| gated_linear | 8 | backbone.blocks.4.mixer | 0.982 ± 0.035 | 0.999 ± 0.002 | 0.998 ± 0.003 | 0.991 ± 0.015 | -0.149 | +5.57 / 1.812 |
| gated_linear | 32 | backbone.blocks.0.mixer | 1.000 ± 0.000 | 0.975 ± 0.059 | 1.000 ± 0.000 | 1.000 ± 0.000 | -1.22 | +5.82 / 4.343 |
| gated_linear | 32 | backbone.blocks.1.mixer | 0.997 ± 0.009 | 1.000 ± 0.001 | 0.999 ± 0.002 | 0.999 ± 0.004 | -0.094 | +5.64 / 2.442 |
| gated_linear | 32 | backbone.blocks.2.mixer | 0.999 ± 0.002 | 1.000 ± 0.005 | 0.999 ± 0.003 | 1.000 ± 0.001 | -0.109 | +5.74 / 3.373 |
| gated_linear | 32 | backbone.blocks.3.mixer | 0.992 ± 0.023 | 0.998 ± 0.010 | 0.996 ± 0.012 | 0.997 ± 0.007 | -0.564 | +5.75 / 3.057 |
| gated_linear | 32 | backbone.blocks.4.mixer | 0.985 ± 0.027 | 0.999 ± 0.002 | 0.999 ± 0.002 | 0.995 ± 0.009 | -0.346 | +5.57 / 1.812 |
| hybrid | 8 | backbone.blocks.0.mixer | 1.000 ± 0.001 | 1.000 ± 0.001 | 0.992 ± 0.080 | 1.000 ± 0.001 | -0.045 | +5.86 / 2.444 |
| hybrid | 8 | backbone.blocks.1.mixer | 0.983 ± 0.016 | 0.985 ± 0.014 | 0.995 ± 0.004 | 0.992 ± 0.007 | -1.535 | +5.61 / 1.159 |
| hybrid | 8 | backbone.blocks.3.mixer | 0.997 ± 0.003 | 0.999 ± 0.001 | 0.998 ± 0.001 | 0.999 ± 0.001 | -0.109 | +5.71 / 1.167 |
| hybrid | 32 | backbone.blocks.0.mixer | 1.000 ± 0.001 | 1.000 ± 0.001 | 0.996 ± 0.056 | 1.000 ± 0.001 | -0.028 | +5.86 / 2.444 |
| hybrid | 32 | backbone.blocks.1.mixer | 0.988 ± 0.013 | 0.987 ± 0.012 | 0.994 ± 0.003 | 0.991 ± 0.006 | -1.06 | +5.61 / 1.159 |
| hybrid | 32 | backbone.blocks.3.mixer | 0.998 ± 0.002 | 0.999 ± 0.002 | 0.998 ± 0.002 | 0.999 ± 0.001 | -0.239 | +5.71 / 1.167 |

## 4g. Gate-clamp intervention (Pass 9): which layers' forgetting does recall need?

Per-query recall with one GLA layer's gate replaced by a constant: `keep` = σ(6) ≈ 0.9975 (no forgetting), `mean` = the head's own average γ (same amount of forgetting, no selectivity). Δ vs the unclamped model on the same prompts, paired 95% CI.

**gated_linear**

| condition | 8 facts | 32 facts | 64 facts |
|---|---|---|---|
| none | 0.850 | 0.374 | 0.085 |
| keep_all | 0.608 (-0.242 [-0.25, -0.23]) | 0.193 (-0.181 [-0.19, -0.17]) | 0.076 (-0.008 [-0.01, -0.00]) |
| keep_L0 | 0.607 (-0.243 [-0.26, -0.23]) | 0.196 (-0.178 [-0.19, -0.17]) | 0.076 (-0.009 [-0.01, -0.00]) |
| mean_L0 | 0.829 (-0.022 [-0.03, -0.02]) | 0.337 (-0.037 [-0.04, -0.03]) | 0.085 (+0.000 [-0.00, +0.00]) |
| keep_L1 | 0.849 (-0.001 [-0.00, +0.00]) | 0.372 (-0.001 [-0.00, +0.00]) | 0.085 (+0.001 [-0.00, +0.00]) |
| mean_L1 | 0.850 (-0.001 [-0.00, +0.00]) | 0.373 (-0.001 [-0.00, +0.00]) | 0.085 (+0.001 [+0.00, +0.00]) |
| keep_L2 | 0.851 (+0.001 [-0.00, +0.00]) | 0.369 (-0.005 [-0.01, -0.00]) | 0.085 (+0.000 [-0.00, +0.00]) |
| mean_L2 | 0.850 (+0.000 [+0.00, +0.00]) | 0.372 (-0.002 [-0.00, +0.00]) | 0.086 (+0.001 [+0.00, +0.00]) |
| keep_L3 | 0.850 (-0.000 [-0.00, +0.00]) | 0.370 (-0.003 [-0.01, +0.00]) | 0.085 (+0.001 [-0.00, +0.00]) |
| mean_L3 | 0.850 (-0.000 [-0.00, +0.00]) | 0.371 (-0.002 [-0.00, -0.00]) | 0.085 (+0.001 [-0.00, +0.00]) |
| keep_L4 | 0.850 (-0.000 [-0.00, +0.00]) | 0.371 (-0.002 [-0.01, +0.00]) | 0.086 (+0.001 [-0.00, +0.00]) |
| mean_L4 | 0.850 (-0.001 [-0.00, +0.00]) | 0.370 (-0.004 [-0.01, -0.00]) | 0.085 (+0.000 [-0.00, +0.00]) |

**hybrid**

| condition | 8 facts | 32 facts | 64 facts |
|---|---|---|---|
| none | 0.850 | 0.357 | 0.117 |
| keep_all | 0.628 (-0.221 [-0.23, -0.21]) | 0.203 (-0.154 [-0.16, -0.14]) | 0.108 (-0.008 [-0.01, -0.00]) |
| keep_L0 | 0.633 (-0.216 [-0.23, -0.21]) | 0.202 (-0.154 [-0.16, -0.14]) | 0.109 (-0.008 [-0.01, -0.00]) |
| mean_L0 | 0.642 (-0.208 [-0.22, -0.20]) | 0.218 (-0.139 [-0.15, -0.13]) | 0.117 (-0.000 [-0.00, +0.00]) |
| keep_L1 | 0.850 (+0.000 [-0.00, +0.00]) | 0.356 (-0.000 [-0.00, +0.00]) | 0.116 (-0.001 [-0.00, +0.00]) |
| mean_L1 | 0.850 (+0.001 [-0.00, +0.00]) | 0.357 (+0.000 [-0.00, +0.00]) | 0.118 (+0.001 [-0.00, +0.00]) |
| keep_L3 | 0.849 (-0.001 [-0.00, +0.00]) | 0.356 (-0.001 [-0.00, +0.00]) | 0.116 (-0.001 [-0.00, +0.00]) |
| mean_L3 | 0.850 (-0.000 [-0.00, +0.00]) | 0.355 (-0.001 [-0.00, +0.00]) | 0.117 (-0.000 [-0.00, +0.00]) |

## 5. Prefill cost (one forward pass, batch 1; mostly kernel quality)

_`cost_profile.csv` not synced yet._

## 5b. Decode cost (one token at a time, batch 1)

_`decode_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
