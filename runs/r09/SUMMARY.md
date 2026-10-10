# iota run `r09`

- **updated:** 2026-10-10 20:52 UTC (last stage: `eval 4`)
- **code:** `ccb5ac0` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | 14000 | 0.904 | 0.809 | 1.000 | 0.600 | 0.0015 | ok |
| gated_linear | 14500 | 0.907 | 0.815 | 1.000 | 0.634 | 0.0015 | ok |
| hybrid | 5000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.00075 | ok |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `logs/train.log` and the `*_sweep.json` files.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| hybrid | 1.000 | 1.000 | 1.000 | 1.000 | pass |
| transformer | 0.809 | 0.380 | 1.000 | 1.000 | pass |
| gated_linear | 0.800 | 0.418 | 1.000 | 1.000 | pass |

n=500 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| n_bindings | true tokens | transformer | gated_linear | hybrid |
|---:|---:|---|---|---|
| 2 | 256 | 0.997 [0.99–1.00] (0.99) | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) |
| 4 | 256 | 0.991 [0.99–0.99] (0.96) | 0.997 [0.99–1.00] (0.99) | 1.000 [1.00–1.00] (1.00) |
| 8 | 256 | 0.975 [0.97–0.98] (0.82) | 0.991 [0.99–0.99] (0.93) | 1.000 [1.00–1.00] (1.00) |
| 16 | 256 | 0.945 [0.94–0.95] (0.43) | 0.963 [0.96–0.97] (0.57) | 1.000 [1.00–1.00] (1.00) |
| 32 | 289 | 0.842 [0.84–0.85] (0.05) | 0.858 [0.85–0.86] (0.09) | 1.000 [1.00–1.00] (1.00) |
| 64 | 386 | 0.620 [0.61–0.63] (0.00) | 0.512 [0.50–0.52] (0.00) | 0.986 [0.98–0.99] (0.79) |
| 128 | 773 | 0.347 [0.34–0.35] (0.00) | 0.222 [0.22–0.23] (0.00) | 0.979 [0.98–0.98] (0.71) |

## 3. Pass 2 — length generalisation

_`pass2_length.csv` not synced yet._

## 4. Pass 3 — state_track control

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| seq_len | true tokens | transformer | gated_linear | hybrid |
|---:|---:|---|---|---|
| 128 | 136 | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) | 0.994 [0.99–1.00] (0.99) |
| 256 | 264 | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) | 0.998 [0.99–1.00] (1.00) |
| 512 | 520 | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) | 0.999 [1.00–1.00] (1.00) |
| 1024 | 1032 | 0.955 [0.94–0.97] (0.95) | 1.000 [1.00–1.00] (1.00) | 0.514 [0.48–0.54] (0.51) |
| 2048 | 2056 | 0.189 [0.17–0.21] (0.19) | 1.000 [1.00–1.00] (1.00) | 0.077 [0.06–0.09] (0.08) |
| 4096 | 4104 | 0.017 [0.01–0.03] (0.02) | 0.999 [1.00–1.00] (1.00) | 0.010 [0.00–0.02] (0.01) |
| 8192 | 8200 | 0.020 [0.01–0.03] (0.02) | 0.996 [0.99–1.00] (1.00) | 0.110 [0.09–0.13] (0.11) |

## 4b. Diagnostic — recall by key length (digits), same prompts as Pass 1

| n_bindings | key digits | transformer | gated_linear | hybrid |
|---:|---:|---:|---:|---:|
| 16 | 1 | 0.948 | 0.961 | 1.000 |
| 16 | 2 | 0.952 | 0.966 | 1.000 |
| 16 | 3 | 0.920 | 0.957 | 1.000 |
| 32 | 1 | 0.874 | 0.866 | 0.998 |
| 32 | 2 | 0.848 | 0.860 | 1.000 |
| 32 | 3 | 0.808 | 0.851 | 1.000 |
| 64 | 1 | 0.674 | 0.507 | 0.978 |
| 64 | 2 | 0.638 | 0.520 | 0.986 |
| 64 | 3 | 0.545 | 0.487 | 0.989 |

If one model's errors pile up on 3-digit keys, its drop with load is partly key resolution, not memory capacity.

## 4f. Decay gates (Pass 8): what γ does on real prompts

Mean γ ± std over tokens × heads, by token role. `retention` = mean log10 of the share of a fact value's write still in the state at the first question. A frozen gate has std ≈ 0; selective forgetting shows as γ(distractor) < γ(fact value).

| model | bindings | layer | fact value | distractor | query | answer | retention (log10) | gate bias / ‖W‖ |
|---|---:|---|---|---|---|---|---:|---|
| gated_linear | 8 | backbone.blocks.0.mixer | 0.268 ± 0.218 | 0.251 ± 0.221 | 0.209 ± 0.211 | 0.284 ± 0.218 | -174.682 | -0.06 / 1.224 |
| gated_linear | 8 | backbone.blocks.1.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.0 | +0.09 / 3.960 |
| gated_linear | 8 | backbone.blocks.2.mixer | 0.577 ± 0.393 | 0.634 ± 0.245 | 0.700 ± 0.280 | 0.585 ± 0.378 | -55.119 | +0.17 / 2.600 |
| gated_linear | 8 | backbone.blocks.3.mixer | 0.961 ± 0.119 | 0.945 ± 0.068 | 0.995 ± 0.012 | 0.996 ± 0.011 | -6.581 | +0.17 / 2.388 |
| gated_linear | 8 | backbone.blocks.4.mixer | 0.997 ± 0.007 | 0.988 ± 0.022 | 0.999 ± 0.004 | 0.999 ± 0.003 | -1.125 | +0.26 / 2.875 |
| gated_linear | 32 | backbone.blocks.0.mixer | 0.266 ± 0.217 | 0.252 ± 0.223 | 0.209 ± 0.213 | 0.283 ± 0.217 | -155.29 | -0.06 / 1.224 |
| gated_linear | 32 | backbone.blocks.1.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.0 | +0.09 / 3.960 |
| gated_linear | 32 | backbone.blocks.2.mixer | 0.580 ± 0.365 | 0.622 ± 0.214 | 0.704 ± 0.241 | 0.594 ± 0.327 | -46.447 | +0.17 / 2.600 |
| gated_linear | 32 | backbone.blocks.3.mixer | 0.967 ± 0.109 | 0.945 ± 0.063 | 0.997 ± 0.007 | 0.998 ± 0.005 | -6.872 | +0.17 / 2.388 |
| gated_linear | 32 | backbone.blocks.4.mixer | 0.997 ± 0.008 | 0.989 ± 0.019 | 0.999 ± 0.004 | 0.999 ± 0.004 | -0.658 | +0.26 / 2.875 |
| hybrid | 8 | backbone.blocks.0.mixer | 0.162 ± 0.177 | 0.152 ± 0.058 | 0.101 ± 0.142 | 0.163 ± 0.179 | -199.649 | -0.11 / 0.955 |
| hybrid | 8 | backbone.blocks.1.mixer | 0.433 ± 0.363 | 0.996 ± 0.041 | 0.991 ± 0.016 | 0.966 ± 0.048 | -12.322 | +0.08 / 1.431 |
| hybrid | 8 | backbone.blocks.3.mixer | 0.790 ± 0.135 | 0.647 ± 0.109 | 0.813 ± 0.149 | 0.824 ± 0.154 | -42.955 | +0.09 / 1.026 |
| hybrid | 32 | backbone.blocks.0.mixer | 0.161 ± 0.176 | 0.152 ± 0.059 | 0.101 ± 0.142 | 0.163 ± 0.179 | -181.413 | -0.11 / 0.955 |
| hybrid | 32 | backbone.blocks.1.mixer | 0.423 ± 0.359 | 0.991 ± 0.060 | 0.991 ± 0.017 | 0.965 ± 0.050 | -52.377 | +0.08 / 1.431 |
| hybrid | 32 | backbone.blocks.3.mixer | 0.791 ± 0.133 | 0.649 ± 0.103 | 0.819 ± 0.144 | 0.820 ± 0.153 | -28.015 | +0.09 / 1.026 |

## 4g. Gate-clamp intervention (Pass 9): which layers' forgetting does recall need?

Per-query recall with one GLA layer's gate replaced by a constant: `keep` = σ(6) ≈ 0.9975 (no forgetting), `mean` = the head's own average γ (same amount of forgetting, no selectivity). Δ vs the unclamped model on the same prompts, paired 95% CI.

**gated_linear**

| condition | 8 facts | 32 facts | 64 facts |
|---|---|---|---|
| none | 0.993 | 0.859 | 0.507 |
| keep_all | 0.014 (-0.979 [-0.98, -0.97]) | 0.012 (-0.848 [-0.86, -0.84]) | 0.009 (-0.498 [-0.51, -0.49]) |
| keep_L0 | 0.013 (-0.979 [-0.98, -0.97]) | 0.011 (-0.848 [-0.86, -0.84]) | 0.010 (-0.497 [-0.51, -0.49]) |
| mean_L0 | 0.841 (-0.152 [-0.16, -0.14]) | 0.486 (-0.373 [-0.38, -0.36]) | 0.245 (-0.262 [-0.27, -0.25]) |
| keep_L1 | 0.994 (+0.001 [-0.00, +0.00]) | 0.841 (-0.018 [-0.02, -0.01]) | 0.465 (-0.042 [-0.05, -0.03]) |
| mean_L1 | 0.993 (+0.000 [+0.00, +0.00]) | 0.859 (+0.000 [+0.00, +0.00]) | 0.507 (+0.000 [+0.00, +0.00]) |
| keep_L2 | 0.992 (-0.001 [-0.00, +0.00]) | 0.850 (-0.009 [-0.01, -0.00]) | 0.496 (-0.010 [-0.02, -0.01]) |
| mean_L2 | 0.993 (+0.000 [+0.00, +0.00]) | 0.859 (-0.001 [-0.00, +0.00]) | 0.506 (-0.001 [-0.00, +0.00]) |
| keep_L3 | 0.990 (-0.003 [-0.01, -0.00]) | 0.853 (-0.006 [-0.01, -0.00]) | 0.539 (+0.032 [+0.02, +0.04]) |
| mean_L3 | 0.992 (-0.001 [-0.00, +0.00]) | 0.826 (-0.034 [-0.04, -0.03]) | 0.493 (-0.014 [-0.02, -0.01]) |
| keep_L4 | 0.993 (-0.000 [-0.00, +0.00]) | 0.859 (+0.000 [-0.00, +0.00]) | 0.502 (-0.005 [-0.01, -0.00]) |
| mean_L4 | 0.993 (+0.000 [+0.00, +0.00]) | 0.859 (+0.000 [+0.00, +0.00]) | 0.506 (-0.000 [-0.00, +0.00]) |

**hybrid**

| condition | 8 facts | 32 facts | 64 facts |
|---|---|---|---|
| none | 1.000 | 1.000 | 0.987 |
| keep_all | 0.016 (-0.984 [-0.99, -0.98]) | 0.011 (-0.990 [-0.99, -0.99]) | 0.011 (-0.976 [-0.98, -0.97]) |
| keep_L0 | 0.017 (-0.984 [-0.99, -0.98]) | 0.012 (-0.988 [-0.99, -0.99]) | 0.011 (-0.976 [-0.98, -0.97]) |
| mean_L0 | 1.000 (+0.000 [+0.00, +0.00]) | 1.000 (-0.000 [-0.00, +0.00]) | 0.985 (-0.002 [-0.00, -0.00]) |
| keep_L1 | 0.489 (-0.510 [-0.52, -0.50]) | 0.189 (-0.811 [-0.82, -0.80]) | 0.109 (-0.878 [-0.88, -0.87]) |
| mean_L1 | 0.659 (-0.341 [-0.35, -0.33]) | 0.731 (-0.269 [-0.28, -0.26]) | 0.847 (-0.140 [-0.15, -0.13]) |
| keep_L3 | 1.000 (+0.000 [+0.00, +0.00]) | 1.000 (+0.000 [+0.00, +0.00]) | 0.987 (-0.000 [-0.00, +0.00]) |
| mean_L3 | 1.000 (+0.000 [+0.00, +0.00]) | 1.000 (+0.000 [+0.00, +0.00]) | 0.987 (-0.000 [-0.00, +0.00]) |

## 5. Prefill cost (one forward pass, batch 1; mostly kernel quality)

_`cost_profile.csv` not synced yet._

## 5b. Decode cost (one token at a time, batch 1)

_`decode_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
