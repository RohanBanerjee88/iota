# iota run `r10`

- **updated:** 2026-10-10 04:47 UTC (last stage: `eval 8,9`)
- **code:** `9150b4e` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | – | – | – | – | – | – | not trained yet |
| gated_linear | 15000 | 0.863 | 0.726 | 1.000 | 0.538 | 0.0015 | ok |
| hybrid | – | – | – | – | – | – | not trained yet |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `logs/train.log` and the `*_sweep.json` files.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| gated_linear | 0.720 | 0.310 | 1.000 | 1.000 | pass |

n=500 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| n_bindings | true tokens | gated_linear |
|---:|---:|---|
| 2 | 256 | 0.998 [0.99–1.00] (0.99) |
| 4 | 256 | 0.985 [0.98–0.99] (0.94) |
| 8 | 256 | 0.971 [0.97–0.97] (0.80) |
| 16 | 256 | 0.912 [0.91–0.92] (0.25) |
| 32 | 289 | 0.752 [0.75–0.76] (0.01) |
| 64 | 386 | 0.419 [0.41–0.43] (0.00) |
| 128 | 773 | 0.184 [0.18–0.19] (0.00) |

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
| 2048 | 2056 | 0.970 [0.96–0.98] (0.97) |
| 4096 | 4104 | 0.298 [0.27–0.33] (0.30) |
| 8192 | 8200 | 0.036 [0.03–0.05] (0.04) |

## 4f. Decay gates (Pass 8): what γ does on real prompts

Mean γ ± std over tokens × heads, by token role. `retention` = mean log10 of the share of a fact value's write still in the state at the first question. A frozen gate has std ≈ 0; selective forgetting shows as γ(distractor) < γ(fact value).

| model | bindings | layer | fact value | distractor | query | answer | retention (log10) | gate bias / ‖W‖ |
|---|---:|---|---|---|---|---|---:|---|
| gated_linear | 8 | backbone.blocks.0.mixer | 0.065 ± 0.099 | 0.643 ± 0.377 | 0.123 ± 0.168 | 0.065 ± 0.098 | -94.798 | -0.08 / 1.684 |
| gated_linear | 8 | backbone.blocks.1.mixer | 0.999 ± 0.002 | 0.968 ± 0.063 | 1.000 ± 0.003 | 0.999 ± 0.003 | -3.206 | +0.30 / 3.202 |
| gated_linear | 8 | backbone.blocks.2.mixer | 0.606 ± 0.403 | 0.635 ± 0.352 | 0.784 ± 0.270 | 0.588 ± 0.417 | -65.668 | +0.11 / 2.266 |
| gated_linear | 8 | backbone.blocks.3.mixer | 0.560 ± 0.401 | 0.695 ± 0.245 | 0.906 ± 0.197 | 0.904 ± 0.179 | -50.647 | +0.09 / 2.502 |
| gated_linear | 8 | backbone.blocks.4.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.999 ± 0.007 | 1.000 ± 0.000 | -0.016 | +0.36 / 3.446 |
| gated_linear | 32 | backbone.blocks.0.mixer | 0.065 ± 0.099 | 0.645 ± 0.377 | 0.123 ± 0.165 | 0.065 ± 0.098 | -144.806 | -0.08 / 1.684 |
| gated_linear | 32 | backbone.blocks.1.mixer | 0.999 ± 0.002 | 0.964 ± 0.076 | 1.000 ± 0.002 | 0.999 ± 0.004 | -1.762 | +0.30 / 3.202 |
| gated_linear | 32 | backbone.blocks.2.mixer | 0.606 ± 0.404 | 0.642 ± 0.344 | 0.791 ± 0.268 | 0.599 ± 0.414 | -54.297 | +0.11 / 2.266 |
| gated_linear | 32 | backbone.blocks.3.mixer | 0.571 ± 0.400 | 0.748 ± 0.243 | 0.922 ± 0.179 | 0.922 ± 0.158 | -53.748 | +0.09 / 2.502 |
| gated_linear | 32 | backbone.blocks.4.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.003 | 1.000 ± 0.000 | -0.003 | +0.36 / 3.446 |

## 4g. Gate-clamp intervention (Pass 9): which layers' forgetting does recall need?

Per-query recall with one GLA layer's gate replaced by a constant: `keep` = σ(6) ≈ 0.9975 (no forgetting), `mean` = the head's own average γ (same amount of forgetting, no selectivity). Δ vs the unclamped model on the same prompts, paired 95% CI.

**gated_linear**

| condition | 8 facts | 32 facts | 64 facts |
|---|---|---|---|
| none | 0.972 | 0.755 | 0.420 |
| keep_all | 0.009 (-0.963 [-0.97, -0.96]) | 0.010 (-0.745 [-0.75, -0.74]) | 0.007 (-0.413 [-0.42, -0.40]) |
| keep_L0 | 0.011 (-0.961 [-0.97, -0.96]) | 0.008 (-0.747 [-0.76, -0.74]) | 0.008 (-0.413 [-0.42, -0.40]) |
| mean_L0 | 0.132 (-0.841 [-0.85, -0.83]) | 0.199 (-0.556 [-0.57, -0.54]) | 0.123 (-0.297 [-0.31, -0.29]) |
| keep_L1 | 0.966 (-0.006 [-0.01, -0.00]) | 0.719 (-0.036 [-0.04, -0.03]) | 0.397 (-0.023 [-0.03, -0.01]) |
| mean_L1 | 0.970 (-0.002 [-0.01, +0.00]) | 0.745 (-0.010 [-0.01, -0.01]) | 0.421 (+0.001 [-0.00, +0.00]) |
| keep_L2 | 0.940 (-0.033 [-0.04, -0.03]) | 0.662 (-0.093 [-0.10, -0.08]) | 0.339 (-0.081 [-0.09, -0.07]) |
| mean_L2 | 0.966 (-0.006 [-0.01, -0.00]) | 0.736 (-0.019 [-0.02, -0.01]) | 0.401 (-0.020 [-0.03, -0.01]) |
| keep_L3 | 0.961 (-0.011 [-0.02, -0.01]) | 0.716 (-0.039 [-0.04, -0.03]) | 0.381 (-0.040 [-0.05, -0.03]) |
| mean_L3 | 0.970 (-0.002 [-0.01, +0.00]) | 0.742 (-0.013 [-0.02, -0.01]) | 0.407 (-0.013 [-0.02, -0.01]) |
| keep_L4 | 0.972 (+0.000 [-0.00, +0.00]) | 0.746 (-0.009 [-0.01, -0.01]) | 0.401 (-0.019 [-0.03, -0.01]) |
| mean_L4 | 0.972 (-0.001 [-0.00, +0.00]) | 0.755 (-0.000 [-0.00, +0.00]) | 0.420 (-0.001 [-0.00, +0.00]) |

## 5. Prefill cost (one forward pass, batch 1; mostly kernel quality)

_`cost_profile.csv` not synced yet._

## 5b. Decode cost (one token at a time, batch 1)

_`decode_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
