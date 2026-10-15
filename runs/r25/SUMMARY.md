# iota run `r25`

- **updated:** 2026-10-15 21:21 UTC (last stage: `eval 8,9,10`)
- **code:** `3fc1484` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | – | – | – | – | – | – | not trained yet |
| gated_linear | 14500 | 0.946 | 0.891 | 1.000 | 0.704 | 0.00075 | ok |
| hybrid | – | – | – | – | – | – | not trained yet |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `*_sweep_history.json` (and `logs/train.log`); `*_sweep.json` stops at the best checkpoint.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| gated_linear | 0.894 | 0.570 | 1.000 | 1.000 | pass |

n=500 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| n_bindings | true tokens | gated_linear |
|---:|---:|---|
| 2 | 256 | 1.000 [1.00–1.00] (1.00) |
| 4 | 256 | 1.000 [1.00–1.00] (1.00) |
| 8 | 256 | 0.995 [0.99–1.00] (0.96) |
| 16 | 256 | 0.984 [0.98–0.99] (0.78) |
| 32 | 289 | 0.929 [0.92–0.93] (0.33) |
| 64 | 386 | 0.708 [0.70–0.72] (0.01) |
| 128 | 773 | 0.448 [0.44–0.46] (0.00) |

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
| 8192 | 8200 | 0.963 [0.95–0.97] (0.96) |

## 4f. Decay gates (Pass 8): what γ does on real prompts

Mean γ ± std over tokens × heads, by token role. `retention` = mean log10 of the share of a fact value's write still in the state at the first question. A frozen gate has std ≈ 0; selective forgetting shows as γ(distractor) < γ(fact value).

| model | bindings | layer | fact value | distractor | query | answer | retention (log10) | gate bias / ‖W‖ |
|---|---:|---|---|---|---|---|---:|---|
| gated_linear | 8 | backbone.blocks.0.mixer | 0.645 ± 0.254 | 0.490 ± 0.305 | 0.428 ± 0.349 | 0.659 ± 0.244 | -104.88 | +1.83 / 1.228 |
| gated_linear | 8 | backbone.blocks.1.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.0 | +1.97 / 2.053 |
| gated_linear | 8 | backbone.blocks.2.mixer | 0.996 ± 0.006 | 0.994 ± 0.016 | 0.996 ± 0.005 | 0.996 ± 0.004 | -0.799 | +1.99 / 1.300 |
| gated_linear | 8 | backbone.blocks.3.mixer | 0.985 ± 0.023 | 0.978 ± 0.037 | 0.988 ± 0.021 | 0.993 ± 0.008 | -2.382 | +2.05 / 1.979 |
| gated_linear | 8 | backbone.blocks.4.mixer | 0.960 ± 0.038 | 0.655 ± 0.069 | 0.991 ± 0.018 | 0.994 ± 0.005 | -39.908 | +1.95 / 1.129 |
| gated_linear | 32 | backbone.blocks.0.mixer | 0.644 ± 0.255 | 0.490 ± 0.305 | 0.429 ± 0.348 | 0.659 ± 0.244 | -94.141 | +1.83 / 1.228 |
| gated_linear | 32 | backbone.blocks.1.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.0 | +1.97 / 2.053 |
| gated_linear | 32 | backbone.blocks.2.mixer | 0.996 ± 0.007 | 0.993 ± 0.021 | 0.996 ± 0.005 | 0.997 ± 0.003 | -1.32 | +1.99 / 1.300 |
| gated_linear | 32 | backbone.blocks.3.mixer | 0.987 ± 0.021 | 0.985 ± 0.026 | 0.992 ± 0.014 | 0.995 ± 0.005 | -1.694 | +2.05 / 1.979 |
| gated_linear | 32 | backbone.blocks.4.mixer | 0.970 ± 0.031 | 0.823 ± 0.050 | 0.995 ± 0.008 | 0.996 ± 0.003 | -12.202 | +1.95 / 1.129 |

## 4g. Gate-clamp intervention (Pass 9): which layers' forgetting does recall need?

Per-query recall with one GLA layer's gate replaced by a constant: `keep` = σ(6) ≈ 0.9975 (no forgetting), `mean` = the head's own average γ (same amount of forgetting, no selectivity). Δ vs the unclamped model on the same prompts, paired 95% CI.

**gated_linear**

| condition | 8 facts | 32 facts | 64 facts |
|---|---|---|---|
| none | 0.996 | 0.929 | 0.716 |
| keep_all | 0.020 (-0.976 [-0.98, -0.97]) | 0.013 (-0.916 [-0.92, -0.91]) | 0.012 (-0.704 [-0.71, -0.69]) |
| keep_L0 | 0.018 (-0.977 [-0.98, -0.97]) | 0.013 (-0.916 [-0.92, -0.91]) | 0.013 (-0.703 [-0.71, -0.69]) |
| mean_L0 | 0.820 (-0.175 [-0.19, -0.16]) | 0.590 (-0.339 [-0.35, -0.33]) | 0.383 (-0.333 [-0.34, -0.32]) |
| keep_L1 | 0.995 (-0.001 [-0.00, +0.00]) | 0.910 (-0.019 [-0.02, -0.01]) | 0.629 (-0.087 [-0.10, -0.08]) |
| mean_L1 | 0.996 (+0.000 [+0.00, +0.00]) | 0.929 (+0.000 [+0.00, +0.00]) | 0.716 (+0.000 [-0.00, +0.00]) |
| keep_L2 | 0.995 (-0.001 [-0.00, +0.00]) | 0.927 (-0.002 [-0.00, -0.00]) | 0.717 (+0.002 [-0.00, +0.01]) |
| mean_L2 | 0.996 (+0.000 [-0.00, +0.00]) | 0.929 (+0.000 [-0.00, +0.00]) | 0.715 (-0.000 [-0.00, +0.00]) |
| keep_L3 | 0.995 (-0.001 [-0.00, +0.00]) | 0.929 (-0.000 [-0.00, +0.00]) | 0.714 (-0.002 [-0.00, +0.00]) |
| mean_L3 | 0.996 (+0.000 [-0.00, +0.00]) | 0.929 (+0.001 [-0.00, +0.00]) | 0.716 (+0.000 [-0.00, +0.00]) |
| keep_L4 | 0.994 (-0.002 [-0.00, -0.00]) | 0.920 (-0.009 [-0.01, -0.01]) | 0.718 (+0.002 [-0.00, +0.01]) |
| mean_L4 | 0.993 (-0.003 [-0.01, -0.00]) | 0.914 (-0.015 [-0.02, -0.01]) | 0.705 (-0.010 [-0.02, -0.00]) |

## 4h. Gate replay (Pass 10): does the timing of forgetting matter?

One layer's recorded gates over the context span are replayed unchanged, shuffled in time (same values, same product), replaced by their geometric mean (same product, no timing) or their arithmetic mean (forgets less). Per-query recall, Δ vs baseline [paired 95% CI].

**gated_linear**

| facts | layer | baseline | replay | shuffle (range) | geomean | arith |
|---:|---|---:|---|---|---|---|
| 8 | L0 | 0.997 | 0.997 (+0.000 [0.00, 0.00]) | 0.816 (0.81–0.82) (-0.182 [-0.19, -0.17]) | 0.925 (-0.072 [-0.08, -0.06]) | 0.884 (-0.113 [-0.12, -0.10]) |
| 8 | L1 | 0.997 | 0.997 (+0.000 [0.00, 0.00]) | 0.997 (1.00–1.00) (+0.000 [0.00, 0.00]) | 0.997 (-0.000 [-0.00, 0.00]) | 0.997 (-0.000 [-0.00, 0.00]) |
| 8 | L2 | 0.997 | 0.997 (+0.000 [0.00, 0.00]) | 0.997 (1.00–1.00) (-0.000 [-0.00, 0.00]) | 0.997 (-0.000 [-0.00, 0.00]) | 0.997 (-0.000 [-0.00, 0.00]) |
| 8 | L3 | 0.997 | 0.997 (+0.000 [0.00, 0.00]) | 0.997 (1.00–1.00) (-0.001 [-0.00, 0.00]) | 0.997 (-0.001 [-0.00, 0.00]) | 0.997 (-0.001 [-0.00, 0.00]) |
| 8 | L4 | 0.997 | 0.997 (+0.000 [0.00, 0.00]) | 0.997 (1.00–1.00) (-0.000 [-0.00, 0.00]) | 0.997 (-0.000 [-0.00, 0.00]) | 0.997 (-0.000 [-0.00, 0.00]) |
| 32 | L0 | 0.928 | 0.928 (+0.000 [0.00, 0.00]) | 0.484 (0.48–0.49) (-0.444 [-0.45, -0.43]) | 0.701 (-0.227 [-0.24, -0.22]) | 0.591 (-0.337 [-0.35, -0.33]) |
| 32 | L1 | 0.928 | 0.928 (+0.000 [0.00, 0.00]) | 0.928 (0.93–0.93) (+0.000 [0.00, 0.00]) | 0.928 (+0.000 [0.00, 0.00]) | 0.928 (+0.000 [0.00, 0.00]) |
| 32 | L2 | 0.928 | 0.928 (+0.000 [0.00, 0.00]) | 0.927 (0.93–0.93) (-0.001 [-0.00, 0.00]) | 0.928 (-0.000 [-0.00, 0.00]) | 0.928 (-0.000 [-0.00, 0.00]) |
| 32 | L3 | 0.928 | 0.928 (+0.000 [0.00, 0.00]) | 0.928 (0.93–0.93) (-0.000 [-0.00, 0.00]) | 0.927 (-0.000 [-0.00, 0.00]) | 0.927 (-0.001 [-0.00, 0.00]) |
| 32 | L4 | 0.928 | 0.928 (+0.000 [0.00, 0.00]) | 0.928 (0.93–0.93) (+0.000 [-0.00, 0.00]) | 0.928 (+0.000 [-0.00, 0.00]) | 0.928 (+0.000 [-0.00, 0.00]) |

## 5. Prefill cost (one forward pass, batch 1; mostly kernel quality)

_`cost_profile.csv` not synced yet._

## 5b. Decode cost (one token at a time, batch 1)

_`decode_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
