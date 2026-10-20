# iota run `r35`

- **updated:** 2026-10-20 19:12 UTC (last stage: `eval 8,9,10`)
- **code:** `4d49fdb` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | – | – | – | – | – | – | not trained yet |
| gated_linear | 14000 | 0.776 | 0.551 | 1.000 | 0.468 | 0.00075 | ok |
| hybrid | – | – | – | – | – | – | not trained yet |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `*_sweep_history.json` (and `logs/train.log`); `*_sweep.json` stops at the best checkpoint.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| gated_linear | 0.521 | 0.176 | 1.000 | 1.000 | pass |

n=500 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| n_bindings | true tokens | gated_linear |
|---:|---:|---|
| 2 | 256 | 0.994 [0.99–1.00] (0.99) |
| 4 | 256 | 0.971 [0.97–0.98] (0.89) |
| 8 | 256 | 0.914 [0.91–0.92] (0.51) |
| 16 | 256 | 0.788 [0.78–0.79] (0.02) |
| 32 | 289 | 0.549 [0.54–0.56] (0.00) |
| 64 | 386 | 0.127 [0.12–0.13] (0.00) |
| 128 | 773 | 0.056 [0.05–0.06] (0.00) |

## 3. Pass 2 — length generalisation

_`pass2_length.csv` not synced yet._

## 4. Pass 3 — state_track control

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| seq_len | true tokens | gated_linear |
|---:|---:|---|
| 128 | 136 | 0.999 [1.00–1.00] (1.00) |
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
| gated_linear | 8 | backbone.blocks.0.mixer | 0.453 ± 0.318 | 0.783 ± 0.082 | 0.443 ± 0.383 | 0.456 ± 0.322 | -35.022 | +1.85 / 1.215 |
| gated_linear | 8 | backbone.blocks.1.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.0 | +2.05 / 2.374 |
| gated_linear | 8 | backbone.blocks.2.mixer | 0.981 ± 0.025 | 0.969 ± 0.039 | 0.986 ± 0.017 | 0.982 ± 0.022 | -3.149 | +2.01 / 1.577 |
| gated_linear | 8 | backbone.blocks.3.mixer | 0.995 ± 0.009 | 0.995 ± 0.007 | 0.996 ± 0.009 | 0.996 ± 0.007 | -0.5 | +2.05 / 1.848 |
| gated_linear | 8 | backbone.blocks.4.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.002 | +2.20 / 2.911 |
| gated_linear | 32 | backbone.blocks.0.mixer | 0.453 ± 0.317 | 0.781 ± 0.086 | 0.441 ± 0.384 | 0.457 ± 0.322 | -64.612 | +1.85 / 1.215 |
| gated_linear | 32 | backbone.blocks.1.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.0 | +2.05 / 2.374 |
| gated_linear | 32 | backbone.blocks.2.mixer | 0.983 ± 0.022 | 0.973 ± 0.036 | 0.989 ± 0.013 | 0.987 ± 0.016 | -2.121 | +2.01 / 1.577 |
| gated_linear | 32 | backbone.blocks.3.mixer | 0.996 ± 0.008 | 0.997 ± 0.005 | 0.997 ± 0.005 | 0.997 ± 0.005 | -0.306 | +2.05 / 1.848 |
| gated_linear | 32 | backbone.blocks.4.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.004 | +2.20 / 2.911 |

## 4g. Gate-clamp intervention (Pass 9): which layers' forgetting does recall need?

Per-query recall with one GLA layer's gate replaced by a constant: `keep` = σ(6) ≈ 0.9975 (no forgetting), `mean` = the head's own average γ (same amount of forgetting, no selectivity). Δ vs the unclamped model on the same prompts, paired 95% CI.

**gated_linear**

| condition | 8 facts | 32 facts | 64 facts |
|---|---|---|---|
| none | 0.917 | 0.551 | 0.128 |
| keep_all | 0.013 (-0.904 [-0.91, -0.89]) | 0.011 (-0.539 [-0.55, -0.53]) | 0.008 (-0.120 [-0.13, -0.11]) |
| keep_L0 | 0.015 (-0.902 [-0.91, -0.89]) | 0.011 (-0.540 [-0.55, -0.53]) | 0.009 (-0.119 [-0.13, -0.11]) |
| mean_L0 | 0.024 (-0.894 [-0.90, -0.88]) | 0.030 (-0.521 [-0.53, -0.51]) | 0.031 (-0.097 [-0.11, -0.09]) |
| keep_L1 | 0.831 (-0.086 [-0.10, -0.08]) | 0.355 (-0.195 [-0.21, -0.18]) | 0.090 (-0.038 [-0.05, -0.03]) |
| mean_L1 | 0.903 (-0.014 [-0.02, -0.01]) | 0.525 (-0.026 [-0.03, -0.02]) | 0.123 (-0.004 [-0.01, -0.00]) |
| keep_L2 | 0.915 (-0.003 [-0.01, +0.00]) | 0.550 (-0.001 [-0.00, +0.00]) | 0.128 (+0.000 [-0.00, +0.00]) |
| mean_L2 | 0.916 (-0.001 [-0.00, +0.00]) | 0.553 (+0.003 [+0.00, +0.01]) | 0.129 (+0.001 [-0.00, +0.00]) |
| keep_L3 | 0.914 (-0.003 [-0.01, +0.00]) | 0.550 (-0.001 [-0.00, +0.00]) | 0.132 (+0.004 [+0.00, +0.01]) |
| mean_L3 | 0.918 (+0.001 [-0.00, +0.00]) | 0.552 (+0.001 [-0.00, +0.00]) | 0.128 (+0.000 [-0.00, +0.00]) |
| keep_L4 | 0.914 (-0.004 [-0.01, +0.00]) | 0.546 (-0.004 [-0.01, -0.00]) | 0.129 (+0.001 [-0.00, +0.00]) |
| mean_L4 | 0.918 (+0.000 [+0.00, +0.00]) | 0.551 (+0.000 [-0.00, +0.00]) | 0.128 (+0.000 [-0.00, +0.00]) |

## 4h. Gate replay (Pass 10): does the timing of forgetting matter?

One layer's recorded gates over the context span are replayed unchanged, shuffled in time (same values, same product), replaced by their geometric mean (same product, no timing) or their arithmetic mean (forgets less). Per-query recall, Δ vs baseline [paired 95% CI].

**gated_linear**

| facts | layer | baseline | replay | shuffle (range) | geomean | arith |
|---:|---|---:|---|---|---|---|
| 8 | L0 | 0.907 | 0.907 (+0.000 [0.00, 0.00]) | 0.027 (0.03–0.03) (-0.880 [-0.89, -0.87]) | 0.028 (-0.878 [-0.89, -0.87]) | 0.017 (-0.889 [-0.90, -0.88]) |
| 8 | L1 | 0.907 | 0.907 (+0.000 [0.00, 0.00]) | 0.881 (0.88–0.88) (-0.025 [-0.03, -0.02]) | 0.887 (-0.019 [-0.03, -0.01]) | 0.888 (-0.019 [-0.03, -0.01]) |
| 8 | L2 | 0.907 | 0.907 (+0.000 [0.00, 0.00]) | 0.906 (0.91–0.91) (-0.001 [-0.00, 0.00]) | 0.905 (-0.001 [-0.00, 0.00]) | 0.905 (-0.002 [-0.00, -0.00]) |
| 8 | L3 | 0.907 | 0.907 (+0.000 [0.00, 0.00]) | 0.908 (0.91–0.91) (+0.001 [-0.00, 0.00]) | 0.908 (+0.001 [-0.00, 0.00]) | 0.908 (+0.001 [-0.00, 0.00]) |
| 8 | L4 | 0.907 | 0.907 (+0.000 [0.00, 0.00]) | 0.907 (0.91–0.91) (+0.000 [0.00, 0.00]) | 0.907 (+0.000 [0.00, 0.00]) | 0.907 (+0.000 [0.00, 0.00]) |
| 32 | L0 | 0.548 | 0.548 (+0.000 [0.00, 0.00]) | 0.060 (0.06–0.06) (-0.488 [-0.50, -0.48]) | 0.170 (-0.378 [-0.39, -0.37]) | 0.024 (-0.524 [-0.54, -0.51]) |
| 32 | L1 | 0.548 | 0.548 (+0.000 [0.00, 0.00]) | 0.477 (0.47–0.48) (-0.071 [-0.08, -0.06]) | 0.512 (-0.036 [-0.04, -0.03]) | 0.518 (-0.030 [-0.04, -0.02]) |
| 32 | L2 | 0.548 | 0.548 (+0.000 [0.00, 0.00]) | 0.548 (0.55–0.55) (+0.001 [-0.00, 0.00]) | 0.549 (+0.001 [-0.00, 0.00]) | 0.549 (+0.001 [-0.00, 0.00]) |
| 32 | L3 | 0.548 | 0.548 (+0.000 [0.00, 0.00]) | 0.547 (0.55–0.55) (-0.000 [-0.00, 0.00]) | 0.548 (-0.000 [-0.00, 0.00]) | 0.547 (-0.000 [-0.00, 0.00]) |
| 32 | L4 | 0.548 | 0.548 (+0.000 [0.00, 0.00]) | 0.548 (0.55–0.55) (-0.000 [-0.00, 0.00]) | 0.548 (-0.000 [-0.00, 0.00]) | 0.548 (-0.000 [-0.00, 0.00]) |

## 5. Prefill cost (one forward pass, batch 1; mostly kernel quality)

_`cost_profile.csv` not synced yet._

## 5b. Decode cost (one token at a time, batch 1)

_`decode_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
