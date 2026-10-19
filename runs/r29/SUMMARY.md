# iota run `r29`

- **updated:** 2026-10-19 03:53 UTC (last stage: `eval 8,9,10`)
- **code:** `36f3d6f` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | – | – | – | – | – | – | not trained yet |
| gated_linear | 15000 | 0.707 | 0.413 | 1.000 | 0.410 | 0.00075 | ok |
| hybrid | – | – | – | – | – | – | not trained yet |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `*_sweep_history.json` (and `logs/train.log`); `*_sweep.json` stops at the best checkpoint.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| gated_linear | 0.381 | 0.108 | 1.000 | 1.000 | pass |

n=500 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| n_bindings | true tokens | gated_linear |
|---:|---:|---|
| 2 | 256 | 0.971 [0.96–0.98] (0.94) |
| 4 | 256 | 0.913 [0.90–0.92] (0.70) |
| 8 | 256 | 0.830 [0.82–0.84] (0.23) |
| 16 | 256 | 0.684 [0.68–0.69] (0.00) |
| 32 | 289 | 0.380 [0.37–0.39] (0.00) |
| 64 | 386 | 0.165 [0.16–0.17] (0.00) |
| 128 | 773 | 0.069 [0.07–0.07] (0.00) |

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
| 4096 | 4104 | 0.998 [0.99–1.00] (1.00) |
| 8192 | 8200 | 0.854 [0.83–0.88] (0.85) |

## 4f. Decay gates (Pass 8): what γ does on real prompts

Mean γ ± std over tokens × heads, by token role. `retention` = mean log10 of the share of a fact value's write still in the state at the first question. A frozen gate has std ≈ 0; selective forgetting shows as γ(distractor) < γ(fact value).

| model | bindings | layer | fact value | distractor | query | answer | retention (log10) | gate bias / ‖W‖ |
|---|---:|---|---|---|---|---|---:|---|
| gated_linear | 8 | backbone.blocks.0.mixer | 0.615 ± 0.180 | 0.481 ± 0.131 | 0.421 ± 0.311 | 0.624 ± 0.182 | -78.258 | +1.80 / 1.235 |
| gated_linear | 8 | backbone.blocks.1.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.0 | +1.98 / 2.479 |
| gated_linear | 8 | backbone.blocks.2.mixer | 0.983 ± 0.051 | 0.999 ± 0.001 | 0.996 ± 0.011 | 0.998 ± 0.002 | -0.127 | +2.05 / 1.753 |
| gated_linear | 8 | backbone.blocks.3.mixer | 0.927 ± 0.057 | 0.711 ± 0.072 | 0.984 ± 0.030 | 0.991 ± 0.009 | -32.073 | +1.97 / 1.185 |
| gated_linear | 8 | backbone.blocks.4.mixer | 0.964 ± 0.140 | 1.000 ± 0.000 | 0.998 ± 0.005 | 0.998 ± 0.003 | -0.046 | +2.25 / 3.002 |
| gated_linear | 32 | backbone.blocks.0.mixer | 0.614 ± 0.180 | 0.481 ± 0.132 | 0.420 ± 0.313 | 0.623 ± 0.183 | -70.144 | +1.80 / 1.235 |
| gated_linear | 32 | backbone.blocks.1.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.001 | +1.98 / 2.479 |
| gated_linear | 32 | backbone.blocks.2.mixer | 0.995 ± 0.027 | 0.999 ± 0.001 | 0.998 ± 0.005 | 0.999 ± 0.001 | -0.135 | +2.05 / 1.753 |
| gated_linear | 32 | backbone.blocks.3.mixer | 0.963 ± 0.040 | 0.845 ± 0.050 | 0.994 ± 0.012 | 0.995 ± 0.004 | -8.438 | +1.97 / 1.185 |
| gated_linear | 32 | backbone.blocks.4.mixer | 0.991 ± 0.074 | 1.000 ± 0.000 | 0.999 ± 0.001 | 0.999 ± 0.002 | -0.036 | +2.25 / 3.002 |

## 4g. Gate-clamp intervention (Pass 9): which layers' forgetting does recall need?

Per-query recall with one GLA layer's gate replaced by a constant: `keep` = σ(6) ≈ 0.9975 (no forgetting), `mean` = the head's own average γ (same amount of forgetting, no selectivity). Δ vs the unclamped model on the same prompts, paired 95% CI.

**gated_linear**

| condition | 8 facts | 32 facts | 64 facts |
|---|---|---|---|
| none | 0.830 | 0.376 | 0.172 |
| keep_all | 0.009 (-0.821 [-0.83, -0.81]) | 0.010 (-0.366 [-0.38, -0.36]) | 0.009 (-0.162 [-0.17, -0.15]) |
| keep_L0 | 0.010 (-0.820 [-0.83, -0.81]) | 0.011 (-0.366 [-0.38, -0.36]) | 0.010 (-0.162 [-0.17, -0.15]) |
| mean_L0 | 0.720 (-0.111 [-0.12, -0.10]) | 0.306 (-0.070 [-0.08, -0.06]) | 0.142 (-0.030 [-0.04, -0.02]) |
| keep_L1 | 0.808 (-0.022 [-0.03, -0.01]) | 0.353 (-0.023 [-0.03, -0.02]) | 0.163 (-0.008 [-0.02, +0.00]) |
| mean_L1 | 0.830 (+0.000 [+0.00, +0.00]) | 0.377 (+0.001 [+0.00, +0.00]) | 0.172 (+0.000 [-0.00, +0.00]) |
| keep_L2 | 0.820 (-0.010 [-0.02, -0.00]) | 0.367 (-0.009 [-0.02, -0.00]) | 0.166 (-0.006 [-0.01, -0.00]) |
| mean_L2 | 0.828 (-0.002 [-0.01, +0.00]) | 0.376 (-0.000 [-0.00, +0.00]) | 0.170 (-0.002 [-0.00, +0.00]) |
| keep_L3 | 0.742 (-0.088 [-0.10, -0.08]) | 0.340 (-0.036 [-0.04, -0.03]) | 0.175 (+0.003 [-0.00, +0.01]) |
| mean_L3 | 0.728 (-0.102 [-0.11, -0.09]) | 0.355 (-0.021 [-0.03, -0.01]) | 0.169 (-0.003 [-0.01, +0.00]) |
| keep_L4 | 0.826 (-0.004 [-0.01, +0.00]) | 0.375 (-0.001 [-0.00, +0.00]) | 0.173 (+0.002 [-0.00, +0.00]) |
| mean_L4 | 0.818 (-0.012 [-0.02, -0.01]) | 0.371 (-0.005 [-0.01, -0.00]) | 0.172 (+0.000 [-0.00, +0.00]) |

## 4h. Gate replay (Pass 10): does the timing of forgetting matter?

One layer's recorded gates over the context span are replayed unchanged, shuffled in time (same values, same product), replaced by their geometric mean (same product, no timing) or their arithmetic mean (forgets less). Per-query recall, Δ vs baseline [paired 95% CI].

**gated_linear**

| facts | layer | baseline | replay | shuffle (range) | geomean | arith |
|---:|---|---:|---|---|---|---|
| 8 | L0 | 0.841 | 0.841 (+0.000 [0.00, 0.00]) | 0.745 (0.74–0.75) (-0.097 [-0.11, -0.09]) | 0.787 (-0.054 [-0.06, -0.04]) | 0.763 (-0.078 [-0.09, -0.07]) |
| 8 | L1 | 0.841 | 0.841 (+0.000 [0.00, 0.00]) | 0.841 (0.84–0.84) (-0.000 [-0.00, 0.00]) | 0.841 (-0.000 [-0.00, 0.00]) | 0.841 (-0.000 [-0.00, 0.00]) |
| 8 | L2 | 0.841 | 0.841 (+0.000 [0.00, 0.00]) | 0.836 (0.84–0.84) (-0.005 [-0.01, -0.00]) | 0.836 (-0.005 [-0.01, -0.00]) | 0.837 (-0.004 [-0.01, 0.00]) |
| 8 | L3 | 0.841 | 0.841 (+0.000 [0.00, 0.00]) | 0.839 (0.84–0.84) (-0.003 [-0.01, 0.00]) | 0.839 (-0.003 [-0.01, 0.00]) | 0.839 (-0.003 [-0.01, -0.00]) |
| 8 | L4 | 0.841 | 0.841 (+0.000 [0.00, 0.00]) | 0.808 (0.81–0.81) (-0.033 [-0.04, -0.03]) | 0.807 (-0.034 [-0.04, -0.03]) | 0.826 (-0.015 [-0.02, -0.01]) |
| 32 | L0 | 0.372 | 0.372 (+0.000 [0.00, 0.00]) | 0.304 (0.30–0.31) (-0.068 [-0.08, -0.06]) | 0.346 (-0.025 [-0.03, -0.02]) | 0.324 (-0.047 [-0.06, -0.04]) |
| 32 | L1 | 0.372 | 0.372 (+0.000 [0.00, 0.00]) | 0.371 (0.37–0.37) (-0.000 [-0.00, 0.00]) | 0.371 (-0.000 [-0.00, 0.00]) | 0.371 (-0.000 [-0.00, 0.00]) |
| 32 | L2 | 0.372 | 0.372 (+0.000 [0.00, 0.00]) | 0.368 (0.37–0.37) (-0.004 [-0.01, 0.00]) | 0.369 (-0.003 [-0.01, 0.00]) | 0.369 (-0.002 [-0.01, 0.00]) |
| 32 | L3 | 0.372 | 0.372 (+0.000 [0.00, 0.00]) | 0.367 (0.37–0.37) (-0.004 [-0.01, 0.00]) | 0.367 (-0.004 [-0.01, 0.00]) | 0.368 (-0.004 [-0.01, 0.00]) |
| 32 | L4 | 0.372 | 0.372 (+0.000 [0.00, 0.00]) | 0.358 (0.36–0.36) (-0.014 [-0.02, -0.01]) | 0.354 (-0.018 [-0.02, -0.01]) | 0.367 (-0.005 [-0.01, 0.00]) |

## 5. Prefill cost (one forward pass, batch 1; mostly kernel quality)

_`cost_profile.csv` not synced yet._

## 5b. Decode cost (one token at a time, batch 1)

_`decode_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
