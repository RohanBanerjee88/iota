# iota run `r40`

- **updated:** 2026-10-26 20:27 UTC (last stage: `eval 8,9,10`)
- **code:** `c9bfe00` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | – | – | – | – | – | – | not trained yet |
| gated_linear | 14500 | 0.900 | 0.801 | 1.000 | 0.606 | 0.0015 | ok |
| hybrid | – | – | – | – | – | – | not trained yet |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `*_sweep_history.json` (and `logs/train.log`); `*_sweep.json` stops at the best checkpoint.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| gated_linear | 0.793 | 0.398 | 1.000 | 1.000 | pass |

n=500 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| n_bindings | true tokens | gated_linear |
|---:|---:|---|
| 2 | 256 | 0.999 [1.00–1.00] (1.00) |
| 4 | 256 | 0.995 [0.99–1.00] (0.98) |
| 8 | 256 | 0.987 [0.98–0.99] (0.91) |
| 16 | 256 | 0.961 [0.96–0.96] (0.54) |
| 32 | 289 | 0.849 [0.84–0.85] (0.08) |
| 64 | 386 | 0.491 [0.48–0.50] (0.00) |
| 128 | 773 | 0.214 [0.21–0.22] (0.00) |

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
| 4096 | 4104 | 0.961 [0.95–0.97] (0.96) |
| 8192 | 8200 | 0.625 [0.59–0.65] (0.62) |

## 4f. Decay gates (Pass 8): what γ does on real prompts

Mean γ ± std over tokens × heads, by token role. `retention` = mean log10 of the share of a fact value's write still in the state at the first question. A frozen gate has std ≈ 0; selective forgetting shows as γ(distractor) < γ(fact value).

| model | bindings | layer | fact value | distractor | query | answer | retention (log10) | gate bias / ‖W‖ |
|---|---:|---|---|---|---|---|---:|---|
| gated_linear | 8 | backbone.blocks.0.mixer | 0.246 ± 0.187 | 0.504 ± 0.219 | 0.301 ± 0.299 | 0.249 ± 0.186 | -84.817 | -0.05 / 1.290 |
| gated_linear | 8 | backbone.blocks.1.mixer | 0.755 ± 0.285 | 0.807 ± 0.287 | 0.828 ± 0.291 | 0.875 ± 0.208 | -41.792 | +0.05 / 1.623 |
| gated_linear | 8 | backbone.blocks.2.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.0 | +0.18 / 3.521 |
| gated_linear | 8 | backbone.blocks.3.mixer | 1.000 ± 0.002 | 1.000 ± 0.000 | 1.000 ± 0.001 | 1.000 ± 0.001 | -0.014 | +0.19 / 2.347 |
| gated_linear | 8 | backbone.blocks.4.mixer | 0.736 ± 0.292 | 0.686 ± 0.092 | 0.930 ± 0.171 | 0.860 ± 0.267 | -37.331 | +0.15 / 1.851 |
| gated_linear | 32 | backbone.blocks.0.mixer | 0.246 ± 0.188 | 0.504 ± 0.219 | 0.302 ± 0.301 | 0.250 ± 0.186 | -91.985 | -0.05 / 1.290 |
| gated_linear | 32 | backbone.blocks.1.mixer | 0.753 ± 0.287 | 0.796 ± 0.295 | 0.828 ± 0.291 | 0.875 ± 0.208 | -72.246 | +0.05 / 1.623 |
| gated_linear | 32 | backbone.blocks.2.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.0 | +0.18 / 3.521 |
| gated_linear | 32 | backbone.blocks.3.mixer | 1.000 ± 0.001 | 1.000 ± 0.000 | 1.000 ± 0.001 | 1.000 ± 0.000 | -0.011 | +0.19 / 2.347 |
| gated_linear | 32 | backbone.blocks.4.mixer | 0.785 ± 0.267 | 0.835 ± 0.079 | 0.948 ± 0.143 | 0.894 ± 0.218 | -14.725 | +0.15 / 1.851 |

## 4g. Gate-clamp intervention (Pass 9): which layers' forgetting does recall need?

Per-query recall with one GLA layer's gate replaced by a constant: `keep` = σ(6) ≈ 0.9975 (no forgetting), `mean` = the head's own average γ (same amount of forgetting, no selectivity). Δ vs the unclamped model on the same prompts, paired 95% CI.

**gated_linear**

| condition | 8 facts | 32 facts | 64 facts |
|---|---|---|---|
| none | 0.985 | 0.845 | 0.501 |
| keep_all | 0.009 (-0.976 [-0.98, -0.97]) | 0.010 (-0.835 [-0.84, -0.83]) | 0.010 (-0.491 [-0.50, -0.48]) |
| keep_L0 | 0.009 (-0.976 [-0.98, -0.97]) | 0.010 (-0.835 [-0.84, -0.83]) | 0.011 (-0.490 [-0.50, -0.48]) |
| mean_L0 | 0.706 (-0.279 [-0.29, -0.26]) | 0.573 (-0.272 [-0.28, -0.26]) | 0.317 (-0.185 [-0.20, -0.17]) |
| keep_L1 | 0.377 (-0.608 [-0.62, -0.59]) | 0.168 (-0.677 [-0.69, -0.67]) | 0.087 (-0.415 [-0.43, -0.40]) |
| mean_L1 | 0.738 (-0.247 [-0.26, -0.23]) | 0.633 (-0.212 [-0.22, -0.20]) | 0.381 (-0.121 [-0.13, -0.11]) |
| keep_L2 | 0.985 (+0.001 [-0.00, +0.00]) | 0.814 (-0.031 [-0.04, -0.02]) | 0.441 (-0.061 [-0.07, -0.05]) |
| mean_L2 | 0.985 (+0.000 [+0.00, +0.00]) | 0.845 (+0.000 [+0.00, +0.00]) | 0.501 (-0.000 [-0.00, +0.00]) |
| keep_L3 | 0.984 (-0.001 [-0.00, +0.00]) | 0.843 (-0.002 [-0.00, +0.00]) | 0.496 (-0.006 [-0.01, -0.00]) |
| mean_L3 | 0.984 (-0.001 [-0.00, +0.00]) | 0.844 (-0.001 [-0.00, +0.00]) | 0.500 (-0.002 [-0.00, +0.00]) |
| keep_L4 | 0.978 (-0.006 [-0.01, -0.00]) | 0.832 (-0.013 [-0.02, -0.01]) | 0.518 (+0.017 [+0.01, +0.02]) |
| mean_L4 | 0.967 (-0.018 [-0.02, -0.01]) | 0.811 (-0.034 [-0.04, -0.03]) | 0.484 (-0.017 [-0.02, -0.01]) |

## 4h. Gate replay (Pass 10): does the timing of forgetting matter?

One layer's recorded gates over the context span are replayed unchanged, shuffled in time (same values, same product), replaced by their geometric mean (same product, no timing) or their arithmetic mean (forgets less). Per-query recall, Δ vs baseline [paired 95% CI].

**gated_linear**

| facts | layer | baseline | replay | shuffle (range) | geomean | arith |
|---:|---|---:|---|---|---|---|
| 8 | L0 | 0.987 | 0.987 (+0.000 [0.00, 0.00]) | 0.680 (0.68–0.68) (-0.308 [-0.32, -0.30]) | 0.802 (-0.185 [-0.20, -0.17]) | 0.693 (-0.294 [-0.31, -0.28]) |
| 8 | L1 | 0.987 | 0.987 (+0.000 [0.00, 0.00]) | 0.770 (0.77–0.77) (-0.217 [-0.23, -0.21]) | 0.905 (-0.083 [-0.09, -0.07]) | 0.762 (-0.225 [-0.24, -0.21]) |
| 8 | L2 | 0.987 | 0.987 (+0.000 [0.00, 0.00]) | 0.987 (0.99–0.99) (+0.000 [0.00, 0.00]) | 0.987 (+0.000 [0.00, 0.00]) | 0.987 (+0.000 [0.00, 0.00]) |
| 8 | L3 | 0.987 | 0.987 (+0.000 [0.00, 0.00]) | 0.986 (0.99–0.99) (-0.001 [-0.00, 0.00]) | 0.987 (-0.001 [-0.00, 0.00]) | 0.987 (-0.001 [-0.00, 0.00]) |
| 8 | L4 | 0.987 | 0.987 (+0.000 [0.00, 0.00]) | 0.988 (0.99–0.99) (+0.000 [0.00, 0.00]) | 0.987 (+0.000 [0.00, 0.00]) | 0.987 (+0.000 [0.00, 0.00]) |
| 32 | L0 | 0.847 | 0.847 (+0.000 [0.00, 0.00]) | 0.414 (0.41–0.42) (-0.433 [-0.44, -0.42]) | 0.620 (-0.228 [-0.24, -0.22]) | 0.482 (-0.364 [-0.38, -0.35]) |
| 32 | L1 | 0.847 | 0.847 (+0.000 [0.00, 0.00]) | 0.621 (0.62–0.62) (-0.227 [-0.23, -0.22]) | 0.781 (-0.066 [-0.07, -0.06]) | 0.720 (-0.127 [-0.14, -0.12]) |
| 32 | L2 | 0.847 | 0.847 (+0.000 [0.00, 0.00]) | 0.847 (0.85–0.85) (+0.000 [-0.00, 0.00]) | 0.847 (+0.000 [0.00, 0.00]) | 0.847 (+0.000 [0.00, 0.00]) |
| 32 | L3 | 0.847 | 0.847 (+0.000 [0.00, 0.00]) | 0.847 (0.85–0.85) (-0.000 [-0.00, 0.00]) | 0.847 (-0.000 [-0.00, 0.00]) | 0.847 (+0.000 [-0.00, 0.00]) |
| 32 | L4 | 0.847 | 0.847 (+0.000 [0.00, 0.00]) | 0.848 (0.85–0.85) (+0.001 [-0.00, 0.00]) | 0.848 (+0.001 [-0.00, 0.00]) | 0.848 (+0.001 [-0.00, 0.00]) |

## 5. Prefill cost (one forward pass, batch 1; mostly kernel quality)

_`cost_profile.csv` not synced yet._

## 5b. Decode cost (one token at a time, batch 1)

_`decode_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
