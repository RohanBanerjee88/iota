# iota run `r39`

- **updated:** 2026-10-26 17:49 UTC (last stage: `eval 8,9,10`)
- **code:** `c9bfe00` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | – | – | – | – | – | – | not trained yet |
| gated_linear | 15000 | 0.994 | 0.988 | 1.000 | 0.942 | 0.0015 | ok |
| hybrid | – | – | – | – | – | – | not trained yet |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `*_sweep_history.json` (and `logs/train.log`); `*_sweep.json` stops at the best checkpoint.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| gated_linear | 0.988 | 0.914 | 1.000 | 1.000 | pass |

n=500 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| n_bindings | true tokens | gated_linear |
|---:|---:|---|
| 2 | 256 | 1.000 [1.00–1.00] (1.00) |
| 4 | 256 | 1.000 [1.00–1.00] (1.00) |
| 8 | 256 | 1.000 [1.00–1.00] (1.00) |
| 16 | 256 | 0.999 [1.00–1.00] (0.98) |
| 32 | 289 | 0.993 [0.99–0.99] (0.90) |
| 64 | 386 | 0.946 [0.94–0.95] (0.41) |
| 128 | 773 | 0.815 [0.81–0.82] (0.04) |

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
| gated_linear | 8 | backbone.blocks.0.mixer | 0.442 ± 0.241 | 0.312 ± 0.326 | 0.308 ± 0.368 | 0.438 ± 0.240 | -242.859 | +1.65 / 1.703 |
| gated_linear | 8 | backbone.blocks.1.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.0 | +2.03 / 2.448 |
| gated_linear | 8 | backbone.blocks.2.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.0 | +1.95 / 2.488 |
| gated_linear | 8 | backbone.blocks.3.mixer | 0.999 ± 0.009 | 1.000 ± 0.001 | 1.000 ± 0.001 | 0.999 ± 0.002 | -0.013 | +1.98 / 2.167 |
| gated_linear | 8 | backbone.blocks.4.mixer | 0.999 ± 0.004 | 0.999 ± 0.001 | 0.997 ± 0.018 | 1.000 ± 0.001 | -0.072 | +1.95 / 2.187 |
| gated_linear | 32 | backbone.blocks.0.mixer | 0.442 ± 0.241 | 0.313 ± 0.326 | 0.309 ± 0.368 | 0.438 ± 0.239 | -174.514 | +1.65 / 1.703 |
| gated_linear | 32 | backbone.blocks.1.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.0 | +2.03 / 2.448 |
| gated_linear | 32 | backbone.blocks.2.mixer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | -0.0 | +1.95 / 2.488 |
| gated_linear | 32 | backbone.blocks.3.mixer | 0.999 ± 0.008 | 1.000 ± 0.001 | 1.000 ± 0.001 | 1.000 ± 0.001 | -0.037 | +1.98 / 2.167 |
| gated_linear | 32 | backbone.blocks.4.mixer | 1.000 ± 0.002 | 1.000 ± 0.001 | 0.998 ± 0.011 | 1.000 ± 0.000 | -0.034 | +1.95 / 2.187 |

## 4g. Gate-clamp intervention (Pass 9): which layers' forgetting does recall need?

Per-query recall with one GLA layer's gate replaced by a constant: `keep` = σ(6) ≈ 0.9975 (no forgetting), `mean` = the head's own average γ (same amount of forgetting, no selectivity). Δ vs the unclamped model on the same prompts, paired 95% CI.

**gated_linear**

| condition | 8 facts | 32 facts | 64 facts |
|---|---|---|---|
| none | 1.000 | 0.992 | 0.952 |
| keep_all | 0.012 (-0.988 [-0.99, -0.98]) | 0.012 (-0.980 [-0.98, -0.98]) | 0.010 (-0.942 [-0.95, -0.94]) |
| keep_L0 | 0.013 (-0.987 [-0.99, -0.98]) | 0.013 (-0.980 [-0.98, -0.98]) | 0.011 (-0.941 [-0.95, -0.94]) |
| mean_L0 | 0.632 (-0.367 [-0.38, -0.35]) | 0.807 (-0.185 [-0.19, -0.18]) | 0.743 (-0.209 [-0.22, -0.20]) |
| keep_L1 | 1.000 (+0.000 [+0.00, +0.00]) | 0.993 (+0.000 [-0.00, +0.00]) | 0.947 (-0.005 [-0.01, -0.00]) |
| mean_L1 | 1.000 (+0.000 [+0.00, +0.00]) | 0.992 (+0.000 [+0.00, +0.00]) | 0.952 (+0.000 [+0.00, +0.00]) |
| keep_L2 | 1.000 (+0.000 [+0.00, +0.00]) | 0.991 (-0.001 [-0.00, +0.00]) | 0.920 (-0.032 [-0.04, -0.03]) |
| mean_L2 | 1.000 (+0.000 [+0.00, +0.00]) | 0.992 (+0.000 [+0.00, +0.00]) | 0.952 (+0.000 [+0.00, +0.00]) |
| keep_L3 | 1.000 (+0.000 [+0.00, +0.00]) | 0.991 (-0.001 [-0.00, +0.00]) | 0.943 (-0.009 [-0.01, -0.01]) |
| mean_L3 | 1.000 (+0.000 [+0.00, +0.00]) | 0.992 (-0.000 [-0.00, +0.00]) | 0.952 (-0.000 [-0.00, +0.00]) |
| keep_L4 | 1.000 (+0.000 [+0.00, +0.00]) | 0.991 (-0.001 [-0.00, -0.00]) | 0.946 (-0.006 [-0.01, -0.00]) |
| mean_L4 | 1.000 (+0.000 [+0.00, +0.00]) | 0.992 (+0.000 [+0.00, +0.00]) | 0.952 (+0.000 [-0.00, +0.00]) |

## 4h. Gate replay (Pass 10): does the timing of forgetting matter?

One layer's recorded gates over the context span are replayed unchanged, shuffled in time (same values, same product), replaced by their geometric mean (same product, no timing) or their arithmetic mean (forgets less). Per-query recall, Δ vs baseline [paired 95% CI].

**gated_linear**

| facts | layer | baseline | replay | shuffle (range) | geomean | arith |
|---:|---|---:|---|---|---|---|
| 8 | L0 | 0.999 | 0.999 (+0.000 [0.00, 0.00]) | 0.329 (0.32–0.34) (-0.671 [-0.68, -0.66]) | 0.289 (-0.711 [-0.72, -0.70]) | 0.568 (-0.432 [-0.45, -0.42]) |
| 8 | L1 | 0.999 | 0.999 (+0.000 [0.00, 0.00]) | 0.999 (1.00–1.00) (+0.000 [0.00, 0.00]) | 0.999 (+0.000 [0.00, 0.00]) | 0.999 (+0.000 [0.00, 0.00]) |
| 8 | L2 | 0.999 | 0.999 (+0.000 [0.00, 0.00]) | 1.000 (1.00–1.00) (+0.000 [0.00, 0.00]) | 1.000 (+0.000 [0.00, 0.00]) | 1.000 (+0.000 [0.00, 0.00]) |
| 8 | L3 | 0.999 | 0.999 (+0.000 [0.00, 0.00]) | 0.999 (1.00–1.00) (+0.000 [0.00, 0.00]) | 0.999 (+0.000 [0.00, 0.00]) | 0.999 (+0.000 [0.00, 0.00]) |
| 8 | L4 | 0.999 | 0.999 (+0.000 [0.00, 0.00]) | 0.999 (1.00–1.00) (+0.000 [0.00, 0.00]) | 0.999 (+0.000 [0.00, 0.00]) | 0.999 (+0.000 [0.00, 0.00]) |
| 32 | L0 | 0.992 | 0.992 (+0.000 [0.00, 0.00]) | 0.362 (0.36–0.37) (-0.630 [-0.64, -0.62]) | 0.269 (-0.723 [-0.73, -0.71]) | 0.795 (-0.197 [-0.21, -0.19]) |
| 32 | L1 | 0.992 | 0.992 (+0.000 [0.00, 0.00]) | 0.992 (0.99–0.99) (+0.000 [0.00, 0.00]) | 0.992 (+0.000 [0.00, 0.00]) | 0.992 (+0.000 [0.00, 0.00]) |
| 32 | L2 | 0.992 | 0.992 (+0.000 [0.00, 0.00]) | 0.992 (0.99–0.99) (+0.000 [0.00, 0.00]) | 0.992 (+0.000 [0.00, 0.00]) | 0.992 (+0.000 [0.00, 0.00]) |
| 32 | L3 | 0.992 | 0.992 (+0.000 [0.00, 0.00]) | 0.992 (0.99–0.99) (+0.000 [0.00, 0.00]) | 0.992 (+0.000 [0.00, 0.00]) | 0.992 (+0.000 [0.00, 0.00]) |
| 32 | L4 | 0.992 | 0.992 (+0.000 [0.00, 0.00]) | 0.992 (0.99–0.99) (+0.000 [0.00, 0.00]) | 0.992 (+0.000 [0.00, 0.00]) | 0.992 (+0.000 [0.00, 0.00]) |

## 5. Prefill cost (one forward pass, batch 1; mostly kernel quality)

_`cost_profile.csv` not synced yet._

## 5b. Decode cost (one token at a time, batch 1)

_`decode_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
