# iota run `r03`

- **updated:** 2026-10-02 17:04 UTC (last stage: `profile`)
- **code:** `f6bbd0f` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 0. LR probe (identical short budget per arch; pick each arch's best)

| arch | lr | best balanced | assoc | state | exact | best step | final balanced | time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| transformer | 0.00075 | 0.912 | 0.824 | 1.000 | 0.621 | 4000 | 0.912 | 1073s |
| transformer ★ | 0.0015 | 0.915 | 0.829 | 1.000 | 0.609 | 4000 | 0.915 | 1071s |
| gated_linear | 0.00075 | 0.541 | 0.082 | 1.000 | 0.363 | 3500 | 0.540 | 1266s |
| gated_linear ★ | 0.0015 | 0.616 | 0.233 | 1.000 | 0.375 | 3500 | 0.613 | 1266s |
| hybrid ★ | 0.00075 | 1.000 | 1.000 | 1.000 | 1.000 | 3000 | 1.000 | 1485s |
| hybrid | 0.0015 | 1.000 | 1.000 | 1.000 | 1.000 | 3500 | 1.000 | 1482s |

4000 steps each, grad_clip 1.0. ★ = best lr for that arch.

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | 10000 | 0.927 | 0.855 | 1.000 | 0.624 | 0.0015 | ok |
| gated_linear | 15000 | 0.916 | 0.831 | 1.000 | 0.640 | 0.0015 | ok |
| hybrid | 6000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.00075 | ok |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `logs/train.log` and the `*_sweep.json` files.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| hybrid | 1.000 | 0.998 | 0.998 | 0.998 | pass |
| transformer | 0.847 | 0.442 | 1.000 | 1.000 | pass |
| gated_linear | 0.811 | 0.412 | 1.000 | 1.000 | pass |

n=500 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| n_bindings | true tokens | transformer | gated_linear | hybrid |
|---:|---:|---|---|---|
| 2 | 256 | 0.998 [0.99–1.00] (0.99) | 0.998 [0.99–1.00] (0.99) | 1.000 [1.00–1.00] (1.00) |
| 4 | 256 | 0.992 [0.99–0.99] (0.97) | 0.992 [0.99–0.99] (0.97) | 1.000 [1.00–1.00] (1.00) |
| 8 | 256 | 0.979 [0.98–0.98] (0.84) | 0.985 [0.98–0.99] (0.89) | 1.000 [1.00–1.00] (1.00) |
| 16 | 256 | 0.952 [0.95–0.96] (0.47) | 0.958 [0.95–0.96] (0.53) | 1.000 [1.00–1.00] (1.00) |
| 32 | 289 | 0.875 [0.87–0.88] (0.10) | 0.866 [0.86–0.87] (0.11) | 1.000 [1.00–1.00] (1.00) |
| 64 | 386 | 0.736 [0.73–0.74] (0.00) | 0.591 [0.58–0.60] (0.00) | 0.996 [1.00–1.00] (0.94) |
| 128 | 773 | 0.521 [0.51–0.53] (0.00) | 0.255 [0.25–0.26] (0.00) | 0.989 [0.99–0.99] (0.84) |

## 3. Pass 2 — length generalisation

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| seq_len | true tokens | transformer | gated_linear | hybrid |
|---:|---:|---|---|---|
| 128 | 128 | 0.979 [0.98–0.98] (0.84) | 0.984 [0.98–0.99] (0.88) | 1.000 [1.00–1.00] (1.00) |
| 256 | 256 | 0.980 [0.98–0.98] (0.85) | 0.987 [0.98–0.99] (0.90) | 1.000 [1.00–1.00] (1.00) |
| 512 | 512 | 0.980 [0.98–0.98] (0.85) | 0.988 [0.99–0.99] (0.91) | 1.000 [1.00–1.00] (1.00) |
| 1024 | 1024 | 0.953 [0.95–0.96] (0.70) | 0.987 [0.98–0.99] (0.90) | 0.996 [0.99–1.00] (0.97) |
| 2048 | 2048 | 0.242 [0.23–0.25] (0.00) | 0.985 [0.98–0.99] (0.89) | 0.384 [0.37–0.39] (0.00) |
| 4096 | 4096 | 0.015 [0.01–0.02] (0.00) | 0.972 [0.97–0.98] (0.81) | 0.026 [0.02–0.03] (0.00) |
| 8192 | 8192 | 0.010 [0.01–0.01] (0.00) | 0.942 [0.94–0.95] (0.62) | 0.010 [0.01–0.01] (0.00) |

## 4. Pass 3 — state_track control

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| seq_len | true tokens | transformer | gated_linear | hybrid |
|---:|---:|---|---|---|
| 128 | 136 | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) |
| 256 | 264 | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) |
| 512 | 520 | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) |
| 1024 | 1032 | 0.834 [0.81–0.86] (0.83) | 1.000 [1.00–1.00] (1.00) | 0.771 [0.75–0.80] (0.77) |
| 2048 | 2056 | 0.080 [0.06–0.10] (0.08) | 1.000 [1.00–1.00] (1.00) | 0.086 [0.07–0.10] (0.09) |
| 4096 | 4104 | 0.015 [0.01–0.02] (0.01) | 1.000 [1.00–1.00] (1.00) | 0.011 [0.01–0.02] (0.01) |
| 8192 | 8200 | 0.107 [0.09–0.13] (0.11) | 1.000 [1.00–1.00] (1.00) | 0.211 [0.19–0.24] (0.21) |

## 5. Cost (forward pass, batch 1)

Peak VRAM MB / median latency ms.

| seq_len | transformer | gated_linear | hybrid |
|---:|---|---|---|
| 128 | 18.6 MB / 3.88 ms | 19.0 MB / 9.55 ms | 21.1 MB / 8.47 ms |
| 256 | 20.0 MB / 5.50 ms | 20.1 MB / 13.81 ms | 22.3 MB / 12.06 ms |
| 512 | 22.8 MB / 4.02 ms | 22.4 MB / 23.08 ms | 24.6 MB / 19.79 ms |
| 1024 | 28.3 MB / 4.74 ms | 26.9 MB / 41.61 ms | 29.8 MB / 34.56 ms |
| 2048 | 39.3 MB / 11.12 ms | 35.9 MB / 78.15 ms | 40.3 MB / 63.80 ms |
| 4096 | 61.3 MB / 23.35 ms | 54.0 MB / 157.52 ms | 61.3 MB / 125.86 ms |
| 8192 | 105.3 MB / 71.52 ms | 90.2 MB / 313.35 ms | 103.4 MB / 251.78 ms |

## 6. Figure

_Not plotted yet._
