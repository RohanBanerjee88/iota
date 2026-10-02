# iota run `r03-smoke`

- **updated:** 2026-10-02 05:48 UTC (last stage: `plot`)
- **code:** `e048745` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 0. LR probe (identical short budget per arch; pick each arch's best)

| arch | lr | best balanced | assoc | state | exact | best step | final balanced | time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| transformer ★ | 0.00075 | 0.004 | 0.008 | 0.000 | 0.000 | 40 | 0.003 | 5s |
| gated_linear ★ | 0.00075 | 0.033 | 0.015 | 0.050 | 0.016 | 60 | 0.033 | 6s |
| hybrid ★ | 0.00075 | 0.006 | 0.013 | 0.000 | 0.000 | 60 | 0.006 | 6s |

4000 steps each, grad_clip 1.0. ★ = best lr for that arch.

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | 60 | 0.004 | 0.008 | 0.000 | 0.000 | 0.0015 | ⚠ control at chance -- figure NOT trustworthy |
| gated_linear | 60 | 0.032 | 0.013 | 0.050 | 0.016 | 0.0015 | ⚠ recall at chance -- model never learned |
| hybrid | 60 | 0.006 | 0.013 | 0.000 | 0.000 | 0.00075 | ⚠ control at chance -- figure NOT trustworthy |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `logs/train.log` and the `*_sweep.json` files.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| hybrid | 0.018 | 0.000 | 0.000 | 0.000 | ⚠ FAIL — a task is at chance |
| transformer | 0.000 | 0.000 | 0.000 | 0.000 | ⚠ FAIL — a task is at chance |
| gated_linear | 0.000 | 0.000 | 0.000 | 0.000 | ⚠ FAIL — a task is at chance |

n=8 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=8 per cell, paired prompts.

| n_bindings | true tokens | transformer | gated_linear | hybrid |
|---:|---:|---|---|---|
| 2 | 256 | 0.000 [0.00–0.00] (0.00) | 0.062 [0.00–0.19] (0.00) | 0.062 [0.00–0.19] (0.00) |
| 4 | 256 | 0.000 [0.00–0.00] (0.00) | 0.000 [0.00–0.00] (0.00) | 0.031 [0.00–0.09] (0.00) |
| 8 | 256 | 0.000 [0.00–0.00] (0.00) | 0.016 [0.00–0.05] (0.00) | 0.016 [0.00–0.05] (0.00) |
| 16 | 256 | 0.008 [0.00–0.02] (0.00) | 0.016 [0.00–0.04] (0.00) | 0.008 [0.00–0.02] (0.00) |
| 32 | 290 | 0.016 [0.00–0.03] (0.00) | 0.023 [0.01–0.05] (0.00) | 0.023 [0.01–0.05] (0.00) |
| 64 | 384 | 0.023 [0.00–0.05] (0.00) | 0.023 [0.00–0.05] (0.00) | 0.008 [0.00–0.02] (0.00) |
| 128 | 774 | 0.000 [0.00–0.00] (0.00) | 0.023 [0.01–0.05] (0.00) | 0.008 [0.00–0.02] (0.00) |

## 3. Pass 2 — length generalisation

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=8 per cell, paired prompts.

| seq_len | true tokens | transformer | gated_linear | hybrid |
|---:|---:|---|---|---|
| 128 | 128 | 0.000 [0.00–0.00] (0.00) | 0.031 [0.00–0.06] (0.00) | 0.000 [0.00–0.00] (0.00) |
| 256 | 256 | 0.016 [0.00–0.05] (0.00) | 0.016 [0.00–0.05] (0.00) | 0.016 [0.00–0.05] (0.00) |
| 512 | 513 | 0.000 [0.00–0.00] (0.00) | 0.000 [0.00–0.00] (0.00) | 0.016 [0.00–0.05] (0.00) |
| 1024 | 1025 | 0.000 [0.00–0.00] (0.00) | 0.016 [0.00–0.05] (0.00) | 0.016 [0.00–0.05] (0.00) |
| 2048 | 2048 | 0.000 [0.00–0.00] (0.00) | 0.000 [0.00–0.00] (0.00) | 0.016 [0.00–0.05] (0.00) |
| 4096 | 4096 | 0.000 [0.00–0.00] (0.00) | 0.016 [0.00–0.05] (0.00) | 0.000 [0.00–0.00] (0.00) |
| 8192 | 8192 | 0.000 [0.00–0.00] (0.00) | 0.000 [0.00–0.00] (0.00) | 0.000 [0.00–0.00] (0.00) |

## 4. Pass 3 — state_track control

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=8 per cell, paired prompts.

| seq_len | true tokens | transformer | gated_linear | hybrid |
|---:|---:|---|---|---|
| 128 | 136 | 0.000 [0.00–0.00] (0.00) | 0.000 [0.00–0.00] (0.00) | 0.000 [0.00–0.00] (0.00) |
| 256 | 264 | 0.000 [0.00–0.00] (0.00) | 0.125 [0.00–0.38] (0.12) | 0.000 [0.00–0.00] (0.00) |
| 512 | 520 | 0.000 [0.00–0.00] (0.00) | 0.000 [0.00–0.00] (0.00) | 0.000 [0.00–0.00] (0.00) |
| 1024 | 1032 | 0.000 [0.00–0.00] (0.00) | 0.125 [0.00–0.38] (0.12) | 0.125 [0.00–0.38] (0.12) |
| 2048 | 2056 | 0.000 [0.00–0.00] (0.00) | 0.000 [0.00–0.00] (0.00) | 0.000 [0.00–0.00] (0.00) |
| 4096 | 4104 | 0.000 [0.00–0.00] (0.00) | 0.000 [0.00–0.00] (0.00) | 0.000 [0.00–0.00] (0.00) |
| 8192 | 8200 | 0.000 [0.00–0.00] (0.00) | 0.000 [0.00–0.00] (0.00) | 0.000 [0.00–0.00] (0.00) |

## 5. Cost (forward pass, batch 1)

Peak VRAM MB / median latency ms.

| seq_len | transformer | gated_linear | hybrid |
|---:|---|---|---|
| 128 | 26.8 MB / 4.01 ms | 27.1 MB / 10.25 ms | 29.2 MB / 8.99 ms |
| 256 | 28.1 MB / 4.07 ms | 28.2 MB / 15.62 ms | 30.4 MB / 13.18 ms |
| 512 | 31.1 MB / 4.25 ms | 30.5 MB / 25.50 ms | 32.9 MB / 20.92 ms |
| 1024 | 36.4 MB / 6.54 ms | 35.0 MB / 48.62 ms | 37.9 MB / 39.15 ms |
| 2048 | 48.2 MB / 8.95 ms | 44.7 MB / 90.63 ms | 49.3 MB / 70.08 ms |
| 4096 | 69.4 MB / 23.81 ms | 62.1 MB / 167.75 ms | 70.1 MB / 136.45 ms |
| 8192 | 113.5 MB / 72.36 ms | 98.3 MB / 334.12 ms | 111.5 MB / 270.94 ms |

## 6. Figure

![money figure](money_figure.png)
