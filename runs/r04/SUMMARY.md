# iota run `r04`

- **updated:** 2026-10-03 02:34 UTC (last stage: `eval 1,3`)
- **code:** `c1fadb5` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | 14000 | 0.904 | 0.809 | 1.000 | 0.600 | 0.0015 | ok |
| gated_linear | 15000 | 0.916 | 0.839 | 0.994 | 0.634 | 0.0015 | ok |
| hybrid | 4500 | 1.000 | 1.000 | 1.000 | 1.000 | 0.00075 | ok |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `logs/train.log` and the `*_sweep.json` files.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| hybrid | 0.999 | 0.992 | 1.000 | 1.000 | pass |
| transformer | 0.809 | 0.380 | 1.000 | 1.000 | pass |
| gated_linear | 0.828 | 0.442 | 1.000 | 1.000 | pass |

n=500 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| n_bindings | true tokens | transformer | gated_linear | hybrid |
|---:|---:|---|---|---|
| 2 | 256 | 0.997 [0.99–1.00] (0.99) | 0.999 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) |
| 4 | 256 | 0.991 [0.99–0.99] (0.96) | 0.995 [0.99–1.00] (0.98) | 1.000 [1.00–1.00] (1.00) |
| 8 | 256 | 0.975 [0.97–0.98] (0.82) | 0.989 [0.99–0.99] (0.92) | 1.000 [1.00–1.00] (1.00) |
| 16 | 256 | 0.945 [0.94–0.95] (0.43) | 0.964 [0.96–0.97] (0.57) | 1.000 [1.00–1.00] (0.99) |
| 32 | 289 | 0.842 [0.84–0.85] (0.05) | 0.882 [0.88–0.89] (0.13) | 1.000 [1.00–1.00] (0.99) |
| 64 | 386 | 0.620 [0.61–0.63] (0.00) | 0.489 [0.48–0.50] (0.00) | 0.997 [1.00–1.00] (0.96) |
| 128 | 773 | 0.347 [0.34–0.35] (0.00) | 0.222 [0.22–0.23] (0.00) | 0.990 [0.99–0.99] (0.85) |

## 3. Pass 2 — length generalisation

_`pass2_length.csv` not synced yet._

## 4. Pass 3 — state_track control

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| seq_len | true tokens | transformer | gated_linear | hybrid |
|---:|---:|---|---|---|
| 128 | 136 | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) |
| 256 | 264 | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) |
| 512 | 520 | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) |
| 1024 | 1032 | 0.955 [0.94–0.97] (0.95) | 1.000 [1.00–1.00] (1.00) | 0.701 [0.67–0.73] (0.70) |
| 2048 | 2056 | 0.189 [0.17–0.21] (0.19) | 1.000 [1.00–1.00] (1.00) | 0.052 [0.04–0.07] (0.05) |
| 4096 | 4104 | 0.017 [0.01–0.03] (0.02) | 0.999 [1.00–1.00] (1.00) | 0.011 [0.01–0.02] (0.01) |
| 8192 | 8200 | 0.020 [0.01–0.03] (0.02) | 0.984 [0.98–0.99] (0.98) | 0.226 [0.20–0.25] (0.23) |

## 5. Cost (forward pass, batch 1)

_`cost_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
