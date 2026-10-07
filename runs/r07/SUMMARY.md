# iota run `r07`

- **updated:** 2026-10-07 00:17 UTC (last stage: `eval 1,3`)
- **code:** `d5e5191` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | 14000 | 0.904 | 0.809 | 1.000 | 0.600 | 0.0015 | ok |
| gated_linear | 10500 | 0.692 | 0.385 | 1.000 | 0.412 | 0.0015 | ok |
| hybrid | 12500 | 0.689 | 0.377 | 1.000 | 0.412 | 0.00075 | ok |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `logs/train.log` and the `*_sweep.json` files.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| hybrid | 0.357 | 0.092 | 1.000 | 1.000 | pass |
| transformer | 0.809 | 0.380 | 1.000 | 1.000 | pass |
| gated_linear | 0.367 | 0.114 | 1.000 | 1.000 | pass |

n=500 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| n_bindings | true tokens | transformer | gated_linear | hybrid |
|---:|---:|---|---|---|
| 2 | 256 | 0.997 [0.99–1.00] (0.99) | 0.978 [0.97–0.98] (0.95) | 0.979 [0.97–0.98] (0.96) |
| 4 | 256 | 0.991 [0.99–0.99] (0.96) | 0.929 [0.92–0.94] (0.74) | 0.926 [0.92–0.93] (0.72) |
| 8 | 256 | 0.975 [0.97–0.98] (0.82) | 0.854 [0.85–0.86] (0.24) | 0.848 [0.84–0.86] (0.23) |
| 16 | 256 | 0.945 [0.94–0.95] (0.43) | 0.718 [0.71–0.72] (0.00) | 0.717 [0.71–0.72] (0.00) |
| 32 | 289 | 0.842 [0.84–0.85] (0.05) | 0.355 [0.35–0.36] (0.00) | 0.357 [0.35–0.36] (0.00) |
| 64 | 386 | 0.620 [0.61–0.63] (0.00) | 0.082 [0.08–0.09] (0.00) | 0.113 [0.11–0.12] (0.00) |
| 128 | 773 | 0.347 [0.34–0.35] (0.00) | 0.045 [0.04–0.05] (0.00) | 0.060 [0.06–0.06] (0.00) |

## 3. Pass 2 — length generalisation

_`pass2_length.csv` not synced yet._

## 4. Pass 3 — state_track control

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| seq_len | true tokens | transformer | gated_linear | hybrid |
|---:|---:|---|---|---|
| 128 | 136 | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) |
| 256 | 264 | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) |
| 512 | 520 | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) |
| 1024 | 1032 | 0.955 [0.94–0.97] (0.95) | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) |
| 2048 | 2056 | 0.189 [0.17–0.21] (0.19) | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) |
| 4096 | 4104 | 0.017 [0.01–0.03] (0.02) | 0.998 [0.99–1.00] (1.00) | 0.927 [0.91–0.94] (0.93) |
| 8192 | 8200 | 0.020 [0.01–0.03] (0.02) | 0.926 [0.91–0.94] (0.93) | 0.180 [0.16–0.20] (0.18) |

## 5. Prefill cost (one forward pass, batch 1; mostly kernel quality)

_`cost_profile.csv` not synced yet._

## 5b. Decode cost (one token at a time, batch 1)

_`decode_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
