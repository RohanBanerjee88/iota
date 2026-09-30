# iota run `r01`

- **updated:** 2026-09-30 07:23 UTC (last stage: `sanity`)
- **code:** `7e1b8ef` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | 14500 | 0.159 | 0.305 | 0.012 | 0.096 | 0.0015 | ⚠ control at chance -- figure NOT trustworthy |
| gated_linear | 19000 | 0.508 | 0.998 | 0.018 | 0.668 | 0.003 | ⚠ control at chance -- figure NOT trustworthy |
| hybrid | 9500 | 0.509 | 0.983 | 0.036 | 0.642 | 0.0015 | ⚠ control at chance -- figure NOT trustworthy |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `logs/train.log` and the `*_sweep.json` files.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| hybrid | 0.988 | 0.964 | 0.016 | 0.016 | ⚠ FAIL — a task is at chance |
| transformer | 0.300 | 0.142 | 0.004 | 0.004 | ⚠ FAIL — a task is at chance |
| gated_linear | 1.000 | 1.000 | 0.008 | 0.008 | ⚠ FAIL — a task is at chance |

n=500 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

_`pass1_capacity.csv` not synced yet._

## 3. Pass 2 — length generalisation

_`pass2_length.csv` not synced yet._

## 4. Pass 3 — state_track control

_`pass3_control.csv` not synced yet._

## 5. Cost (forward pass, batch 1)

_`cost_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
