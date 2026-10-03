# iota run `r04`

- **updated:** 2026-10-03 02:26 UTC (last stage: `train`)
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

_`sanity_indist.csv` not synced yet._

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
