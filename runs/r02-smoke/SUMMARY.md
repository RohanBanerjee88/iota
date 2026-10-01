# iota run `r02-smoke`

- **updated:** 2026-10-01 02:55 UTC (last stage: `tune:gated_linear:0.00075`)
- **code:** `d218e42` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 0. LR probe (identical short budget per arch; pick each arch's best)

| arch | lr | best balanced | assoc | state | exact | best step | final balanced | time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| transformer ★ | 0.00075 | 0.001 | 0.003 | 0.000 | 0.000 | 40 | 0.000 | 5s |
| gated_linear ★ | 0.00075 | 0.005 | 0.010 | 0.000 | 0.000 | 60 | 0.005 | 6s |
| hybrid ★ | 0.00075 | 0.005 | 0.010 | 0.000 | 0.000 | 40 | 0.005 | 5s |

4000 steps each, grad_clip 1.0. ★ = best lr for that arch.

## 1. Training (best checkpoint, full-difficulty held-out set)

_No training runs synced yet._

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
