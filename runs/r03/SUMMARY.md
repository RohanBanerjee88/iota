# iota run `r03`

- **updated:** 2026-10-02 11:25 UTC (last stage: `train`)
- **code:** `e048745` on `claude/wonderful-ritchie-hbm5zn`
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
