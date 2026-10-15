# iota run `r19`

- **updated:** 2026-10-15 04:48 UTC (last stage: `tune:gated_linear:0.00075`)
- **code:** `3f12ce3` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 0. LR probe (identical short budget per arch; pick each arch's best)

| arch | lr | best balanced | assoc | state | exact | best step | final balanced | time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gated_linear ★ | 0.00075 | 0.670 | 0.340 | 1.000 | 0.387 | 5250 | 0.670 | 2406s |

6000 steps each, grad_clip 1.0. ★ = best lr for that arch.

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

## 5. Prefill cost (one forward pass, batch 1; mostly kernel quality)

_`cost_profile.csv` not synced yet._

## 5b. Decode cost (one token at a time, batch 1)

_`decode_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
