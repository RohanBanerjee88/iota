# iota run `r09-smoke`

- **updated:** 2026-10-07 19:06 UTC (last stage: `train:gated_linear`)
- **code:** `494960f` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 0. LR probe (identical short budget per arch; pick each arch's best)

| arch | lr | best balanced | assoc | state | exact | best step | final balanced | time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| transformer ★ | 0.00075 | 0.001 | 0.003 | 0.000 | 0.000 | 60 | 0.001 | 6s |
| gated_linear ★ | 0.00075 | 0.009 | 0.018 | 0.000 | 0.000 | 60 | 0.009 | 9s |
| hybrid ★ | 0.00075 | 0.006 | 0.013 | 0.000 | 0.000 | 60 | 0.006 | 7s |

4000 steps each, grad_clip 1.0. ★ = best lr for that arch.

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | 60 | 0.001 | 0.003 | 0.000 | 0.000 | 0.0015 | ⚠ control at chance -- figure NOT trustworthy |
| gated_linear | 60 | 0.032 | 0.013 | 0.050 | 0.016 | 0.0015 | ⚠ recall at chance -- model never learned |
| hybrid | 60 | 0.006 | 0.013 | 0.000 | 0.000 | 0.00075 | ⚠ control at chance -- figure NOT trustworthy |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `logs/train.log` and the `*_sweep.json` files.

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
