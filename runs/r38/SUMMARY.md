# iota run `r38`

- **updated:** 2026-10-25 19:48 UTC (last stage: `eval 1,3`)
- **code:** `45e8d0d` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | – | – | – | – | – | – | not trained yet |
| gated_linear | 13000 | 0.639 | 0.277 | 1.000 | 0.382 | 0.0015 | ok |
| hybrid | – | – | – | – | – | – | not trained yet |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `*_sweep_history.json` (and `logs/train.log`); `*_sweep.json` stops at the best checkpoint.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| gated_linear | 0.260 | 0.066 | 1.000 | 1.000 | pass |

n=500 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| n_bindings | true tokens | gated_linear |
|---:|---:|---|
| 2 | 256 | 0.950 [0.94–0.96] (0.90) |
| 4 | 256 | 0.861 [0.85–0.87] (0.54) |
| 8 | 256 | 0.704 [0.69–0.71] (0.05) |
| 16 | 256 | 0.466 [0.46–0.47] (0.00) |
| 32 | 289 | 0.225 [0.22–0.23] (0.00) |
| 64 | 386 | 0.108 [0.10–0.11] (0.00) |
| 128 | 773 | 0.057 [0.05–0.06] (0.00) |

## 3. Pass 2 — length generalisation

_`pass2_length.csv` not synced yet._

## 4. Pass 3 — state_track control

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| seq_len | true tokens | gated_linear |
|---:|---:|---|
| 128 | 136 | 0.999 [1.00–1.00] (1.00) |
| 256 | 264 | 0.999 [1.00–1.00] (1.00) |
| 512 | 520 | 1.000 [1.00–1.00] (1.00) |
| 1024 | 1032 | 1.000 [1.00–1.00] (1.00) |
| 2048 | 2056 | 0.968 [0.96–0.98] (0.97) |
| 4096 | 4104 | 0.112 [0.09–0.13] (0.11) |
| 8192 | 8200 | 0.023 [0.01–0.03] (0.02) |

## 5. Prefill cost (one forward pass, batch 1; mostly kernel quality)

_`cost_profile.csv` not synced yet._

## 5b. Decode cost (one token at a time, batch 1)

_`decode_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
