# iota run `r01`

- **updated:** 2026-09-30 07:29 UTC (last stage: `eval 1,3`)
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

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| n_bindings | true tokens | transformer | gated_linear | hybrid |
|---:|---:|---|---|---|
| 2 | 254 | 0.977 [0.97–0.98] (0.95) | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) |
| 4 | 252 | 0.582 [0.57–0.59] (0.03) | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) |
| 8 | 247 | 0.287 [0.28–0.29] (0.00) | 1.000 [1.00–1.00] (1.00) | 0.998 [1.00–1.00] (0.99) |
| 16 | 244 | 0.128 [0.12–0.13] (0.00) | 0.991 [0.99–0.99] (0.87) | 0.962 [0.96–0.96] (0.56) |
| 32 | 275 | 0.065 [0.06–0.07] (0.00) | 0.410 [0.40–0.42] (0.00) | 0.443 [0.44–0.45] (0.00) |
| 64 | 367 | 0.025 [0.02–0.03] (0.00) | 0.161 [0.16–0.17] (0.00) | 0.209 [0.20–0.21] (0.00) |
| 128 | 773 | 0.009 [0.01–0.01] (0.00) | 0.069 [0.06–0.07] (0.00) | 0.044 [0.04–0.05] (0.00) |

## 3. Pass 2 — length generalisation

_`pass2_length.csv` not synced yet._

## 4. Pass 3 — state_track control

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| seq_len | true tokens | transformer | gated_linear | hybrid |
|---:|---:|---|---|---|
| 128 | 144 | 0.018 [0.01–0.03] (0.02) | 0.012 [0.01–0.02] (0.01) | 0.007 [0.00–0.01] (0.01) |
| 256 | 272 | 0.015 [0.01–0.02] (0.01) | 0.017 [0.01–0.03] (0.02) | 0.008 [0.00–0.01] (0.01) |
| 512 | 528 | 0.008 [0.00–0.01] (0.01) | 0.011 [0.01–0.02] (0.01) | 0.016 [0.01–0.02] (0.02) |
| 1024 | 1040 | 0.013 [0.01–0.02] (0.01) | 0.016 [0.01–0.02] (0.02) | 0.007 [0.00–0.01] (0.01) |
| 2048 | 2064 | 0.015 [0.01–0.02] (0.01) | 0.013 [0.01–0.02] (0.01) | 0.006 [0.00–0.01] (0.01) |
| 4096 | 4112 | 0.010 [0.00–0.02] (0.01) | 0.009 [0.00–0.02] (0.01) | 0.011 [0.01–0.02] (0.01) |
| 8192 | 8208 | 0.013 [0.01–0.02] (0.01) | 0.007 [0.00–0.01] (0.01) | 0.016 [0.01–0.02] (0.02) |

## 5. Cost (forward pass, batch 1)

_`cost_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
