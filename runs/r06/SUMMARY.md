# iota run `r06`

- **updated:** 2026-10-05 17:37 UTC (last stage: `eval 4`)
- **code:** `5026708` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | 12500 | 0.901 | 0.802 | 1.000 | 0.602 | 0.0015 | ok |
| gated_linear | 12500 | 0.977 | 0.955 | 1.000 | 0.830 | 0.0015 | ok |
| hybrid | 6000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.00075 | ok |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `logs/train.log` and the `*_sweep.json` files.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| hybrid | 1.000 | 0.998 | 1.000 | 1.000 | pass |
| transformer | 0.797 | 0.364 | 1.000 | 1.000 | pass |
| gated_linear | 0.946 | 0.728 | 1.000 | 1.000 | pass |

n=500 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| n_bindings | true tokens | transformer | gated_linear | hybrid |
|---:|---:|---|---|---|
| 2 | 256 | 0.997 [0.99–1.00] (0.99) | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) |
| 4 | 256 | 0.992 [0.99–0.99] (0.97) | 0.999 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) |
| 8 | 256 | 0.981 [0.98–0.98] (0.85) | 0.999 [1.00–1.00] (0.99) | 1.000 [1.00–1.00] (1.00) |
| 16 | 256 | 0.949 [0.95–0.95] (0.42) | 0.995 [0.99–1.00] (0.93) | 1.000 [1.00–1.00] (1.00) |
| 32 | 289 | 0.841 [0.84–0.85] (0.05) | 0.971 [0.97–0.97] (0.63) | 1.000 [1.00–1.00] (1.00) |
| 64 | 386 | 0.610 [0.60–0.62] (0.00) | 0.555 [0.55–0.56] (0.00) | 0.991 [0.99–0.99] (0.86) |
| 128 | 773 | 0.320 [0.31–0.33] (0.00) | 0.307 [0.30–0.31] (0.00) | 0.988 [0.99–0.99] (0.82) |

## 3. Pass 2 — length generalisation

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| seq_len | true tokens | transformer | gated_linear | hybrid |
|---:|---:|---|---|---|
| 128 | 128 | 0.977 [0.97–0.98] (0.83) | 0.999 [1.00–1.00] (0.99) | 1.000 [1.00–1.00] (1.00) |
| 256 | 256 | 0.981 [0.98–0.98] (0.86) | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) |
| 512 | 512 | 0.979 [0.98–0.98] (0.85) | 0.998 [1.00–1.00] (0.99) | 1.000 [1.00–1.00] (1.00) |
| 1024 | 1024 | 0.950 [0.95–0.96] (0.68) | 0.997 [1.00–1.00] (0.98) | 0.998 [1.00–1.00] (0.98) |
| 2048 | 2048 | 0.219 [0.21–0.23] (0.00) | 0.993 [0.99–0.99] (0.94) | 0.390 [0.38–0.40] (0.00) |
| 4096 | 4096 | 0.019 [0.02–0.02] (0.00) | 0.972 [0.97–0.98] (0.79) | 0.025 [0.02–0.03] (0.00) |
| 8192 | 8192 | 0.009 [0.01–0.01] (0.00) | 0.919 [0.91–0.93] (0.50) | 0.011 [0.01–0.01] (0.00) |

## 4. Pass 3 — state_track control

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| seq_len | true tokens | transformer | gated_linear | hybrid |
|---:|---:|---|---|---|
| 128 | 136 | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) |
| 256 | 264 | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) |
| 512 | 520 | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) |
| 1024 | 1032 | 0.803 [0.78–0.83] (0.80) | 1.000 [1.00–1.00] (1.00) | 0.962 [0.95–0.97] (0.96) |
| 2048 | 2056 | 0.119 [0.10–0.14] (0.12) | 0.993 [0.99–1.00] (0.99) | 0.206 [0.18–0.23] (0.21) |
| 4096 | 4104 | 0.019 [0.01–0.03] (0.02) | 0.901 [0.88–0.92] (0.90) | 0.062 [0.05–0.08] (0.06) |
| 8192 | 8200 | 0.003 [0.00–0.01] (0.00) | 0.594 [0.56–0.62] (0.59) | 0.221 [0.20–0.25] (0.22) |

## 4b. Diagnostic — recall by key length (digits), same prompts as Pass 1

| n_bindings | key digits | transformer | gated_linear | hybrid |
|---:|---:|---:|---:|---:|
| 16 | 1 | 0.951 | 0.995 | 1.000 |
| 16 | 2 | 0.953 | 0.996 | 1.000 |
| 16 | 3 | 0.936 | 0.994 | 1.000 |
| 32 | 1 | 0.873 | 0.990 | 0.999 |
| 32 | 2 | 0.847 | 0.972 | 1.000 |
| 32 | 3 | 0.810 | 0.963 | 1.000 |
| 64 | 1 | 0.649 | 0.740 | 0.987 |
| 64 | 2 | 0.626 | 0.563 | 0.990 |
| 64 | 3 | 0.545 | 0.466 | 0.996 |

If one model's errors pile up on 3-digit keys, its drop with load is partly key resolution, not memory capacity.

## 5. Prefill cost (one forward pass, batch 1; mostly kernel quality)

_`cost_profile.csv` not synced yet._

## 5b. Decode cost (one token at a time, batch 1)

_`decode_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
