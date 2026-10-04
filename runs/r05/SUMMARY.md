# iota run `r05`

- **updated:** 2026-10-04 06:32 UTC (last stage: `eval 2`)
- **code:** `e713514` on `main`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | 12500 | 0.926 | 0.853 | 1.000 | 0.618 | 0.0015 | ok |
| gated_linear | 15000 | 0.957 | 0.913 | 1.000 | 0.742 | 0.0015 | ok |
| hybrid | 4000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.00075 | ok |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `logs/train.log` and the `*_sweep.json` files.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| hybrid | 1.000 | 0.998 | 1.000 | 1.000 | pass |
| transformer | 0.849 | 0.420 | 1.000 | 1.000 | pass |
| gated_linear | 0.900 | 0.580 | 1.000 | 1.000 | pass |

n=500 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| n_bindings | true tokens | transformer | gated_linear | hybrid |
|---:|---:|---|---|---|
| 2 | 256 | 0.996 [0.99–1.00] (0.99) | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) |
| 4 | 256 | 0.990 [0.99–0.99] (0.96) | 0.998 [1.00–1.00] (0.99) | 1.000 [1.00–1.00] (1.00) |
| 8 | 256 | 0.978 [0.97–0.98] (0.83) | 0.994 [0.99–1.00] (0.96) | 1.000 [1.00–1.00] (1.00) |
| 16 | 256 | 0.949 [0.95–0.95] (0.42) | 0.983 [0.98–0.98] (0.77) | 1.000 [1.00–1.00] (1.00) |
| 32 | 289 | 0.871 [0.87–0.88] (0.10) | 0.935 [0.93–0.94] (0.36) | 1.000 [1.00–1.00] (1.00) |
| 64 | 386 | 0.728 [0.72–0.74] (0.00) | 0.754 [0.75–0.76] (0.00) | 0.982 [0.98–0.98] (0.72) |
| 128 | 773 | 0.512 [0.50–0.52] (0.00) | 0.456 [0.45–0.46] (0.00) | 0.973 [0.97–0.98] (0.65) |

## 3. Pass 2 — length generalisation

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| seq_len | true tokens | transformer | gated_linear | hybrid |
|---:|---:|---|---|---|
| 128 | 128 | 0.978 [0.97–0.98] (0.83) | 0.996 [0.99–1.00] (0.97) | 1.000 [1.00–1.00] (1.00) |
| 256 | 256 | 0.980 [0.98–0.98] (0.85) | 0.995 [0.99–1.00] (0.96) | 1.000 [1.00–1.00] (1.00) |
| 512 | 512 | 0.977 [0.97–0.98] (0.83) | 0.996 [0.99–1.00] (0.97) | 1.000 [1.00–1.00] (1.00) |
| 1024 | 1024 | 0.913 [0.91–0.92] (0.51) | 0.994 [0.99–1.00] (0.96) | 0.992 [0.99–0.99] (0.94) |
| 2048 | 2048 | 0.019 [0.02–0.02] (0.00) | 0.994 [0.99–1.00] (0.96) | 0.226 [0.22–0.23] (0.00) |
| 4096 | 4096 | 0.011 [0.01–0.01] (0.00) | 0.991 [0.99–0.99] (0.93) | 0.046 [0.04–0.05] (0.00) |
| 8192 | 8192 | 0.011 [0.01–0.01] (0.00) | 0.986 [0.98–0.99] (0.90) | 0.013 [0.01–0.02] (0.00) |

## 4. Pass 3 — state_track control

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| seq_len | true tokens | transformer | gated_linear | hybrid |
|---:|---:|---|---|---|
| 128 | 136 | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) | 0.998 [0.99–1.00] (1.00) |
| 256 | 264 | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) |
| 512 | 520 | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) | 0.999 [1.00–1.00] (1.00) |
| 1024 | 1032 | 0.214 [0.19–0.24] (0.21) | 1.000 [1.00–1.00] (1.00) | 0.760 [0.73–0.79] (0.76) |
| 2048 | 2056 | 0.111 [0.09–0.13] (0.11) | 0.981 [0.97–0.99] (0.98) | 0.062 [0.05–0.08] (0.06) |
| 4096 | 4104 | 0.011 [0.01–0.02] (0.01) | 0.802 [0.78–0.83] (0.80) | 0.011 [0.01–0.02] (0.01) |
| 8192 | 8200 | 0.161 [0.14–0.18] (0.16) | 0.495 [0.47–0.53] (0.49) | 0.072 [0.06–0.09] (0.07) |

## 4b. Diagnostic — recall by key length (digits), same prompts as Pass 1

| n_bindings | key digits | transformer | gated_linear | hybrid |
|---:|---:|---:|---:|---:|
| 16 | 1 | 0.954 | 0.991 | 1.000 |
| 16 | 2 | 0.953 | 0.980 | 1.000 |
| 16 | 3 | 0.937 | 0.989 | 1.000 |
| 32 | 1 | 0.894 | 0.957 | 1.000 |
| 32 | 2 | 0.877 | 0.928 | 1.000 |
| 32 | 3 | 0.844 | 0.950 | 1.000 |
| 64 | 1 | 0.777 | 0.811 | 0.982 |
| 64 | 2 | 0.745 | 0.734 | 0.981 |
| 64 | 3 | 0.660 | 0.798 | 0.987 |

If one model's errors pile up on 3-digit keys, its drop with load is partly key resolution, not memory capacity.

## 5. Prefill cost (one forward pass, batch 1; mostly kernel quality)

_`cost_profile.csv` not synced yet._

## 5b. Decode cost (one token at a time, batch 1)

_`decode_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
