# iota run `r08`

- **updated:** 2026-10-07 08:24 UTC (last stage: `eval 4`)
- **code:** `aa84c56` on `claude/wonderful-ritchie-hbm5zn`
- **device:** Tesla T4 · torch 2.10.0+cu128 · Kaggle Batch

## 1. Training (best checkpoint, full-difficulty held-out set)

| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| transformer | 14000 | 0.904 | 0.809 | 1.000 | 0.600 | 0.0015 | ok |
| gated_linear | 8000 | 0.597 | 0.194 | 1.000 | 0.362 | 0.0015 | ok |
| hybrid | 5000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.00075 | ok |

Healthy = assoc high **and** state well above chance (~0.01). Full per-eval curves are in `logs/train.log` and the `*_sweep.json` files.

## 1b. Sanity gate — each model on its own training distribution

| model | assoc per-query | assoc exact | state per-query | state exact | gate |
|---|---:|---:|---:|---:|---|
| hybrid | 0.999 | 0.994 | 1.000 | 1.000 | pass |
| transformer | 0.809 | 0.380 | 1.000 | 1.000 | pass |
| gated_linear | 0.164 | 0.034 | 1.000 | 1.000 | pass |

n=500 per mode. Both per-query columns must be well above chance (~0.01) before any sweep number means anything.

## 2. Pass 1 — capacity (the headline)

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| n_bindings | true tokens | transformer | gated_linear | hybrid |
|---:|---:|---|---|---|
| 2 | 256 | 0.997 [0.99–1.00] (0.99) | 0.851 [0.84–0.87] (0.70) | 1.000 [1.00–1.00] (1.00) |
| 4 | 256 | 0.991 [0.99–0.99] (0.96) | 0.705 [0.69–0.72] (0.20) | 1.000 [1.00–1.00] (1.00) |
| 8 | 256 | 0.975 [0.97–0.98] (0.82) | 0.531 [0.52–0.54] (0.00) | 1.000 [1.00–1.00] (1.00) |
| 16 | 256 | 0.945 [0.94–0.95] (0.43) | 0.360 [0.35–0.37] (0.00) | 1.000 [1.00–1.00] (0.99) |
| 32 | 289 | 0.842 [0.84–0.85] (0.05) | 0.143 [0.14–0.15] (0.00) | 1.000 [1.00–1.00] (0.99) |
| 64 | 386 | 0.620 [0.61–0.63] (0.00) | 0.079 [0.08–0.08] (0.00) | 0.988 [0.99–0.99] (0.81) |
| 128 | 773 | 0.347 [0.34–0.35] (0.00) | 0.049 [0.05–0.05] (0.00) | 0.986 [0.98–0.99] (0.80) |

## 3. Pass 2 — length generalisation

_`pass2_length.csv` not synced yet._

## 4. Pass 3 — state_track control

Per-query accuracy [95% CI], exact-all-queries in parentheses. n=1000 per cell, paired prompts.

| seq_len | true tokens | transformer | gated_linear | hybrid |
|---:|---:|---|---|---|
| 128 | 136 | 1.000 [1.00–1.00] (1.00) | 0.999 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) |
| 256 | 264 | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) |
| 512 | 520 | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) | 1.000 [1.00–1.00] (1.00) |
| 1024 | 1032 | 0.955 [0.94–0.97] (0.95) | 1.000 [1.00–1.00] (1.00) | 0.967 [0.96–0.98] (0.97) |
| 2048 | 2056 | 0.189 [0.17–0.21] (0.19) | 1.000 [1.00–1.00] (1.00) | 0.613 [0.58–0.64] (0.61) |
| 4096 | 4104 | 0.017 [0.01–0.03] (0.02) | 0.998 [0.99–1.00] (1.00) | 0.184 [0.16–0.21] (0.18) |
| 8192 | 8200 | 0.020 [0.01–0.03] (0.02) | 0.970 [0.96–0.98] (0.97) | 0.386 [0.36–0.42] (0.39) |

## 4b. Diagnostic — recall by key length (digits), same prompts as Pass 1

| n_bindings | key digits | transformer | gated_linear | hybrid |
|---:|---:|---:|---:|---:|
| 16 | 1 | 0.948 | 0.695 | 0.998 |
| 16 | 2 | 0.952 | 0.286 | 1.000 |
| 16 | 3 | 0.920 | 0.489 | 1.000 |
| 32 | 1 | 0.874 | 0.431 | 0.998 |
| 32 | 2 | 0.848 | 0.088 | 1.000 |
| 32 | 3 | 0.808 | 0.215 | 1.000 |
| 64 | 1 | 0.674 | 0.223 | 0.985 |
| 64 | 2 | 0.638 | 0.057 | 0.987 |
| 64 | 3 | 0.545 | 0.100 | 0.991 |

If one model's errors pile up on 3-digit keys, its drop with load is partly key resolution, not memory capacity.

## 5. Prefill cost (one forward pass, batch 1; mostly kernel quality)

_`cost_profile.csv` not synced yet._

## 5b. Decode cost (one token at a time, batch 1)

_`decode_profile.csv` not synced yet._

## 6. Figure

_Not plotted yet._
