# iota seed sweep `sw1`

- **updated:** 2026-10-27 22:08 UTC (last: `sweep bias2_s05 stopped`)
- **code:** `87bccab` on `claude/wonderful-ritchie-hbm5zn` · **device:** Tesla T4
- **on track** = held-out recall (assoc, full difficulty) ≥ 0.4 at the stop step. At 9k it matched the
  15k outcome in 23 of 24 full runs (PROGRESS.md 2026-10-27). Jobs are the full schedule stopped early.

## Per cell

| cell | lr | gate bias | jobs done | failed | on track | mean @1.5k | mean @3k | mean @last | median @last, on-track runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bias0 | 0.0015 | 0.0 | 1 | 0 | 1/1 | 0.012 | 0.333 | 0.832 | 0.832 |
| bias2 | 0.0015 | 2.0 | 1 | 0 | 1/1 | 0.070 | 0.102 | 0.521 | 0.521 |

Fisher exact (two-sided), on track bias0 vs bias2: **p = 1.000** (1/1 vs 1/1).
Mann-Whitney (two-sided), held-out recall at the stop step among on-track runs: p = 1.000 (n = 1 vs 1).

## Jobs

| job | status | last step | @1.5k | @3k | @6k | @last | on track | first step ≥ thresh | minutes | gpu |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| bias0_s05 | stopped | 9000 | 0.0121 | 0.3331 | 0.5804 | 0.8323 | 1 | 4000 | 83 | 0 |
| bias2_s05 | stopped | 9000 | 0.0698 | 0.102 | 0.1779 | 0.5212 | 1 | 8000 | 84 | 1 |
