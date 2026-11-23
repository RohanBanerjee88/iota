# iota seed sweep `rep1`

- **updated:** 2026-11-23 03:32 UTC (last: `sweep nudge_s09 stopped`)
- **code:** `3249da9` on `claude/wonderful-ritchie-hbm5zn` · **device:** Tesla T4
- **on track** = held-out recall (assoc, full difficulty) ≥ 0.4 at the stop step. At 9k it matched the
  15k outcome in 23 of 24 full runs (PROGRESS.md 2026-10-27). Jobs are the full schedule stopped early.

## Per cell

| cell | lr | gate bias | jobs done | failed | on track | mean @1.5k | mean @3k | mean @last | median @last, on-track runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bias0 | 0.0015 | 0.0 | 2 | 0 | 2/2 | 0.056 | 0.322 | 0.849 | 0.849 |
| nudge | 0.0015 | 0.001 | 3 | 0 | 1/3 | 0.039 | 0.120 | 0.379 | 0.910 |

Fisher exact (two-sided), on track bias0 vs nudge: **p = 0.400** (2/2 vs 1/3).
Mann-Whitney (two-sided), held-out recall at the stop step among on-track runs: p = 0.540 (n = 2 vs 1).

## Jobs

| job | status | last step | @1.5k | @3k | @6k | @last | on track | first step ≥ thresh | minutes | gpu |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| bias0_s05 | stopped | 9000 | 0.0121 | 0.3331 | 0.5804 | 0.8323 | 1 | 4000 | 83 | 1 |
| nudge_s05 | stopped | 9000 | 0.0118 | 0.022 | 0.0315 | 0.0376 | 0 |  | 80 | 0 |
| bias0_s08 | stopped | 9000 | 0.0994 | 0.3115 | 0.7147 | 0.8665 | 1 | 4000 | 80 | 0 |
| nudge_s08 | stopped | 9000 | 0.0307 | 0.1149 | 0.1684 | 0.1885 | 0 |  | 83 | 1 |
| nudge_s09 | stopped | 9000 | 0.0755 | 0.2231 | 0.747 | 0.9101 | 1 | 4000 | 80 | 0 |
