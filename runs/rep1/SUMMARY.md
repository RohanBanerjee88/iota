# iota seed sweep `rep1`

- **updated:** 2026-11-23 00:51 UTC (last: `sweep nudge_s05 stopped`)
- **code:** `3249da9` on `claude/wonderful-ritchie-hbm5zn` · **device:** Tesla T4
- **on track** = held-out recall (assoc, full difficulty) ≥ 0.4 at the stop step. At 9k it matched the
  15k outcome in 23 of 24 full runs (PROGRESS.md 2026-10-27). Jobs are the full schedule stopped early.

## Per cell

| cell | lr | gate bias | jobs done | failed | on track | mean @1.5k | mean @3k | mean @last | median @last, on-track runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nudge | 0.0015 | 0.001 | 1 | 0 | 0/1 | 0.012 | 0.022 | 0.038 | – |

## Jobs

| job | status | last step | @1.5k | @3k | @6k | @last | on track | first step ≥ thresh | minutes | gpu |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| nudge_s05 | stopped | 9000 | 0.0118 | 0.022 | 0.0315 | 0.0376 | 0 |  | 80 | 0 |
