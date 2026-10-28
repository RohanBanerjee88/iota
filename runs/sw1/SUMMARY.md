# iota seed sweep `sw1`

- **updated:** 2026-10-28 02:15 UTC (last: `sweep bias0_s08 stopped`)
- **code:** `87bccab` on `claude/wonderful-ritchie-hbm5zn` · **device:** Tesla T4
- **on track** = held-out recall (assoc, full difficulty) ≥ 0.4 at the stop step. At 9k it matched the
  15k outcome in 23 of 24 full runs (PROGRESS.md 2026-10-27). Jobs are the full schedule stopped early.

## Per cell

| cell | lr | gate bias | jobs done | failed | on track | mean @1.5k | mean @3k | mean @last | median @last, on-track runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bias0 | 0.0015 | 0.0 | 4 | 0 | 4/4 | 0.061 | 0.267 | 0.775 | 0.834 |
| bias2 | 0.0015 | 2.0 | 3 | 0 | 3/3 | 0.057 | 0.129 | 0.556 | 0.532 |

Fisher exact (two-sided), on track bias0 vs bias2: **p = 1.000** (4/4 vs 3/3).
Mann-Whitney (two-sided), held-out recall at the stop step among on-track runs: p = 0.112 (n = 4 vs 3).

## Jobs

| job | status | last step | @1.5k | @3k | @6k | @last | on track | first step ≥ thresh | minutes | gpu |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| bias0_s05 | stopped | 9000 | 0.0121 | 0.3331 | 0.5804 | 0.8323 | 1 | 4000 | 83 | 0 |
| bias2_s05 | stopped | 9000 | 0.0698 | 0.102 | 0.1779 | 0.5212 | 1 | 8000 | 84 | 1 |
| bias0_s06 | stopped | 9000 | 0.1047 | 0.2401 | 0.4439 | 0.566 | 1 | 6000 | 83 | 0 |
| bias2_s06 | stopped | 9000 | 0.0285 | 0.1665 | 0.3308 | 0.6153 | 1 | 7000 | 84 | 1 |
| bias0_s07 | stopped | 9000 | 0.0281 | 0.1813 | 0.6271 | 0.835 | 1 | 5500 | 83 | 0 |
| bias2_s07 | stopped | 9000 | 0.0725 | 0.1191 | 0.1885 | 0.5322 | 1 | 7500 | 84 | 1 |
| bias0_s08 | stopped | 9000 | 0.0994 | 0.3115 | 0.7147 | 0.8665 | 1 | 4000 | 83 | 0 |
