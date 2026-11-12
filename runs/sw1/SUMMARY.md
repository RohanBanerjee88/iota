# iota seed sweep `sw1`

- **updated:** 2026-11-12 03:59 UTC (last: `sweep bias2_s16 stopped`)
- **code:** `ccef679` on `claude/wonderful-ritchie-hbm5zn` · **device:** Tesla T4
- **on track** = held-out recall (assoc, full difficulty) ≥ 0.4 at the stop step. At 9k it matched the
  15k outcome in 23 of 24 full runs (PROGRESS.md 2026-10-27). Jobs are the full schedule stopped early.

## Per cell

| cell | lr | gate bias | jobs done | failed | on track | mean @1.5k | mean @3k | mean @last | median @last, on-track runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bias0 | 0.0015 | 0.0 | 12 | 0 | 9/12 | 0.063 | 0.232 | 0.640 | 0.832 |
| bias2 | 0.0015 | 2.0 | 12 | 0 | 10/12 | 0.091 | 0.209 | 0.639 | 0.715 |

Fisher exact (two-sided), on track bias0 vs bias2: **p = 1.000** (9/12 vs 10/12).
Mann-Whitney (two-sided), held-out recall at the stop step among on-track runs: p = 0.775 (n = 9 vs 10).

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
| bias2_s08 | stopped | 9000 | 0.0296 | 0.0994 | 0.1233 | 0.1301 | 0 |  | 84 | 1 |
| bias0_s09 | stopped | 9000 | 0.0303 | 0.0994 | 0.162 | 0.187 | 0 |  | 83 | 0 |
| bias2_s09 | stopped | 9000 | 0.0789 | 0.1142 | 0.1764 | 0.2246 | 0 |  | 84 | 1 |
| bias0_s10 | stopped | 9000 | 0.0319 | 0.1627 | 0.4378 | 0.7083 | 1 | 6000 | 83 | 0 |
| bias2_s10 | stopped | 9000 | 0.1146 | 0.3539 | 0.8103 | 0.9568 | 1 | 3500 | 84 | 1 |
| bias0_s11 | stopped | 9000 | 0.0994 | 0.3323 | 0.4784 | 0.8001 | 1 | 4000 | 91 | 1 |
| bias2_s11 | stopped | 9000 | 0.1726 | 0.2872 | 0.4738 | 0.8156 | 1 | 4500 | 84 | 0 |
| bias0_s12 | stopped | 9000 | 0.0114 | 0.0239 | 0.1415 | 0.1772 | 0 |  | 84 | 0 |
| bias2_s12 | stopped | 9000 | 0.0728 | 0.0899 | 0.1684 | 0.4829 | 1 | 8500 | 92 | 1 |
| bias0_s13 | stopped | 9000 | 0.0463 | 0.3031 | 0.5804 | 0.8824 | 1 | 5000 | 91 | 1 |
| bias2_s13 | stopped | 9000 | 0.1502 | 0.2026 | 0.7807 | 0.9294 | 1 | 4000 | 84 | 0 |
| bias0_s14 | stopped | 9000 | 0.1108 | 0.3961 | 0.7443 | 0.9378 | 1 | 3500 | 84 | 0 |
| bias2_s14 | stopped | 9000 | 0.1055 | 0.4006 | 0.8782 | 0.9458 | 1 | 3000 | 91 | 1 |
| bias0_s15 | stopped | 9000 | 0.0448 | 0.1165 | 0.1608 | 0.2379 | 0 |  | 91 | 1 |
| bias2_s15 | stopped | 9000 | 0.1161 | 0.4272 | 0.8729 | 0.9488 | 1 | 3000 | 84 | 0 |
| bias0_s16 | stopped | 9000 | 0.1423 | 0.2891 | 0.3392 | 0.6548 | 1 | 8500 | 84 | 0 |
| bias2_s16 | stopped | 9000 | 0.0797 | 0.1415 | 0.352 | 0.5694 | 1 | 6500 | 92 | 1 |
