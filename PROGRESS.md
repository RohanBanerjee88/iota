# iota — progress log

The lab notebook. **Newest first.** Every Kaggle run gets an entry in the run log with
its outcome and what we changed because of it. The raw numbers for each run live on the
[`kaggle-results`](https://github.com/RohanBanerjee88/iota/tree/kaggle-results) branch
under `runs/<run_id>/SUMMARY.md`. This file holds the *decisions*.

## Where we are (2026-09-30)

All code for the deliverable is built and CPU-verified: data, verifier, three models,
training, the three eval passes, cost profiling and the figure. Earlier GPU attempts
(June–July) each exposed a flaw in the *experiment design*, not the code, and every one
has been fixed (see the timeline). **No trustworthy figure exists yet.** The next step is a
clean run `r01` from scratch on Kaggle, with results published automatically to the
`kaggle-results` branch.

**Next action:** run `notebooks/kaggle_train.ipynb` with `PLAN="smoke"`, then `"A"`, then `"B"`.

## Run log

| run | date | code | plan | outcome | decision |
|---|---|---|---|---|---|
| r01 | — | — | smoke → A → B | pending | — |

How to read a run: open `runs/<id>/SUMMARY.md` on the `kaggle-results` branch. Check the
**sanity gate first** (section 1b). If any model has `state` or `assoc` near 0.01, stop:
the figure is not trustworthy.

## Timeline

**2026-09-30 — Kaggle → GitHub link, fresh-run prep.**
- `scripts/sync_results.py`: after every stage (and after each model in training), the
  run's CSVs, run jsons, logs, figure and an auto-generated `SUMMARY.md` are committed to
  the `kaggle-results` branch. Kaggle and Claude's sandbox can both reach GitHub, while
  Claude cannot reach Kaggle or the HF Hub, so GitHub is the shared channel.
- `run_all --smoke`: the whole pipeline in minutes, into `experiments/smoke/`. It fails
  loudly if the GitHub link is broken.
- New `sanity` stage writes `sanity_indist.csv`, the per-mode gate.
- Fixed: Session B's plot stage dropped Session A's panels. Kaggle wipes the disk
  between sessions, and the CSVs were pushed to HF but never pulled back.
- Fixed: an eval OOM at batch 32 was recorded as an architecture result. The batch now
  halves down to 1 first.
- Profiling now matches the spec: 3 warmups, median of 5 synced runs.
- New notebook `notebooks/kaggle_train.ipynb` (the old one was a Phase-2 stub).
- 59 CPU tests pass; the Phase 2 data gate passes (10,000/10,000).

**2026-07-24 — Raise ceilings, resumable driver, figure + profiling.**
The capacity pass evaluates to 128 bindings / 16 queries, but models trained on ≤8 / ≤3.
A 16× extrapolation risked flattening all three models and erasing the crossover.
Raised to ≤16 bindings / ≤6 queries. Added `run_all.py`, which detects stale checkpoints
trained on an old curriculum. Built `plot.py`, `profile.py`, and the cluster-bootstrap CI
for the per-query metric.

**2026-07-16 — The inverted sweep (the big lesson).**
The first full sweep showed *linear beating the transformer*. The cause was a confound.
Recall with 128 bindings needs about 773 tokens, but models trained only to 256 tokens
and the capacity pass ran at 512. So attention models failed on **length extrapolation**
before capacity ever mattered. Fixes:
- train to seq 640 and hold the capacity pass at seq 256
- the `state_track` control had been **at chance the whole time**, hidden by the pooled
  metric; the trainer now reports and early-stops on the balanced per-mode mean

**2026-07-10 → 07-13 — Transformer instability.**
The transformer crashed at the difficulty ramp (0.486 → 0.156). Lowering lr to 1.2e-3
stopped the crash but plateaued at 0.59. The fix was lr 1.5e-3 with grad_clip 0.5. Also:
eval loads from weights and config if the run json is lost, and there is an
in-distribution sanity script.

**2026-06-20 — The sweep curriculum didn't train.**
The first GPU run left all three models at about 1%. The easy §0 gate had not covered
the full curriculum. Three causes:
- modular multiplication in the control (unlearnable at 2M params)
- a hard query/answer layout
- lr too high for the ramp

Fixes: an additive-only control, interleaved MQAR, an easy phase followed by a ramp, and
per-query early-stop. GLA needs lr 3e-3 and a long easy phase (it learns 3–5× slower), and
patience only counts after the ramp.

**2026-06-19 — Phases 3–5 + the §0 gate.**
- Tokenizer (vocab 99) and three models (2.1 / 2.1 / 2.7M params).
- The transformer reached 98.8% on easy recall.
- The gate: all three learn easy recall (0.95 / 0.94 / 1.00).
- Two real GLA bugs found: a backward NaN from `0*inf`, and a decay gate initialised so
  low it forgot everything.

**2026-06-18 — Phases 0–2.** Scaffold, DSL generator, independent oracle + verifier.
10,000/10,000 agreement.

## Known gaps / open questions

- **Cost panel is forward-pass only.** The spec asks for prefill and decode latency
  separately; `profile.py` measures one forward pass.
- **One seed per model.** CIs cover eval sampling, not training variance. If the crossover
  is marginal, run a second seed before claiming it.
- **Per-arch lr differs** (GLA 3e-3 vs 1.5e-3). This is allowed by the fairness rule, but
  we need to state how each was tuned.
- **Phase 9 (Gradio demo)** not started. It waits for a trustworthy figure.
