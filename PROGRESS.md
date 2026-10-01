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

**Next action:** r01 failed the sanity gate (see the 2026-09-30 r01 entry). The three flaws are fixed and CPU-validated. Run r02 with `PLAN="tune"` first.

## Run log

| run | date | code | plan | outcome | decision |
|---|---|---|---|---|---|
| r02 | — | — | tune → A → B | pending: run `PLAN="tune"` | — |
| r01 | 2026-09-30 | `7e1b8ef` | A (train + sanity + pass 1, 3) | ❌ gate failed: control at chance for all 3; transformer recall 0.30 in-distribution vs 0.998 GLA / 0.98 hybrid | don't run B; fix the 3 design flaws below, CPU-validate, then r02 |
| r01-smoke | 2026-09-30 | `1453249` | smoke | ✅ whole pipeline ran on a Kaggle T4 in ~90 s; all 8 publishes landed on `kaggle-results` | link works → launch plan A |

How to read a run: open `runs/<id>/SUMMARY.md` on the `kaggle-results` branch. Check the
**sanity gate first** (section 1b). If any model has `state` or `assoc` near 0.01, stop:
the figure is not trustworthy.

## Timeline

**2026-10-01 — r02 lr probe (Kaggle T4, 3 h): the new control works; the pure transformer still can't learn recall.**

| arch | lr 7.5e-4 (assoc / state) | lr 1.5e-3 | lr 3e-3 |
|---|---|---|---|
| transformer | 0.10 / 1.00 | **0.18 / 0.99** | 0.02 / 0.68 |
| hybrid | **1.00 / 1.00** (by step 3000) | 1.00 / 1.00 | 0.05 / 0.95 |
| gated_linear | **0.07 / 1.00** | 0.04 / 1.00 | 0.02 / 1.00 |

- **The overwrite control is learned by all three** (GLA 1.00 at every lr). It is now a real control, and the
  "linear can't forget" risk did not materialise.
- **lr 3e-3 is worst for every arch**, which drops it from the grid.
- **The transformer has the r01 signature again.** Easy-phase loss reaches 0.10 (it learns 2-binding
  recall), then jumps to about 1.0–1.4 once length and load ramp up and stays there. The hybrid, with the
  *same* attention layers plus GLA layers in front, learns everything by step 3000.
  Hypothesis: multi-digit keys (`1 0 0`) need neighbouring tokens fused before attention can match them;
  GLA layers supply that local mixing and pure attention has to discover it through RoPE alone.
- **CPU test of the hypothesis** (d128, 2 layers, identical 3000-step budget): adding a 4-tap short causal
  conv (`short_conv: 4`, the Based/Mamba/Zoology ingredient) took transformer recall from 0.16 → **0.73** at
  step 1000 and 0.26 (final, no conv) → **0.83** by step 1500. `short_conv` now exists for every mixer
  (default off, causality-tested).
- GLA's recall stays low at 4000 steps. Whether that is slow learning or a real capacity limit can only be
  read from the per-n_bindings breakdown in Pass 1 after full training.

**2026-09-30 — r02 design: fix the three r01 flaws (approved), CPU-validated.**
- **Control → overwrite.** `state_track` now uses `ops_kinds: [set]` (`x = 88`): hold one value across
  distance and keep the latest, with no arithmetic. CPU probe (d128, 2 layers): transformer 0.01 → **1.00**
  on the control. Risk to watch: overwriting the same key needs a *learned forget* in linear attention. If
  GLA fails this control while acing recall, fall back to single-binding recall as the control.
- **Keys → random + shuffled** from [0,128). There is no counting shortcut, and every eval key has been
  seen in training. The CPU transformer's recall climbs steadily (0.14 → 0.32 by step 4000): harder
  (multi-digit key matching), not stuck.
- **Ceilings → n_bindings ≤ 64, n_queries ≤ 16.** The capacity pass is in-distribution to 64; only 128
  (~870 tok) extrapolates.
- **LR probe** (`run_all --stage tune`, notebook `PLAN="tune"`): every arch × lr {7.5e-4, 1.5e-3, 3e-3},
  identical 4000-step budget, grad_clip 1.0 for all (the transformer's 0.5 was a one-off). This yields
  `tune.csv`, from which each sweep config's lr is set. Estimated at ~3–3.5 h on a T4.
- The small CPU GLA learned nothing in 3000 steps (neither task). That fits its known slow start and is
  not informative; the full-size GPU probe decides.
- Pipeline smoke (with the tune stage) passes end to end on CPU; 62 tests pass; Phase 2 gate 10,000/10,000.

**2026-09-30 — r01 plan A: the pipeline works; the experiment design still has three flaws.**
Trained on a Kaggle T4 (hybrid 79 min, transformer 70 min, GLA 111 min). Sanity gate (in-distribution, n=500):

| model | assoc per-query | state (control) per-query |
|---|---:|---:|
| hybrid | 0.988 | 0.016 |
| transformer | **0.300** | 0.004 |
| gated_linear | 1.000 | 0.008 |

1. **The control is not a control.** `state_track` never rose above chance for *any* model, even in the
   easy phase (2 ops). The task asks for chained multi-digit modular arithmetic done silently
   (`10 − 22 + 54 mod 97 = 42`, digits split), which is an arithmetic test, not a state-holding test.
   So it cannot show "linear ≈ dense on single-slot memory".
2. **The dense baseline is broken, not beaten.** Transformer loss jumped from 0.35 to about 1.3 when the
   difficulty ramp started and never recovered. Its Pass-1 per-query accuracy is ≈2/n_bindings at every
   load (0.58, 0.29, 0.13, 0.065 at nb = 4, 8, 16, 32), the signature of picking among the bound values
   without doing the key→value lookup. The hybrid has the same attention layers and learned fine, so this
   is optimisation (recipe or grad_clip 0.5), not architecture.
3. **Pass 1 past 16 bindings measures extrapolation, not capacity.** Keys are always `0..n-1` in order,
   so "GET k" can be solved by counting to the k-th SET line. Training never sees keys ≥ 16 or more than
   6 queries. Both GLA *and* the full-attention hybrid fall off the same cliff right at the training edge
   (nb 16 → 32: 0.99 → 0.41 and 0.96 → 0.44). If it were state capacity, the hybrid would hold. GLA stores
   16 bindings perfectly, so its real capacity limit has not been reached yet.

No figure from r01. Plan B (length + cost) was not run: it would measure a broken baseline.

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
