# iota — progress log

The lab notebook. **Newest first.** Every Kaggle run gets an entry in the run log with
its outcome and what we changed because of it. The raw numbers for each run live on the
[`kaggle-results`](https://github.com/RohanBanerjee88/iota/tree/kaggle-results) branch
under `runs/<run_id>/SUMMARY.md`. This file holds the *decisions*.

## Where we are (2026-10-05)

**Done:** the deliverable figure exists, with three seeds and every run passing the sanity gate:
[`figures/seeds/money_figure_seeds.png`](figures/seeds/money_figure_seeds.png), with every number in
[`figures/seeds/SEEDS.md`](figures/seeds/SEEDS.md). The final-result entry (2026-10-05) below states what it shows
and its caveats. The paper audit (passes 5 and 6, three seeds) is in: the linear model's failure is load, not
distance, and teacher forcing does not inflate any headline number.

Pass 7 (2026-10-06) rejected the "asking a question damages a recurrent memory" hypothesis: no architecture
recalls worse after answered questions than after matched filler.

r07 (2026-10-07) showed the gate-init "bug" was load-bearing: with the configured init (γ₀ ≈ 0.9975) recall
collapses for GLA *and* the hybrid (32 facts: 0.88 → 0.36 and 1.00 → 0.36).

Pass 8 (2026-10-07) confirmed r07's gates stayed pinned near 1. The legacy init's models learned what r07's
couldn't: a fast-forgetting first layer, and layers that forget filler more than facts.

**Next:** run pass 9 (gate-clamp intervention, built) on r04–r07 to find which layers' forgetting recall needs.

## Run log

| run | date | code | plan | outcome | decision |
|---|---|---|---|---|---|
| pass 8 | 2026-10-07 | `4101680` | gate stats on r04–r07 | r07: every gate 0.98–1.00, bias 6 → 5.6–5.9; r04–r06: layer 0 forgets within a few tokens (γ 0.12–0.45), some layers forget filler > facts; hybrid's linear layers never hold facts to the question | saturation confirmed (with a correction); clamp intervention next |
| r07 | 2026-10-07 | `d5e5191` | A ✅ (+ pass 4) | intended gate init: GLA recall 0.88 → 0.36 @32, hybrid 1.00 → 0.36 @32; both plateau at assoc ≈ 0.37 by step ~4k; control 1.00; transformer = r04's (reused) | the legacy init is what made recall work; gate diagnostic next |
| audit | 2026-10-06 | `1a8050f` | pass 7 on r04–r06 | answered questions never hurt later recall (any arch); dense *gains* up to 0.07 from them; GLA within ±0.02; hybrid 1.00 throughout | drop the read-is-a-write hypothesis; r07 next |
| audit | 2026-10-05 | `28c0e1e` | passes 5, 6 on r04–r06 | GLA: load costs 0.55, 8× distance costs ≤0.05; dense/hybrid collapse past 1024 at any load; free-running ≈ teacher-forced (max gap 0.038, dense) | headline numbers stand; pass 7 next |
| r06 | 2026-10-05 | `a66cc35`/`5026708` | A ✅ (+ pass 4) → B ✅ | gate passes; 3-seed crossover: GLA > dense ≤32 (all seeds), dense > GLA @128 (all seeds) | final three-seed figure committed (`figures/seeds/`) |
| r05 | 2026-10-04 | `e713514` | A ✅ (+ pass 4) → B ✅ | seed swings up to 0.26 at high load; GLA@64 0.49→0.75 flips the order vs dense; hybrid robust | narrow headline; r06 = seed 2; r05 B for pass 2 |
| r04 | 2026-10-03 | `c1fadb5`/`f6b589e` | A ✅ → B ✅ | gate passes; matched size doesn't help the transformer; crossover at 32–64 bindings; hybrid 1.00 | B: full 4-panel figure; decode: GLA 0.33 MB flat vs dense 640 MB @64k | pass 4: transformer drop is load-driven, not key length | seed replicate / write-up / merge |
| r03 | 2026-10-02 | `e048745` | tune ✅ → A ✅ → B ✅ | conv fixed the transformer (recall 0.18 → 0.83); hybrid 1.00; GLA 0.23; control 1.00 for all | A: gate PASSES; capacity gap linear < dense at 64/128, hybrid ≈ 1.00 everywhere; GLA length-gen perfect B: GLA length-gen 0.94 at 8192 vs attention ~0.01; cost panel uninformative (prefill only) | r04 param-matched; add decode profiling |
| r02 | 2026-10-01 | `d218e42` | tune | control learned by all 3; hybrid recall 1.00, transformer 0.18, GLA 0.07 | pure attention can't learn multi-digit key matching → short conv for all (option 1) → r03 |
| r01 | 2026-09-30 | `7e1b8ef` | A (train + sanity + pass 1, 3) | ❌ gate failed: control at chance for all 3; transformer recall 0.30 in-distribution vs 0.998 GLA / 0.98 hybrid | don't run B; fix the 3 design flaws below, CPU-validate, then r02 |
| r01-smoke | 2026-09-30 | `1453249` | smoke | ✅ whole pipeline ran on a Kaggle T4 in ~90 s; all 8 publishes landed on `kaggle-results` | link works → launch plan A |

How to read a run: open `runs/<id>/SUMMARY.md` on the `kaggle-results` branch. Check the
**sanity gate first** (section 1b). If any model has `state` or `assoc` near 0.01, stop:
the figure is not trustworthy.

## Timeline

**2026-10-07 — Pass 9 built: gate-clamp intervention (eval only).**
`iota/clamp.py` replaces one GLA layer's gate with a constant per head at eval time and measures per-query recall
on pass-1-style prompts (8 / 32 / 64 facts, n = 500). Δ is measured against the unclamped model on the same
prompts, with a paired bootstrap CI over examples. Two clamps per layer, plus `keep` on all layers at once:
- **`keep`:** γ = σ(6) ≈ 0.9975, the configured init. Removes the layer's forgetting *and* its selectivity.
- **`mean`:** γ = the head's own average γ on these prompts. Keeps how much the layer forgets, removes what it
  chooses to forget.

Reading: a drop under `keep` only means the *amount* of forgetting matters; a drop under `mean` too means the
*selectivity* matters. On r07 every `keep` clamp should be a no-op (built-in control). Tests check that the clamp
imposes exactly the requested γ, touches no other layer, undoes itself on exit, is a no-op on an intended-init
model, and that `mean` uses each head's own mean. Notebook audit: `AUDIT_PASSES = "9"` over r04–r07.

**2026-10-07 — Pass 8 results: r07's gates are pinned near 1; the legacy models learned to forget.**
Data: `runs/r0{4,5,6,7}/pass8_gates.csv` on `kaggle-results`, 500 prompts, 32 facts (8 facts gives the same
picture). γ = mean ± std over tokens × heads. "Retention" = log10 of the share of a fact's write still in the
state at the first question (0 = all of it, −3 = a thousandth).

*GLA, r04 (legacy init) vs r07 (configured init):*

| layer | r04 γ fact value | r04 γ distractor | r04 retention | r07 γ fact value | r07 γ distractor | r07 retention |
|---|---|---|---:|---|---|---:|
| 0 | 0.40 ± 0.34 | 0.28 ± 0.21 | −141 | 1.000 ± 0.000 | 0.975 ± 0.059 | −1.2 |
| 1 | 1.000 ± 0.000 | 0.93 ± 0.14 | −3.7 | 0.997 ± 0.009 | 1.000 ± 0.001 | −0.09 |
| 2 | 1.000 | 1.000 | 0.00 | 0.999 | 1.000 | −0.11 |
| 3 | 0.997 ± 0.006 | 0.92 ± 0.10 | −3.8 | 0.992 ± 0.023 | 0.998 ± 0.010 | −0.56 |
| 4 | 1.000 | 1.000 | −0.01 | 0.985 ± 0.027 | 0.999 ± 0.002 | −0.35 |

- **Saturation confirmed, with a correction.** Every r07 gate, in both the GLA and the hybrid, sits at 0.98–1.00
  with std ≤ 0.06. The gate bias barely moved (6 → 5.6–5.9). But the gate *weights* did learn (‖W‖ 1.2–4.3 per
  head, the same size as r04's), so the gate is not literally frozen. Rather, a bias of ~6 keeps σ near 1 whatever
  W·x adds. The prediction "‖W‖ ≈ 0" was wrong; "γ ≈ 1 everywhere" was right.
- **r07 does not forget facts; it forgets nothing.** Facts keep 6–100% of their write until the question. Its
  failure is clutter: ~200 distractor tokens and every fact pile up in one state with no way to drop anything.
- **What all six legacy models learned (r04–r06, GLA and hybrid):**
  - **Layer 0 forgets fast:** γ 0.12–0.45, retention ≤ −140, so it only sees the last few tokens. It works like a
    second short convolution.
  - **Selective forgetting in some layers:** r04 GLA layers 1 and 3 keep facts at 1.00 but let distractors decay
    (0.93, 0.92). r06 GLA layer 1 shows the same (0.91 vs 0.77). In the hybrid, layer 0 keeps fact values longer
    than filler in all three seeds (0.25–0.31 vs 0.12–0.19).
  - **Long-memory layers:** GLA layers with γ ≈ 1.000 and retention ≈ 0, like r07's. r05 instead spreads γ over
    0.6–0.75, and its layer 3 has bimodal heads, some always 0 and some always 1. There is more than one solution.
- **Correction to the r07 entry:** in r04–r06, **no linear layer of the hybrid keeps facts until the question**
  (retention −33 to −195 in all three layers and all seeds). The hybrid's recall must come from its attention
  layers, and its linear layers do short-range processing. So the earlier hypothesis ("the hybrid's recall was
  carried by its GLA layers") is wrong. Revised: with the legacy init the hybrid's linear layers become local
  processors that feed attention; with the configured init they become uniform accumulators, and the hybrid's
  attention never learns recall. Why that blocks attention is open.

**Decision:** the mechanism is "the configured init pins γ near 1 and the model never learns to forget", not
"the model forgets facts". This is still correlational. Next: a gate-clamp intervention on r04 (eval only). Clamp
one layer's γ to σ(6) at a time, and measure the recall drop. That shows which layers' forgetting recall actually
needs.

**2026-10-07 — Pass 8 built: decay-gate statistics (tests the r07 mechanism, no training).**
`iota/gates.py` hooks every GLA layer's gate and records γ = σ(g_proj(x)) on pass-1-style prompts (8 and 32
facts, n = 500). It splits γ by token role: fact key, **fact value**, **distractor**, question, answer. For each
fact value it also records how much of the write survives to the first question (mean log10 of the product of
γ), plus each layer's gate bias and weight norm.
- **Predictions if the saturation story is right:**
  - r07: γ ≈ 0.9975 on every role, std ≈ 0, bias still ≈ 6, ‖W‖ near 0.
  - r04–r06: γ varies with the input, with γ(distractor) and γ(fact value) differing and a wide spread.
- Tests check the role labels against the dataset's answer spans, that a fresh intended-init model reads exactly
  σ(6) everywhere with std 0, that a fresh legacy model sits around 0.5 and varies, and the retention formula for a
  constant gate.
- Notebook `PLAN="audit"` with `AUDIT_PASSES = "8"` runs r04, r05, r06 (main configs) and r07 (`configs/fixinit`).

**2026-10-07 — r07 plan A: the configured gate init kills recall, for GLA and the hybrid.**
r07 is r04 (seed 0, same lr, data and schedule) with one change: the decay gate keeps its configured init (bias 6,
zero weight, so γ₀ ≈ 0.9975) instead of being overwritten (bias 0, random weight, γ₀ ≈ 0.5). The transformer is
r04's checkpoint, reused, and its numbers match r04 to the third decimal, which confirms the pairing.

| facts | GLA r04 → r07 | hybrid r04 → r07 | transformer (both) |
|---:|---|---|---|
| 8 | 0.989 → 0.854 | 1.000 → 0.848 | 0.975 |
| 16 | 0.964 → 0.718 | 1.000 → 0.717 | 0.945 |
| 32 | 0.882 → 0.355 | 1.000 → 0.357 | 0.842 |
| 64 | 0.489 → 0.082 | 0.997 → 0.113 | 0.620 |
| 128 | 0.222 → 0.045 | 0.990 → 0.060 | 0.347 |

- **Training curves (held-out assoc):** r07 GLA starts faster (0.10 vs 0.01 at step 500) but is flat at
  0.36–0.38 from step ~3.5k to the stop at 10.5k. r04 GLA starts slow and climbs to 0.84. r07 hybrid is flat at
  ~0.37 by step ~8k; r04 hybrid hit 0.98 by step 2.5k.
- **The control is unaffected:** state_track is 1.00 for every model. GLA holds 0.93 at 8192 tokens.
- **The hybrid falls to exactly the GLA curve** (0.357 vs 0.355 at 32 facts, 0.717 vs 0.718 at 16). Its two
  attention layers do not take over recall, even though a 5-layer transformer learns it to 0.84. Hypothesis: the
  hybrid's recall in r04–r06 was carried by its GLA layers, and the attention layers never learned a recall
  circuit of their own.
- **Likely mechanism (unverified): a saturated sigmoid gate.** At bias 6 the gate's gradient is σ'(6) ≈ 0.0025,
  101× smaller than σ'(0) = 0.25 at the legacy init. With zero weight it starts input-independent and can barely
  move, so γ stays ≈ 0.9975 everywhere. That gives near-uniform accumulation with no selective forgetting, and a
  hard ceiling on how many bindings the state can separate. The legacy init starts in the gate's responsive range
  and learns input-dependent forgetting. To check: measure γ per layer and per token type on the r04 vs r07
  checkpoints.
- **Caveats:** one seed (the effect is 0.5 at 32 facts, far outside r04–r06's seed spread); lr not re-tuned for
  the new init; plateau early stop. The plateau sits flat for ~7k steps, so more steps would not obviously help.

**Decision:** the "init bug" is a finding, not a footnote. Every r01–r06 result depends on the γ₀ ≈ 0.5 init,
and the documented init gives the opposite conclusion about linear-attention recall. The paper needs a
gate-init section. Next: a gate diagnostic to test the saturation mechanism (cheap, no training).

**2026-10-06 — Pass 7 on r04–r06: answering questions does not damage recall. Hypothesis dropped.**
Numbers: [`figures/seeds/SEEDS.md`](figures/seeds/SEEDS.md) and `figures/seeds/pass7_history.csv` (n = 500 per cell
per seed). Δ = condition − oracle, paired per item. The hypothesis predicted **neutral > oracle**, especially for GLA.

| model | 32 facts, 15 prior Qs: none* / oracle / neutral / generated | 64 facts, 15 prior Qs: oracle / neutral |
|---|---|---|
| GLA | 0.917 / 0.937 / 0.920 / 0.927 | 0.779 / 0.760 |
| dense | 0.815 / 0.899 / 0.825 / 0.847 | 0.697 / 0.661 |
| hybrid | 1.00 everywhere | 1.00 |

\* "none" = the 0-prior cell, which uses different items, so it is an unpaired reference.

- **Every sign is the wrong way for the hypothesis.** Correctly answered questions never hurt, in any architecture,
  cell or seed. For dense attention they *help*, by up to 0.074 (3/3 seeds' CIs exclude 0 at 32 facts / 15 Qs).
  For GLA, oracle − neutral is +0.00 to +0.02 and mostly inside the CIs. The hybrid sits at ceiling, so it is
  uninformative.
- **Order doesn't matter** (reordered within ±0.005). **The model's own answers** fall between oracle and neutral.
- **Why answered questions help: elimination, a training-distribution prior.** Training prompts never ask the same
  key twice (`rng.sample` in `dsl.py`), so a model can learn that an already-asked key is not the target. Split the
  errors into "copies the value of an asked key" vs "any other wrong value":
  - **Dense, 32 facts / 15 Qs:** copy-errors 0.08–0.12 → 0.02–0.03.
  - **GLA, same cell:** copy-errors 0.02–0.07 → 0.00–0.03.
  - **Other errors:** stay flat or rise by ~0.01–0.02 in **both** architectures, as expected if the remaining
    confusion just moves to the keys that are left.
- **No recurrent-specific damage.** Dense attention only appends on a question, yet it shows the same small rise in
  "other" errors as GLA. So the dense → hybrid → linear ladder gives no recurrent-specific signal.
- **Caveat:** elimination could in principle mask a small GLA write cost. Ruling that out cleanly would need a model
  trained with repeated-key queries. Given effects of ≤ 0.02, that is not worth a training run now.

**Decision** (the pre-registered rule): oracle-clean histories do not hurt beyond matched filler, so drop the
"queries write into memory and damage it" direction. Keep the eval (cheap, tested) as a negative result and as a
note on eval design: multi-query training teaches elimination, so sequential-use evals must control for it.
Next: r07.

**2026-10-05 — Audit results: passes 5 + 6 on r04–r06 (three seeds, n = 500 per cell).**
Every number: [`figures/seeds/SEEDS.md`](figures/seeds/SEEDS.md), `figures/seeds/pass5_grid.csv`,
`figures/seeds/pass6_freerun.csv`. Mean per-query accuracy [min–max over seeds].

*Pass 5, joint load × distance grid (8 queries, true length within ±11 tokens of the column):*

| bindings | GLA 1024 → 8192 | dense 1024 → 2048 | hybrid 1024 → 2048 |
|---:|---|---|---|
| 8 | 0.994 → 0.959 | 0.931 → 0.110 | 0.992 → 0.316 |
| 32 | 0.923 → 0.870 | 0.763 → 0.061 | 0.970 → 0.178 |
| 64 | 0.759 → 0.706 | 0.581 → 0.049 | 0.878 → 0.149 |
| 128 | 0.447 [0.34–0.54] → 0.412 | 0.379 [0.29–0.51] → 0.064 | 0.856 → 0.220 |

- **GLA: load and distance are nearly separable, and load dominates.** Going 8 → 128 bindings at 1024 tokens costs
  0.55. Going 1024 → 8192 tokens costs 0.035–0.053 at *every* load. What breaks the linear model is how many facts
  it holds, not how far back they are.
- **Dense and hybrid fail on length at every load.** Both are still fine at 1024 tokens (past the 640 they trained
  on), drop to 0.05–0.32 at 2048 and to ≤ 0.03 at 4096 and beyond. This is pass 2's result, now shown not to depend
  on load. The hybrid's recurrent layers do not rescue it: its attention layers set its length limit.
- **Caveat:** even the shortest column (1024) is past the training length, so every attention cell in this grid
  extrapolates. That is why the hybrid scores 0.856 at 128 bindings here but 0.984 in pass 1. At 1024 tokens and
  128 bindings, GLA (0.447) and dense (0.379) overlap across seeds, so the grid does **not** flip pass 1's
  "dense > GLA at 128". For an in-distribution load × distance panel we would need a ≤ 640 column (128 bindings
  alone take ~770 tokens, so that column could only go to 64 bindings).

*Pass 6, free-running vs teacher-forced (pass 1's prompts; the model's own answers stay in the context):*
- **Hybrid:** identical (gap 0.000 on every seed).
- **GLA:** gap ≤ 0.012 on any seed.
- **Dense:** loses the most: 0.949 → 0.913 at 16 bindings (worst single seed 0.038).
- So teacher forcing does not inflate any headline number, and the GLA is barely hurt by its own mistakes staying
  in context. That is a first hint for pass 7's *generated* condition: earlier errors are unlikely to be what
  damages linear recall.

*A hint that pass 7 tests properly:* at 64 bindings the GLA scores 0.60 in pass 1 (386 tokens, 16 queries) but
0.76 in pass 5 (1024 tokens, 8 queries), so it does better on a prompt 2.6× longer. Two things differ: the
number of queries and how densely the facts are packed. Pass 7 holds both fixed and varies only the history.

**Decision:** the paper's capacity and length claims stand as measured; describe the grid as an extrapolation
grid. Next: pass 7 on r04–r06 (notebook `PLAN="audit"`, `AUDIT_PASSES = "7"`), then r07.

**2026-10-05 — r07 prepared: the gate-init ablation, paired with r04.**
- `configs/fixinit/sweep_*.yaml` are r04's configs (seed 0, same lr, data, schedule and size) with one change:
  `legacy_gate_init: false` for GLA and hybrid. A test checks that the train block and fingerprint differ from
  the main configs in exactly that way. With the same seed every other weight starts identical and the data
  stream is the same, so r07 − r04 is a paired comparison.
- `IOTA_CONFIG_DIR=configs/fixinit` selects them (training, eval fallback, profile). The main configs are
  untouched, so the r04–r06 checkpoints stay valid for pass 7.
- The transformer has no gate, so its fixinit config matches r04's exactly, and `--reuse
  transformer=BanerjeeRohan44/iota-r04` copies r04's checkpoint into the r07 repo instead of retraining. Reuse refuses any checkpoint whose fingerprint doesn't match (tested).
- lr stays at the legacy-init tuned values: this is a one-factor ablation, not a re-tune. If the fixed init trains
  much worse at that lr, that is itself the finding, and a re-tune is the follow-up.
- Plan B now runs passes 2, 4 (n = 1000, as r04) and 5, 6, 7 (n = 500, as the audit), so every r04 number has an
  r07 partner.

**2026-10-05 — Pass 7 built: matched histories (paper direction, Stage 1 screen).**
Question: does answering earlier questions change later recall? Every query/answer token also updates a
recurrent state, so a "read" is a write. Design (`iota/history.py`): per item, a fixed fact prefix
(SET lines + distractors, exactly 512 tokens), a held-out target key, and a history region of exactly T
tokens before the same final query at the same absolute position:
- **neutral:** T distractor tokens
- **oracle:** n_prior other questions with correct answers
- **generated:** the same questions with the model's own greedy answers
- **reordered:** the oracle blocks permuted

Cells: 8/32/64 bindings × 0/3/7/15 prior questions (q < n), ≤ ~610 tokens total, so everything is
in-distribution. Metrics: final-answer greedy accuracy, true-answer log-prob, error-copies-a-history-value
rate, and paired Δ vs oracle with an item-bootstrap CI. Dense attention only appends on a question, the
hybrid partly overwrites and linear fully overwrites, so the three archs form a natural control ladder.
Tests check the invariants: identical prefix, length and final-query position; target never asked earlier;
reordered is a permutation; neutral has no questions; generated differs only inside answer spans.
`aggregate_seeds` reports per-seed Δ and how many seeds' CIs exclude 0. Decision rule (from the draft): if
oracle-clean histories don't hurt beyond matched filler, drop the "query writes damage memory" hypothesis.

**2026-10-05 — Correction: the GLA decay-gate init never took effect (found in the paper-draft audit).**
- `GatedLinearAttention.__init__` sets the gate bias to `decay_bias_init` (6.0 → γ ≈ 0.9975) with zero weight.
  `LMBackbone` then applies its generic init to *every* linear layer, which overwrites that with **bias 0 and
  random weight (γ = 0.5 at init)**. Confirmed by building the r04–r06 configs: every gate bias is 0.0.
- So **every linear and hybrid model in r01–r06 trained with γ₀ = 0.5**, not the configured 0.9975. The
  2026-06-19 entry's "decay-gate init fix" only ever reached the standalone layer that the unit test checks,
  never a full model. Whatever made GLA learn recall in June, it was not that init.
- The results stand as measurements, but must be described with the init they actually used. Configs and the
  paper's method section must not claim γ₀ ≈ 0.9975.
- **Fix, without rewriting history:** a new model key `legacy_gate_init`, defaulting to `true` (also when the key
  is absent), reproduces the historical behaviour bit-for-bit (verified: same seed → identical weights to old
  `main` for all three archs). `legacy_gate_init: false` applies the intended init. The checkpoint fingerprint
  treats a missing key as `true`, so r01–r06 checkpoints stay valid. Tests cover both paths.
- Planned ablation: retrain GLA and hybrid with `legacy_gate_init: false` (3 seeds) to measure whether γ₀
  changes the load/distance trade-off or the seed lottery.

**2026-10-05 — Audit evals for the paper, no retraining needed.**
- **Pass 5, joint load × distance grid:** {8, 32, 64, 128} bindings × {1024, 2048, 4096, 8192} tokens, with 8
  queries in every cell. Building it exposed a bookkeeping bias: `seq_len` budgets whitespace *words*, but numbers
  split into digit tokens, so more bindings mean more tokens (128 bindings at "1024" was ~1270 real tokens).
  Grid cells now regenerate until the true length is within ±11 tokens of target. (Pass 1's existing cells are
  unchanged and keep their recorded true lengths.)
- **Pass 6, free-running vs teacher-forced:** on Pass 1's prompts, the model's own answers stay in the context for
  later queries. Query 0 is identical under both protocols (tested); a model that memorised its data scores the
  same either way (tested).
- Notebook `PLAN="audit"` runs passes 5+6 on r04, r05, r06 in turn (wiping local checkpoints between runs).
  `aggregate_seeds` and `SUMMARY.md` tabulate both.
- Also verified for the draft: validation prompts, eval prompts and 60k sampled training prompts do not overlap
  (val∩eval = 0, train∩eval = 0).

**2026-10-05 — r06 plan B + the final three-seed figure (`figures/seeds/`).**
Pass 2 with three seeds: GLA recall **0.953 [0.92–0.99] at 8192**, 0.991 at 2048; transformer 0.111 / 0.010;
hybrid 0.316 / 0.012. Every run (r04, r05, r06) passed the sanity gate. `scripts/aggregate_seeds.py --runs r04
r05 r06` → `figures/seeds/` (committed): money_figure_seeds.png/pdf (mean, bands = seed range), SEEDS.md
(every per-seed value), aggregated CSVs.

**Final result (three seeds, ~2.67M params / 5 layers each):**
1. **Hybrid (2 of 5 attention layers) ≥ 0.97 recall at every load up to 128 bindings, in every seed**, at ~40%
   of dense decode memory/latency at 64k context.
2. **Linear vs dense cross over:** linear > dense up to 32 bindings in every seed; dense > linear at 128 in every
   seed (margin 0.01–0.13); 64 is the crossover zone.
3. **Length:** linear recall 0.95 at 12.8× training length in every seed; both RoPE-attention models collapse
   past ~1024.
4. **Decode cost:** linear keeps a constant 0.33 MB state at ~5.5 ms/token; dense keeps a 640 MB KV cache at
   56 ms/token at 64k context.

**Caveats:** single scale (~2.7M); the pure transformer plateaus at ~0.80–0.85 in-distribution (load-driven,
not key length, so a trainability observation at this scale); lr tuned at 4 layers; GLA still creeping up at
its 15000-step ceiling; the control's far-length behaviour (8192) is seed-dependent.

**2026-10-05 — r06 plan A (seed 2): with three seeds the capacity crossover is real.**
Gate passes (in-dist assoc: hybrid 1.00, GLA 0.946, transformer 0.797; control 1.00 for all).
Per-seed paired difference (GLA − transformer, per-query recall, same prompts):

| n_bindings | s0 | s1 | s2 | mean T / GLA / hybrid |
|---:|---:|---:|---:|---|
| 32 | +0.04 | +0.06 | +0.13 | 0.851 / 0.929 / 1.000 |
| 64 | −0.13 | +0.03 | −0.06 | 0.653 / 0.600 / 0.990 |
| 128 | −0.13 | −0.06 | −0.01 | 0.393 / 0.329 / 0.983 |

- **GLA > dense up to 32 bindings in every seed; dense > GLA at 128 in every seed (margin 0.01–0.13); 64 is
  the crossover zone (2 of 3 seeds favour dense).**
- **Hybrid ≥ 0.97 at every load in every seed.**
- **Control:** GLA ≥ 0.98 at 2048 in every seed vs ≤ 0.21 for both attention models; beyond that GLA varies
  (8192: 0.98 / 0.50 / 0.59).
- Pending: r06 plan B (pass 2), then the final three-seed figure.

**2026-10-05 — r05 plan B: recall length-generalisation holds on both seeds.**
Pass 2 (recall at 8 bindings), mean [range] over r04 + r05: GLA 0.991 [0.99–0.99] at 2048 and **0.970
[0.95–0.99] at 8192**; transformer 0.057 / 0.011, hybrid 0.278 / 0.013. Both RoPE models collapse past ~1024
in both seeds. The seed-dependence seen at 8192 (0.98 vs 0.50) is specific to the single-value *control*;
recall itself generalises in both seeds. Pass 4 repeats on seed 1. Decode cost is identical (architecture-only).

**2026-10-04 — r06 prepared (seed 2) + multi-seed figure.**
- `seed: 2` in all three configs: a third replicate (r04 = 0, r05 = 1, r06 = 2). Nothing else changes.
- `scripts/aggregate_seeds.py --runs r04 r05 r06` pulls each run's CSVs from `kaggle-results`, averages
  every eval cell across seeds, and draws the money figure with **bands = min–max across seeds**. The caption
  states the seed count per panel; decode cost is copied, not averaged (it is architecture-only). `SEEDS.md`
  lists every per-seed value next to the mean, so a disagreement is never hidden.
- Preview on r04 + r05: at 64 bindings the transformer [0.62–0.73] and GLA [0.49–0.75] bands overlap; the
  hybrid stays at [0.98–1.00].

**2026-10-04 — r05 plan A (seed-1 replicate of r04): seed variance is large; the headline must narrow.**

Paired with r04 (seed 0), same prompts, per-query recall:

| n_bindings | transformer s0 / s1 | gated_linear s0 / s1 | hybrid s0 / s1 |
|---:|---:|---:|---:|
| 16 | 0.945 / 0.949 | 0.964 / 0.983 | 1.00 / 1.00 |
| 32 | 0.842 / 0.871 | 0.882 / 0.935 | 1.00 / 1.00 |
| 64 | 0.620 / 0.728 | **0.489 / 0.754** | 0.997 / 0.982 |
| 128 | 0.347 / 0.512 | 0.222 / 0.456 | 0.990 / 0.973 |

Control vs length: GLA 1.00 / 0.98 at 2048 in both seeds, but 0.98 → **0.50** at 8192. Both attention models
collapse past ~1024 in both seeds (transformer at 1024: 0.955 / 0.214).

- **Seed swings reach 0.26 at high load** (GLA @ 64), bigger than the GLA-vs-dense gap. At 64 the order
  flips with the seed (s0: dense ahead by 0.13; s1: GLA ahead by 0.03).
- **Robust across both seeds:**
  - the hybrid holds ≥ 0.97 at every load
  - GLA ≥ dense up to 32 bindings
  - dense modestly ahead at 128
  - GLA ≫ attention at 2× training length
  - decode memory/latency (architecture-only, seed-independent)
- **Not robust:** "linear degrades faster than dense at 64", and "GLA length-generalises perfectly to 8192".
- Key-length (pass 4) pattern repeats: transformer ~10 points worse on 3-digit keys; GLA shows no 3-digit penalty.
- Next: a third seed (r06) to put a real spread on the 64/128 cells, and r05 plan B (pass 2) for paired length
  numbers. Decode cost does not need re-running (it depends on architecture only).

**2026-10-03 — r05 prepared: seed replicate of r04.** `seed: 1` in all three configs; nothing else changes.
The seed now picks the training-example stream as well as the weight init (before, every run read the same
examples, so a seed only changed the init). Seed 0 is byte-identical to before (tested), so r01–r04 stay
reproducible. The held-out eval set and every eval pass keep the same prompts, so r04 and r05 are paired.

**2026-10-03 — r04 pass 4 (key-length diagnostic): the key-resolution hypothesis is mostly rejected.**

| n_bindings | transformer 1 / 2 / 3-digit keys | gated_linear 1 / 2 / 3 | hybrid 1 / 2 / 3 |
|---:|---|---|---|
| 16 | 0.95 / 0.95 / 0.92 | 0.97 / 0.96 / 0.97 | 1.00 / 1.00 / 1.00 |
| 32 | 0.87 / 0.85 / 0.81 | 0.92 / 0.87 / 0.90 | 1.00 / 1.00 / 1.00 |
| 64 | 0.67 / 0.64 / 0.55 | 0.62 / 0.45 / 0.55 | 1.00 / 1.00 / 1.00 |

- 3-digit keys cost the transformer only 6–12 points. Even 1-digit keys, which the conv sees whole, fall
  from 0.95 → 0.67 with load, so its drop is genuinely load-driven, not a key-resolution artifact.
- GLA shows no 3-digit penalty; it is worst on 2-digit keys (70% of keys, the most look-alike neighbours).
  That is consistent with interference between similar keys in a fixed-size state (plausible, not proven).
- The hybrid is flat at ~1.00 for every key length and load.
- **Framing for the write-up:** the dense ceiling is a *trainability* observation, not a capability claim.
  At ~2.7M params and an identical budget, pure attention did not learn high-load recall fully; the hybrid
  did by step 4500. The literature shows attention *can* solve MQAR, so claim only what we measured.

**2026-10-03 — r04 plan B: the full four-panel figure. The first complete, defensible result.**

Decode (T4, batch 1; exact memory kept per sequence / median ms per token):

| context | transformer | hybrid | gated_linear |
|---:|---:|---:|---:|
| 1024 | 10.4 MB / 5.2 ms | 4.3 MB / 5.5 ms | 0.33 MB / 5.6 ms |
| 8192 | 80 MB / 8.2 ms | 32 MB / 7.0 ms | 0.33 MB / 5.5 ms |
| 65536 | 640 MB / 56 ms | 256 MB / 25 ms | 0.33 MB / 5.5 ms |

Below ~4k context every model sits at ~5 ms/token (per-step launch overhead dominates); above it the KV-cache
models grow linearly and GLA stays flat. Pass 2 confirms r03: GLA 0.95 at 8192, both RoPE models ~0.01.

**Headline (all models ~2.67M params / 5 layers, n=1000 per point):** gated linear attention matches dense up
to 32 bindings, then degrades faster (64: 0.49 vs 0.62; 128: 0.22 vs 0.35). A hybrid with 2 of 5 attention
layers holds ≥ 0.99 up to 128 bindings at ~40% of dense decode memory and ~45% of its per-token latency at
64k context. On the single-value control linear is never worse, it generalises to 12.8× its training length
where RoPE attention collapses, and it decodes with a constant 0.33 MB state, ~2000× smaller than dense's KV
cache at 64k context.

Pass 4 (key-length diagnostic) did not run (the notebook copy still had `--passes 2`); run it alone (~15 min).
Caveats for a write-up: single seed; lr tuned at 4 layers; GLA still creeping up at its 15000-step ceiling;
the pure transformer plateaus at ~0.81 in-distribution (pass 4 tests the key-resolution hypothesis).

**2026-10-03 — r04 plan A (depth/param-matched, 5 layers each): the size confound is gone; a clean
crossover at 32–64 bindings; the pure transformer still caps at ~0.81.**

| n_bindings | transformer r03 (4L) → r04 (5L) | gated_linear r03 → r04 | hybrid r04 |
|---:|---:|---:|---:|
| 8 | 0.979 → 0.975 | 0.985 → 0.989 | 1.000 |
| 16 | 0.952 → 0.945 | 0.958 → 0.964 | 1.000 |
| 32 | 0.875 → **0.842** | 0.866 → **0.882** | 1.000 |
| 64 | 0.736 → **0.620** | 0.591 → **0.489** | 0.997 |
| 128 | 0.521 → 0.347 | 0.255 → 0.222 | 0.990 |
| sanity (in-dist assoc) | 0.847 → 0.809 | 0.811 → 0.828 | 0.999 |

- **The hybrid's lead is architectural, not size:** a matched-size transformer (2.665M, 5 layers) is no
  better than the 4-layer one; the hybrid reaches 1.00 by step 4500.
- **Crossover:** GLA ≥ transformer up to 32 bindings (0.882 vs 0.842), transformer > GLA at 64 (0.62 vs
  0.49) and 128 (0.35 vs 0.22). Control: GLA 0.98–1.00 at every length; both RoPE models fail past ~1024.
- **Open:** the pure transformer plateaus at ~0.81–0.85 in-distribution at both sizes, while the
  literature has attention solving MQAR cleanly. Hypothesis: key resolution. Keys are 1–3 digits; a 4-tap
  conv cannot see a whole 3-digit key with its value, so partial matches collide more as more look-alike
  keys share the context. That looks like a capacity drop but isn't memory.
  **Test:** new pass 4 (`--passes 2,4`) splits Pass-1 recall by key digit count. It needs no retraining and
  runs with r04 plan B.

**2026-10-02 — Decode benchmark (fills the BUILD_PLAN §6 prefill/decode gap).**
- Every mixer now has a one-token `step` (attention: preallocated KV cache + RoPE at position t; GLA: one
  update of the (S, z) state; both: a rolling buffer for the short conv), plus `init_decode_cache` /
  `decode_step` on every model. No new weights or buffers, so existing checkpoints load unchanged. A test
  checks that token-by-token decoding reproduces the full forward to 1e-5 for all archs, with and without
  the conv (measured: ≤ 3e-7).
- `run_decode_profile` → `decode_profile.csv`: the exact bytes each model must keep, and the median ms per
  generated token, at context 128 … 65536 (synthetic context; cost depends on size, not contents). It runs
  inside the profile stage, so r04 plan B picks it up.
- CPU numbers (5-layer configs): at 32768 context, transformer 320 MB of KV cache, hybrid 128 MB (2 attention
  layers), **GLA 0.33 MB at every length**.
- Figure: panels C/D are now decode memory and decode latency (the old prefill panel is the fallback, and
  its numbers stay in SUMMARY.md).

**2026-10-02 — r03 plan B: length generalisation is the strongest result; the cost panel does not measure
what the thesis needs.**

Pass 2, recall at n_bindings = 8 vs length (trained ≤ 640 tokens):

| tokens | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---:|---:|---:|---:|---:|
| transformer | 0.980 | 0.953 | **0.242** | 0.015 | 0.010 |
| gated_linear | 0.988 | 0.987 | **0.985** | 0.972 | **0.942** |
| hybrid | 1.000 | 0.996 | **0.384** | 0.026 | 0.010 |

- **GLA keeps ~0.94–0.99 recall out to 12.8× its training length;** both RoPE-attention models collapse
  between 1024 and 2048 tokens. Same pattern as the Pass-3 control.
- **The cost panel (forward pass, batch 1) does not show "linear is cheaper".** VRAM is nearly identical
  (8192 tokens: 105 / 90 / 103 MB), because SDPA's memory-efficient kernel never materialises the T×T matrix.
  Latency favours dense (72 vs 313 ms), because dense uses a fused CUDA kernel and our GLA is a pure-PyTorch
  chunk loop. This measures *implementations*, not architectures. Linear attention's real advantage is
  **decode**: a constant-size state vs a KV cache that grows with context. BUILD_PLAN §6 asked for
  prefill and decode separately, and we only measure prefill. That is a known gap, now visible.

**2026-10-02 — r04 prepared: depth/param-matched.** Transformer and GLA go from 4 → 5 layers (2.665M / 2.670M
vs hybrid 2.668M). Data, conv, lr (tuned at 4 layers in r03), schedule and seed are unchanged, so depth/size
is the only difference from r03.

**2026-10-02 — r03 plan A: the first run to pass the sanity gate. A real (modest) linear-vs-dense gap, a
dominant hybrid, and a clean length result.**
Training (T4): hybrid hit 1.00 by step 6000 and stopped at 10000; the transformer **plateaued at ~0.84
recall from step 3000 to 14000**; GLA climbed slowly to 0.83 and was still creeping up (+0.002 per 1000
steps) when it hit the 15000 ceiling. Sanity (in-distribution): assoc 1.00 / 0.85 / 0.81
(hybrid / transformer / GLA), control 1.00 for all → gate passes.

Pass 1, capacity (per-query, n=1000, CI ≈ ±0.01):

| n_bindings | transformer | gated_linear | hybrid |
|---:|---:|---:|---:|
| 2 / 8 / 16 | 0.998 / 0.979 / 0.952 | 0.998 / 0.985 / 0.958 | 1.00 / 1.00 / 1.00 |
| 32 | 0.875 | 0.866 | 1.00 |
| 64 | **0.736** | **0.591** | 0.996 |
| 128 (extrapolated) | **0.521** | **0.255** | 0.989 |

Pass 3, control (state_track) vs length: **GLA 1.00 at every length up to 8192.** The transformer and hybrid
(RoPE attention) are 1.00 up to 512, then 0.83 / 0.77 at 1024, ~0.08 at 2048, ~0.01–0.2 beyond. They fail
past ~1.6× the 640-token training length.

What this supports:
- **Linear ≈ dense up to 32 bindings, then linear degrades faster** (−0.15 at 64, −0.27 at 128). That is the
  predicted direction, but a modest gap.
- **The hybrid (2 of 5 layers attention) beats everything at every load.**
- **The control rules out "linear is just worse":** on single-value memory, linear is never worse, and it is
  far better at length extrapolation, where RoPE attention collapses.

What it does NOT yet support, the open confound:
- **The dense baseline is weaker than the hybrid** (0.85 vs 1.00 in-distribution) **and smaller** (4 layers,
  2.14M params vs 5 layers, 2.67M). "Hybrid recovers dense-level accuracy" needs a param- and depth-matched
  comparison before it is a claim. The transformer's flat plateau (3000 → 14000) says more steps won't fix it.

**2026-10-02 — r03 lr probe (with short conv): the transformer is fixed; the recipe is locked.**

| arch | lr 7.5e-4 (assoc / state) | lr 1.5e-3 | chosen |
|---|---|---|---|
| transformer | 0.82 / 1.00 | **0.83 / 1.00** | 1.5e-3 |
| hybrid | **1.00 / 1.00** (step 3000) | 1.00 / 1.00 (step 3500) | 7.5e-4 (tie → reached 1.0 first) |
| gated_linear | 0.08 / 1.00 | **0.23 / 1.00** | 1.5e-3 |

- **The short conv fixed the dense baseline:** transformer recall went from 0.18 (r02, no conv) to 0.83 at the
  same budget. GLA also improved (0.07 → 0.23). The hybrid was already at 1.00 and still is.
- **The control is learned by all three at every lr.**
- **Recipe for plan A, identical protocol for all three:** the chosen lr, grad_clip 1.0, easy 1000 / ramp
  2000 (the probe's schedule), and one common 15000-step ceiling with plateau early-stop. This replaces the
  per-arch leftover schedules from earlier fixes.
- GLA's pooled recall (0.23) is dominated by high-load examples (up to 64 bindings, 16 queries). Whether that
  is a capacity limit or slow learning is exactly what Pass 1's per-n_bindings curve will show.

**2026-10-01 — Decision (option 1): short conv on for all three, re-tune as r03.**
- `short_conv: 4` in every sweep config (all mixers: attention and GLA). This is a scope change: the
  contender is now "gated linear attention + short conv" (the standard modern design), not "pure" linear.
  It is the same for all three, so the comparison stays fair.
- The checkpoint fingerprint now covers every model key, not just the curriculum, so a no-conv checkpoint
  can never be silently reused (tested).
- lr grid → {7.5e-4, 1.5e-3}: 3e-3 lost for every arch in r02. New run id **r03** (fresh HF repo), so the
  r02 no-conv probe stays a clean record. CPU smoke with conv passes end to end; 64 tests pass.

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

- ~~Cost panel is forward-pass only~~ → fixed: decode benchmark (2026-10-02).
- **One seed per model.** CIs cover eval sampling, not training variance. If the crossover
  is marginal, run a second seed before claiming it.
- **Per-arch lr differs** (GLA 3e-3 vs 1.5e-3). This is allowed by the fairness rule, but
  we need to state how each was tuned.
- **GLA gate init** (2026-10-05): every r01–r06 run used γ₀ = 0.5, not the configured 0.9975; ablation = r07: the configured init collapses recall (2026-10-07).
- **Phase 9 (Gradio demo)** not started. It waits for a trustworthy figure.
