# iota

A custom linear-time architecture for bounded formal reasoning.

The repo exists to produce **one deliverable**: a graph of exact, verifier-checked
accuracy vs. recall-load / sequence-length, per architecture, with a cost axis
(VRAM / latency). It answers one question: *where does a cheap fixed-memory model stop
being correct, and what is the smallest fix that keeps it correct?*

> "On verifier-checked multi-query recall, pure linear attention holds dense-level
> accuracy up to ~N bindings, then degrades; a hybrid with k full-attention layers
> recovers it at ~W% of dense memory. On a single-accumulator control, linear matches
> dense, so the gap is specifically associative recall."

- [`BUILD_PLAN.md`](BUILD_PLAN.md): the original spec and guardrails
- [`PHASE6_EVAL_SPEC.md`](PHASE6_EVAL_SPEC.md): how the experiment is designed so the figure is interpretable
- [`KAGGLE_RUN_PLAN.md`](KAGGLE_RUN_PLAN.md): how to run it on a GPU
- [`PROGRESS.md`](PROGRESS.md): **what happened, what we learned, what's next** (start here)

## Status

| Phase | What | State |
|---|---|---|
| 0–2 | scaffold, task generator, independent oracle + verifier | ✅ 10,000/10,000 agreement |
| 3–5 | tokenizer (vocab 99), three models, training; transformer hits 98.8% on easy recall | ✅ |
| 6 §0 | all three architectures learn easy recall (0.95 / 0.94 / 1.00) | ✅ |
| 6–8 | sweep training, 3 eval passes, cost profiling, figure | 🟡 code done + CPU-verified; clean GPU run `r01` pending |
| 9 | Gradio demo | ⬜ waits for a trustworthy figure |

Earlier GPU attempts each exposed an experiment-design flaw. The worst was a
length/capacity confound that made linear *look* better than the transformer, along
with a control task stuck at chance. All are fixed; see [`PROGRESS.md`](PROGRESS.md).

## The three contenders

| model | idea | params |
|---|---|---|
| `transformer` | dense causal attention (SDPA/Flash), RoPE, the baseline | 2.13M |
| `gated_linear` | gated linear attention: a fixed-size decaying memory, chunk-parallel | 2.14M |
| `hybrid` | gated linear with 2 of 5 layers swapped for full attention | 2.66M |

## The tasks

**`assoc_recall`, the real test.** Facts are defined early and buried in noise, then
retrieved (several queries per prompt: MQAR). A fixed-size state must hold them all.
Each answer follows its query and is always 2 digits; the model is scored on the digits
after each `=` of a `GET`:

```
SET 0 = 25
SET 1 = 6
SET 2 = 91
DISTRACTOR mn gg zz
DISTRACTOR pp hh pp zz rr hh qx qx
GET 0 = 25
GET 1 = 06
```

**`state_track`, the control.** One running value; linear attention's home turf.
If linear matches dense here but not on recall, the gap is specifically recall.

```
START x = 10
x = ( x - 44 ) mod 97
DISTRACTOR pp hh tt
x = ( x + 88 ) mod 97
ANSWER x     → 54
```

Every answer is checked by an independent oracle (`iota/data/oracle.py`), never by
hand-written labels.

## Running it

**Locally (CPU):**

```bash
pip install -r requirements.txt pytest
python -m pytest -q                                   # 59 tests, ~20s
python tasks.py report                                # Phase 2 data gate (10k examples)
python -m scripts.run_all --stage all --smoke --no-gh # whole pipeline, tiny, ~4 min CPU
```

**On Kaggle (the real run):** open [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb),
add the `GH_TOKEN` and `HF_TOKEN` secrets, and run with `PLAN="smoke"`, then `"A"`, then `"B"`.
Details in [`KAGGLE_RUN_PLAN.md`](KAGGLE_RUN_PLAN.md).

**Where results go:** every stage publishes CSVs, logs, the figure and a `SUMMARY.md` to the
[`kaggle-results`](https://github.com/RohanBanerjee88/iota/tree/kaggle-results) branch under
`runs/<run_id>/`. Weights go to the HF Hub. Nothing needs to be copied out of Kaggle by hand.

## Repo map

```
iota/data/      dsl.py (generator) · oracle.py · verifier.py · tokenizer.py · dataset.py
iota/models/    base.py (SeqModel) · transformer.py · gated_linear.py · hybrid.py
iota/           train.py · eval.py (passes 1-3) · profile.py (cost) · plot.py (figure)
scripts/        run_all.py (resumable driver) · sync_results.py (→ kaggle-results)
                sanity_indist.py (per-mode gate) · push_to_hf.py · retrain_all.py (legacy)
configs/        sweep_*.yaml (the real run) · milestone_*.yaml (§0 gate) · tune_*.yaml (lr probes)
notebooks/      kaggle_train.ipynb
```
