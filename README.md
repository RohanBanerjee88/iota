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
| 6–8 | sweep training, 3 eval passes, cost profiling, figure | ✅ r04: first complete figure (gate passes, size-matched models, decode cost) |
| 9 | Gradio demo | ⬜ waits for a trustworthy figure |

**Current result (two seeds, r04 + r05; all models ~2.67M params / 5 layers, n=1000 per point per seed):**
- **The hybrid** (2 of 5 layers attention) holds **≥ 0.97 recall at every load up to 128 bindings** in both
  seeds. That is the most robust finding.
- **Pure gated linear vs pure dense** are close and both degrade with load:
  - linear ≥ dense up to 32 bindings in both seeds
  - at 64 the order flips with the seed (0.49 vs 0.62; 0.75 vs 0.73)
  - dense is modestly ahead at 128
  - seed-to-seed swings reach 0.26 at high load, so a third seed is running before any finer claim
- **Length:** linear stays at ≥ 0.98 at 2× training length where RoPE attention collapses (≤ 0.2), in both seeds;
  further out (8192) it varies by seed (0.98 / 0.50).
- **Decode cost** (seed-independent): linear keeps a constant 0.33 MB state at 5.5 ms/token; dense keeps a
  640 MB KV cache at 56 ms/token at 64k context; the hybrid sits at ~40% of dense.

Figure (seed 0): [`runs/r04/money_figure.png`](https://github.com/RohanBanerjee88/iota/blob/kaggle-results/runs/r04/money_figure.png).
The pure transformer plateaus at ~0.81–0.85 in-distribution. A key-length breakdown shows that drop is load-driven,
so it is a trainability observation at this scale, not a claim that attention can't do the task.

Earlier GPU attempts each exposed an experiment-design flaw. The worst was a
length/capacity confound that made linear *look* better than the transformer, along
with a control task stuck at chance. All are fixed; see [`PROGRESS.md`](PROGRESS.md).

## The three contenders

| model | idea | params |
|---|---|---|
| `transformer` | dense causal attention (SDPA/Flash), RoPE, the baseline | 2.14M |
| `gated_linear` | gated linear attention: a fixed-size decaying memory, chunk-parallel | 2.14M |
| `hybrid` | gated linear with 2 of 5 layers swapped for full attention | 2.67M |

All three put a 4-tap short causal conv in front of every mixer (`short_conv: 4`). It fuses
neighbouring tokens so multi-digit keys can be matched; without it pure attention never learned
the recall task (r02). It is on for all three so the comparison stays fair.

## The tasks

**`assoc_recall`, the real test.** Facts are defined early and buried in noise, then
retrieved (several queries per prompt: MQAR). Keys are random numbers in shuffled order, so the
only way to answer is to match the key. A fixed-size state must hold them all.
Each answer follows its query and is always 2 digits; the model is scored on the digits
after each `=` of a `GET`:

```
SET 100 = 89
SET 26 = 33
SET 81 = 3
DISTRACTOR lk tt mn pp lk
DISTRACTOR hh zz gg zz
GET 81 = 03
GET 100 = 89
```

**`state_track`, the control.** One value, overwritten a few times across distance; keep the
latest. Linear attention's home turf. If linear matches dense here but not on recall, the gap is
specifically recall.

```
START x = 17
DISTRACTOR zz rr qx
x = 65
DISTRACTOR hh tt zz
x = 30
ANSWER x     → 30
```

Every answer is checked by an independent oracle (`iota/data/oracle.py`), never by
hand-written labels.

## Running it

**Locally (CPU):**

```bash
pip install -r requirements.txt pytest
python -m pytest -q                                   # 66 tests, ~20s
python tasks.py report                                # Phase 2 data gate (10k examples)
python -m scripts.run_all --stage all --smoke --no-gh # whole pipeline, tiny, ~4 min CPU
```

**On Kaggle (the real run):** open [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb),
add the `GH_TOKEN` and `HF_TOKEN` secrets, and run with `PLAN="smoke"`, then `"tune"`, `"A"`, `"B"`.
Details in [`KAGGLE_RUN_PLAN.md`](KAGGLE_RUN_PLAN.md).

**Where results go:** every stage publishes CSVs, logs, the figure and a `SUMMARY.md` to the
[`kaggle-results`](https://github.com/RohanBanerjee88/iota/tree/kaggle-results) branch under
`runs/<run_id>/`. Weights go to the HF Hub. Nothing needs to be copied out of Kaggle by hand.

## Repo map

```
iota/data/      dsl.py (generator) · oracle.py · verifier.py · tokenizer.py · dataset.py
iota/models/    base.py (SeqModel) · transformer.py · gated_linear.py · hybrid.py
iota/           train.py · eval.py (passes 1-3) · profile.py (prefill + decode cost) · plot.py (figure)
scripts/        run_all.py (resumable driver) · sync_results.py (→ kaggle-results)
                sanity_indist.py (per-mode gate) · push_to_hf.py · retrain_all.py (legacy)
configs/        sweep_*.yaml (the real run) · milestone_*.yaml (§0 gate) · tune_*.yaml (lr probes)
notebooks/      kaggle_train.ipynb
```
