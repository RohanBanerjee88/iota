# iota

A custom linear-time architecture for bounded formal reasoning.

The repo exists to produce **one deliverable**: a graph of exact, verifier-checked
accuracy vs. recall-load / sequence-length, per architecture, with a cost axis
(VRAM / latency). See [`BUILD_PLAN.md`](BUILD_PLAN.md) for the full working spec and
phase order.

## Status

**Phases 0–5 complete** — pipeline validated end to end (data → tokenizer → model
→ train → verifier-eval). Stopped at the Phase 5 milestone; the full
length/recall sweep (Phase 6) is next.

- **Phase 0 — scaffold**: repo tree, `seed_everything`, `tasks.py` runner.
- **Phase 1 — `iota/data/dsl.py`**: deterministic task generator, two modes
  (`state_track` control, `assoc_recall` MQAR-style recall test).
- **Phase 2 — `oracle.py` + `verifier.py`**: an independent recursive-descent
  interpreter and a verifier. **Gate: 10,000 examples, 100% oracle/target
  agreement, verifier catches every corruption.** Plus 10 hand-authored
  adversarial cases (unusual spacing, nested parens, negative intermediates,
  mod-boundary values, unparenthesized precedence) proving oracle independence.
- **Phase 3 — `tokenizer.py`**: word/keyword level, numbers digit-split →
  **vocab = 99 (< 100)**, lossless round-trip. Exposes the *true* tokenized
  length (post digit-split), threaded through dataset/eval so real token length
  (not nominal `seq_len`) is the x-axis.
- **Phase 4 — `models/`**: `transformer` (dense, FlashAttention via SDPA),
  `gated_linear` (chunk-parallel GLA + recurrent reference check), `hybrid`
  (mostly-linear + k full-attention layers) behind one `SeqModel` interface.
  Defaults land at **2.1 / 2.1 / 2.7 M params**. The chunked GLA matches its
  recurrent reference to 1e-4 across chunk sizes.
- **Phase 5 — `train.py` + `eval.py`**: config-driven, masked next-token CE,
  verifier-checked exact-answer eval, safetensors + run-json checkpoints.
  **Milestone: the 2.1M dense transformer reaches 98.8% exact-answer accuracy
  (n=1000 held-out) on the in-distribution `assoc_recall` slice**, early-stopping
  at ~500 steps (~3.5 min on 4 CPU cores). This validates the whole pipeline.

> Milestone scope: in-distribution = short sequences, low recall load, pure-recall
> (`GET`) queries — the cleanest pipeline validation. Arithmetic-op queries,
> `state_track`, and the length/recall sweep are Phase 6.

## Quickstart

```bash
pip install -r requirements.txt
python -m iota.data.dsl --smoke           # one tiny example of each mode
python -m iota.data.tokenizer             # vocab size + round-trip check
python tasks.py test                      # full suite (CPU, ~10s)
python tasks.py report                    # Phase 2 acceptance summary
python -m iota.train --config configs/milestone_transformer.yaml --smoke   # pipeline smoke
python tasks.py train                     # reproduce the Phase 5 milestone (~3.5 min CPU)
```

## Task modes

**Mode A — `state_track`** (control). A single accumulator carried across distance:

```
START x = 7
x = ( x * 3 + 5 ) mod 97
DISTRACTOR qx lk mn
x = ( x + 41 ) mod 97
ANSWER x
```

**Mode B — `assoc_recall`** (the real test). Many bindings defined early, buried
under distractors, then one is retrieved — a fixed-size recurrent state must hold all
bindings at once, so linear-attention accuracy is expected to degrade as
`n_bindings` grows while a transformer holds:

```
SET v0 = 14
SET v1 = 3
...
DISTRACTOR ...
GET v0            # or: ANSWER ( v0 + v1 ) mod 97
```

## Next — Phase 6 (the sweep) and carry-forwards

Phase 6 evaluates exact accuracy across the length grid (128→8192, OOD beyond
training) and recall-load grid (`n_bindings` 2→64) for all three models, writing a
tidy CSV. Two carry-forwards are already hooked in `dsl.py` (see the TODO there):

- `assoc_recall` eval must query keys **uniformly** across all `n_bindings`
  (`query_pos="uniform"`, implemented) rather than only the early half.
- Support a **multi-query** variant (several `GET`s per prompt, each verified) —
  generation + oracle support is still TODO.

Then Phase 7 profile (latency/VRAM), Phase 8 the money graph, Phase 9 demo. Do not
start a phase before the previous one's acceptance test passes.
