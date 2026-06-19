# iota

A custom linear-time architecture for bounded formal reasoning.

The repo exists to produce **one deliverable**: a graph of exact, verifier-checked
accuracy vs. recall-load / sequence-length, per architecture, with a cost axis
(VRAM / latency). See [`BUILD_PLAN.md`](BUILD_PLAN.md) for the full working spec and
phase order.

## Status

**Phases 0–2 complete** (the data layer is proven; the first-session hard stop).

- **Phase 0 — scaffold**: repo tree, `seed_everything`, `tasks.py` runner.
- **Phase 1 — `iota/data/dsl.py`**: deterministic task generator, two modes
  (`state_track` control, `assoc_recall` MQAR-style recall test), parameterized by
  `n_bindings`, `distractor_density`, `seq_len`.
- **Phase 2 — `iota/data/oracle.py` + `iota/data/verifier.py`**: a symbolic
  interpreter that recomputes the exact answer, and a verifier that checks
  predictions against it. **Gate: 10,000 examples across both modes and all
  difficulty levels — 100% oracle/target agreement, verifier catches every
  corruption.**

Everything downstream trusts this layer, so no model code is written until it passes.

## Quickstart (CPU, seconds)

```bash
pip install numpy pytest          # data layer needs no torch
python -m iota.data.dsl --smoke   # one tiny example of each mode
python tasks.py test              # full suite incl. the 10k-example gate
python tasks.py report            # Phase 2 acceptance summary
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

## Next (later sessions)

Phase 3 tokenizer → Phase 4 three models (`transformer`, `gated_linear`, `hybrid`)
behind one `SeqModel` interface → train / eval / profile / the money graph. Do not
start a phase before the previous one's acceptance test passes.
