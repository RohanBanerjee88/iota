iota — Build Plan (spec for Claude Code)

This is the working spec. Read it before writing any code. The repo exists to
produce one deliverable: a graph of exact accuracy vs. recall-load/sequence-length,
per architecture, with a cost axis (VRAM / latency). Everything else is in service
of that graph. Do not build anything that does not move toward it.

# 0. Guardrails (read first, obey throughout)

* Smallest thing that works. Models are 1–3M params. No distributed training, no
  mixed-precision heroics, no config frameworks beyond plain YAML. Iteration speed
  beats sophistication.
* Everything is verifier-checked. No metric is ever based on hand-written labels. A
  Python verifier (and a symbolic oracle) is the source of truth.
* Determinism. Every script takes `--seed` (default 0) and seeds `random`, `numpy`,
  and `torch`. Data generation is reproducible from a seed.
* Runs on CPU for tests. All unit tests and a `--smoke` mode of every script must run
  on CPU in seconds. GPU is only for real training/profiling.
* Minimal deps. `torch`, `numpy`, `pyyaml`, `matplotlib`, `tqdm`.
  `flash-linear-attention` is OPTIONAL and isolated behind a try/except — never a
  hard dependency, never blocks a run.
* Do not run ahead. Build in the phase order below. Each phase has an acceptance
  test. Do not start the next phase until the current one passes. Stop after Phase 2
  in the first session and report.

# 1. Repo structure

```
iota/
  README.md
  BUILD_PLAN.md              # this file
  requirements.txt
  configs/
    tiny_transformer.yaml
    gated_linear.yaml
    hybrid.yaml
  iota/
    __init__.py
    data/
      dsl.py                 # task generator (BOTH modes — see §3)
      verifier.py            # executable correctness checker
      oracle.py              # symbolic solver (ground-truth answerer)
      tokenizer.py           # tiny char/word-level tokenizer
    models/
      base.py                # shared SeqModel interface (see §4)
      transformer.py         # dense baseline (FlashAttention if available)
      gated_linear.py        # the contender (see §5)
      hybrid.py              # mostly-linear + k full-attention layers
    train.py
    eval.py
    profile.py
    plot.py
  tests/
    test_dsl.py
    test_verifier.py
    test_oracle_agrees.py
  experiments/
    results/                 # json/csv per run
    plots/
  notebooks/
    kaggle_train.ipynb       # thin wrapper that calls train.py/eval.py
```

# 2. Phased build order with acceptance criteria

Phase 0 — Scaffold. Create the tree above with empty/stub files,
`requirements.txt`, a `seed_everything(seed)` util, and a `Makefile` or `tasks.py`
with `smoke`, `test`, `train`, `eval`, `profile`, `plot` targets. Done when:
`pytest -q` runs (even with 0 real tests) and `python -m iota.data.dsl --smoke`
prints a tiny example.

Phase 1 — Data generator (`dsl.py`) — the foundation, build this first.
Implements two task modes (§3). Difficulty is parameterized by `n_bindings`,
`distractor_density`, and `seq_len`. Done when: `gen(mode, n_bindings,
distractor_density, seq_len, seed)` returns `(prompt_str, target_str, meta)`, is
deterministic per seed, and `--smoke` prints one example of each mode.

Phase 2 — Verifier + symbolic oracle (`verifier.py`, `oracle.py`).
* `oracle.solve(prompt_str) -> answer_str`: parses the DSL and computes the correct
  answer exactly. This is ground truth.
* `verifier.check(prompt_str, predicted_str) -> bool`: recomputes via the oracle and
  compares. Done when (this is the gate to stop the first session):
  1. `tests/test_oracle_agrees.py` generates 10,000 examples across both modes and
     all difficulty levels; for every one, `oracle.solve(prompt) == target`. 100%
     agreement required.
  2. The verifier returns `True` for the oracle's answer and `False` for a
     deliberately corrupted answer, on every example.
  3. STOP and report. Print a summary: examples generated, agreement rate, a couple
     of sample prompts/answers per mode. Do not proceed to models yet.

Phase 3 — Tokenizer (`tokenizer.py`). Tiny deterministic vocab (digits, operators,
variable names, keywords, distractor tokens, pad/eos). `encode/decode` round-trip.
Done when: round-trip is lossless on 1,000 random examples; vocab size printed and
< ~100.

Phase 4 — Three models behind one interface (`base.py` + the three). All implement
the `SeqModel` interface in §4. Each is config-driven. Done when: each model does a
forward pass on a CPU smoke batch and returns logits of shape `(B, T, vocab)`; param
counts print and are in the 1–3M range for the default configs.

Phase 5 — Train (`train.py`). Config-driven, trains on short sequences only (train
`seq_len` 128–512, low–mid `n_bindings`). Next-token cross-entropy. Saves
`safetensors` + a run json. Done when: a tiny transformer reaches near-100%
exact-answer accuracy (verifier-checked) on the in-distribution eval split.

Phase 6 — Eval (`eval.py`). Loads a checkpoint, evaluates exact-answer accuracy
(verifier-checked) on a grid:
* sequence lengths: `128, 256, 512, 1024, 2048, 4096, 8192` (OOD beyond training)
* recall loads (`n_bindings`): a sweep, e.g. `2, 4, 8, 16, 32, 64`
* per cell: accuracy + a few captured failure examples. Writes tidy CSV: `model,
  mode, seq_len, n_bindings, distractor_density, accuracy, n`.

Phase 7 — Profile (`profile.py`). Per model, per sequence length: prefill latency,
decode latency, peak VRAM, tokens/sec. Profiling correctness is mandatory (see §6).
Writes CSV.

Phase 8 — The money graph (`plot.py`). Joins eval + profile CSVs:
* Primary: accuracy vs. recall-load (and vs. seq_len), one line per architecture.
* Secondary: cost (VRAM / latency) vs. seq_len, same architectures.
* Annotate the crossover.

Phase 9 — Demo. Gradio app: problem → model answers → verifier marks correct/incorrect
live → shows latency + VRAM vs the dense baseline. Target: HF Spaces free CPU Basic.

# 3. Task spec

Two modes. The discriminating axis is recall load (`n_bindings`), not sequence
length alone.

Mode A — `state_track` (control / sanity). A single accumulator updated step by step.

```
START x = 7
x = (x * 3 + 5) mod 97
DISTRACTOR qx lk mn        # ignored tokens, density-controlled
x = (x + 41) mod 97
ANSWER x
```

Target: final value of `x`.

Mode B — `assoc_recall` (the real test — MQAR-style). Define many variables early,
bury them under distractors, then require retrieval of a specific one (optionally
with one arithmetic op).

```
SET a = 14
SET b = 3
SET c = 91
... (n_bindings total) ...
DISTRACTOR ...
GET c                      # or: ANSWER (a + c) mod 97
```

Target: the requested binding's value (or the small op over two bindings).

Difficulty knobs (sweep all three): `n_bindings` (primary discriminator, 2→64),
`distractor_density`, `seq_len` (train short ≤512, eval long →8192). The generator
must guarantee, for Mode B, that the queried binding is defined early and the answer
genuinely depends on recalling it across distance.

# 4. `SeqModel` interface

```python
class SeqModel(nn.Module):
    def forward(self, tokens: LongTensor[B, T]) -> FloatTensor[B, T, vocab]: ...
    @classmethod
    def from_config(cls, cfg: dict) -> "SeqModel": ...
    def num_params(self) -> int: ...
```

`train.py`, `eval.py`, `profile.py` only ever touch this interface.

# 5. Gated linear attention spec

```
S_t = γ_t · S_{t-1} + φ(k_t) v_tᵀ        # d×d state
z_t = γ_t · z_{t-1} + φ(k_t)             # normalizer
o_t = (φ(q_t)ᵀ S_t) / (φ(q_t)ᵀ z_t + ε)
```

Critical gotcha — fair latency: implement chunk-parallel (intra-chunk in parallel,
carry state across chunks) so wall-clock scales linearly. Keep the simple recurrent
version as a `--reference` correctness check (small float tolerance; not bit-exact
with FLA either). Hybrid: `cfg.full_attention_layers` replaces those layers with
standard attention.

# 6. Profiling correctness (non-negotiable)

≥3 warmup iters; time with `torch.cuda.synchronize()` or CUDA events; VRAM via
`reset_peak_memory_stats()` then `max_memory_allocated()`; separate prefill from
decode; report median of ≥5 runs.

# 7. Compute / deployment

Train: Kaggle (primary), Colab (backup). Hosting: HF Hub (safetensors). Demo: HF
Spaces free CPU Basic. Cost target: $0.

# 8. First session = Phases 0 → 2 only

Hard stop after Phase 2's acceptance gate. Report results and sample data. Do not
write a single line of model code until the data layer is proven.

# Definition of done for the whole PoC

One figure: "On verifier-checked recall tasks, pure linear attention holds accuracy
up to ~N bindings / ~L tokens, then degrades; a hybrid with K full-attention layers
recovers dense-level accuracy at ~X% of dense memory/latency."
