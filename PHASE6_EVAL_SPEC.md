# iota — Phase 6 / 7 Eval Spec (the crossover experiment)

> The deliverable is one figure: **exact accuracy vs. difficulty, per architecture, overlaid with cost.**
> This spec exists so the figure is *interpretable* — so that when a line drops, we know *why*.

---

## 0. Pre-sweep gate (do this first, on CPU, before any Kaggle time)

Train **all three** architectures (`transformer`, `gated_linear`, `hybrid`) on the **same easy milestone slice**, **same budget** (same steps, optimizer, data seed). Require each to reach **≥95% exact-answer** on the easy in-distribution eval.

- All three should be **near-equal** here. The easy regime is where they must NOT diverge.
- If `gated_linear` lags the transformer *on the easy slice*, that's a training/implementation bug (decay-gate saturation, lr, init, normalizer) — **fix it on CPU before the sweep**, not after burning GPU hours.
- This converts the milestone from "pipeline works for one model" to "all three contenders are trainable," which is the actual precondition for a meaningful sweep.

---

## 1. Core principle — separate the two difficulty axes, and add a control

There are two *independent* reasons a model can fail, and they must never be swept together:

| Axis | Knob | What it tests | Who should fail |
|---|---|---|---|
| **Capacity** | `n_bindings` | Can a fixed-size state hold many bindings at once? | pure linear (MQAR limit) |
| **Length-gen** | `seq_len` (distractor padding) | Can it hold a binding across distance beyond training length? | possibly all, differently |
| **Control** | `state_track` mode | Single accumulator — linear attention's home turf | nobody (linear should match dense) |

The control is not optional. If linear **matches** the transformer on `state_track` but **loses** on `assoc_recall`, you've proven the gap is *specifically associative recall* — which rules out "our linear model is just worse at everything" and makes the whole result defensible.

---

## 2. Training (once per architecture, identical budget — fairness is non-negotiable)

- Train each model on a **mixed in-distribution curriculum**: `assoc_recall` (multi-query) + `state_track`, `n_bindings ∈ [2..16]`, `seq_len ≤ 512`.
- **Identical** steps, optimizer, schedule, and data seed across all three. Any budget asymmetry invalidates the comparison.
- Training ceiling on `n_bindings` is **16**; the sweep evaluates *beyond* that (to 128). This makes the capacity curve a test of **extrapolation** (architecturally determined), not memorization of seen difficulty.
- Save one checkpoint per architecture to `experiments/results/`.

---

## 3. Eval passes (all OOD relative to training)

**Pass 1 — Recall-capacity (the money curve).**
Mode `assoc_recall`, **multi-query**. Fix `seq_len ≈ 512` (near training length, so length-gen isn't a confound). Sweep:
```
n_bindings ∈ {2, 4, 8, 16, 32, 64, 128}
```
Expectation: transformer ~flat-high; pure linear degrades as `n_bindings` climbs past its state capacity; hybrid recovers most of the gap. **This is the headline.**

**Pass 2 — Length-generalization.**
Mode `assoc_recall`, fix `n_bindings = 8` (a level all models pass in-distribution). Sweep via distractor padding:
```
seq_len ∈ {128, 256, 512, 1024, 2048, 4096, 8192}
```
Train was ≤512, so ≥1024 is extrapolation. Isolates "holds the binding across distance."

**Pass 3 — Control.**
Mode `state_track`, sweep `seq_len` same grid as Pass 2. Expectation: **linear ≈ transformer**. This is the interpretability anchor.

---

## 4. Multi-query spec (the canonical MQAR stressor)

Single-query under-tests capacity — a model can survive long enough to answer one question without holding all bindings. So Pass 1 asks **K queries per example** (K = all bindings, or a random subset):

- Generator emits multiple `GET`/`ANSWER` queries over distinct, uniformly-chosen keys.
- Oracle returns the **ordered list** of correct answers.
- Metrics:
  - `accuracy_exact` — all K correct (all-or-nothing per example). **Primary.**
  - `accuracy_per_query` — fraction of individual queries correct. Secondary, for resolution.

---

## 5. Per-cell hygiene

- **Paired eval:** all three architectures see the **identical prompts** in each cell (fixed seed per cell), so differences are signal, not sampling noise.
- **Query position uniform** across all `n_bindings` (already implemented as `query_pos="uniform"`) — prevents the model from cheating by only learning recent or only-first bindings.
- `n = 500–1000` per cell; report a **Wilson or bootstrap CI** alongside the point estimate.
- Record **true tokenized length** (post digit-split) per example — that is the real x-axis, not nominal `seq_len`.
- Save ~5 **failure examples** per cell for qualitative inspection.
- **Identical early-stop / decoding rule** across architectures.

---

## 6. Cost (Phase 7 — joined to accuracy)

Per architecture, per `seq_len`: **prefill latency**, **decode latency**, **peak VRAM**, **tokens/sec**. Profiling hygiene from `BUILD_PLAN.md §6` is mandatory: ≥3 warmup iters, `torch.cuda.synchronize()` around timing, `reset_peak_memory_stats()` + `max_memory_allocated()`, median of ≥5 runs. A timing without warmup+synchronize is a bug, not a result.

Join cost CSV to eval CSV on `(model, seq_len)` for the overlay.

---

## 7. The deliverable figure + sentence

- **Panel A:** accuracy vs `n_bindings` (Pass 1), one line per architecture. Annotate where pure-linear crosses below dense, and where hybrid recovers.
- **Panel B:** cost (VRAM / prefill latency) vs `seq_len`, same architectures — dense ~quadratic, linear ~linear.
- **Panel C (small):** the `state_track` control — linear overlapping dense.

The result reads:

> "On verifier-checked multi-query recall, pure linear holds dense-level accuracy up to ~N bindings, then degrades to Z%; a hybrid with k full-attention layers recovers dense-level accuracy at ~W% of dense memory/latency. On the single-accumulator control, linear matches dense — so the gap is specifically associative recall, exactly as predicted. Here is where cheap reasoning stops being correct, and the minimal architecture that stays correct."

---

## 8. CSV schema (one row per eval cell)

```
model, mode, pass, seq_len_nominal, seq_len_true_tokens,
n_bindings, n_queries, distractor_density,
accuracy_exact, accuracy_per_query, ci_low, ci_high, n, seed
```

Cost CSV:
```
model, seq_len_nominal, seq_len_true_tokens,
prefill_ms, decode_ms_per_tok, peak_vram_mb, tokens_per_sec, n_runs
```

---

### One-line summary

Train all three identically on easy → sweep capacity and length **separately** → keep a `state_track` control → multi-query for the recall passes → paired prompts + CIs → overlay accuracy with cost. That's a publishable result, not a vibe.
