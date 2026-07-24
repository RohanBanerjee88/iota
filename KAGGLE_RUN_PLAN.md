# iota — Kaggle Sweep Run Plan (Phase 6/7 on GPU)

> Pipeline is CPU-validated end to end; §0 gate passed for all three architectures. Nothing here is debugging — every GPU minute is experiment. Goal: produce the crossover figure (accuracy vs. capacity / length, overlaid with cost).

---

## 0. The fairness rule that governs everything (read first)

"Identical budget" does **NOT** mean identical step count. GLA groks recall ~3–5× slower than the transformer (its own §0 curve: flat to ~step 1000, clean by ~2000; transformer done by ~500). If you fix steps, GLA is undertrained and the capacity curve is contaminated — a reviewer says "you just undertrained the contender" and the result dies.

**Fair = same data, same optimizer family, same tuning effort, each trained to its own plateau under a common generous ceiling.** Concretely:
- Common max-steps ceiling (generous, sized for the slowest learner — GLA).
- Early-stop on a **large fixed held-out set (n≥500)**, patience-based, identical rule for all three.
- Report the plateau accuracy, not the step-count-matched accuracy.
- Same LR schedule shape; per-arch LR is allowed to differ if tuned by the same protocol (note it).

---

## 1. Training config for the sweep (one model per architecture)

| Knob | Value | Why |
|---|---|---|
| Params | 2–3M each (2.13 / 2.14 / 2.66M) | tiny, fast, fits free T4 easily |
| Train task mix | `assoc_recall` (multi-query) + `state_track` | recall is the test, state_track is the control |
| **Training ceiling `n_bindings ≤ 16`, `n_queries ≤ 6`** | **hard rule** | eval extrapolates to 128 bindings / 16 queries → the curve measures architectural capacity, not memorized difficulty. Raised from 8/3: a 16× binding extrapolation risked flattening *all three* models and erasing the crossover. |
| Train `seq_len ≤ 640` | hard rule | eval extrapolates to 8192 → measures length-gen. 640 covers the capacity sweep's longest cell (nb=128 ⇒ ~773 tok) at a mild 1.2× RoPE extrapolation, so **capacity is not confounded with length** — the flaw that inverted the first sweep. |
| Max-steps ceiling | ~8–10k (GLA-sized) | transformer/hybrid early-stop well before; GLA uses it |
| Early-stop | n≥500 held-out, patience ~5 evals | the fixed-large-n rule that avoids the §0 early-stop artifact |
| Precision | fp32 first run | correctness over speed; revisit bf16 only if time-bound |
| Seed | fixed, logged | reproducibility |

---

## 2. The three eval passes (from PHASE6_EVAL_SPEC.md)

| Pass | Mode | Fixed | Swept |
|---|---|---|---|
| **1 — capacity (headline)** | assoc_recall, multi-query | seq_len=256 (in-distribution; true_len stays ~245–370 for nb ≤ 64) | n_bindings {2,4,8,16,32,64,128} |
| **2 — length-gen** | assoc_recall | n_bindings=8 | seq_len {128…8192} |
| **3 — control** | state_track | n_bindings n/a | seq_len {128…8192} |

Per cell: paired prompts across all 3 archs (same seed), n=500–1000, Wilson/bootstrap CI, true tokenized length recorded, ~5 failure examples saved.

---

## 3. How to run it (Kaggle: 9-hr sessions, ~30 GPU-hr/week)

Everything goes through one resumable driver, `scripts/run_all.py`. **Every stage is safe to re-run** — completed work is skipped, so a session timeout costs only the in-progress model.

**Session A — train + Pass 1 + Pass 3 (cheap, decisive).**
```bash
git pull origin main
python -m scripts.run_all --stage train --repo <user>/iota-sweep   # HF backup after each model
python -m scripts.run_all --stage eval  --passes 1,3
```

**Session B — Pass 2 (length-gen) + cost profiling + the figure.**
```bash
python -m scripts.run_all --stage eval --passes 2   # dense may OOM at 8192 — that's a result
python -m scripts.run_all --stage profile
python -m scripts.run_all --stage plot
```

Useful at any time:
```bash
python -m scripts.run_all --stage status      # what's trained / stale / missing
python -m scripts.run_all --only gated_linear --stage train   # finish one model
python -m scripts.sanity_indist               # per-mode in-distribution check
```

**Resume logic.** A checkpoint counts as done only if the weights exist, its *saved curriculum matches the current config*, and its history shows real training. The curriculum check is what stops an old checkpoint (trained at the superseded `seq_len ≤ 256` / `n_bindings ≤ 8` ceilings) from being silently reused and reproducing the broken sweep — those are reported `stale` and retrained.

**The gate before any figure.** The train summary prints per-mode accuracy. Every model must show `assoc` high **and** `state` well above chance (~0.01). A control at chance means the figure is not trustworthy yet — this was the failure the pooled per-query metric hid the first time round.

---

## 4. Kaggle gotchas

- **Persist before timeout** — push checkpoints + CSVs to HF Hub *during* the run.
- **Enable GPU** and confirm `torch.cuda.is_available()` in cell 1.
- **Pin threads** to avoid oversubscription in the data/eval loops.
- **`flash-linear-attention` stays optional** — never blocks the run.
- **dense at 8192** may OOM on a 16GB T4 — expected; record the length where it dies.
- **Greedy-gen eval is slow for GLA** — the sweep uses teacher-forced exact-match (one forward pass), which is *identical* to greedy for these query→answer tasks and sidesteps the per-token scan.

---

## 5. Definition of done for the GPU phase

A single figure where every clause has numbers + CIs:
> "Multi-query recall: pure linear holds dense-level accuracy to ~N bindings, then falls to Z%; hybrid recovers dense-level accuracy at ~W% of dense VRAM/latency. On the state_track control, linear matches dense — so the gap is specifically associative recall. Dense OOMs/spikes at L tokens where linear stays flat."
