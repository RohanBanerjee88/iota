"""Task runner (plain stdlib, no framework).

Usage:
    python tasks.py smoke     # tiny end-to-end data demo (CPU, seconds)
    python tasks.py test      # pytest -q
    python tasks.py report    # Phase 2 acceptance summary (10k examples)
    python tasks.py train     # Phase 5+ (not built in the first session)
    python tasks.py eval
    python tasks.py profile
    python tasks.py plot
"""

from __future__ import annotations

import subprocess
import sys


def smoke() -> int:
    from iota.data import dsl, oracle, verifier

    print("== DSL smoke (one example per mode) ==\n")
    dsl._smoke(seed=0)
    print("== oracle/verifier round-trip ==")
    for mode in dsl.MODES:
        prompt, target, _ = dsl.gen(mode, 6, 0.3, 96, seed=0)
        ans = oracle.solve(prompt)
        ok = verifier.check(prompt, ans)
        print(f"  {mode:13s} target={target:>2s} oracle={ans:>2s} verifier={ok}")
    return 0


def test() -> int:
    return subprocess.call([sys.executable, "-m", "pytest", "-q"])


def report() -> int:
    """Phase 2 acceptance summary: 10k examples, agreement rate, samples."""
    from iota.data import dsl, oracle, verifier

    modes = dsl.MODES
    n_bindings = [2, 4, 8, 16, 32, 64]
    densities = [0.0, 0.3, 0.6]
    seq_lens = [128, 256, 512]
    target_n = 10_000

    n = agree = corrupt_caught = 0
    mismatches = []
    seed = 0
    done = False
    while not done:
        for mode in modes:
            for nb in n_bindings:
                for dens in densities:
                    for sl in seq_lens:
                        prompt, tgt, _ = dsl.gen(mode, nb, dens, sl, seed)
                        ans = oracle.solve(prompt)
                        n += 1
                        if ans == tgt:
                            agree += 1
                        elif len(mismatches) < 5:
                            mismatches.append((mode, nb, dens, sl, seed, tgt, ans))
                        if verifier.check(prompt, tgt) and not verifier.check(
                            prompt, str((int(tgt) + 1) % dsl.MOD)
                        ):
                            corrupt_caught += 1
                        if n >= target_n:
                            done = True
                            break
                    if done:
                        break
                if done:
                    break
            if done:
                break
        seed += 1

    print("=" * 64)
    print("PHASE 2 ACCEPTANCE REPORT")
    print("=" * 64)
    print(f"examples generated      : {n}")
    print(f"oracle == target        : {agree}/{n}  ({100.0 * agree / n:.2f}%)")
    print(f"verifier true+/false-   : {corrupt_caught}/{n}  ({100.0 * corrupt_caught / n:.2f}%)")
    print(f"mismatches              : {mismatches if mismatches else 'none'}")
    print("\n--- sample prompts/answers per mode ---")
    for mode in modes:
        prompt, tgt, meta = dsl.gen(mode, 4, 0.3, 48, seed=0)
        print(f"\n[{mode}]  target={tgt}  query={meta['query']}")
        print(prompt)
    print("=" * 64)
    ok = agree == n == 10_000 and corrupt_caught == n
    print("GATE:", "PASS ✅" if ok else "FAIL ❌")
    return 0 if ok else 1


def train() -> int:
    """Train the Phase 5 milestone config (dense transformer -> near-100%)."""
    import yaml

    from iota.train import train as run_train

    cfg = yaml.safe_load(open("configs/milestone_transformer.yaml"))
    out = run_train(cfg)
    print(f"\nMILESTONE final exact-answer acc = {out['final_acc']:.4f}")
    return 0 if out["final_acc"] >= 0.95 else 1


def _not_built(name: str):
    def _fn() -> int:
        print(f"[{name}] is a later phase (Phase 6+). Stopping at the Phase 5 milestone.")
        return 0

    return _fn


TARGETS = {
    "smoke": smoke,
    "test": test,
    "report": report,
    "train": train,
    "eval": _not_built("eval"),
    "profile": _not_built("profile"),
    "plot": _not_built("plot"),
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in TARGETS:
        print(__doc__)
        return 1
    return TARGETS[sys.argv[1]]()


if __name__ == "__main__":
    raise SystemExit(main())
