"""Eval (Phase 5 milestone slice).

Exact-answer accuracy, verifier-checked (never loss). For Phase 5 this measures
in-distribution accuracy to validate the pipeline. The full length/recall grid
and CSV output is Phase 6 (see BUILD_PLAN.md §6 and the carry-forward TODOs).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import torch

from .data.dataset import DataSampler, EVAL_OFFSET, Example
from .data.tokenizer import Tokenizer, get_tokenizer
from .data.verifier import check


@torch.no_grad()
def _greedy_generate(model, prompt_id_batch: torch.Tensor, eos_id: int, max_new: int, device) -> List[List[int]]:
    """Greedy-decode `max_new` tokens for a batch of equal-length prompts."""
    model.eval()
    seq = prompt_id_batch.to(device)
    B = seq.shape[0]
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    generated = [[] for _ in range(B)]
    for _ in range(max_new):
        logits = model(seq)[:, -1, :]
        nxt = logits.argmax(dim=-1)
        for b in range(B):
            if not finished[b]:
                tokid = int(nxt[b])
                if tokid == eos_id:
                    finished[b] = True
                else:
                    generated[b].append(tokid)
        if bool(finished.all()):
            break
        seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
    return generated


@torch.no_grad()
def exact_answer_accuracy(
    model,
    sampler: DataSampler,
    n: int = 256,
    offset: int = EVAL_OFFSET,
    max_new: int = 4,
    device: str = "cpu",
    tok: Tokenizer = None,
    capture_failures: int = 5,
) -> Tuple[float, List[Dict]]:
    """Generate answers for `n` held-out examples; score with the verifier.

    Batches examples that share a prompt length so no padding is needed (correct
    for both attention and recurrent architectures). Returns (accuracy, failures).
    """
    tok = tok or get_tokenizer()
    examples: List[Example] = [sampler.example(i, offset) for i in range(n)]

    groups: Dict[int, List[int]] = defaultdict(list)
    for idx, ex in enumerate(examples):
        groups[ex.true_len].append(idx)

    correct = 0
    failures: List[Dict] = []
    for length, idxs in groups.items():
        batch = torch.stack([torch.tensor(examples[i].tokens[:length], dtype=torch.long) for i in idxs])
        gens = _greedy_generate(model, batch, tok.eos_id, max_new, device)
        for local, gidx in enumerate(idxs):
            ex = examples[gidx]
            pred = tok.decode(gens[local])
            ok = check(ex.prompt, pred)
            correct += int(ok)
            if not ok and len(failures) < capture_failures:
                failures.append(
                    {"target": ex.target, "pred": pred, "true_len": ex.true_len, "n_bindings": ex.meta.get("n_bindings")}
                )
    return correct / max(1, len(examples)), failures


def load_checkpoint(run_name: str, results_dir: str = "experiments/results", device: str = "cpu"):
    """Rebuild a model from a saved run json + weights (safetensors or .pt)."""
    import json
    import os

    import yaml  # noqa: F401

    from .models import build_model

    with open(os.path.join(results_dir, f"{run_name}.json")) as fh:
        run = json.load(fh)
    cfg = run["config"]
    model = build_model(cfg).to(device)
    wpath = os.path.join(results_dir, run["weights"])
    if wpath.endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(wpath)
    else:
        state = torch.load(wpath, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, cfg
