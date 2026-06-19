"""Train (Phase 5).

Config-driven, trains on short sequences only. Next-token cross-entropy masked to
the answer span. Saves safetensors + a run json (config, loss/acc curve, seed) to
experiments/results/. Architecture-agnostic: only touches the SeqModel interface
and the data sampler.

Phase 5 done-criterion: a tiny transformer reaches near-100% exact-answer
(verifier-checked) accuracy on the in-distribution eval split. That milestone
validates the whole pipeline (data -> tokenizer -> model -> train -> verifier-eval).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict

import torch
import torch.nn.functional as F
import yaml

from .data.dataset import (
    EVAL_OFFSET,
    TRAIN_OFFSET,
    CurriculumSampler,
    DataSampler,
    collate,
    collate_sweep,
)
from .data.tokenizer import get_tokenizer
from .eval import exact_answer_accuracy, teacher_forced_scores
from .models import build_model
from .util import seed_everything

RESULTS_DIR = "experiments/results"


def _lr_at(step: int, base_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * min(1.0, prog)))


def _save(model, cfg: Dict, history, run_name: str):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # clone() breaks the shared storage from weight tying (embed.weight ==
    # head.weight); safetensors refuses tensors that share memory.
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    try:
        from safetensors.torch import save_file

        save_file(state, os.path.join(RESULTS_DIR, f"{run_name}.safetensors"))
        weights = f"{run_name}.safetensors"
    except Exception:
        torch.save(state, os.path.join(RESULTS_DIR, f"{run_name}.pt"))
        weights = f"{run_name}.pt"
    with open(os.path.join(RESULTS_DIR, f"{run_name}.json"), "w") as fh:
        json.dump({"config": cfg, "history": history, "weights": weights}, fh, indent=2)
    return weights


def train_sweep(cfg: Dict, smoke: bool = False) -> Dict:
    """Train to PLATEAU on the mixed curriculum (Phase 6 sweep model).

    Fairness rule: NOT a fixed step count. A common generous max-steps ceiling +
    patience-based early-stop on a large fixed held-out set (teacher-forced exact
    accuracy), identical for all three architectures. Report plateau accuracy.
    """
    tcfg = cfg.get("train", {})
    seed = int(tcfg.get("seed", 0))
    seed_everything(seed)
    device = tcfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    tok = get_tokenizer()
    cfg["vocab_size"] = tok.vocab_size
    model = build_model(cfg).to(device)

    sampler = CurriculumSampler(cfg["curriculum"], tok)
    max_steps = 60 if smoke else int(tcfg.get("max_steps", 10000))
    batch_size = 8 if smoke else int(tcfg.get("batch_size", 64))
    base_lr = float(tcfg.get("lr", 3e-3))
    warmup = int(tcfg.get("warmup", 200))
    wd = float(tcfg.get("weight_decay", 0.01))
    eval_every = 20 if smoke else int(tcfg.get("eval_every", 500))
    eval_n = 64 if smoke else int(tcfg.get("eval_n", 500))
    patience = int(tcfg.get("patience", 5))
    run_name = tcfg.get("run_name", f"{cfg['arch']}_sweep")

    # Fixed held-out eval set (disjoint stream via EVAL_OFFSET), built once.
    eval_examples = sampler.batch(list(range(eval_n)), EVAL_OFFSET)
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=wd, betas=(0.9, 0.95))

    print(f"[sweep] model={cfg['arch']} params={model.num_params()/1e6:.2f}M device={device} "
          f"max_steps={max_steps} bs={batch_size} patience={patience}", flush=True)

    history = {"eval_acc": []}
    best_acc, bad, cursor, t0 = -1.0, 0, 0, time.time()
    for step in range(max_steps):
        model.train()
        lr = _lr_at(step, base_lr, warmup, max_steps)
        for g in opt.param_groups:
            g["lr"] = lr
        ex = sampler.batch(list(range(cursor, cursor + batch_size)), TRAIN_OFFSET)
        cursor += batch_size
        inp, tgt, mask = collate_sweep(ex, tok.pad_id)
        inp, tgt, mask = inp.to(device), tgt.to(device), mask.to(device)
        logits = model(inp)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1),
                             reduction="none").view_as(tgt)
        loss = (ce * mask).sum() / mask.sum().clamp_min(1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % eval_every == 0 or step == max_steps - 1:
            scores = teacher_forced_scores(model, eval_examples, tok.pad_id, device)
            acc = sum(1 for s in scores if all(s)) / max(1, len(scores))
            history["eval_acc"].append({"step": step + 1, "acc": acc})
            improved = acc > best_acc + 1e-4
            print(f"[sweep] {cfg['arch']} step {step+1:5d} loss {loss.item():.3f} "
                  f"plateau_acc {acc:.3f} {'*' if improved else ''} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if improved:
                best_acc, bad = acc, 0
                if not smoke:
                    _save(model, cfg, history, run_name)  # checkpoint the best
            else:
                bad += 1
                if bad >= patience:
                    print(f"[sweep] {cfg['arch']} plateaued (patience {patience}) -> stop", flush=True)
                    break
    if smoke:
        _save(model, cfg, history, run_name)
    print(f"[sweep] {cfg['arch']} BEST plateau_acc = {best_acc:.4f} -> {RESULTS_DIR}/{run_name}", flush=True)
    return {"final_acc": best_acc, "best_acc": best_acc, "history": history}


def train(cfg: Dict, smoke: bool = False) -> Dict:
    if "curriculum" in cfg:
        return train_sweep(cfg, smoke=smoke)
    tcfg = cfg.get("train", {})
    dcfg = cfg["data"]
    seed = int(tcfg.get("seed", 0))
    seed_everything(seed)
    device = tcfg.get("device", "cpu")

    tok = get_tokenizer()
    cfg["vocab_size"] = tok.vocab_size
    model = build_model(cfg).to(device)

    steps = 30 if smoke else int(tcfg.get("steps", 3000))
    batch_size = 8 if smoke else int(tcfg.get("batch_size", 64))
    base_lr = float(tcfg.get("lr", 3e-4))
    warmup = int(tcfg.get("warmup", 200))
    wd = float(tcfg.get("weight_decay", 0.01))
    eval_every = 10 if smoke else int(tcfg.get("eval_every", 500))
    eval_n = 32 if smoke else int(tcfg.get("eval_n", 256))
    target_acc = float(tcfg.get("target_acc", 0.98))

    sampler = DataSampler(dcfg, tok)
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=wd, betas=(0.9, 0.95))

    print(f"model={cfg['arch']} params={model.num_params()/1e6:.2f}M vocab={tok.vocab_size} "
          f"device={device} steps={steps} bs={batch_size}", flush=True)
    print(f"data={dcfg}", flush=True)

    history = {"step": [], "loss": [], "eval_acc": []}
    cursor = 0
    t0 = time.time()
    best_acc = 0.0
    for step in range(steps):
        model.train()
        lr = _lr_at(step, base_lr, warmup, steps)
        for g in opt.param_groups:
            g["lr"] = lr
        idxs = list(range(cursor, cursor + batch_size))
        cursor += batch_size
        examples = sampler.batch_examples(idxs, TRAIN_OFFSET)
        input_ids, target_ids, loss_mask = collate(examples, tok.pad_id)
        input_ids, target_ids, loss_mask = input_ids.to(device), target_ids.to(device), loss_mask.to(device)

        logits = model(input_ids)
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1), reduction="none"
        ).view_as(target_ids)
        loss = (ce * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % max(1, eval_every // 5) == 0:
            history["step"].append(step)
            history["loss"].append(loss.item())

        if (step + 1) % eval_every == 0 or step == steps - 1:
            acc, failures = exact_answer_accuracy(model, sampler, n=eval_n, device=device, tok=tok)
            history["eval_acc"].append({"step": step + 1, "acc": acc})
            best_acc = max(best_acc, acc)
            dt = time.time() - t0
            print(f"step {step+1:5d}/{steps}  loss {loss.item():.4f}  eval_acc {acc:.3f}  "
                  f"({dt:.0f}s, {(step+1)/dt:.1f} it/s)", flush=True)
            if acc >= target_acc and not smoke:
                print(f"  reached target_acc {target_acc} -> early stop", flush=True)
                break

    # robust final eval on a larger held-out set for the reported milestone number
    final_n = 32 if smoke else int(tcfg.get("final_eval_n", 1000))
    final_acc, failures = exact_answer_accuracy(model, sampler, n=final_n, device=device, tok=tok)
    history["final_eval"] = {"n": final_n, "acc": final_acc, "failures": failures}
    print(f"final eval (n={final_n}): exact-answer acc = {final_acc:.4f}", flush=True)

    run_name = tcfg.get("run_name", f"{cfg['arch']}_milestone")
    if not smoke:
        weights = _save(model, cfg, history, run_name)
        print(f"saved -> {RESULTS_DIR}/{run_name}.json ({weights})", flush=True)
    return {"final_acc": final_acc, "best_acc": max(best_acc, final_acc), "history": history}


def main():
    ap = argparse.ArgumentParser(description="iota trainer")
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run in seconds")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.seed is not None:
        cfg.setdefault("train", {})["seed"] = args.seed
    train(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
