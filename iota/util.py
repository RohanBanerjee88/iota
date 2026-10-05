"""Shared utilities. Kept dependency-light so the data layer runs on CPU in seconds."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 0) -> int:
    """Seed `random`, `numpy`, and (if installed) `torch`.

    torch is imported lazily and behind a try/except so the data layer and its
    tests never hard-depend on it. Returns the seed for convenience.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional for everything in Phases 0-2
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    return seed


def sweep_config_path(arch: str) -> str:
    """configs/sweep_<arch>.yaml, or the same file under $IOTA_CONFIG_DIR.

    Lets an ablation (e.g. configs/fixinit/) train and evaluate without editing the
    main sweep configs, so the checkpoints of earlier runs stay valid for re-eval.
    """
    return os.path.join(os.environ.get("IOTA_CONFIG_DIR", "configs"), f"sweep_{arch}.yaml")
