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
