"""Model implementations behind one architecture-agnostic SeqModel interface."""

from .base import SeqModel, build_model

__all__ = ["SeqModel", "build_model"]
