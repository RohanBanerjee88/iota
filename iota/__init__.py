"""iota — a custom linear-time architecture for bounded, verifier-checkable reasoning.

This package is built in strict phase order (see BUILD_PLAN.md). The data layer
(dsl + oracle + verifier) is proven before any model code is written, because
every downstream metric trusts it.
"""

from .util import seed_everything

__all__ = ["seed_everything"]
