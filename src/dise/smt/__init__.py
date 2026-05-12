"""SMT abstraction layer for DiSE.

Public API:

* :class:`SMTBackend`   — abstract base class
* :class:`MockBackend`  — conservative fallback (always available)
* :class:`Z3Backend`    — z3-backed (None if z3-solver not installed)
* :func:`default_backend` — return Z3Backend() when available, else MockBackend()
"""

from __future__ import annotations

from .base import (
    ARITH_OPS,
    COMPARE_OPS,
    LOGIC_OPS,
    SUPPORTED_OPS,
    Clause,
    SatResult,
    SMTBackend,
    SMTExpr,
)
from .cache import CachedBackend, CacheStats
from .mock import MockBackend, MockExpr

try:
    from .z3_backend import Z3Backend
    _HAS_Z3 = True
except ImportError:
    Z3Backend = None  # type: ignore[assignment,misc]
    _HAS_Z3 = False


def default_backend() -> SMTBackend:
    """Return Z3Backend() when z3-solver is installed, else MockBackend()."""
    if _HAS_Z3 and Z3Backend is not None:
        return Z3Backend()
    return MockBackend()


def has_z3() -> bool:
    """True iff the z3 backend is available."""
    return _HAS_Z3


__all__ = [
    "ARITH_OPS",
    "COMPARE_OPS",
    "LOGIC_OPS",
    "SUPPORTED_OPS",
    "CachedBackend",
    "CacheStats",
    "Clause",
    "MockBackend",
    "MockExpr",
    "SMTBackend",
    "SMTExpr",
    "SatResult",
    "Z3Backend",
    "default_backend",
    "has_z3",
]
