"""Benchmark protocol shared by all DiSE benchmark scripts.

Each benchmark exposes:

* a ``name`` (used in tables/plots),
* a ``description`` (one-line; rendered into ``--help`` and reports),
* a ``program(**kwargs) -> Any`` callable,
* a ``distribution()`` factory returning a dict of factor distributions,
* a ``property_fn(out) -> bool`` callable,
* (optionally) closed-form or expected ground-truth values.

The :class:`Benchmark` dataclass bundles these. The module also exposes
a global :data:`REGISTRY` mapping benchmark names to factories, so the
``dise`` CLI and the experiment runner can enumerate them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from dise.distributions import Distribution


@dataclass
class Benchmark:
    """A single benchmark configuration."""

    name: str
    description: str
    program: Callable[..., Any]
    distribution: Mapping[str, Distribution]
    property_fn: Callable[[Any], bool]
    # Heuristic budgets for default experiments
    suggested_budget: int = 5000
    suggested_bootstrap: int = 200
    suggested_batch_size: int = 50
    # Optional: closed-form ground truth or analytic notes.
    closed_form_mu: float | None = None
    notes: str = ""

    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry — populated by individual benchmark modules at import time.
# ---------------------------------------------------------------------------


_REGISTRY: dict[str, Callable[[], Benchmark]] = {}


def register(factory: Callable[[], Benchmark]) -> Callable[[], Benchmark]:
    """Decorator: register a Benchmark factory.

    Idempotent on the same logical factory (same ``__qualname__``) — this
    avoids spurious errors when a module is loaded both as
    ``benchmarks.foo`` and as ``__main__``. Re-registering a *different*
    factory under an existing name raises :class:`ValueError`.

    Usage::

        @register
        def gcd_geometric() -> Benchmark:
            return Benchmark(...)
    """
    bench = factory()
    existing = _REGISTRY.get(bench.name)
    if existing is None:
        _REGISTRY[bench.name] = factory
    elif existing.__qualname__ != factory.__qualname__:
        raise ValueError(f"duplicate benchmark name: {bench.name!r}")
    return factory


def list_benchmarks() -> list[str]:
    """All registered benchmark names, sorted."""
    return sorted(_REGISTRY)


def get_benchmark(name: str) -> Benchmark:
    """Return the freshly-constructed :class:`Benchmark` registered under ``name``."""
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown benchmark {name!r}; choose from {', '.join(list_benchmarks())}"
        )
    return _REGISTRY[name]()


__all__ = ["Benchmark", "register", "list_benchmarks", "get_benchmark"]
