"""DiSE benchmarks.

Each benchmark module exposes:

* a :class:`~dise.benchmarks._base.Benchmark` factory registered via
  :func:`~dise.benchmarks._base.register`,
* a ``main()`` entry point so the module can be invoked as
  ``python -m benchmarks.<name>``.

Importing this package registers every benchmark, so callers can
enumerate them via :func:`~dise.benchmarks._base.list_benchmarks` or fetch
them via :func:`~dise.benchmarks._base.get_benchmark`.

Use :class:`dise.estimate` directly for ad-hoc programs, and the top-
level ``dise`` CLI (``dise benchmark <name>``) or the experiment runner
:func:`dise.experiment.run_experiment` for systematic evaluation.
"""

from __future__ import annotations

# Importing each module triggers @register side effects.
from . import (  # noqa: F401  (side-effect imports)
    bitvector_kernels,
    collatz,
    gcd_geometric,
    integer_sqrt,
    miller_rabin,
    modular_exp,
    sieve_primality,
    sparse_trie_depth,
)
from ._base import Benchmark, get_benchmark, list_benchmarks, register

__all__ = ["Benchmark", "get_benchmark", "list_benchmarks", "register"]
