"""Region base types: :class:`Status`, :class:`SampleBatch`, :class:`Region`.

This module defines the abstract surface every concrete region in
:mod:`dise.regions._concrete` implements. A *region*
:math:`R_\\pi \\subseteq \\mathcal X` is the geometric counterpart of
a path condition :math:`\\pi`; the frontier of regions partitions the
input space and supports the stratified estimator at the heart of
ASIP (see :doc:`/docs/algorithm` §2).

The :class:`Region` ABC requires:

* a **formula** (an opaque ``SMTExpr`` characterizing the region;
  used for SMT-driven refinement and closure);
* a **mass estimator** ``mass(distribution, smt, rng, n_mc)`` that
  returns :math:`(\\hat w_\\pi, \\widehat{\\mathrm{Var}}(\\hat w_\\pi))`;
* a **constrained sampler** ``sample(distribution, smt, rng, n)``
  that draws :math:`n` samples from :math:`D \\,|\\, R_\\pi`;
* a **membership test** ``contains(x)`` used by ``find_leaf_for``;
* a flag ``is_axis_aligned`` distinguishing closed-form mass from
  importance-sampled mass.

For axis-aligned regions the mass is exact and the variance is zero —
this is the structural variance-reduction lever that distinguishes
DiSE from plain Monte Carlo (cf. Theorem 1 in
:doc:`/docs/algorithm` §4).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..distributions import ProductDistribution
from ..smt import SMTBackend, SMTExpr


class Status(Enum):
    """Lifecycle status of a frontier leaf.

    A leaf is in exactly one of these states at any moment. ``OPEN``
    is the only state under which the scheduler will allocate
    additional samples or refine.
    """

    #: Sampled but not yet resolved; eligible for allocate / refine.
    OPEN = "open"
    #: SMT-proved (or sample-determined) :math:`\mu_\pi = 1`.
    CLOSED_TRUE = "closed_true"
    #: SMT-proved (or sample-determined) :math:`\mu_\pi = 0`.
    CLOSED_FALSE = "closed_false"
    #: SMT-proved empty region; mass exactly 0 — never contributes
    #: to :math:`\hat\mu` and is excluded from open / closed counts.
    EMPTY = "empty"
    #: Reserved: concolic execution exceeded ``max_concolic_branches``
    #: on every sample drawn from this leaf. **Currently unreached**
    #: by the scheduler — diverged samples are dropped silently (see
    #: :class:`~dise.scheduler.ASIPScheduler._record_observation`).
    #: Wired in the type system for future use; do not rely on it.
    DIVERGED = "diverged"


@dataclass
class SampleBatch:
    """A batch of constrained samples drawn from a region under ``D``.

    ``n`` may be less than the requested count if sampling was difficult
    (rare-event regions). ``rejection_ratio`` is the acceptance rate from
    the proposal, or ``None`` for closed-form (axis-aligned) sampling.
    """

    inputs: dict[str, np.ndarray]
    n: int
    rejection_ratio: float | None = None

    def __post_init__(self) -> None:
        for k, arr in self.inputs.items():
            if len(arr) != self.n:
                raise ValueError(
                    f"SampleBatch: var {k!r} has {len(arr)} entries, expected {self.n}"
                )

    def iter_assignments(self):
        """Yield each sample as a ``dict[str, int]``."""
        for i in range(self.n):
            yield {k: int(v[i]) for k, v in self.inputs.items()}


class Region(ABC):
    """A symbolic region of input space.

    Concrete regions:

    * :class:`UnconstrainedRegion`  — the full distribution support (mass 1).
    * :class:`EmptyRegion`          — SMT-proved empty (mass 0).
    * :class:`AxisAlignedBox`       — conjunction of per-variable intervals;
      closed-form mass.
    * :class:`GeneralRegion`        — axis-aligned envelope + a predicate;
      importance-sampled mass.
    """

    @property
    @abstractmethod
    def formula(self) -> SMTExpr:
        """The SMT formula characterizing this region."""

    @property
    @abstractmethod
    def is_axis_aligned(self) -> bool:
        """True iff this region is an axis-aligned box (no general predicate)."""

    @abstractmethod
    def mass(
        self,
        distribution: ProductDistribution,
        smt: SMTBackend,
        rng: np.random.Generator,
        n_mc: int = 1000,
    ) -> tuple[float, float]:
        """Estimate :math:`w_\\pi = P_D[X \\in R_\\pi]` along with its variance.

        Returns ``(w_hat, w_var)``. For axis-aligned regions this is closed
        form with ``w_var = 0``. For general regions, uses ``n_mc`` IS
        samples from the axis-aligned envelope.
        """

    @abstractmethod
    def sample(
        self,
        distribution: ProductDistribution,
        smt: SMTBackend,
        rng: np.random.Generator,
        n: int,
    ) -> SampleBatch:
        """Draw ``n`` samples from ``D | R_\\pi`` (best-effort)."""

    @abstractmethod
    def contains(self, x: dict[str, int]) -> bool:
        """Return True iff ``x`` is in this region."""


__all__ = ["Status", "SampleBatch", "Region"]
