"""Region base types: Status, SampleBatch, Region ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..distributions import ProductDistribution
from ..smt import SMTBackend, SMTExpr


class Status(Enum):
    """Lifecycle status of a frontier leaf."""

    OPEN = "open"
    CLOSED_TRUE = "closed_true"     # phi == 1 throughout
    CLOSED_FALSE = "closed_false"   # phi == 0 throughout
    EMPTY = "empty"                 # SMT-proved region is empty (mass 0)
    DIVERGED = "diverged"           # concolic exceeded max_branches


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
