"""Univariate and product discrete distributions for DiSE.

The base class :class:`Distribution` defines the contract: PMF, CDF, mass of
an interval, sampling (raw and truncated), and a support-bounds query. The
concrete classes are immutable (frozen dataclasses), hashable, and safe to
use as keys in caches.

A :class:`ProductDistribution` is a joint over named integer variables with
independent factors; this is class (D1) of the DiSE problem hierarchy.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy import stats as _sp_stats


class Distribution(ABC):
    """A univariate discrete distribution over (some subset of) the integers."""

    @abstractmethod
    def pmf(self, x: int) -> float:
        """P(X == x)."""

    @abstractmethod
    def cdf(self, x: int) -> float:
        """P(X <= x)."""

    def mass(self, lo: int, hi: int) -> float:
        """P(lo <= X <= hi). Returns 0 if lo > hi."""
        if lo > hi:
            return 0.0
        return max(0.0, self.cdf(hi) - self.cdf(lo - 1))

    @abstractmethod
    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw ``n`` iid samples as an int64 ndarray of shape (n,)."""

    def sample_truncated(
        self,
        rng: np.random.Generator,
        lo: int,
        hi: int,
        n: int,
    ) -> np.ndarray:
        """Draw ``n`` samples conditional on ``lo <= X <= hi``.

        Default implementation: bounded rejection sampling. Subclasses with
        closed-form CDFs should override with an inverse-CDF method.
        """
        if lo > hi or self.mass(lo, hi) <= 0.0:
            raise ValueError(f"empty truncation [{lo}, {hi}] for {self!r}")
        out = np.empty(n, dtype=np.int64)
        i = 0
        attempts = 0
        max_attempts = max(1000, 200 * n)
        while i < n and attempts < max_attempts:
            batch = self.sample(rng, max(16, n - i))
            ok = batch[(batch >= lo) & (batch <= hi)]
            k = min(len(ok), n - i)
            out[i : i + k] = ok[:k]
            i += k
            attempts += len(batch)
        if i < n:
            raise RuntimeError(
                f"rejection-truncated sampling exhausted: got {i}/{n} in [{lo},{hi}]"
            )
        return out

    @abstractmethod
    def support_bounds(self, eps: float = 1e-10) -> tuple[int, int]:
        """Smallest [a, b] with P(a <= X <= b) >= 1 - eps.

        Returned bounds are inclusive integers. Distributions with finite
        support may ignore ``eps`` and return their natural support.
        """


@dataclass(frozen=True)
class Geometric(Distribution):
    """Geometric distribution on {1, 2, ...} with success probability ``p``."""

    p: float

    def __post_init__(self) -> None:
        if not (0.0 < self.p < 1.0):
            raise ValueError(f"Geometric: p must be in (0, 1), got {self.p}")

    def pmf(self, x: int) -> float:
        if x < 1:
            return 0.0
        return (1.0 - self.p) ** (x - 1) * self.p

    def cdf(self, x: int) -> float:
        if x < 1:
            return 0.0
        return 1.0 - (1.0 - self.p) ** x

    def mass(self, lo: int, hi: int) -> float:
        if lo > hi:
            return 0.0
        lo_eff = max(1, lo)
        if lo_eff > hi:
            return 0.0
        q = 1.0 - self.p
        return q ** (lo_eff - 1) - q**hi

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.geometric(self.p, size=n).astype(np.int64)

    def sample_truncated(
        self, rng: np.random.Generator, lo: int, hi: int, n: int
    ) -> np.ndarray:
        lo = max(1, lo)
        if lo > hi or self.mass(lo, hi) <= 0.0:
            raise ValueError(f"empty truncation [{lo}, {hi}]")
        F_lo = self.cdf(lo - 1)
        F_hi = self.cdf(hi)
        u = rng.uniform(F_lo, F_hi, size=n)
        # F(k) = 1 - (1-p)^k  =>  k = ceil( log(1-u) / log(1-p) )
        log_one_minus_u = np.log1p(-u)
        log_one_minus_p = math.log1p(-self.p)
        k = np.ceil(log_one_minus_u / log_one_minus_p)
        out = k.astype(np.int64)
        np.clip(out, lo, hi, out=out)
        return out

    def support_bounds(self, eps: float = 1e-10) -> tuple[int, int]:
        if eps <= 0.0:
            eps = 1e-300
        if eps >= 1.0:
            return (1, 1)
        # P(X > b) = (1-p)^b <= eps  =>  b >= log(eps)/log(1-p)
        b = math.ceil(math.log(eps) / math.log1p(-self.p))
        return (1, max(1, b))


@dataclass(frozen=True)
class BoundedGeometric(Distribution):
    """Geometric distribution truncated to the finite support {1, ..., N}."""

    p: float
    N: int

    def __post_init__(self) -> None:
        if not (0.0 < self.p < 1.0):
            raise ValueError(f"BoundedGeometric: p must be in (0, 1), got {self.p}")
        if self.N < 1:
            raise ValueError(f"BoundedGeometric: N must be >= 1, got {self.N}")

    def _Z(self) -> float:
        return 1.0 - (1.0 - self.p) ** self.N

    def pmf(self, x: int) -> float:
        if x < 1 or x > self.N:
            return 0.0
        return (1.0 - self.p) ** (x - 1) * self.p / self._Z()

    def cdf(self, x: int) -> float:
        if x < 1:
            return 0.0
        if x >= self.N:
            return 1.0
        return (1.0 - (1.0 - self.p) ** x) / self._Z()

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return self.sample_truncated(rng, 1, self.N, n)

    def sample_truncated(
        self, rng: np.random.Generator, lo: int, hi: int, n: int
    ) -> np.ndarray:
        lo = max(1, lo)
        hi = min(self.N, hi)
        if lo > hi or self.mass(lo, hi) <= 0.0:
            raise ValueError(f"empty truncation [{lo}, {hi}]")
        F_lo = self.cdf(lo - 1)
        F_hi = self.cdf(hi)
        Z = self._Z()
        u = rng.uniform(F_lo, F_hi, size=n)
        # F(k) = (1 - (1-p)^k) / Z  =>  k = ceil( log(1 - u*Z) / log(1-p) )
        log_term = np.log1p(-u * Z)
        log_one_minus_p = math.log1p(-self.p)
        k = np.ceil(log_term / log_one_minus_p)
        out = k.astype(np.int64)
        np.clip(out, lo, hi, out=out)
        return out

    def support_bounds(self, eps: float = 1e-10) -> tuple[int, int]:
        return (1, self.N)


@dataclass(frozen=True)
class Uniform(Distribution):
    """Discrete uniform on {lo, ..., hi} (inclusive)."""

    lo: int
    hi: int

    def __post_init__(self) -> None:
        if self.hi < self.lo:
            raise ValueError(f"Uniform: hi {self.hi} < lo {self.lo}")

    def _size(self) -> int:
        return self.hi - self.lo + 1

    def pmf(self, x: int) -> float:
        if x < self.lo or x > self.hi:
            return 0.0
        return 1.0 / self._size()

    def cdf(self, x: int) -> float:
        if x < self.lo:
            return 0.0
        if x >= self.hi:
            return 1.0
        return (x - self.lo + 1) / self._size()

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.integers(self.lo, self.hi + 1, size=n, dtype=np.int64)

    def sample_truncated(
        self, rng: np.random.Generator, lo: int, hi: int, n: int
    ) -> np.ndarray:
        a = max(self.lo, lo)
        b = min(self.hi, hi)
        if a > b:
            raise ValueError(f"empty truncation [{lo}, {hi}]")
        return rng.integers(a, b + 1, size=n, dtype=np.int64)

    def support_bounds(self, eps: float = 1e-10) -> tuple[int, int]:
        return (self.lo, self.hi)


@dataclass(frozen=True)
class Categorical(Distribution):
    """Categorical on {0, ..., k-1} with explicit probabilities."""

    probs: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.probs) == 0:
            raise ValueError("Categorical: probs must be non-empty")
        if any(p < 0 for p in self.probs):
            raise ValueError("Categorical: negative probability")
        total = sum(self.probs)
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError(f"Categorical: probs sum to {total}, not 1")

    def pmf(self, x: int) -> float:
        if 0 <= x < len(self.probs):
            return self.probs[x]
        return 0.0

    def cdf(self, x: int) -> float:
        if x < 0:
            return 0.0
        x = min(x, len(self.probs) - 1)
        return sum(self.probs[: x + 1])

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.choice(len(self.probs), size=n, p=list(self.probs)).astype(np.int64)

    def sample_truncated(
        self, rng: np.random.Generator, lo: int, hi: int, n: int
    ) -> np.ndarray:
        a = max(0, lo)
        b = min(len(self.probs) - 1, hi)
        if a > b:
            raise ValueError(f"empty truncation [{lo}, {hi}]")
        sub = np.array(self.probs[a : b + 1], dtype=np.float64)
        Z = sub.sum()
        if Z <= 0.0:
            raise ValueError("zero-mass truncation")
        sub /= Z
        return (rng.choice(b - a + 1, size=n, p=sub) + a).astype(np.int64)

    def support_bounds(self, eps: float = 1e-10) -> tuple[int, int]:
        return (0, len(self.probs) - 1)


@dataclass(frozen=True)
class Poisson(Distribution):
    """Poisson distribution with mean ``lam`` on {0, 1, 2, ...}."""

    lam: float

    def __post_init__(self) -> None:
        if self.lam < 0:
            raise ValueError(f"Poisson: lam must be >= 0, got {self.lam}")

    def pmf(self, x: int) -> float:
        if x < 0:
            return 0.0
        return float(_sp_stats.poisson.pmf(x, self.lam))

    def cdf(self, x: int) -> float:
        if x < 0:
            return 0.0
        return float(_sp_stats.poisson.cdf(x, self.lam))

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.poisson(self.lam, size=n).astype(np.int64)

    def sample_truncated(
        self, rng: np.random.Generator, lo: int, hi: int, n: int
    ) -> np.ndarray:
        a = max(0, lo)
        if a > hi:
            raise ValueError(f"empty truncation [{lo}, {hi}]")
        F_lo = self.cdf(a - 1) if a > 0 else 0.0
        F_hi = self.cdf(hi)
        if F_hi <= F_lo:
            raise ValueError(f"zero-mass truncation [{lo}, {hi}]")
        u = rng.uniform(F_lo, F_hi, size=n)
        out = _sp_stats.poisson.ppf(u, self.lam).astype(np.int64)
        np.clip(out, a, hi, out=out)
        return out

    def support_bounds(self, eps: float = 1e-10) -> tuple[int, int]:
        if eps <= 0.0:
            eps = 1e-300
        lo = int(_sp_stats.poisson.ppf(eps / 2.0, self.lam))
        hi = int(_sp_stats.poisson.ppf(1.0 - eps / 2.0, self.lam))
        return (max(0, lo), hi)


@dataclass(frozen=True)
class ProductDistribution:
    """A joint distribution over named integer variables with independent factors.

    Represents class (D1) of the DiSE problem hierarchy. PMF is the product
    of factor PMFs; sampling is independent per factor.
    """

    factors: Mapping[str, Distribution]

    def __post_init__(self) -> None:
        if len(self.factors) == 0:
            raise ValueError("ProductDistribution: factors must be non-empty")

    def pmf(self, x: Mapping[str, int]) -> float:
        p = 1.0
        for name, dist in self.factors.items():
            p *= dist.pmf(x[name])
        return p

    def sample(self, rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
        return {name: dist.sample(rng, n) for name, dist in self.factors.items()}

    def sample_one(self, rng: np.random.Generator) -> dict[str, int]:
        return {name: int(dist.sample(rng, 1)[0]) for name, dist in self.factors.items()}

    def support_bounds(self, eps: float = 1e-10) -> dict[str, tuple[int, int]]:
        return {name: d.support_bounds(eps) for name, d in self.factors.items()}

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(self.factors.keys())


__all__ = [
    "Distribution",
    "Geometric",
    "BoundedGeometric",
    "Uniform",
    "Categorical",
    "Poisson",
    "ProductDistribution",
]
