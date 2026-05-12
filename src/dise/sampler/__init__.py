"""Samplers for constrained regions.

The :class:`Sampler` abstract class wraps the "draw n samples from
``D | R_pi``" operation. Two implementations:

* :class:`RejectionSampler` — uses the region's natural sampling
  (closed-form for axis-aligned, rejection from the envelope for
  general regions).
* :class:`IntegerLatticeMHSampler` — Metropolis-Hastings on the integer
  lattice for rare-event general regions where rejection's acceptance
  rate becomes too low.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np

from ..distributions import ProductDistribution
from ..regions import GeneralRegion, Region, SampleBatch
from ..smt import SMTBackend


class Sampler(ABC):
    """Abstract sampler interface."""

    @abstractmethod
    def sample(
        self,
        region: Region,
        distribution: ProductDistribution,
        smt: SMTBackend,
        rng: np.random.Generator,
        n: int,
    ) -> SampleBatch:
        """Draw ``n`` samples from ``D | R_pi``. May return a partial batch
        if sampling is hard (rare-event region); the returned
        :class:`SampleBatch` reports the actual count and rejection ratio.
        """


class RejectionSampler(Sampler):
    """Sample by rejection from the region's axis-aligned envelope.

    For axis-aligned regions, this is closed-form (no rejection needed).
    For general regions, draws from the envelope and accepts by the
    region's predicate. Configurable ``max_attempts_per_sample`` controls
    the budget; if the acceptance rate is too low to produce ``n``
    accepted samples within ``n * max_attempts_per_sample`` attempts, the
    returned batch is partial.
    """

    def __init__(self, max_attempts_per_sample: int = 200) -> None:
        if max_attempts_per_sample < 1:
            raise ValueError("max_attempts_per_sample must be >= 1")
        self.max_attempts_per_sample = max_attempts_per_sample

    def sample(
        self,
        region: Region,
        distribution: ProductDistribution,
        smt: SMTBackend,
        rng: np.random.Generator,
        n: int,
    ) -> SampleBatch:
        if n <= 0:
            empty = {v: np.empty(0, dtype=np.int64) for v in distribution.variables}
            return SampleBatch(inputs=empty, n=0, rejection_ratio=None)
        if isinstance(region, GeneralRegion):
            return self._rejection_general(region, distribution, smt, rng, n)
        # Axis-aligned / unconstrained / empty: delegate to region's own method.
        return region.sample(distribution, smt, rng, n)

    def _rejection_general(
        self,
        region: GeneralRegion,
        distribution: ProductDistribution,
        smt: SMTBackend,
        rng: np.random.Generator,
        n: int,
    ) -> SampleBatch:
        accepted: dict[str, list[int]] = {v: [] for v in distribution.variables}
        max_attempts = max(64, self.max_attempts_per_sample * n)
        attempts = 0
        accepts = 0
        while accepts < n and attempts < max_attempts:
            chunk = max(16, n - accepts)
            chunk = min(chunk, max_attempts - attempts)
            if chunk <= 0:
                break
            batch = region.base.sample(distribution, smt, rng, chunk)
            for x in batch.iter_assignments():
                attempts += 1
                try:
                    ok = smt.evaluate(region.formula, x)
                except (ValueError, KeyError, ZeroDivisionError, ArithmeticError):
                    ok = False
                if ok:
                    for v in distribution.variables:
                        accepted[v].append(x[v])
                    accepts += 1
                    if accepts >= n:
                        break
                if attempts >= max_attempts:
                    break
        rej_ratio = accepts / attempts if attempts > 0 else 0.0
        inputs = {
            v: np.array(accepted[v], dtype=np.int64) for v in distribution.variables
        }
        return SampleBatch(inputs=inputs, n=accepts, rejection_ratio=rej_ratio)


class IntegerLatticeMHSampler(Sampler):
    """Metropolis-Hastings sampler on the integer lattice.

    Targets :math:`\\pi(x) \\propto D(x) \\cdot \\mathbf{1}[x \\in R_\\pi]`
    using a symmetric integer-Gaussian proposal:

        :math:`x_i' = x_i + \\mathrm{round}(\\mathcal{N}(0, \\sigma_i^2))`,
        clipped to the envelope ``base.bounds``.

    Acceptance probability is

        :math:`\\alpha(x \\to x') = \\min\\!\\left(1,
            \\dfrac{D(x') \\cdot \\mathbf{1}[x' \\in R_\\pi]}{D(x) \\cdot \\mathbf{1}[x \\in R_\\pi]}
        \\right).`

    Initialization: a brief rejection-sampling pass to find a starting
    point inside the region. If that fails (rare-event region with no
    feasible initial draw), the sampler returns an empty batch with
    ``rejection_ratio = 0.0``.

    Parameters
    ----------
    n_burn_in : int
        Number of MH steps before recording any sample.
    thin : int
        Keep one of every ``thin`` post-burn-in steps.
    sigma_scale : float
        Proposal step-size scaling: per-variable
        ``sigma = sigma_scale * (hi - lo)`` over its envelope range.
    init_attempts : int
        Number of rejection draws to try when finding the initial chain
        state. Falling back to a uniform envelope draw if none satisfy.
    """

    def __init__(
        self,
        n_burn_in: int = 200,
        thin: int = 5,
        sigma_scale: float = 0.15,
        init_attempts: int = 2000,
    ) -> None:
        if n_burn_in < 0:
            raise ValueError("n_burn_in must be >= 0")
        if thin < 1:
            raise ValueError("thin must be >= 1")
        if sigma_scale <= 0:
            raise ValueError("sigma_scale must be > 0")
        if init_attempts < 1:
            raise ValueError("init_attempts must be >= 1")
        self.n_burn_in = n_burn_in
        self.thin = thin
        self.sigma_scale = sigma_scale
        self.init_attempts = init_attempts

    def sample(
        self,
        region: Region,
        distribution: ProductDistribution,
        smt: SMTBackend,
        rng: np.random.Generator,
        n: int,
    ) -> SampleBatch:
        if n <= 0:
            empty = {v: np.empty(0, dtype=np.int64) for v in distribution.variables}
            return SampleBatch(inputs=empty, n=0, rejection_ratio=None)
        if not isinstance(region, GeneralRegion):
            # Use the region's own sampler for non-general cases (closed-form).
            return region.sample(distribution, smt, rng, n)
        base = region.base
        var_names = list(distribution.variables)
        # Per-variable proposal sigma and clip bounds (envelope).
        sigmas: dict[str, float] = {}
        clip_bounds: dict[str, tuple[int, int]] = {}
        for v in var_names:
            lo, hi = base.bounds[v]
            clip_bounds[v] = (lo, hi)
            sigmas[v] = max(1.0, self.sigma_scale * (hi - lo))

        # Find an initial state in R_pi.
        init_state: dict[str, int] | None = self._initialize(
            region, distribution, smt, rng, var_names
        )
        if init_state is None:
            empty = {v: np.empty(0, dtype=np.int64) for v in var_names}
            return SampleBatch(inputs=empty, n=0, rejection_ratio=0.0)

        # MH chain.
        x = dict(init_state)
        log_target_x = self._log_target(x, distribution, smt, region)
        # Burn-in
        n_proposed = 0
        n_accepted = 0
        for _ in range(self.n_burn_in):
            x, log_target_x, accepted = self._mh_step(
                x, log_target_x, distribution, smt, region,
                sigmas, clip_bounds, var_names, rng,
            )
            n_proposed += 1
            n_accepted += int(accepted)
        # Sampling: collect n samples, one every `thin` steps.
        collected: dict[str, list[int]] = {v: [] for v in var_names}
        while len(collected[var_names[0]]) < n:
            for _ in range(self.thin):
                x, log_target_x, accepted = self._mh_step(
                    x, log_target_x, distribution, smt, region,
                    sigmas, clip_bounds, var_names, rng,
                )
                n_proposed += 1
                n_accepted += int(accepted)
            for v in var_names:
                collected[v].append(x[v])
        inputs = {v: np.array(collected[v], dtype=np.int64) for v in var_names}
        rej_ratio = n_accepted / n_proposed if n_proposed > 0 else 0.0
        return SampleBatch(inputs=inputs, n=n, rejection_ratio=rej_ratio)

    def _initialize(
        self,
        region: GeneralRegion,
        distribution: ProductDistribution,
        smt: SMTBackend,
        rng: np.random.Generator,
        var_names: list[str],
    ) -> dict[str, int] | None:
        # Rejection from the base envelope.
        attempts = 0
        while attempts < self.init_attempts:
            chunk = min(64, self.init_attempts - attempts)
            batch = region.base.sample(distribution, smt, rng, chunk)
            for x in batch.iter_assignments():
                try:
                    if smt.evaluate(region.formula, x):
                        return x
                except (ValueError, KeyError, ZeroDivisionError, ArithmeticError):
                    pass
            attempts += chunk
        return None

    def _log_target(
        self,
        x: dict[str, int],
        distribution: ProductDistribution,
        smt: SMTBackend,
        region: GeneralRegion,
    ) -> float:
        try:
            if not smt.evaluate(region.formula, x):
                return float("-inf")
        except (ValueError, KeyError, ZeroDivisionError, ArithmeticError):
            return float("-inf")
        # Also: x must lie in the envelope (per construction).
        if not region.base.contains(x):
            return float("-inf")
        # log D(x)
        p = distribution.pmf(x)
        if p <= 0.0:
            return float("-inf")
        return math.log(p)

    def _mh_step(
        self,
        x: dict[str, int],
        log_target_x: float,
        distribution: ProductDistribution,
        smt: SMTBackend,
        region: GeneralRegion,
        sigmas: dict[str, float],
        clip_bounds: dict[str, tuple[int, int]],
        var_names: list[str],
        rng: np.random.Generator,
    ) -> tuple[dict[str, int], float, bool]:
        # Symmetric proposal: rounded Gaussian on each coordinate.
        x_new: dict[str, int] = {}
        for v in var_names:
            step = int(round(float(rng.normal(0.0, sigmas[v]))))
            lo, hi = clip_bounds[v]
            x_new[v] = max(lo, min(hi, x[v] + step))
        log_target_new = self._log_target(x_new, distribution, smt, region)
        if log_target_new == float("-inf"):
            return x, log_target_x, False
        log_alpha = log_target_new - log_target_x
        if log_alpha >= 0.0 or rng.random() < math.exp(log_alpha):
            return x_new, log_target_new, True
        return x, log_target_x, False


__all__ = ["Sampler", "RejectionSampler", "IntegerLatticeMHSampler"]
