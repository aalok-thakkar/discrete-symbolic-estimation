"""State-of-the-art (SoTA) sampling baselines for empirical comparison.

DiSE solves: estimate :math:`\\mu = \\Pr_{x \\sim D}[\\varphi(P(x)) = 1]`
with a certified two-sided half-width at confidence :math:`1 - \\delta`.

The two baselines in :mod:`dise.baselines` (``PlainMonteCarlo`` and
``StratifiedRandomMC``) are deliberately weak — they fix the sample
count up front. The methods here are *adaptive*: each draws samples
in batches and decides at every batch whether the certified
interval is tight enough to stop. They share with DiSE the
"anytime-valid, data-dependent stopping" property; only the
*structural* (symbolic) component is missing. This is the cleanest
apples-to-apples comparison for the headline claim that
SMT-driven stratification beats adaptive concentration alone.

Three methods are implemented:

* :class:`AdaptiveHoeffding` — sequential Hoeffding bound with a
  Basel-mass union-bound-in-time correction. Algorithmic core of
  Sampson, Panchekha, Mytkowicz, McKinley, Grossman, Ceze,
  *Expressing and verifying probabilistic assertions* (PLDI 2014).
* :class:`EmpiricalBernsteinStopping` — Maurer-Pontil
  empirical-Bernstein bound with a geometric union-bound-in-time
  schedule. Mnih, Szepesvári, Audibert,
  *Empirical Bernstein Stopping* (ICML 2008).
* :class:`BettingConfidenceSequence` — Hedged-capital betting
  martingale with predictable-plug-in betting fractions, inverted
  via Ville's inequality. Waudby-Smith & Ramdas,
  *Estimating means of bounded random variables by betting*
  (JRSS-B 2024). This is the current SoTA for tight anytime-valid
  intervals on bounded means.

All three are *sound* under data-dependent stopping at confidence
:math:`1 - \\delta`. They share a common driver (sample in batches,
update a certified interval, stop when half-width :math:`\\le
\\varepsilon` or budget exhausted).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from ..distributions import Distribution, ProductDistribution
from ..estimator import (
    empirical_bernstein_halfwidth_mp,
    wilson_halfwidth_for_leaf,
)
from . import Baseline, BaselineResult

# ---------------------------------------------------------------------------
# Shared driver
# ---------------------------------------------------------------------------


def _sample_one(
    rng: np.random.Generator,
    dist: ProductDistribution,
    keys: list[str],
    program: Callable[..., Any],
    property_fn: Callable[[Any], bool],
) -> bool:
    """Draw one input, run the program, return the Boolean property value."""
    batch = dist.sample(rng, 1)
    x = {k: int(batch[k][0]) for k in keys}
    return bool(property_fn(program(**x)))


# ---------------------------------------------------------------------------
# Adaptive Hoeffding (Sampson et al, PLDI 2014)
# ---------------------------------------------------------------------------


def _hoeffding_anytime_halfwidth(n: int, delta: float) -> float:
    r"""Hoeffding bound with a Basel-mass union-bound-in-time correction.

    For an iid stream of [0,1]-bounded variables, returns :math:`h_n`
    such that

    .. math::
        \Pr\!\big[\,\exists n \ge 1 : |\bar X_n - \mu| > h_n\,\big]
        \;\le\; \delta.

    Construction: at sample :math:`n`, use the fixed-:math:`n`
    Hoeffding bound at confidence :math:`\delta_n = 6\delta/(\pi^2 n^2)`;
    Basel identity :math:`\sum_n 6/(\pi^2 n^2) = 1` then yields the
    union bound. This is the bound used (in spirit) by Sampson et
    al. (PLDI 2014, §4) to make the sequential ``passert`` check
    sound under adaptive sample sizes.
    """
    if n <= 0:
        return 1.0
    delta_n = 6.0 * delta / (math.pi * math.pi * n * n)
    delta_n = min(delta_n, 0.999)
    return math.sqrt(math.log(2.0 / delta_n) / (2.0 * n))


class AdaptiveHoeffding(Baseline):
    """Sequential Hoeffding stopping (Sampson et al., PLDI 2014).

    Algorithm. Draw samples one at a time (in batches of ``batch_size``
    for efficiency). After each batch, compute the empirical mean
    :math:`\\hat\\mu_n` and a Hoeffding-style half-width :math:`h_n` valid
    under data-dependent stopping. Stop when either:

    * :math:`h_n \\le \\varepsilon` (target accuracy reached), or
    * the sample budget is exhausted.

    The half-width uses a Basel-mass union-bound-in-time correction
    (:func:`_hoeffding_anytime_halfwidth`) so the bound is sound under
    arbitrary adaptive stopping.

    Notes
    -----
    Sampson et al.'s original ``passert`` paper uses a sequential
    Hoeffding test on each ``passert`` site; the bound here is the
    standard anytime-valid version of that test. This is the *direct*
    SoTA comparator for the assertion-violation framing (DiSE's
    :func:`failure_probability` API).
    """

    name = "adaptive_hoeffding"

    def __init__(self, epsilon: float = 0.05, batch_size: int = 50) -> None:
        if epsilon <= 0.0 or epsilon >= 1.0:
            raise ValueError("epsilon must be in (0, 1)")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.epsilon = epsilon
        self.batch_size = batch_size

    def run(
        self,
        program: Callable[..., Any],
        distribution: Mapping[str, Distribution],
        property_fn: Callable[[Any], bool],
        budget: int,
        delta: float,
        seed: int,
    ) -> BaselineResult:
        rng = np.random.default_rng(seed)
        dist = ProductDistribution(factors=dict(distribution))
        keys = list(distribution.keys())
        hits = 0
        n = 0
        t0 = time.perf_counter()
        terminated_reason = "budget_exhausted"
        half = 1.0
        while n < budget:
            remaining = budget - n
            this_batch = min(self.batch_size, remaining)
            batch = dist.sample(rng, this_batch)
            for i in range(this_batch):
                x = {k: int(batch[k][i]) for k in keys}
                if bool(property_fn(program(**x))):
                    hits += 1
                n += 1
            half = _hoeffding_anytime_halfwidth(n, delta)
            if half <= self.epsilon:
                terminated_reason = "epsilon_reached"
                break
        wall = time.perf_counter() - t0
        mu_hat = hits / n if n > 0 else 0.0
        lo = max(0.0, mu_hat - half)
        hi = min(1.0, mu_hat + half)
        return BaselineResult(
            name=self.name,
            mu_hat=mu_hat,
            interval=(lo, hi),
            samples_used=n,
            wall_clock_s=wall,
            delta=delta,
            extras={
                "hits": hits,
                "half_width": half,
                "terminated_reason": terminated_reason,
                "epsilon": self.epsilon,
            },
        )


# ---------------------------------------------------------------------------
# Empirical Bernstein Stopping (Mnih, Szepesvári, Audibert, ICML 2008)
# ---------------------------------------------------------------------------


def _eb_anytime_halfwidth(n: int, v_emp: float, delta: float) -> float:
    r"""Maurer-Pontil empirical-Bernstein bound with Basel-mass union-bound-in-time.

    Returns :math:`h_n` such that for an iid stream of [0,1]-bounded
    variables with empirical variance :math:`\widehat V_n`,

    .. math::
        \Pr\!\big[\,\exists n \ge 2 : |\bar X_n - \mu| > h_n\,\big]
        \;\le\; \delta.

    The same union-bound device as :func:`_hoeffding_anytime_halfwidth`
    but applied to the MP empirical-Bernstein bound, which is tight
    when :math:`\mu \in \{0, 1\}` (variance vanishes) — a regime the
    Hoeffding bound cannot exploit.
    """
    if n <= 1:
        return 1.0
    delta_n = 6.0 * delta / (math.pi * math.pi * n * n)
    delta_n = min(delta_n, 0.999)
    return empirical_bernstein_halfwidth_mp(v_emp, n, delta_n, range_bound=1.0)


class EmpiricalBernsteinStopping(Baseline):
    """Sequential empirical-Bernstein stopping (Mnih–Szepesvári–Audibert, ICML 2008).

    Algorithm. Same outer loop as :class:`AdaptiveHoeffding` but uses
    the Maurer-Pontil empirical-Bernstein bound instead of Hoeffding.
    The EB bound adapts to the *observed* sample variance, which
    matters dramatically for Bernoulli observations near 0 or 1
    (e.g. ``popcount_w6`` with property "popcount ≥ 0" — variance
    is *zero* on every sample, so EB closes the interval in O(1/ε)
    samples while Hoeffding takes O(1/ε²)).

    Notes
    -----
    The original 2008 paper uses a *geometric* union-bound-in-time
    schedule (sample sizes :math:`n_k = \\beta^k`) for theoretical
    optimality. Our implementation uses the Basel-mass schedule
    (:math:`\\delta_n = 6\\delta/(\\pi^2 n^2)`) for simplicity and
    direct comparability with :class:`AdaptiveHoeffding`. Both are
    sound and within constant factors of each other.
    """

    name = "ebstop"

    def __init__(self, epsilon: float = 0.05, batch_size: int = 50) -> None:
        if epsilon <= 0.0 or epsilon >= 1.0:
            raise ValueError("epsilon must be in (0, 1)")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.epsilon = epsilon
        self.batch_size = batch_size

    def run(
        self,
        program: Callable[..., Any],
        distribution: Mapping[str, Distribution],
        property_fn: Callable[[Any], bool],
        budget: int,
        delta: float,
        seed: int,
    ) -> BaselineResult:
        rng = np.random.default_rng(seed)
        dist = ProductDistribution(factors=dict(distribution))
        keys = list(distribution.keys())
        hits = 0
        sum_sq = 0.0  # for tracking sum of X_i^2 (Bernoulli: X^2 = X)
        n = 0
        t0 = time.perf_counter()
        terminated_reason = "budget_exhausted"
        half = 1.0
        while n < budget:
            remaining = budget - n
            this_batch = min(self.batch_size, remaining)
            batch = dist.sample(rng, this_batch)
            for i in range(this_batch):
                x = {k: int(batch[k][i]) for k in keys}
                obs = 1.0 if bool(property_fn(program(**x))) else 0.0
                hits += int(obs)
                sum_sq += obs * obs
                n += 1
            mu_hat = hits / n
            # Bernoulli empirical variance: E[X^2] - E[X]^2 = p - p^2 = p(1-p).
            v_emp = max(0.0, mu_hat * (1.0 - mu_hat))
            half = _eb_anytime_halfwidth(n, v_emp, delta)
            if half <= self.epsilon:
                terminated_reason = "epsilon_reached"
                break
        wall = time.perf_counter() - t0
        mu_hat = hits / n if n > 0 else 0.0
        lo = max(0.0, mu_hat - half)
        hi = min(1.0, mu_hat + half)
        return BaselineResult(
            name=self.name,
            mu_hat=mu_hat,
            interval=(lo, hi),
            samples_used=n,
            wall_clock_s=wall,
            delta=delta,
            extras={
                "hits": hits,
                "half_width": half,
                "terminated_reason": terminated_reason,
                "epsilon": self.epsilon,
                "empirical_variance": v_emp if n > 0 else None,
            },
        )


# ---------------------------------------------------------------------------
# Betting Confidence Sequence (Waudby-Smith & Ramdas, JRSS-B 2024)
# ---------------------------------------------------------------------------


class BettingConfidenceSequence(Baseline):
    r"""Hedged-capital betting CS (Waudby-Smith & Ramdas, JRSS-B 2024).

    The SoTA construction for anytime-valid confidence sequences on
    bounded means. For each candidate mean :math:`m \in [0, 1]` we
    maintain a *capital process*

    .. math::
        \mathcal{K}_t^+(m) \;=\; \prod_{i \le t}\, (1 + \lambda_i^+ \cdot (X_i - m)), \\
        \mathcal{K}_t^-(m) \;=\; \prod_{i \le t}\, (1 - \lambda_i^- \cdot (X_i - m)),

    with predictable betting fractions :math:`\lambda_i^\pm \in [0, 1/m)`
    or :math:`[0, 1/(1-m))`. By Ville's inequality, the set

    .. math::
        \mathcal{C}_t \;=\; \big\{m : \max(\mathcal{K}_t^+(m), \mathcal{K}_t^-(m)) < 1/\delta\big\}

    is a :math:`(1-\delta)`-coverage anytime-valid CS for :math:`\mu`.

    Implementation. We use the "PrPl" (predictable plug-in)
    betting fractions of WS-R §3.2: at step :math:`t`,

    .. math::
        \lambda_t^*(m) \;=\; \sqrt{\frac{2 \log(2/\delta)}{\hat\sigma_{t-1}^2 \,t \,\log(1+t)}}

    truncated to :math:`[0, c/m]` (resp. :math:`[0, c/(1-m)]`) with
    :math:`c = 0.5`. The :math:`m`-axis is discretized on a 1024-point
    grid; the CS endpoints are the leftmost/rightmost grid points
    still inside :math:`\mathcal C_t`.

    Notes
    -----
    This baseline is the *most-comparable-to-DiSE* member of the
    sampling family: like DiSE's ``method="anytime"`` it provides
    anytime-valid certified intervals under arbitrary adaptive
    stopping, and like DiSE it adapts its sampling cost to the
    intrinsic variance of the property. Unlike DiSE it does no
    symbolic reasoning, so the comparison isolates the
    structural-stratification contribution.

    References
    ----------
    Waudby-Smith & Ramdas, *Estimating means of bounded random
    variables by betting*, JRSS-B 86(1):1–27, 2024.
    """

    name = "betting_cs"

    def __init__(
        self,
        epsilon: float = 0.05,
        batch_size: int = 50,
        grid_size: int = 1024,
        c_truncate: float = 0.5,
    ) -> None:
        if epsilon <= 0.0 or epsilon >= 1.0:
            raise ValueError("epsilon must be in (0, 1)")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if grid_size < 8:
            raise ValueError("grid_size must be >= 8")
        if not (0.0 < c_truncate < 1.0):
            raise ValueError("c_truncate must be in (0, 1)")
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.grid_size = grid_size
        self.c_truncate = c_truncate

    def _step(
        self,
        log_cap_plus: np.ndarray,
        log_cap_minus: np.ndarray,
        m_grid: np.ndarray,
        x_t: float,
        mu_hat_prev: float,
        var_hat_prev: float,
        t: int,
        log_2_over_delta: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update log-capital processes by one observation.

        PrPl betting fraction:
            lambda_t* = sqrt(2 log(2/delta) / (sigma_hat^2 * t * log(1+t)))
        truncated to [0, c/m]  (resp. [0, c/(1-m)]).
        """
        # Use mu_hat_prev and a regularized variance estimate. The
        # WS-R recipe uses sample variance with a small Laplace prior
        # to avoid lambda blowup at t=0.
        sigma2 = max(var_hat_prev, 1e-4)
        denom = sigma2 * max(t, 1) * math.log(1.0 + max(t, 1))
        lam_star = math.sqrt(2.0 * log_2_over_delta / denom)

        # Per-m truncation: lambda^+ <= c/m,  lambda^- <= c/(1-m).
        # On the boundary m=0 or m=1, the corresponding bet would
        # blow up; clip via numpy.
        eps = 1e-12
        lam_plus = np.minimum(lam_star, self.c_truncate / np.maximum(m_grid, eps))
        lam_minus = np.minimum(lam_star, self.c_truncate / np.maximum(1.0 - m_grid, eps))

        # Increment: log(1 + lambda^+ (x - m)) and log(1 + lambda^- (m - x)).
        # For Bernoulli x ∈ {0, 1} both factors are positive when
        # lambda is in [0, 1/m) or [0, 1/(1-m)] respectively.
        plus_inc = 1.0 + lam_plus * (x_t - m_grid)
        minus_inc = 1.0 + lam_minus * (m_grid - x_t)
        # Numerical guard.
        plus_inc = np.maximum(plus_inc, 1e-300)
        minus_inc = np.maximum(minus_inc, 1e-300)
        log_cap_plus = log_cap_plus + np.log(plus_inc)
        log_cap_minus = log_cap_minus + np.log(minus_inc)
        return log_cap_plus, log_cap_minus

    def run(
        self,
        program: Callable[..., Any],
        distribution: Mapping[str, Distribution],
        property_fn: Callable[[Any], bool],
        budget: int,
        delta: float,
        seed: int,
    ) -> BaselineResult:
        rng = np.random.default_rng(seed)
        dist = ProductDistribution(factors=dict(distribution))
        keys = list(distribution.keys())
        # Grid of candidate means.
        m_grid = np.linspace(0.0, 1.0, self.grid_size)
        log_cap_plus = np.zeros_like(m_grid)
        log_cap_minus = np.zeros_like(m_grid)
        log_threshold = math.log(1.0 / delta)
        log_2_over_delta = math.log(2.0 / delta)
        hits = 0
        sum_sq = 0.0
        n = 0
        t0 = time.perf_counter()
        terminated_reason = "budget_exhausted"
        half = 1.0
        lo, hi = 0.0, 1.0
        while n < budget:
            remaining = budget - n
            this_batch = min(self.batch_size, remaining)
            batch = dist.sample(rng, this_batch)
            for i in range(this_batch):
                x = {k: int(batch[k][i]) for k in keys}
                obs = 1.0 if bool(property_fn(program(**x))) else 0.0
                # Running stats *before* updating with the new obs
                # (predictable: lambda_t depends on F_{t-1}).
                mu_hat_prev = (hits / n) if n > 0 else 0.5
                if n >= 2:
                    # Bernoulli variance estimate.
                    var_hat_prev = max(0.0, mu_hat_prev * (1.0 - mu_hat_prev))
                    var_hat_prev = max(var_hat_prev, 1.0 / (4.0 * n))  # Laplace floor
                else:
                    var_hat_prev = 0.25  # max Bernoulli variance prior
                log_cap_plus, log_cap_minus = self._step(
                    log_cap_plus, log_cap_minus, m_grid, obs,
                    mu_hat_prev, var_hat_prev, n + 1, log_2_over_delta,
                )
                hits += int(obs)
                sum_sq += obs * obs
                n += 1
            # CS membership: m is in CS iff max(K+, K-) < 1/delta
            # equivalently log_cap_plus < log(1/delta) AND log_cap_minus < log(1/delta).
            in_cs = (log_cap_plus < log_threshold) & (log_cap_minus < log_threshold)
            if not in_cs.any():
                # Pathological — CS empty. Fall back to [mu_hat, mu_hat].
                mu_hat = hits / n
                lo, hi = mu_hat, mu_hat
                half = 0.0
            else:
                idx = np.nonzero(in_cs)[0]
                lo = float(m_grid[idx[0]])
                hi = float(m_grid[idx[-1]])
                half = (hi - lo) / 2.0
            if half <= self.epsilon:
                terminated_reason = "epsilon_reached"
                break
        wall = time.perf_counter() - t0
        mu_hat = hits / n if n > 0 else 0.5
        # Snap point estimate inside the CS.
        lo = max(0.0, lo)
        hi = min(1.0, hi)
        # If mu_hat is outside the CS (rare but possible), project it.
        if mu_hat < lo:
            mu_hat = lo
        elif mu_hat > hi:
            mu_hat = hi
        return BaselineResult(
            name=self.name,
            mu_hat=mu_hat,
            interval=(lo, hi),
            samples_used=n,
            wall_clock_s=wall,
            delta=delta,
            extras={
                "hits": hits,
                "half_width": half,
                "terminated_reason": terminated_reason,
                "epsilon": self.epsilon,
                "grid_size": self.grid_size,
            },
        )


__all__ = [
    "AdaptiveHoeffding",
    "EmpiricalBernsteinStopping",
    "BettingConfidenceSequence",
]
