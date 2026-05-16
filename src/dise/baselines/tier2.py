"""Tier-2 sampling baselines: ablations isolating the bound vs.
stratification contributions.

Five baselines that share the :class:`Baseline` protocol and slot
directly into the experiment harness.  Together they form a clean
ladder against which DiSE's gain decomposes:

1. :class:`PlainMonteCarloHoeffding`
   --- the textbook statistical-model-checking bound.  Strictly
   conservative; the reference every SMC paper compares against.

2. :class:`PlainMonteCarloEmpiricalBernstein`
   --- Maurer-Pontil 2009 empirical-Bernstein on the plain sample
   mean.  Variance-adaptive but not anytime-valid.

3. :class:`PlainMonteCarloBetting`
   --- WSR 2024 predictable-plug-in empirical-Bernstein on the plain
   sample mean.  Anytime-valid, variance-adaptive.  Paired with
   ``dise_betting`` this isolates *what the bound contributes* from
   *what the symbolic stratification contributes*.

4. :class:`QuasiMonteCarloSobol`
   --- Low-discrepancy (Sobol) point set + Wilson interval.  Standard
   variance-reduction baseline from numerical simulation.  Reported
   interval uses the i.i.d. Wilson bound, which is technically a
   heuristic for deterministic point sets (Koksma-Hlawka would be
   tighter); we note the caveat in the ``extras`` dict.

5. :class:`AdaptiveStratifiedMC`
   --- Two-pass Neyman-allocation stratification (Carpentier-Munos
   2011 spirit): pilot pass to estimate per-stratum variance, then
   allocate the remaining budget proportional to ``w_k * sigma_k``.
   The strongest *pure-sampling* stratifier --- no SMT guidance ---
   so the gap to DiSE measures what symbolic refinement adds beyond
   adaptive sampling.
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
    prpl_eb_halfwidth_anytime,
    wilson_halfwidth_for_leaf,
)
from . import Baseline, BaselineResult


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _draw_and_run(
    program: Callable[..., Any],
    distribution: Mapping[str, Distribution],
    property_fn: Callable[[Any], bool],
    n: int,
    rng: np.random.Generator,
) -> tuple[list[int], int]:
    """Draw ``n`` iid samples from the product distribution, run the
    program on each, and return ``(phi_values, hits)``.

    The ``phi_values`` list is the per-sample Bernoulli stream
    (used by the WSR PrPl-EB construction).
    """
    dist = ProductDistribution(factors=dict(distribution))
    batch = dist.sample(rng, n)
    keys = list(distribution.keys())
    phis: list[int] = []
    hits = 0
    for i in range(n):
        x = {k: int(batch[k][i]) for k in keys}
        out = program(**x)
        v = 1 if bool(property_fn(out)) else 0
        phis.append(v)
        hits += v
    return phis, hits


# ---------------------------------------------------------------------------
# 1. Plain MC + Hoeffding
# ---------------------------------------------------------------------------


def _hoeffding_halfwidth(n: int, delta: float) -> float:
    """Two-sided Hoeffding half-width for Bernoulli observations:

    .. math::
       h = \\sqrt{\\log(2/\\delta) / (2 n)}.
    """
    if n <= 0:
        return 1.0
    return math.sqrt(math.log(2.0 / delta) / (2.0 * n))


class PlainMonteCarloHoeffding(Baseline):
    """Plain MC with the textbook one-sided Hoeffding-union bound.

    Used as the SMC reference in PLASMA-Lab, Storm SMC, and the
    Sampson passert paper.  The bound does not require knowing the
    variance and is sound at any fixed sample size, but does not
    adapt to low-variance regimes.
    """

    name = "plain_mc_hoeffding"

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
        t0 = time.perf_counter()
        phis, hits = _draw_and_run(program, distribution, property_fn, budget, rng)
        wall = time.perf_counter() - t0
        mu_hat = hits / budget if budget > 0 else 0.0
        half = _hoeffding_halfwidth(budget, delta)
        lo = max(0.0, mu_hat - half)
        hi = min(1.0, mu_hat + half)
        return BaselineResult(
            name=self.name,
            mu_hat=mu_hat,
            interval=(lo, hi),
            samples_used=budget,
            wall_clock_s=wall,
            delta=delta,
            extras={"hits": hits, "half_width": half, "bound": "hoeffding"},
        )


# ---------------------------------------------------------------------------
# 2. Plain MC + Maurer-Pontil empirical Bernstein
# ---------------------------------------------------------------------------


class PlainMonteCarloEmpiricalBernstein(Baseline):
    """Plain MC with the Maurer-Pontil 2009 empirical-Bernstein bound.

    Variance-adaptive but not anytime-valid.  Strictly tighter than
    Hoeffding when the empirical variance is small.
    """

    name = "plain_mc_eb"

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
        t0 = time.perf_counter()
        phis, hits = _draw_and_run(program, distribution, property_fn, budget, rng)
        wall = time.perf_counter() - t0
        mu_hat = hits / budget if budget > 0 else 0.0
        # Wilson-smoothed Bernoulli variance plug-in (also used by DiSE
        # to avoid the n=0 / extreme-p degeneracy).
        p_tilde = (hits + 1) / (budget + 2) if budget > 0 else 0.5
        v_hat = p_tilde * (1.0 - p_tilde)
        half = empirical_bernstein_halfwidth_mp(v_hat, budget, delta, range_bound=1.0)
        half = min(half, 1.0)
        lo = max(0.0, mu_hat - half)
        hi = min(1.0, mu_hat + half)
        return BaselineResult(
            name=self.name,
            mu_hat=mu_hat,
            interval=(lo, hi),
            samples_used=budget,
            wall_clock_s=wall,
            delta=delta,
            extras={
                "hits": hits,
                "half_width": half,
                "bound": "maurer-pontil-eb",
                "v_hat": v_hat,
            },
        )


# ---------------------------------------------------------------------------
# 3. Plain MC + WSR PrPl-EB betting CI
# ---------------------------------------------------------------------------


class PlainMonteCarloBetting(Baseline):
    """Plain MC with the WSR 2024 predictable-plug-in
    empirical-Bernstein anytime-valid bound.

    Crucial ablation against ``dise_betting``: shares the *bound*,
    differs in the *stratification*.  Any gap is attributable to
    DiSE's symbolic partitioning.
    """

    name = "plain_mc_betting"

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
        t0 = time.perf_counter()
        phis, hits = _draw_and_run(program, distribution, property_fn, budget, rng)
        wall = time.perf_counter() - t0
        mu_hat = hits / budget if budget > 0 else 0.0
        half = prpl_eb_halfwidth_anytime(phis, delta)
        half = min(half, 1.0)
        lo = max(0.0, mu_hat - half)
        hi = min(1.0, mu_hat + half)
        return BaselineResult(
            name=self.name,
            mu_hat=mu_hat,
            interval=(lo, hi),
            samples_used=budget,
            wall_clock_s=wall,
            delta=delta,
            extras={
                "hits": hits,
                "half_width": half,
                "bound": "wsr-prpl-eb",
            },
        )


# ---------------------------------------------------------------------------
# 4. Quasi-Monte Carlo with a Sobol low-discrepancy sequence
# ---------------------------------------------------------------------------


def _sobol_int_batch(
    n: int,
    distribution: Mapping[str, Distribution],
    seed: int,
) -> dict[str, np.ndarray]:
    """Generate ``n`` samples per coordinate from a scrambled Sobol
    sequence, transformed onto each marginal's support via the inverse CDF.

    Falls back to plain iid sampling if SciPy is unavailable or the
    support is too small (Sobol degenerates on small alphabets).
    """
    try:
        from scipy.stats import qmc
    except ImportError:  # pragma: no cover
        rng = np.random.default_rng(seed)
        dist = ProductDistribution(factors=dict(distribution))
        return dist.sample(rng, n)

    d = len(distribution)
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    # Sobol prefers power-of-2; round up.
    m = max(1, int(math.ceil(math.log2(max(n, 1)))))
    pts = sampler.random_base2(m)[:n]  # uniform [0, 1]^d
    out: dict[str, np.ndarray] = {}
    for i, (var_name, marginal) in enumerate(distribution.items()):
        lo, hi = marginal.support_bounds(eps=1e-12)
        # Map [0,1) -> integer support [lo, hi] via floor.
        u = pts[:, i]
        vals = np.floor(lo + u * (hi - lo + 1)).astype(int)
        vals = np.clip(vals, lo, hi)
        out[var_name] = vals
    return out


class QuasiMonteCarloSobol(Baseline):
    """Quasi-Monte Carlo with a scrambled Sobol point set.

    Standard variance-reduction baseline from numerical simulation.
    The reported interval uses the i.i.d. Wilson bound, which is a
    *heuristic* for deterministic point sets (Koksma-Hlawka would be
    tighter, but requires bounded-variation assumptions on the
    integrand).  We document the caveat in ``extras["caveat"]``.
    """

    name = "quasi_mc_sobol"

    def run(
        self,
        program: Callable[..., Any],
        distribution: Mapping[str, Distribution],
        property_fn: Callable[[Any], bool],
        budget: int,
        delta: float,
        seed: int,
    ) -> BaselineResult:
        t0 = time.perf_counter()
        batch = _sobol_int_batch(budget, distribution, seed)
        keys = list(distribution.keys())
        hits = 0
        for i in range(budget):
            x = {k: int(batch[k][i]) for k in keys}
            out = program(**x)
            if bool(property_fn(out)):
                hits += 1
        wall = time.perf_counter() - t0
        mu_hat = hits / budget if budget > 0 else 0.0
        half = wilson_halfwidth_for_leaf(budget, hits, delta)
        lo = max(0.0, mu_hat - half)
        hi = min(1.0, mu_hat + half)
        return BaselineResult(
            name=self.name,
            mu_hat=mu_hat,
            interval=(lo, hi),
            samples_used=budget,
            wall_clock_s=wall,
            delta=delta,
            extras={
                "hits": hits,
                "half_width": half,
                "bound": "wilson",
                "caveat": "Wilson is heuristic for non-iid Sobol points",
            },
        )


# ---------------------------------------------------------------------------
# 5. Adaptive stratified MC (Carpentier-Munos 2011 spirit)
# ---------------------------------------------------------------------------


class AdaptiveStratifiedMC(Baseline):
    """Two-pass adaptive stratification with Neyman allocation.

    Implements the spirit of Carpentier & Munos (NeurIPS 2011)
    *Finite Time Analysis of Stratified Sampling for Monte Carlo*:

    1. **Pilot pass** ($n_0 = $ ``pilot_frac * budget`` samples,
       distributed across ``n_strata`` random hash buckets).  Estimate
       per-stratum hit-rate and variance.
    2. **Adaptive pass** (remaining samples).  Allocate each stratum
       $n_k \\propto w_k \\sqrt{v_k}$ where $w_k$ is the empirical
       bucket weight and $v_k$ the Wilson-smoothed variance.  Resample
       from the bucket (rejection-sample by drawing fresh inputs and
       hashing).

    The reported interval is the standard per-stratum Wilson +
    Bonferroni union bound, as in :class:`StratifiedRandomMC`.

    Strata here are still hash-based (no symbolic guidance) --- the
    *adaptive allocation* is the only difference.  The gap between
    this baseline and DiSE measures the value of SMT-driven
    refinement beyond pure variance-adaptive sampling.
    """

    name = "adaptive_stratified"

    def __init__(self, n_strata: int = 16, pilot_frac: float = 0.3) -> None:
        if n_strata < 1:
            raise ValueError("n_strata must be >= 1")
        if not 0.0 < pilot_frac < 1.0:
            raise ValueError("pilot_frac must be in (0, 1)")
        self.n_strata = n_strata
        self.pilot_frac = pilot_frac

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

        def _stratum_of(x: dict[str, int]) -> int:
            h = hash(tuple(sorted(x.items())))
            return (h % self.n_strata + self.n_strata) % self.n_strata

        n_pilot = max(self.n_strata, int(self.pilot_frac * budget))
        n_pilot = min(n_pilot, budget)
        counts = [0] * self.n_strata
        hits = [0] * self.n_strata

        t0 = time.perf_counter()
        # Pilot pass: uniform iid sampling, record bucket / hit counts.
        pilot_batch = dist.sample(rng, n_pilot)
        for i in range(n_pilot):
            x = {k: int(pilot_batch[k][i]) for k in keys}
            k_idx = _stratum_of(x)
            counts[k_idx] += 1
            if bool(property_fn(program(**x))):
                hits[k_idx] += 1

        remaining = budget - n_pilot
        if remaining > 0:
            # Neyman allocation on remaining budget.
            n0 = sum(counts)
            weights = [c / n0 if n0 else 1.0 / self.n_strata for c in counts]
            sigmas: list[float] = []
            for k in range(self.n_strata):
                n_k = counts[k]
                if n_k == 0:
                    sigmas.append(0.5)  # max-variance prior on a "wasted" bucket
                    continue
                p_tilde = (hits[k] + 1) / (n_k + 2)
                sigmas.append(math.sqrt(max(p_tilde * (1.0 - p_tilde), 1e-9)))
            scores = [weights[k] * sigmas[k] for k in range(self.n_strata)]
            total = sum(scores) or 1.0
            allocations = [int(round(remaining * s / total)) for s in scores]
            # Fix rounding drift.
            drift = remaining - sum(allocations)
            for k in sorted(range(self.n_strata), key=lambda i: -scores[i]):
                if drift == 0:
                    break
                step = 1 if drift > 0 else -1
                if allocations[k] + step >= 0:
                    allocations[k] += step
                    drift -= step

            # Adaptive pass: for each stratum, rejection-sample fresh
            # inputs that hash to it.  This preserves the conditional
            # distribution P[x | bucket(x) = k] = P[x] (since strata
            # are hash partitions).
            for k in range(self.n_strata):
                need = allocations[k]
                if need <= 0:
                    continue
                drawn = 0
                # Cap the attempts to avoid infinite loops on tiny strata.
                max_attempts = need * 32
                attempts = 0
                while drawn < need and attempts < max_attempts:
                    block = max(need - drawn, 1)
                    sub_batch = dist.sample(rng, block)
                    for j in range(block):
                        attempts += 1
                        if attempts > max_attempts:
                            break
                        x = {kk: int(sub_batch[kk][j]) for kk in keys}
                        if _stratum_of(x) != k:
                            continue
                        counts[k] += 1
                        drawn += 1
                        if bool(property_fn(program(**x))):
                            hits[k] += 1
                        if drawn >= need:
                            break

        wall = time.perf_counter() - t0
        n_total = sum(counts)
        n_nonempty = sum(1 for c in counts if c > 0)
        delta_per = delta / max(n_nonempty, 1)
        mu_hat = 0.0
        eps_stat = 0.0
        for k in range(self.n_strata):
            n_k = counts[k]
            if n_k == 0:
                continue
            w_k = n_k / n_total
            p_k = hits[k] / n_k
            mu_hat += w_k * p_k
            eps_stat += w_k * wilson_halfwidth_for_leaf(n_k, hits[k], delta_per)
        eps_stat = min(eps_stat, 1.0)
        lo = max(0.0, mu_hat - eps_stat)
        hi = min(1.0, mu_hat + eps_stat)
        return BaselineResult(
            name=self.name,
            mu_hat=mu_hat,
            interval=(lo, hi),
            samples_used=n_total,
            wall_clock_s=wall,
            delta=delta,
            extras={
                "n_strata": self.n_strata,
                "n_pilot": n_pilot,
                "bucket_counts": counts,
                "bucket_hits": hits,
                "bound": "wilson-per-stratum",
                "allocation": "neyman-w_sigma",
            },
        )


__all__ = [
    "AdaptiveStratifiedMC",
    "PlainMonteCarloBetting",
    "PlainMonteCarloEmpiricalBernstein",
    "PlainMonteCarloHoeffding",
    "QuasiMonteCarloSobol",
]
