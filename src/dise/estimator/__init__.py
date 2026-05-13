"""Statistical machinery for DiSE.

Aggregates a :class:`~dise.regions.Frontier` into a certified two-sided
interval. Implements:

* Theorem 1 (variance identity) — already exposed via
  ``FrontierNode.variance_contribution``.
* Bernstein half-width — bounds total estimator variance into a confidence
  half-width at confidence ``1 - delta``.
* Wilson half-width — per-leaf, for sanity-checking.
* The final certified interval:

  .. math::
    [\\max(0, \\hat\\mu - \\varepsilon_{\\text{stat}} - W_{\\text{open}}),
     \\min(1, \\hat\\mu + \\varepsilon_{\\text{stat}} + W_{\\text{open}})]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..regions import Frontier, Status


@dataclass
class EstimatorState:
    """Snapshot of the certified estimator at a given moment."""

    mu_hat: float
    variance: float
    eps_stat: float
    eps_mass: float
    W_open: float
    W_close: float
    delta: float
    interval: tuple[float, float]
    n_total_samples: int
    n_leaves: int
    n_open_leaves: int
    n_closed_leaves: int

    def __repr__(self) -> str:
        lo, hi = self.interval
        return (
            f"EstimatorState(mu_hat={self.mu_hat:.4f}, "
            f"interval=[{lo:.4f}, {hi:.4f}], "
            f"eps_stat={self.eps_stat:.4g}, W_open={self.W_open:.4g}, "
            f"W_close={self.W_close:.4g}, "
            f"delta={self.delta}, "
            f"samples={self.n_total_samples}, "
            f"leaves={self.n_leaves} (open={self.n_open_leaves}, closed={self.n_closed_leaves}))"
        )


def bernstein_halfwidth(
    variance: float,
    delta: float,
    per_sample_bound: float = 1.0,
) -> float:
    """Bernstein-style half-width on a sum of bounded independent r.v.s.

    Given a bound on total estimator variance ``V`` and a per-contribution
    range bound ``B``, returns

        sqrt(2 * V * log(2 / delta)) + (B / 3) * log(2 / delta).

    This is the classical Bernstein bound. It is *tight* on the variance
    term but its linear-in-``B`` correction is conservative when the
    sample-mean estimator is over many samples per leaf. Use
    :func:`empirical_bernstein_halfwidth_mp` for the Maurer-Pontil
    refinement when the per-leaf empirical variance is available.
    """
    if delta <= 0.0 or delta >= 1.0:
        raise ValueError("delta must be in (0, 1)")
    if variance < 0.0:
        variance = 0.0
    log_term = math.log(2.0 / delta)
    return math.sqrt(2.0 * variance * log_term) + (per_sample_bound / 3.0) * log_term


def empirical_bernstein_halfwidth_mp(
    empirical_variance: float,
    n: int,
    delta: float,
    range_bound: float = 1.0,
) -> float:
    """Maurer-Pontil empirical-Bernstein half-width.

    For iid samples :math:`X_1, ..., X_n` with :math:`|X_i| \\le M/2`
    (range ``range_bound = M``) and empirical mean :math:`\\bar X_n`,
    empirical variance :math:`\\widehat V_n`,

        :math:`\\Pr\\left[\\bar X_n - \\mathbb{E}[X] \\ge
        \\sqrt{\\frac{2 \\widehat V_n \\log(2/\\delta)}{n}}
        + \\frac{7 M \\log(2/\\delta)}{3(n - 1)}\\right] \\le \\delta.`

    Reference: Maurer & Pontil, "Empirical Bernstein bounds and sample
    variance penalization" (COLT 2009).
    """
    if delta <= 0.0 or delta >= 1.0:
        raise ValueError("delta must be in (0, 1)")
    if n <= 1:
        return range_bound
    if empirical_variance < 0.0:
        empirical_variance = 0.0
    log_term = math.log(2.0 / delta)
    return math.sqrt(2.0 * empirical_variance * log_term / n) + (
        7.0 * range_bound * log_term
    ) / (3.0 * (n - 1))


def wilson_halfwidth_for_leaf(n: int, h: int, delta: float) -> float:
    """Wilson-score half-width for a single Bernoulli leaf at fixed ``n``.

    Valid only at a *fixed* (non-data-dependent) sample size ``n``.
    For data-adaptive ``n`` use :func:`wilson_halfwidth_anytime`.

    For ``h == 0`` or ``h == n`` returns a non-zero half-width (no
    Bernoulli collapse).
    """
    if delta <= 0.0 or delta >= 1.0:
        raise ValueError("delta must be in (0, 1)")
    if n == 0:
        return 1.0
    z = _phi_inv_one_sided(1.0 - delta / 2.0)
    p = h / n
    denom = 1.0 + z * z / n
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(half, 1.0 / (n + 2))


def betting_halfwidth_anytime(
    n: int,
    h: int,
    delta: float,
    *,
    grid_size: int = 512,
    lambdas: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
) -> float:
    r"""Hedged-capital betting CS half-width for a Bernoulli leaf.

    Anytime-valid bound on the per-leaf disagreement rate, tighter than
    :func:`wilson_halfwidth_anytime` by a constant factor for moderate
    :math:`n` and :math:`\\bar X_n \\in (0.05, 0.95)`. Construction: a
    fixed-design hedged-capital betting confidence sequence (Waudby-
    Smith & Ramdas 2024, §3) with a small :math:`\\lambda`-grid and a
    Bonferroni-corrected per-:math:`\\lambda` test. For each candidate
    mean :math:`m`, every betting fraction either confirms membership
    of :math:`m` or excludes it; the :math:`(1-\\delta)`-CS is the
    intersection of non-exclusions.

    Returns the symmetric half-width
    :math:`\\max(\\bar X_n - \\inf \\mathcal C_n,\\ \\sup \\mathcal C_n -
    \\bar X_n)`. The empirical mean is always inside the CS by
    construction.

    Reference: Waudby-Smith & Ramdas, *Estimating means of bounded
    random variables by betting*, JRSS-B 2024.
    """
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")
    if n <= 0:
        return 1.0
    if h < 0 or h > n:
        raise ValueError(f"h={h!r} not in [0, n={n!r}]")
    if grid_size < 8:
        raise ValueError("grid_size must be >= 8")
    n_lam = len(lambdas)
    delta_per = delta / (2.0 * n_lam)
    log_thresh = math.log(1.0 / delta_per)
    m_grid = np.linspace(1e-6, 1.0 - 1e-6, grid_size)
    eps = 1e-12
    in_cs = np.ones(grid_size, dtype=bool)
    for lam in lambdas:
        lam_plus = np.minimum(lam, 1.0 / (m_grid + eps))
        factor_h_plus = 1.0 + lam_plus * (1.0 - m_grid)
        factor_nh_plus = np.maximum(1.0 - lam_plus * m_grid, eps)
        log_K_plus = h * np.log(factor_h_plus) + (n - h) * np.log(factor_nh_plus)
        lam_minus = np.minimum(lam, 1.0 / (1.0 - m_grid + eps))
        factor_h_minus = np.maximum(1.0 - lam_minus * (1.0 - m_grid), eps)
        factor_nh_minus = 1.0 + lam_minus * m_grid
        log_K_minus = h * np.log(factor_h_minus) + (n - h) * np.log(factor_nh_minus)
        in_cs &= (log_K_plus < log_thresh) & (log_K_minus < log_thresh)
    if not in_cs.any():
        return 1.0
    idx = np.nonzero(in_cs)[0]
    lo = float(m_grid[idx[0]])
    hi = float(m_grid[idx[-1]])
    mu_hat = h / n
    half = max(mu_hat - lo, hi - mu_hat)
    return max(half, 1.0 / (n + 2))


def wilson_halfwidth_anytime(n: int, h: int, delta: float) -> float:
    r"""Time-uniform Wilson-score half-width.

    For an iid Bernoulli stream :math:`X_1, X_2, \ldots` with mean
    :math:`\mu` and empirical mean :math:`\bar X_n = h/n`, returns a
    bound such that

    .. math::

        \Pr\!\big[\,\exists n \ge 1 : |\bar X_n - \mu| > h_n\,\big] \;\le\; \delta.

    Construction: apply the fixed-``n`` Wilson interval at the
    Bonferroni-in-time confidence
    :math:`\delta_n = 6\,\delta / (\pi^2 n^2)`. Since
    :math:`\sum_{n=1}^\infty 6 / (\pi^2 n^2) = 1` (Basel), the union
    bound

    .. math::

        \Pr\!\bigg[\,\bigcup_{n=1}^\infty \{|\bar X_n - \mu| > h_n\}\bigg]
        \;\le\; \sum_{n=1}^\infty \delta_n \;=\; \delta

    yields anytime validity — valid at every (data-dependent) stopping
    time on the sample-count axis. The half-width is slightly larger
    than the fixed-``n`` Wilson bound by roughly a
    :math:`\sqrt{\log n}` factor.

    This is the bound to use when sample sizes are chosen adaptively or
    the algorithm uses an optional-stopping rule on the observed
    interval — both true of ASIP.

    References
    ----------
    Howard, Ramdas, McAuliffe, Sekhon (2021), "Time-uniform Chernoff
    bounds via nonnegative supermartingales", Probab. Surv.;
    Robbins (1970), "Statistical methods related to the law of the
    iterated logarithm".
    """
    if delta <= 0.0 or delta >= 1.0:
        raise ValueError("delta must be in (0, 1)")
    if n <= 0:
        return 1.0
    # 6 / pi^2 normalizing constant from the Basel identity.
    delta_n = 6.0 * delta / (math.pi * math.pi * n * n)
    delta_n = min(delta_n, 0.999)  # numerical guard
    return wilson_halfwidth_for_leaf(n, h, delta_n)


def _phi_inv_one_sided(p: float) -> float:
    """Inverse standard-normal CDF via Acklam's rational approximation.

    Returns :math:`\\Phi^{-1}(p)` for :math:`p \\in (0, 1)`. Accurate to
    roughly seven significant digits across the full open interval —
    sufficient for :func:`wilson_halfwidth_for_leaf`, which needs
    :math:`z_{1 - \\delta/2}` at modest :math:`\\delta`. Avoids a
    ``scipy.special`` dependency on the hot path.

    References
    ----------
    P. Acklam, "An algorithm for computing the inverse normal
    cumulative distribution function" (2003), web-published note.
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in (0, 1)")
    # Coefficients from Acklam's algorithm.
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(
        ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
    ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def compute_estimator_state(
    frontier: Frontier,
    delta: float,
    method: Literal["bernstein", "wilson", "empirical-bernstein", "anytime"] = "wilson",
) -> EstimatorState:
    """Compute the certified estimator state from the current frontier.

    Methods:

    * ``"wilson"`` (default) — sum of per-open-leaf Wilson half-widths
      with Bonferroni correction. Tight for Bernoulli leaves at *fixed*
      sample counts. **Use this if your stopping rule is non-adaptive**
      (e.g. a fixed budget reached on every run).
    * ``"anytime"`` — sum of per-open-leaf *time-uniform* Wilson half-
      widths (Howard-Ramdas-McAuliffe-style; see
      :func:`wilson_halfwidth_anytime`). Valid under data-dependent
      stopping (e.g. ``epsilon_reached``) and adaptive per-leaf sample
      sizes. Slightly looser than ``"wilson"`` (by a
      :math:`\\sqrt{\\log n}` factor). **Use this for ATVA-style
      soundness claims under ASIP's adaptive schedule.**
    * ``"bernstein"`` — classical Bernstein bound on the total estimator
      variance. Conservative; soundness-only.
    * ``"empirical-bernstein"`` — Maurer-Pontil empirical-Bernstein
      bound. Tighter than ``"bernstein"`` when per-leaf empirical
      variance is available.

    Returns an :class:`EstimatorState` snapshot. When the frontier is
    fully resolved (no open leaves, all masses closed-form), ``eps_stat``
    is exactly 0.
    """
    leaves = frontier.leaves()
    open_leaves = [n for n in leaves if n.status == Status.OPEN]
    closed_leaves = [
        n
        for n in leaves
        if n.status in (Status.CLOSED_TRUE, Status.CLOSED_FALSE)
    ]
    empty_leaves = [n for n in leaves if n.status == Status.EMPTY]
    n_total_samples = sum(n.n_samples for n in leaves)

    mu_hat, variance = frontier.compute_mu_hat()
    W_open = frontier.open_mass()

    # When the estimator is fully resolved, eps_stat = 0.
    if not open_leaves and variance == 0.0:
        eps_stat = 0.0
    elif method == "bernstein":
        per_sample_bound = max((leaf.w_hat for leaf in open_leaves), default=1.0)
        per_sample_bound = max(per_sample_bound, 1e-12)
        eps_stat = bernstein_halfwidth(variance, delta, per_sample_bound)
    elif method == "empirical-bernstein":
        # Bonferroni: each open leaf gets delta / max(1, K).
        K = max(len(open_leaves), 1)
        delta_per_leaf = delta / K
        eps_stat = 0.0
        for leaf in open_leaves:
            if leaf.n_samples < 2:
                # MP requires n >= 2; fall back to leaf range (w_hat).
                eps_stat += leaf.w_hat
                continue
            # Empirical Bernoulli variance with Wilson smoothing.
            n, h = leaf.n_samples, leaf.n_hits
            p_tilde = (h + 1) / (n + 2)
            v_hat = p_tilde * (1.0 - p_tilde)
            eps_stat += leaf.w_hat * empirical_bernstein_halfwidth_mp(
                v_hat, n, delta_per_leaf, range_bound=1.0
            )
    elif method == "wilson":
        # Bonferroni over leaves; fixed-n Wilson per leaf.
        K = max(len(open_leaves), 1)
        delta_per_leaf = delta / K
        eps_stat = sum(
            leaf.w_hat
            * wilson_halfwidth_for_leaf(leaf.n_samples, leaf.n_hits, delta_per_leaf)
            for leaf in open_leaves
        )
    elif method == "anytime":
        # Bonferroni over leaves; per-leaf bound is min(Wilson-anytime,
        # betting CS), each evaluated at ``delta_per_leaf / 2`` so the
        # union of their failure events is bounded by delta_per_leaf.
        # The Wilson-anytime bound is tighter on extreme leaves
        # (h = 0 or h = n via the smoothing trick); the betting CS is
        # tighter in the interior of [0, 1] (where Bernoulli variance
        # is large and Wilson's union-bound-in-time penalty bites).
        # The min recovers the best of both.
        K = max(len(open_leaves), 1)
        delta_per_leaf = delta / K
        delta_inner = delta_per_leaf / 2.0
        eps_stat = 0.0
        for leaf in open_leaves:
            n_l, h_l = leaf.n_samples, leaf.n_hits
            h_w = wilson_halfwidth_anytime(n_l, h_l, delta_inner)
            h_b = betting_halfwidth_anytime(n_l, h_l, delta_inner)
            eps_stat += leaf.w_hat * min(h_w, h_b)
    else:
        raise ValueError(f"unknown method: {method!r}")

    # Cap eps_stat at 1 (the interval is clipped anyway).
    eps_stat = min(eps_stat, 1.0)
    eps_mass = sum(math.sqrt(leaf.w_var) for leaf in leaves)
    # Closure-uncertainty accumulator. Sample-based closures of leaves
    # contribute `closure_epsilon * w_leaf` each into this budget; SMT-
    # verified closures contribute 0. See :meth:`dise.regions.Frontier.try_close`.
    W_close = getattr(frontier, "W_close_accumulated", 0.0)
    # Mass-attribution uncertainty for open leaves. The deterministic
    # decomposition |mu - mu_hat| ≤ |w_pi - hat_w_pi| * mu_pi +
    # hat_w_pi * |mu_pi - hat_mu_pi| (Theorem 2 of docs/algorithm.md)
    # bounds the second term via Wilson per leaf (already captured in
    # eps_stat) and the first term via |w_pi - hat_w_pi|. For
    # axis-aligned regions hat_w_pi = w_pi (exact mass), so the mass
    # term is 0. For general regions we use a per-leaf 95% Wilson upper
    # bound on the proportion via sqrt(w_var); summed gives ``eps_mass``.
    # We use ``z = 2.0`` as a uniform multiplier (≈ 1-δ at δ ≈ 0.05).
    z_mass = 2.0
    eps_mass_certified = z_mass * eps_mass
    lo = max(0.0, mu_hat - eps_stat - eps_mass_certified - W_close)
    hi = min(1.0, mu_hat + eps_stat + eps_mass_certified + W_close)

    return EstimatorState(
        mu_hat=mu_hat,
        variance=variance,
        eps_stat=eps_stat,
        eps_mass=eps_mass,
        W_open=W_open,
        W_close=W_close,
        delta=delta,
        interval=(lo, hi),
        n_total_samples=n_total_samples,
        n_leaves=len(leaves) - len(empty_leaves),
        n_open_leaves=len(open_leaves),
        n_closed_leaves=len(closed_leaves),
    )


__all__ = [
    "EstimatorState",
    "bernstein_halfwidth",
    "betting_halfwidth_anytime",
    "compute_estimator_state",
    "empirical_bernstein_halfwidth_mp",
    "wilson_halfwidth_anytime",
    "wilson_halfwidth_for_leaf",
]
