"""ASIP (Adaptive Symbolic Importance Partitioning) scheduler.

The :class:`ASIPScheduler` ties together concolic execution, sampling,
refinement, and the certified estimator. At each iteration it picks one
action — allocate more samples to an open leaf, or refine an open leaf
on a divergent branch — based on expected variance reduction per unit
cost.

Termination is reached when:

* ``samples_used >= budget_samples``  → ``terminated_reason='budget_exhausted'``
* ``eps_stat + W_open <= epsilon``    → ``terminated_reason='epsilon_reached'``
* No action with positive gain exists → ``terminated_reason='no_actions_available'``
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..concolic import BranchRecord, ConcolicResult, run_concolic
from ..distributions import ProductDistribution
from ..estimator import EstimatorState, compute_estimator_state
from ..regions import Frontier, FrontierNode, Status
from ..sampler import RejectionSampler, Sampler
from ..smt import SMTBackend, SMTExpr

# -----------------------------------------------------------------------------
# Config / logging / results
# -----------------------------------------------------------------------------


@dataclass
class SchedulerConfig:
    """Configuration for :class:`ASIPScheduler`.

    Termination is governed by four orthogonal knobs. The algorithm
    halts at the **first** condition that fires:

    * ``epsilon`` (always active) — stop when
      :math:`\\varepsilon_{\\text{stat}} + W_{\\text{open}} \\le \\varepsilon`.
    * ``budget_samples`` — optional cap on concolic runs.
      ``None`` disables the cap (recommended for soundness-mode runs;
      see :class:`SchedulerResult.terminated_reason`).
    * ``budget_seconds`` — optional wall-clock cap. ``None`` disables.
    * ``min_gain_per_cost`` — diminishing-returns floor; the algorithm
      declares ``"no_actions_available"`` when the best candidate
      action's expected gain-per-cost falls below this threshold.

    Setting *all* of ``budget_samples``, ``budget_seconds``, and
    ``min_gain_per_cost`` to permissive values yields a "soundness-
    only" run: the algorithm runs until ``epsilon_reached``. For
    pathologically hard targets this may not terminate; configure at
    least one cap unless you trust the target is reachable.
    """

    epsilon: float = 0.05
    delta: float = 0.05
    # ``None`` disables the cap; the algorithm runs until the target
    # ``epsilon`` is reached or another cap fires.
    budget_samples: int | None = 10_000
    # ``None`` disables the wall-clock cap.
    budget_seconds: float | None = None
    # The algorithm stops when the best action's expected
    # gain-per-cost falls below this threshold. Default 0 keeps
    # backward-compatible behavior.
    min_gain_per_cost: float = 0.0
    # Certified half-width method (see ``compute_estimator_state``).
    # ``"wilson"`` is tightest at fixed n; ``"anytime"`` is sound under
    # the scheduler's adaptive stopping rule (recommended for ATVA-
    # style certificates — see docs/algorithm.md §13).
    method: Literal["wilson", "anytime", "bernstein", "empirical-bernstein"] = "wilson"
    bootstrap_samples: int = 200
    batch_size: int = 50
    refinement_cost_in_samples: float = 1.0
    max_refinement_depth: int = 50
    n_mass_samples: int = 1000
    smt_timeout_ms: int = 5000
    closure_min_samples: int = 5
    # Sound concentration-bounded closure parameters. The closure rule
    # (see :meth:`dise.regions.Frontier.try_close`) fires only when an
    # anytime-valid Wilson upper bound on the per-leaf disagreement rate
    # is at most ``closure_epsilon`` at confidence ``1 - delta_close``.
    # Each sample-based closure contributes ``closure_epsilon * w_leaf``
    # to the certified-interval half-width via the W_close accumulator.
    # SMT-verified closures contribute zero (they are exact).
    #
    # Defaults: closure_epsilon = 0.02 requires roughly 1000 samples per
    # leaf to fire; delta_close = 0.005 is the per-leaf failure
    # probability budget (the union bound over closed leaves is the
    # caller's responsibility — see :class:`Frontier`).
    delta_close: float = 0.005
    closure_epsilon: float = 0.02
    # Minimum fraction of variance contribution a refinement must
    # eliminate for the refine action to be proposed. Below this
    # threshold the refinement is "marginal" — the new children will
    # have similar variance to the parent, but each needs its own
    # ~``log(1/delta_close)/closure_epsilon^2`` samples to clear the
    # concentration check at closure time. Marginal refinements
    # over-fragment the frontier and starve every leaf of the closure
    # budget. Default 0.25 (require at least a 25 % variance reduction).
    min_refine_variance_reduction: float = 0.25
    max_concolic_branches: int = 10_000
    verbose: bool = False


@dataclass
class IterationLog:
    iter_idx: int
    action_kind: str
    leaf_depth: int
    samples_used_after: int
    mu_hat: float
    eps_stat: float
    W_open: float
    interval: tuple[float, float]
    n_leaves: int
    n_open_leaves: int


@dataclass
class SchedulerResult:
    final_estimator: EstimatorState
    iterations: list[IterationLog]
    samples_used: int
    refinements_done: int
    smt_calls: int
    terminated_reason: str
    frontier: Frontier

    def __repr__(self) -> str:
        lo, hi = self.final_estimator.interval
        return (
            f"SchedulerResult(mu_hat={self.final_estimator.mu_hat:.4f}, "
            f"interval=[{lo:.4f}, {hi:.4f}], "
            f"samples={self.samples_used}, refinements={self.refinements_done}, "
            f"terminated={self.terminated_reason!r})"
        )


# -----------------------------------------------------------------------------
# Internal action representation
# -----------------------------------------------------------------------------


@dataclass
class _Action:
    kind: Literal["allocate", "refine"]
    leaf: FrontierNode
    expected_gain: float
    cost: float
    k: int = 0
    clause: SMTExpr | None = None

    @property
    def gain_per_cost(self) -> float:
        if self.cost <= 0:
            return float("inf") if self.expected_gain > 0 else 0.0
        return self.expected_gain / self.cost


# -----------------------------------------------------------------------------
# Scheduler
# -----------------------------------------------------------------------------


class ASIPScheduler:
    """The main DiSE driver."""

    def __init__(
        self,
        program: Callable[..., Any],
        distribution: ProductDistribution,
        property_fn: Callable[[Any], bool],
        smt: SMTBackend,
        config: SchedulerConfig,
        rng: np.random.Generator,
        sampler: Sampler | None = None,
    ) -> None:
        self.program = program
        self.distribution = distribution
        self.property_fn = property_fn
        self.smt = smt
        self.config = config
        self.rng = rng
        self.sampler: Sampler = sampler if sampler is not None else RejectionSampler()
        self.frontier = Frontier(distribution, smt, n_mc_for_mass=config.n_mass_samples)
        self.samples_used = 0
        self.refinements_done = 0
        self.smt_calls = 0
        self.iterations: list[IterationLog] = []
        self.terminated_reason = "not_terminated"
        # Wall-clock timer, set at `run` time.
        self._start_time: float | None = None
        # Per-leaf raw path conditions and phi values (used to choose
        # refinement clauses). Scheduler-local; not part of Frontier state.
        self._leaf_paths: dict[int, list[list[BranchRecord]]] = {}
        self._leaf_path_phis: dict[int, list[int]] = {}

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _run_concolic(self, x: dict[str, int]) -> ConcolicResult:
        return run_concolic(
            self.program,
            x,
            self.property_fn,
            self.smt,
            max_branches=self.config.max_concolic_branches,
        )

    def _leaf_key(self, leaf: FrontierNode) -> int:
        return id(leaf)

    def _record_observation(
        self, leaf: FrontierNode, result: ConcolicResult
    ) -> None:
        if not result.terminated:
            return
        canonical = tuple(
            self.smt.repr_expr(br.clause_taken) for br in result.path_condition
        )
        path_clauses = tuple(br.clause_taken for br in result.path_condition)
        self.frontier.add_observation(
            leaf, canonical, result.phi_value, path_clauses=path_clauses
        )
        key = self._leaf_key(leaf)
        self._leaf_paths.setdefault(key, []).append(list(result.path_condition))
        self._leaf_path_phis.setdefault(key, []).append(int(result.phi_value))

    def _budget_remaining(self) -> int | None:
        """Concolic-run budget left before the sample cap fires, or
        ``None`` if no sample cap is configured."""
        if self.config.budget_samples is None:
            return None
        return self.config.budget_samples - self.samples_used

    def _time_exhausted(self) -> bool:
        if self.config.budget_seconds is None or self._start_time is None:
            return False
        return (time.perf_counter() - self._start_time) >= self.config.budget_seconds

    def _allocate_one_batch(self, leaf: FrontierNode, k: int) -> int:
        """Draw ``k`` samples from ``leaf.region``, run concolic on each,
        record observations. Returns the number of concolic runs actually
        performed (may be < k if the sample cap or time cap fires).
        """
        if leaf.status != Status.OPEN:
            return 0
        remaining = self._budget_remaining()
        if remaining is not None:
            if remaining <= 0:
                return 0
            k = min(k, remaining)
        batch = self.sampler.sample(
            leaf.region, self.distribution, self.smt, self.rng, k
        )
        n_run = 0
        for x in batch.iter_assignments():
            remaining = self._budget_remaining()
            if remaining is not None and remaining <= 0:
                break
            if self._time_exhausted():
                break
            result = self._run_concolic(x)
            self.samples_used += 1
            n_run += 1
            target = self.frontier.find_leaf_for(x)
            self._record_observation(target, result)
        return n_run

    # ------------------------------------------------------------------
    # Closure
    # ------------------------------------------------------------------

    def _try_close_all(self) -> int:
        n_closed = 0
        for leaf in self.frontier.open_leaves():
            if self.frontier.try_close(
                leaf,
                self.config.closure_min_samples,
                delta_close=self.config.delta_close,
                closure_epsilon=self.config.closure_epsilon,
            ):
                n_closed += 1
                # Drop the leaf's local paths once closed.
                key = self._leaf_key(leaf)
                self._leaf_paths.pop(key, None)
                self._leaf_path_phis.pop(key, None)
        return n_closed

    # ------------------------------------------------------------------
    # Refinement
    # ------------------------------------------------------------------

    def _find_divergent_clause(
        self, paths: list[list[BranchRecord]]
    ) -> SMTExpr | None:
        """Fallback: return the clause at the first position where the
        observed paths first disagree."""
        if len(paths) < 2:
            return None
        canonical = [
            [self.smt.repr_expr(br.clause_taken) for br in path] for path in paths
        ]
        max_len = max((len(c) for c in canonical), default=0)
        for k in range(max_len):
            seen: set[str | None] = set()
            for c in canonical:
                seen.add(c[k] if k < len(c) else None)
            if len(seen) > 1:
                for path in paths:
                    if k < len(path):
                        return path[k].clause_taken
                return None
        return None

    def _predicted_no_refine_halfwidth(
        self, leaf: FrontierNode, n_observed: int, h_observed: int, K_current: int
    ) -> float:
        """Predicted contribution of this leaf to eps_stat if we *don't*
        refine: scale current empirical-mean estimate forward to the
        full remaining sample budget.

        Uses min(Wilson-anytime, betting CS) at the same delta_inner the
        estimator will use.
        """
        from ..estimator import betting_halfwidth_anytime, wilson_halfwidth_anytime

        remaining = self._budget_remaining()
        if remaining is None or remaining <= 0:
            n_future = n_observed
        else:
            # Optimistic: assume all remaining budget goes to this leaf.
            n_future = n_observed + remaining
        # Maintain h/n ratio under the forecast.
        if n_observed > 0:
            h_future = int(h_observed * n_future / n_observed)
        else:
            h_future = 0
        delta_per_leaf = self.config.delta / max(K_current, 1)
        delta_inner = delta_per_leaf / 2.0
        h_w = wilson_halfwidth_anytime(n_future, h_future, delta_inner)
        h_b = betting_halfwidth_anytime(n_future, h_future, delta_inner)
        return leaf.w_hat * min(h_w, h_b)

    def _predicted_refine_halfwidth(
        self,
        leaf: FrontierNode,
        n_true: int, h_true: int,
        n_false: int, h_false: int,
        K_current: int,
    ) -> float:
        """Predicted contribution to eps_stat if we DO refine on this clause:
        budget is split proportional to the observed bootstrap fraction, each
        child uses the (K+1)-leaf delta budget.

        Uses naive maintenance of the per-side h/n ratio to forecast
        post-allocation child statistics. This is *optimistic* on the
        empirically-uniform side (the selection effect biases ``h = 0``
        observations low) — the predictor consequently over-recommends
        refinement on benchmarks where the chosen clause's uniform side
        turns out to be only ~10% phi-1 after more samples. Accepted as
        a known limitation; correcting requires either a Bayesian
        prior on per-side ``mu`` or a multiple-testing-corrected upper
        bound, both of which over-suppress beneficial refinements
        elsewhere.
        """
        from ..estimator import betting_halfwidth_anytime, wilson_halfwidth_anytime

        remaining = self._budget_remaining()
        if remaining is None or remaining <= 0:
            remaining = 0
        n_total_obs = max(n_true + n_false, 1)
        f_t = n_true / n_total_obs
        f_f = n_false / n_total_obs
        n_t_future = n_true + int(remaining * f_t)
        n_f_future = n_false + int(remaining * f_f)
        h_t_future = int(h_true * n_t_future / n_true) if n_true else 0
        h_f_future = int(h_false * n_f_future / n_false) if n_false else 0
        delta_per_leaf = self.config.delta / (K_current + 1)
        delta_inner = delta_per_leaf / 2.0
        w_t = leaf.w_hat * f_t
        w_f = leaf.w_hat * f_f
        half_t = min(
            wilson_halfwidth_anytime(n_t_future, h_t_future, delta_inner),
            betting_halfwidth_anytime(n_t_future, h_t_future, delta_inner),
        )
        half_f = min(
            wilson_halfwidth_anytime(n_f_future, h_f_future, delta_inner),
            betting_halfwidth_anytime(n_f_future, h_f_future, delta_inner),
        )
        return w_t * half_t + w_f * half_f

    def _best_refinement_clause(
        self,
        leaf: FrontierNode,
        paths: list[list[BranchRecord]],
        path_phis: list[int],
        require_uniform_child: bool = True,
    ) -> tuple[SMTExpr | None, float]:
        """Pick the best refinement clause for ``leaf``.

        For each candidate clause ``b`` (drawn from the observed paths
        beyond ``leaf.depth``), partition the observed samples by whether
        their path included ``b`` taken. Compute the *informational
        gain*

            G(b) = V_pi - V_{pi ^ b} - V_{pi ^ ~b}

        where

            V_pi   = w_pi^2 * mu_var_pi / n_pi
            V_b    = (w_pi * f_b)^2  * mu_var_b   / max(n_b, 1)
            V_~b   = (w_pi * (1 - f_b))^2 * mu_var_~b / max(n_~b, 1)

        with ``f_b`` the empirical fraction of samples taking ``b`` (and
        Wilson smoothing on the per-side mu_var). Returns the
        ``(clause, gain)`` pair maximizing ``G``; if no candidate beats
        the current variance, falls back to ``_find_divergent_clause``
        (gain = ``leaf.variance_contribution``).
        """
        if len(paths) < 2:
            return None, 0.0
        # Collect candidate clauses (those appearing strictly past leaf.depth
        # in at least one path — earlier clauses are already in F_pi).
        depth = leaf.depth
        candidates: dict[str, SMTExpr] = {}
        for path in paths:
            for k in range(depth, len(path)):
                key = self.smt.repr_expr(path[k].clause_taken)
                candidates.setdefault(key, path[k].clause_taken)
        if not candidates:
            return None, 0.0

        canonical_paths = [
            [self.smt.repr_expr(br.clause_taken) for br in path] for path in paths
        ]
        w_pi = leaf.w_hat
        v_pi = leaf.variance_contribution

        best_clause: SMTExpr | None = None
        best_gain = 0.0

        for key, clause_expr in candidates.items():
            n_true = 0
            h_true = 0
            n_false = 0
            h_false = 0
            for _path, phi, can in zip(paths, path_phis, canonical_paths, strict=True):
                # A sample's path "takes" the clause iff `key` appears in its
                # canonical sequence.
                if key in can:
                    n_true += 1
                    if phi:
                        h_true += 1
                else:
                    n_false += 1
                    if phi:
                        h_false += 1
            n_total = n_true + n_false
            if n_total == 0 or n_true == 0 or n_false == 0:
                # Doesn't separate the samples; not informative.
                continue
            # Under sound concentration-bounded closure, refinement into
            # two phi-disagreeing children is a *net loss*: the per-leaf
            # samples drop by half but the Bonferroni-corrected Wilson /
            # betting bound only shrinks by ~sqrt(2 log(2/delta) / log(1/delta))
            # ≈ 0.7 — so the weighted sum w_child * half_child summed over
            # the children is roughly 2*0.5*0.7 = 0.7 *larger* than the
            # parent's per-leaf half-width. Refinement only pays off when
            # at least one child becomes phi-uniform (and can close).
            # We skip clauses whose children are both empirically
            # phi-mixed unless the caller opts out (e.g. when the gain
            # is so large the structural reduction wins anyway).
            if require_uniform_child:
                true_uniform = h_true == 0 or h_true == n_true
                false_uniform = h_false == 0 or h_false == n_false
                if not (true_uniform or false_uniform):
                    continue
            f_b = n_true / n_total
            # Wilson-smoothed per-side mu_var
            p_t = (h_true + 1) / (n_true + 2)
            p_f = (h_false + 1) / (n_false + 2)
            mu_var_t = p_t * (1.0 - p_t)
            mu_var_f = p_f * (1.0 - p_f)
            w_t = w_pi * f_b
            w_f = w_pi * (1.0 - f_b)
            v_b = w_t * w_t * mu_var_t / max(n_true, 1)
            v_nb = w_f * w_f * mu_var_f / max(n_false, 1)
            gain = v_pi - (v_b + v_nb)
            if gain > best_gain:
                best_gain = gain
                best_clause = clause_expr

        if best_clause is None:
            # No clause showed positive informational gain. Fall back to first
            # divergent clause but with a smaller (heuristic) gain estimate.
            clause = self._find_divergent_clause(paths)
            if clause is None:
                return None, 0.0
            return clause, max(v_pi, 1e-12)
        return best_clause, best_gain

    def _execute_refine(self, leaf: FrontierNode, clause: SMTExpr) -> None:
        if leaf.depth >= self.config.max_refinement_depth:
            return
        try:
            children = self.frontier.refine(leaf, clause, self.rng)
        except ValueError:
            return
        self.smt_calls += 2  # two satisfiability checks per refinement
        self.refinements_done += 1
        # Drop scheduler-side paths for the parent; children start fresh.
        parent_key = self._leaf_key(leaf)
        self._leaf_paths.pop(parent_key, None)
        self._leaf_path_phis.pop(parent_key, None)
        # Seed each non-empty child with at least a few samples so closure
        # can be evaluated.
        seed_n = min(self.config.batch_size, self.config.closure_min_samples * 2)
        for c in children:
            if c.status == Status.OPEN:
                self._allocate_one_batch(c, seed_n)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def _candidate_actions(self) -> list[_Action]:
        actions: list[_Action] = []
        for leaf in self.frontier.open_leaves():
            n = leaf.n_samples
            w = leaf.w_hat
            mu_var = leaf.mu_var
            k = self.config.batch_size

            # Allocate action
            if n == 0:
                alloc_gain = w * w * 0.25
            else:
                alloc_gain = w * w * mu_var * k / (n * (n + k))
            actions.append(
                _Action(
                    kind="allocate",
                    leaf=leaf,
                    expected_gain=alloc_gain,
                    cost=float(k),
                    k=k,
                )
            )

            # Refine action — only if we have observed paths AND the
            # observations show non-trivial phi-disagreement.
            #
            # A leaf with all observations agreeing on phi-value cannot
            # benefit from refinement, *regardless* of whether path
            # sequences agree: every child it could produce inherits
            # the same all-agree phi statistics, the per-child
            # Wilson-smoothed mu_var stays positive at the same rate
            # (=> the fallback "any divergent clause" gain
            # ``max(v_pi, 1e-12)`` keeps re-proposing refinement
            # indefinitely), and the leaf fragments into deeper
            # phi-uniform children that all need their own
            # closure-bound certification. This is the dominant cause
            # of over-fragmentation under sound closure: a leaf with
            # 200 samples all giving ``phi=1`` (e.g. ``popcount(x)≥0``)
            # would refine 50× over the budget, never accumulating
            # enough samples per child to clear the concentration
            # check.
            #
            # The right action for a phi-uniform leaf is to *allocate*
            # more samples, which lets the sound concentration-bounded
            # closure rule eventually fire on the leaf as a whole.
            paths = self._leaf_paths.get(self._leaf_key(leaf), [])
            phis = self._leaf_path_phis.get(self._leaf_key(leaf), [])
            unique_phis = set(phis)
            phi_uniform = len(unique_phis) <= 1 and len(phis) >= 1

            # Budget-aware leaf cap.
            n_close_per_leaf = max(
                50,
                int(
                    1.0 / max(self.config.closure_epsilon, 1e-6) ** 2
                    * (1.0 + 0.5)
                ),
            )
            budget_for_leaves = self.config.budget_samples or 10_000
            max_leaves = max(2, budget_for_leaves // n_close_per_leaf)
            n_current_leaves = sum(
                1 for _ in self.frontier.open_leaves()
            ) + sum(1 for _ in self.frontier.closed_leaves())
            over_budget = n_current_leaves >= max_leaves

            if (
                len(paths) >= 2
                and leaf.depth < self.config.max_refinement_depth
                and not phi_uniform
                and not over_budget
            ):
                clause, refine_gain = self._best_refinement_clause(leaf, paths, phis)
                # Refinement-or-allocate decision: predict the *bound
                # contribution* under each choice and refine only if
                # refinement produces a strictly tighter predicted
                # eps_stat contribution. The simple variance-reduction
                # heuristic ignores the Bonferroni-vs-sample-size
                # tradeoff that hurts on benchmarks like miller_rabin
                # / sieve_primality where one informative refinement
                # still *increases* the certified half-width because
                # each child's per-leaf sample count halves while the
                # per-leaf Wilson / betting bound only shrinks by
                # ~sqrt(log(K+1) / log(K)).
                if clause is not None and refine_gain > 0:
                    # Recompute split stats for the chosen clause.
                    canon = [
                        [self.smt.repr_expr(br.clause_taken) for br in p]
                        for p in paths
                    ]
                    key = self.smt.repr_expr(clause)
                    n_t = h_t = n_f = h_f = 0
                    for can, phi in zip(canon, phis, strict=True):
                        if key in can:
                            n_t += 1
                            if phi:
                                h_t += 1
                        else:
                            n_f += 1
                            if phi:
                                h_f += 1
                    K_now = len(self.frontier.open_leaves())
                    pred_no_refine = self._predicted_no_refine_halfwidth(
                        leaf, leaf.n_samples, leaf.n_hits, K_now
                    )
                    pred_refine = self._predicted_refine_halfwidth(
                        leaf, n_t, h_t, n_f, h_f, K_now
                    )
                    if pred_refine >= pred_no_refine:
                        clause = None  # Don't refine.
                # Under sound concentration-bounded closure, a "marginal"
                # refinement (one that splits the leaf without making
                # either child phi-uniform) hurts: every new child needs
                # its own ~``log(1/delta_close)/closure_epsilon^2``
                # samples to clear the concentration check, and the
                # leaf's contribution to the total ``W_close`` stays the
                # same (mass conservation). Only propose refine when
                # the candidate clause is *highly informative* — i.e.
                # the children are substantially closer to phi-uniform
                # than the parent. We measure this by requiring the
                # variance to drop by at least ``min_variance_reduction``
                # times the parent's variance contribution.
                v_pi = leaf.variance_contribution
                min_variance_reduction = self.config.min_refine_variance_reduction
                worthwhile = (
                    clause is not None
                    and refine_gain > 0
                    and (v_pi <= 0 or refine_gain >= min_variance_reduction * v_pi)
                )
                if worthwhile:
                    actions.append(
                        _Action(
                            kind="refine",
                            leaf=leaf,
                            expected_gain=refine_gain,
                            cost=self.config.refinement_cost_in_samples,
                            clause=clause,
                        )
                    )
        return actions

    def _pick_best_action(self, actions: list[_Action]) -> _Action | None:
        if not actions:
            return None
        # Sort by gain_per_cost desc; tie-break preferring "refine" (structural).
        def key(a: _Action) -> tuple[float, int]:
            return (a.gain_per_cost, 1 if a.kind == "refine" else 0)
        actions.sort(key=key, reverse=True)
        best = actions[0]
        if best.gain_per_cost <= self.config.min_gain_per_cost:
            return None
        return best

    # ------------------------------------------------------------------
    # Bootstrap + main loop
    # ------------------------------------------------------------------

    def bootstrap(self) -> None:
        """Initial sampling pass: draw ``bootstrap_samples`` from ``D``
        and attribute each to the root."""
        n = self.config.bootstrap_samples
        if self.config.budget_samples is not None:
            n = min(n, self.config.budget_samples)
        self._allocate_one_batch(self.frontier.root, n)

    def _should_terminate(self) -> tuple[bool, EstimatorState]:
        state = compute_estimator_state(self.frontier, self.config.delta, method=self.config.method)
        if (
            self.config.budget_samples is not None
            and self.samples_used >= self.config.budget_samples
        ):
            self.terminated_reason = "budget_exhausted"
            return True, state
        if self._time_exhausted():
            self.terminated_reason = "time_exhausted"
            return True, state
        # The certified half-width is `eps_stat + eps_mass_certified +
        # W_close` (see ``compute_estimator_state`` and docs/algorithm.md
        # §1). Use the actual interval rather than approximating from
        # individual fields so the termination check matches whatever
        # construction the estimator implements.
        lo, hi = state.interval
        cert_half = max((hi - lo) / 2.0, 0.0)
        if cert_half <= self.config.epsilon:
            self.terminated_reason = "epsilon_reached"
            return True, state
        return False, state

    def _log_iteration(
        self, iter_idx: int, action: _Action, state: EstimatorState
    ) -> None:
        self.iterations.append(
            IterationLog(
                iter_idx=iter_idx,
                action_kind=action.kind,
                leaf_depth=action.leaf.depth,
                samples_used_after=self.samples_used,
                mu_hat=state.mu_hat,
                eps_stat=state.eps_stat,
                W_open=state.W_open,
                interval=state.interval,
                n_leaves=state.n_leaves,
                n_open_leaves=state.n_open_leaves,
            )
        )

    def run(self) -> SchedulerResult:
        self._start_time = time.perf_counter()
        self.bootstrap()
        self._try_close_all()

        # Safety upper bound on iterations. We use:
        #   - the sample budget (each iteration spends at least 1 sample
        #     in the allocate branch), plus
        #   - a generous slack for refinement-only iterations, plus
        #   - a fallback of 10^7 when there is no sample budget.
        if self.config.budget_samples is not None:
            max_iters = self.config.budget_samples + 1000
        else:
            max_iters = 10_000_000

        iter_idx = 0
        for iter_idx in range(max_iters):
            done, state = self._should_terminate()
            if done:
                break
            actions = self._candidate_actions()
            best = self._pick_best_action(actions)
            if best is None:
                self.terminated_reason = "no_actions_available"
                break
            if best.kind == "allocate":
                self._allocate_one_batch(best.leaf, best.k)
            else:
                assert best.clause is not None
                self._execute_refine(best.leaf, best.clause)
            self._try_close_all()
            self._log_iteration(iter_idx, best, state)

        final_state = compute_estimator_state(self.frontier, self.config.delta, method=self.config.method)
        return SchedulerResult(
            final_estimator=final_state,
            iterations=self.iterations,
            samples_used=self.samples_used,
            refinements_done=self.refinements_done,
            smt_calls=self.smt_calls,
            terminated_reason=self.terminated_reason,
            frontier=self.frontier,
        )


__all__ = [
    "ASIPScheduler",
    "IterationLog",
    "SchedulerConfig",
    "SchedulerResult",
]
