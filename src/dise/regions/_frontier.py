"""Frontier: tree of :class:`FrontierNode`s tracking the current ASIP state."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..distributions import ProductDistribution
from ..smt import SMTBackend, SMTExpr
from ._base import Region, Status
from ._concrete import EmptyRegion, UnconstrainedRegion, build_region


@dataclass
class FrontierNode:
    """One leaf or internal node in the frontier tree.

    A node owns a :class:`Region`, a :class:`Status`, statistics about the
    samples drawn from it (in the OPEN case), and a list of child nodes if
    refinement has occurred. Variance / mu / mu_var are derived properties
    that always reflect the current sample counts.
    """

    region: Region
    status: Status = Status.OPEN
    parent: FrontierNode | None = None
    depth: int = 0
    refining_clause: SMTExpr | None = None  # clause that produced this from parent

    # Cached mass (computed at construction time via Frontier.ensure_mass)
    w_hat: float = 0.0
    w_var: float = 0.0
    mass_computed: bool = False

    # Sample statistics for OPEN nodes
    n_samples: int = 0
    n_hits: int = 0
    observed_sequences: list[tuple] = field(default_factory=list)
    observed_phis: list[int] = field(default_factory=list)
    # Raw clause-takens from each observation, parallel to observed_sequences.
    # Used by Frontier.try_close to validate path determinism symbolically.
    observed_paths: list[tuple] = field(default_factory=list)

    children: list[FrontierNode] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def formula(self) -> SMTExpr:
        return self.region.formula

    @property
    def mu_hat(self) -> float:
        """Best current estimate of :math:`\\mu_\\pi` for this leaf."""
        if self.status == Status.CLOSED_TRUE:
            return 1.0
        if self.status in (Status.CLOSED_FALSE, Status.EMPTY):
            return 0.0
        if self.n_samples == 0:
            return 0.5
        return self.n_hits / self.n_samples

    @property
    def mu_var(self) -> float:
        """Wilson-smoothed per-sample Bernoulli variance plug-in:
        ``tilde_p * (1 - tilde_p)`` with ``tilde_p = (h + 1) / (n + 2)``.

        Never collapses to 0 from finite samples (closure of an open leaf
        requires symbolic evidence, not just agreement).
        """
        if self.status in (Status.CLOSED_TRUE, Status.CLOSED_FALSE, Status.EMPTY):
            return 0.0
        n, h = self.n_samples, self.n_hits
        if n == 0:
            return 0.25
        p_tilde = (h + 1) / (n + 2)
        return p_tilde * (1.0 - p_tilde)

    @property
    def mu_mean_var(self) -> float:
        """Variance of the sample-mean estimator of mu_pi: ``mu_var / n``."""
        if self.status in (Status.CLOSED_TRUE, Status.CLOSED_FALSE, Status.EMPTY):
            return 0.0
        if self.n_samples == 0:
            return 0.25
        return self.mu_var / self.n_samples

    @property
    def variance_contribution(self) -> float:
        """Per-Theorem-1 contribution to total estimator variance::

            Var(w_hat * mu_hat) = w_hat^2 * Var(mu_hat)
                                + mu_hat^2 * Var(w_hat)
                                + Var(w_hat) * Var(mu_hat)
        """
        w, w_v = self.w_hat, self.w_var
        m = self.mu_hat
        mmv = self.mu_mean_var
        return w * w * mmv + m * m * w_v + w_v * mmv

    def reset_observations(self) -> None:
        """Drop sample-based statistics. Used after refinement."""
        self.n_samples = 0
        self.n_hits = 0
        self.observed_sequences.clear()
        self.observed_phis.clear()
        self.observed_paths.clear()


class Frontier:
    """The full tree of :class:`FrontierNode`s.

    Invariants:

    * The root is an :class:`UnconstrainedRegion` over the full distribution.
    * Internal nodes' children partition the parent's region (modulo
      backend-``unknown`` cases where refinement may produce overlapping
      :class:`GeneralRegion` envelopes — sampling still correctly attributes
      each input via :meth:`find_leaf_for`).
    """

    def __init__(
        self,
        distribution: ProductDistribution,
        smt: SMTBackend,
        n_mc_for_mass: int = 1000,
    ) -> None:
        self.distribution = distribution
        self.smt = smt
        self.n_mc_for_mass = n_mc_for_mass
        root_region = UnconstrainedRegion(smt.true())
        self.root = FrontierNode(
            region=root_region,
            status=Status.OPEN,
            depth=0,
        )
        self.root.w_hat = 1.0
        self.root.w_var = 0.0
        self.root.mass_computed = True

    # ---- tree walks ----

    def leaves(self) -> list[FrontierNode]:
        out: list[FrontierNode] = []

        def walk(node: FrontierNode) -> None:
            if node.is_leaf:
                out.append(node)
                return
            for c in node.children:
                walk(c)

        walk(self.root)
        return out

    def open_leaves(self) -> list[FrontierNode]:
        return [n for n in self.leaves() if n.status == Status.OPEN]

    def closed_leaves(self) -> list[FrontierNode]:
        return [
            n
            for n in self.leaves()
            if n.status in (Status.CLOSED_TRUE, Status.CLOSED_FALSE)
        ]

    def all_nodes(self) -> list[FrontierNode]:
        out: list[FrontierNode] = []

        def walk(node: FrontierNode) -> None:
            out.append(node)
            for c in node.children:
                walk(c)

        walk(self.root)
        return out

    def n_leaves(self) -> int:
        return len(self.leaves())

    # ---- aggregate quantities ----

    def open_mass(self) -> float:
        """W_open = sum of w_hat over OPEN leaves."""
        return sum(n.w_hat for n in self.open_leaves())

    def total_leaf_mass(self) -> float:
        """Sum of w_hat over all leaves (should equal ~1.0 if frontier
        partitions the input space)."""
        return sum(n.w_hat for n in self.leaves() if n.status != Status.EMPTY)

    def compute_mu_hat(self) -> tuple[float, float]:
        """Return ``(mu_hat, total_estimator_variance)``.

        Sum over all leaves of ``w_hat_pi * mu_hat_pi`` (point estimate) and
        ``variance_contribution`` (Theorem 1).
        """
        mu = 0.0
        var = 0.0
        for leaf in self.leaves():
            if leaf.status == Status.EMPTY:
                continue
            mu += leaf.w_hat * leaf.mu_hat
            var += leaf.variance_contribution
        return mu, var

    # ---- attribution ----

    def find_leaf_for(self, x: dict[str, int]) -> FrontierNode:
        """Walk the tree to find the leaf whose region contains ``x``.

        If multiple children claim ``x`` (overlapping :class:`GeneralRegion`
        envelopes are possible under ``unknown`` SMT), the first matching
        child is chosen. If no child matches, returns the deepest ancestor.
        """
        node = self.root
        while not node.is_leaf:
            matched: FrontierNode | None = None
            for c in node.children:
                if c.status == Status.EMPTY:
                    continue
                if c.region.contains(x):
                    matched = c
                    break
            if matched is None:
                return node
            node = matched
        return node

    # ---- mutations ----

    def ensure_mass(self, node: FrontierNode, rng: np.random.Generator) -> None:
        """Compute and cache ``(w_hat, w_var)`` for ``node`` if not already."""
        if node.mass_computed:
            return
        w_hat, w_var = node.region.mass(
            self.distribution, self.smt, rng, self.n_mc_for_mass
        )
        node.w_hat = w_hat
        node.w_var = w_var
        node.mass_computed = True

    def refine(
        self,
        node: FrontierNode,
        clause: SMTExpr,
        rng: np.random.Generator,
    ) -> list[FrontierNode]:
        """Split ``node`` into two children by ``clause`` (and its negation).

        Mass-conservation policy:

        * If both children are axis-aligned (closed-form masses), use the
          closed-form values directly — these sum to the parent's mass
          *exactly*.
        * If at least one child is a :class:`GeneralRegion`, derive the
          children's masses by drawing one batch of IS samples from the
          *parent* region and counting how many satisfy ``clause`` vs
          its negation. The split proportion ``p_true`` is Wilson-smoothed.
          The children's masses are
          ``w_true = w_parent * p_true``, ``w_false = w_parent * (1 - p_true)``,
          which sum to ``w_parent`` exactly.

        This restores the partition invariant
        ``sum_{pi in leaves} w_hat_pi = 1`` modulo IS noise concentrated in
        the parent's mass (root mass is exactly 1).

        Empty children (SMT-proved unsat) get status ``EMPTY`` immediately
        and contribute zero mass; the other child receives the parent's
        full mass.

        The parent's sample-based observations are dropped — the brief
        calls for not re-attributing branch records across refinement.
        """
        if not node.is_leaf:
            raise ValueError("can only refine a leaf")
        if node.status != Status.OPEN:
            raise ValueError(f"can only refine an OPEN leaf (got {node.status})")
        smt = self.smt
        neg_clause = smt.negation(clause)

        # Build the two child regions
        new_formula_true = smt.conjunction(node.region.formula, clause)
        new_formula_false = smt.conjunction(node.region.formula, neg_clause)
        region_true = build_region(new_formula_true, self.distribution, smt)
        region_false = build_region(new_formula_false, self.distribution, smt)

        # Always build a child node (even for empty regions, marked EMPTY).
        child_true = FrontierNode(
            region=region_true,
            status=Status.EMPTY if isinstance(region_true, EmptyRegion) else Status.OPEN,
            parent=node,
            depth=node.depth + 1,
            refining_clause=clause,
        )
        child_false = FrontierNode(
            region=region_false,
            status=Status.EMPTY if isinstance(region_false, EmptyRegion) else Status.OPEN,
            parent=node,
            depth=node.depth + 1,
            refining_clause=neg_clause,
        )
        children = [child_true, child_false]

        # Mass assignment policy
        true_empty = child_true.status == Status.EMPTY
        false_empty = child_false.status == Status.EMPTY
        if true_empty and false_empty:
            child_true.w_hat = child_true.w_var = 0.0
            child_false.w_hat = child_false.w_var = 0.0
            child_true.mass_computed = child_false.mass_computed = True
        elif true_empty:
            child_true.w_hat = child_true.w_var = 0.0
            child_true.mass_computed = True
            child_false.w_hat = node.w_hat
            child_false.w_var = node.w_var
            child_false.mass_computed = True
        elif false_empty:
            child_false.w_hat = child_false.w_var = 0.0
            child_false.mass_computed = True
            child_true.w_hat = node.w_hat
            child_true.w_var = node.w_var
            child_true.mass_computed = True
        else:
            # Both children non-empty. Choose between closed-form mass and
            # proportional split based on what kinds of region we got.
            children_axis_aligned = (
                region_true.is_axis_aligned and region_false.is_axis_aligned
            )
            if children_axis_aligned:
                # Closed-form is exact and partitions cleanly.
                self.ensure_mass(child_true, rng)
                self.ensure_mass(child_false, rng)
            else:
                # Mass-conservative proportional split from parent samples.
                self._proportional_split_mass(node, child_true, child_false, clause, rng)

        node.children = children
        node.reset_observations()
        return children

    def _proportional_split_mass(
        self,
        parent: FrontierNode,
        child_true: FrontierNode,
        child_false: FrontierNode,
        clause: SMTExpr,
        rng: np.random.Generator,
    ) -> None:
        """Draw one IS batch from ``parent.region``, count the fraction
        satisfying ``clause`` (Wilson-smoothed), and split ``parent.w_hat``
        between the two children proportionally. Variance of the split
        proportion is also accounted for in each child's ``w_var``.
        """
        n_mc = self.n_mc_for_mass
        try:
            sample_batch = parent.region.sample(
                self.distribution, self.smt, rng, n_mc
            )
        except Exception:
            # Fall back to independent estimation
            self.ensure_mass(child_true, rng)
            self.ensure_mass(child_false, rng)
            return
        n_total = sample_batch.n
        if n_total == 0:
            self.ensure_mass(child_true, rng)
            self.ensure_mass(child_false, rng)
            return
        n_true = 0
        for x in sample_batch.iter_assignments():
            try:
                if self.smt.evaluate(clause, x):
                    n_true += 1
            except (ValueError, KeyError, ZeroDivisionError, ArithmeticError):
                # Conservative: treat as "not in either side", but since
                # clause and ¬clause partition the world, this is unusual.
                # Default to NOT in the true side.
                continue
        # Wilson-smoothed split proportion
        p_tilde = (n_true + 1) / (n_total + 2)
        bern_var = p_tilde * (1.0 - p_tilde)
        split_mean_var = bern_var / n_total  # variance of the proportion estimator
        w_parent = parent.w_hat
        # Use the un-smoothed empirical proportion for the point estimate
        # (the proportional masses must sum exactly to w_parent).
        p_emp = n_true / n_total
        child_true.w_hat = w_parent * p_emp
        child_false.w_hat = w_parent * (1.0 - p_emp)
        # Variance: parent's mass variance plus the split-proportion variance,
        # scaled by w_parent^2. Children inherit parent's w_var component.
        child_true.w_var = parent.w_var + w_parent * w_parent * split_mean_var
        child_false.w_var = parent.w_var + w_parent * w_parent * split_mean_var
        child_true.mass_computed = True
        child_false.mass_computed = True

    def try_close(self, node: FrontierNode, min_samples: int) -> bool:
        """Apply the closure rule.

        A leaf is closed iff *all four* conditions hold:

        1. ``n_samples >= min_samples`` (filter: don't call SMT on a
           single observation).
        2. All observed branch sequences at the leaf are identical
           (so there is one well-defined "observed path" to certify).
        3. All observed :math:`\\varphi`-values agree (so closure is
           CLOSED_TRUE vs CLOSED_FALSE is unambiguous).
        4. **The SMT backend proves the region implies the observed
           path** — concretely, ``is_satisfiable(F_pi ∧ ¬path)`` returns
           ``"unsat"``. Returns ``"sat"`` or ``"unknown"`` → keep open.

        Condition (4) is the only soundness-relevant check; (1)-(3) are
        cheap pre-filters that avoid wasted SMT calls. There is **no
        sample-based fallback**: a leaf the SMT backend cannot certify
        stays open and contributes to ``W_open`` (the certified-interval
        cushion). This is the trade-off — wider intervals on programs
        whose closure is symbolically intractable, in exchange for a
        clean one-line soundness story.

        Returns ``True`` if the node was closed.
        """
        if node.status != Status.OPEN:
            return False
        if node.n_samples < min_samples:
            return False
        if not node.observed_sequences or not node.observed_phis:
            return False
        first_seq = node.observed_sequences[0]
        if any(seq != first_seq for seq in node.observed_sequences):
            return False
        first_phi = node.observed_phis[0]
        if any(p != first_phi for p in node.observed_phis):
            return False
        # SMT closure proof — *required*. Empty path clauses are
        # admissible: an observed run that recorded no branches has
        # path formula ``True``, so the closure check
        # ``F_pi ∧ ¬True ≡ F_pi ∧ False`` is trivially unsat. Any
        # well-behaved backend reports ``unsat`` in that case.
        path_clauses = node.observed_paths[0] if node.observed_paths else ()
        path_formula = self.smt.conjunction(*path_clauses)
        check = self.smt.conjunction(
            node.region.formula, self.smt.negation(path_formula)
        )
        result = self.smt.is_satisfiable(check)
        if result != "unsat":
            # ``"sat"`` (counterexample) or ``"unknown"`` (backend can't
            # decide) — neither certifies path-determinism. Keep open.
            return False
        node.status = Status.CLOSED_TRUE if first_phi else Status.CLOSED_FALSE
        return True

    def add_observation(
        self,
        node: FrontierNode,
        branch_sequence: tuple,
        phi_value: int,
        path_clauses: tuple = (),
    ) -> None:
        """Record one concolic-run observation against ``node``.

        ``branch_sequence`` is the canonical (string) sequence used for
        equality checks. ``path_clauses`` (optional) is the raw tuple of
        SMT clauses, used by :meth:`try_close` for symbolic verification.
        """
        node.n_samples += 1
        if phi_value:
            node.n_hits += 1
        node.observed_sequences.append(tuple(branch_sequence))
        node.observed_phis.append(int(phi_value))
        node.observed_paths.append(tuple(path_clauses))


__all__ = ["Frontier", "FrontierNode"]
