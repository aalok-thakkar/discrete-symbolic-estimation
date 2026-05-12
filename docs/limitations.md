# Limitations and open extensions

DiSE is a research prototype. The implementation deliberately covers
class (D1) — product-form discrete distributions over deterministic
integer / bitvector programs — and leaves several axes for future
work. This document is exhaustive: every regret is listed here.

## 1. Data types and programs

* **Strings, floats, byte buffers.** Integer / bitvector inputs only.
  Floats would require an SMT theory for floating-point (Z3 has
  `FloatingPoint`, but the concolic wrapper does not).
* **Symbolic memory.** Heap or array accesses indexed by symbolic
  values are not modeled. Programs that use lists / dicts in a way
  the concolic wrapper doesn't see (no comparisons) still run, but
  the path condition won't capture them.
* **Non-deterministic programs.** DiSE assumes $P$ is a deterministic
  function; if `program(...)` uses external randomness, the closure
  rule's path determinism is violated and certified-interval
  guarantees break.

## 2. Distribution classes

* **Class (D2) Bayes-net-structured discrete distributions** — not
  supported. Would require ancestor-conditioned constrained samplers
  and mass estimators that handle dependence.
* **Class (D3) general discrete** — out of scope.
* **Continuous distributions** of any kind — out of scope.

## 3. Algorithmic gaps

* **Rare-event regions.** When acceptance under the envelope drops
  below ~1 %, `RejectionSampler` returns partial batches. The
  `IntegerLatticeMHSampler` (Metropolis-Hastings on integers) is
  shipped as a working alternative, but its mixing is heuristic and
  not theoretically tied to a $\delta$-certified bound on a
  burn-in/thin schedule. Future work: replace with a perfect-sampling
  scheme or a verified-mixing one.
* **`MockBackend` path-determinism shortcut.** The closure rule's
  symbolic shortcut returns `"unknown"` for non-axis-aligned
  arithmetic under `MockBackend`. The scheduler falls back to a
  sample-based heuristic, which can close path-non-deterministic
  leaves prematurely (≈ 1 % bias on hard benchmarks). Using
  `Z3Backend` resolves this. The certified-interval *correctness*
  claim holds at $1 - \delta$ only when `is_satisfiable` is truly
  sound (which both backends guarantee) *and* closure is
  symbolically-validated (which only Z3 provides for arithmetic).
* **Mass conservation on `GeneralRegion`s** — partially resolved. The
  proportional-split refinement (see
  [`algorithm.md`](algorithm.md) §5) guarantees that children's
  masses sum to the parent's mass exactly, modulo Wilson-smoothed IS
  noise from the split-proportion estimator. The *root* mass is
  exactly $1$ by construction. Earlier prototype builds had a
  systematic mass drift of ~30 % on `popcount`; this is fixed.
* **Refinement clause selection.** The current heuristic
  (`_best_refinement_clause`) scores a *finite* set of candidate
  clauses harvested from observed paths. It does not invent new
  clauses (e.g. via Craig interpolation), so it cannot refine on
  predicates the program never branches on.
* **Concolic divergence.** A run that exceeds `max_concolic_branches`
  is dropped (not aggregated). Programs with unbounded inputs or
  inadvertent infinite loops thus quietly waste sample budget. The
  estimator state's `terminated_reason` and the scheduler's iteration
  log are the diagnostic.

## 4. Tooling gaps

* **CLI:** `dise run` accepts only registered benchmarks; ad-hoc
  programs go through the Python `estimate()` API.
* **Tighter Bernstein bounds.** We implement classical Bernstein and
  Maurer-Pontil empirical-Bernstein. The Audibert-Munos-Szepesvári
  bound and the Howard-Ramdas-McAuliffe time-uniform bound would be
  natural next additions.
* **Parallel execution.** All concolic runs are serial. Threading the
  outer loop is straightforward (each run is independent) but not yet
  implemented.

## 5. Verification gaps

* DiSE produces **certified intervals** under the soundness of its
  SMT oracle and the closure rule. We do not produce *machine-checked
  proofs* of those certificates (no Lean / Coq export).
* Theorem 2 assumes mass and sample-mean estimators are
  *independent* per leaf. The proportional-split refinement
  introduces *implicit* coupling (the split proportion is a function
  of the IS batch, which uses the same RNG stream). In practice this
  coupling is benign — the split is fixed *before* per-child
  sampling begins — but a formal independence argument would
  benefit from a stronger RNG-partition scheme.

## 6. Open extensions, ranked by impact

1. **Bayes-net (D2) distributions.** Adds a whole class of structured
   joints without sacrificing closed-form mass.
2. **Craig-interpolation refinement.** Augments the candidate-clause
   set with predicates the SMT solver invents.
3. **Parallel concolic.** Linear speed-up.
4. **Counter-example-guided MCMC.** Use SMT to find feasible starts
   for the integer-lattice MH chain; tightens the acceptance rate on
   very rare events.
5. **Machine-checked certificates.** Export `(F_pi, n_pi, h_pi,
   path_pi, mu_hat, eps_total)` tuples in a format consumable by a
   proof assistant.
6. **Formal handling of `unknown`.** Right now `unknown` falls back
   to sample-based closure; a more conservative variant would
   *refuse* to close and accept the wider interval.
