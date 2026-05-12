# API reference

A consolidated tour of DiSE's public Python surface, module by
module. Symbols re-exported at the top level (``from dise import …``)
are marked **★**.

For a step-by-step walkthrough, see [`tutorial.md`](tutorial.md). For
the CLI surface, see [`cli-reference.md`](cli-reference.md).

## 1. Top-level — :mod:`dise`

The most commonly used names are re-exported from the package root:

```python
from dise import (
    # estimation
    estimate, failure_probability, EstimationResult,
    # distributions
    Distribution, Geometric, BoundedGeometric, Uniform,
    Categorical, Poisson, ProductDistribution,
    # SMT
    SMTBackend, MockBackend, Z3Backend,
    CachedBackend, CacheStats,
    default_backend, has_z3,
    # baselines
    Baseline, BaselineResult,
    PlainMonteCarlo, StratifiedRandomMC, DiSEBaseline,
)
```

### `estimate(program, distribution, property_fn, **kwargs) -> EstimationResult` ★

The headline entry point. Estimates
:math:`\mu = \Pr_D[\varphi(P(x)) = 1]` and returns a certified
two-sided interval at confidence :math:`1 - \delta`.

```python
estimate(
    program,                 # Callable[..., Any]
    distribution,            # Mapping[str, Distribution]
    property_fn,             # Callable[[Any], bool]
    epsilon=0.05,            # target half-width
    delta=0.05,              # confidence parameter
    budget=10_000,           # None = unbounded sample cap
    budget_seconds=None,     # None = no wall-clock cap
    min_gain_per_cost=0.0,   # diminishing-returns floor
    method="wilson",         # "wilson" | "anytime" | "bernstein" | "empirical-bernstein"
    bootstrap=200,
    batch_size=50,
    seed=0,
    backend=None,            # None = default_backend()
    verbose=False,
    max_refinement_depth=50,
    closure_min_samples=5,
    max_concolic_branches=10_000,
) -> EstimationResult
```

For ATVA-grade soundness under ASIP's adaptive stopping, use
``method="anytime"``.

### `failure_probability(program, distribution, *, catch=AssertionError, **kwargs) -> EstimationResult` ★

Convenience wrapper for the classical assertion-violation framing.
Instruments ``program`` so that exceptions of type ``catch`` (default
:class:`AssertionError`; pass a tuple for multiple) become Boolean
failures, then calls :func:`estimate`. All :func:`estimate` kwargs
forward through.

The default ``budget=None`` is unbounded — prefer to also set at
least one of ``budget_seconds`` or ``min_gain_per_cost > 0`` for
hard-bounded runs.

### `EstimationResult` ★

Dataclass returned by :func:`estimate` and :func:`failure_probability`.

| Attribute            | Type                  | Meaning                                                                  |
|----------------------|-----------------------|--------------------------------------------------------------------------|
| ``mu_hat``           | ``float``             | Point estimate.                                                          |
| ``interval``         | ``tuple[float, float]`` | Certified two-sided interval at confidence :math:`1 - \delta`.        |
| ``eps_stat``         | ``float``             | Statistical half-width contribution.                                     |
| ``W_open``           | ``float``             | Mass of unresolved (OPEN) leaves at termination.                         |
| ``delta``            | ``float``             | The confidence parameter passed in.                                      |
| ``samples_used``     | ``int``               | Concolic-run count.                                                      |
| ``refinements_done`` | ``int``               | Number of SMT refinement events.                                         |
| ``n_leaves``         | ``int``               | Total partition leaves (excluding EMPTY).                                |
| ``n_open_leaves``    | ``int``               | OPEN leaves at termination.                                              |
| ``n_closed_leaves``  | ``int``               | CLOSED_TRUE ∪ CLOSED_FALSE.                                              |
| ``terminated_reason``| ``str``               | One of ``epsilon_reached``, ``budget_exhausted``, ``time_exhausted``, ``no_actions_available``. |
| ``iterations``       | ``list[IterationLog]``| Trajectory (suppressed from ``repr`` — access directly).                |

Property: ``result.half_width`` returns ``(hi - lo) / 2``.

## 2. Distributions — :mod:`dise.distributions`

Immutable discrete distributions with closed-form PMF/CDF/mass and
seeded sampling.

### `Distribution` ★ — abstract base

Methods every distribution implements: ``pmf(x)``, ``cdf(x)``,
``mass(lo, hi)``, ``sample(rng, n)``, ``sample_truncated(rng, lo, hi, n)``,
``support_bounds(eps=1e-10)``.

### Concrete distributions ★

| Class               | Constructor                  | Support                                  |
|---------------------|------------------------------|------------------------------------------|
| ``Geometric``       | ``Geometric(p)``             | :math:`\{1, 2, \ldots\}`                 |
| ``BoundedGeometric``| ``BoundedGeometric(p, N)``   | :math:`\{1, \ldots, N\}` (renormalized)  |
| ``Uniform``         | ``Uniform(lo, hi)``          | :math:`\{l, l+1, \ldots, h\}` inclusive  |
| ``Categorical``     | ``Categorical(probs)``       | :math:`\{0, \ldots, k-1\}` for ``len(probs) == k`` |
| ``Poisson``         ``Poisson(lam)``              | :math:`\{0, 1, 2, \ldots\}`              |

All are frozen dataclasses; instances are hashable and safe to use
as dict keys.

### `ProductDistribution(factors: Mapping[str, Distribution])` ★

Joint distribution over named integer variables with independent
factors. Class (D1) in the brief's hierarchy.

```python
ProductDistribution(
    factors={"a": Uniform(1, 31), "b": Uniform(1, 31)}
)
```

Methods: ``pmf(x)``, ``sample(rng, n) -> dict[str, np.ndarray]``,
``sample_one(rng) -> dict[str, int]``, ``support_bounds(eps)``,
``variables``.

## 3. SMT — :mod:`dise.smt`

Abstraction layer for symbolic reasoning.

### `SMTBackend` ★ — abstract base

Defines the contract: ``make_int_var``, ``const``, ``op``,
``is_satisfiable``, ``evaluate``, ``free_vars``,
``is_axis_aligned``, ``project_to_variable``, ``extract_var_bound``,
``top_level_conjuncts``, ``conjunction``, ``negation``,
``repr_expr``, plus ``true()`` / ``false()``.

Soundness contract: ``is_satisfiable`` never returns ``"sat"`` /
``"unsat"`` incorrectly; may return ``"unknown"``.

### Backends

* ``MockBackend()`` ★ — conservative; handles axis-aligned LIA
  formulas exactly and returns ``"unknown"`` for non-axis-aligned
  arithmetic.
* ``Z3Backend()`` ★ — full LIA via ``z3-solver``. May be ``None`` if
  z3 is not installed; guard with ``has_z3()`` or use
  ``default_backend()``.
* ``CachedBackend(inner, max_entries=50_000)`` ★ — memoizing facade
  with FIFO eviction. Exposes ``stats`` (a :class:`CacheStats`
  instance with hit-rate metrics).

### Helpers ★

* ``default_backend() -> SMTBackend`` — Z3 if installed, else Mock.
* ``has_z3() -> bool`` — runtime check.

## 4. Regions — :mod:`dise.regions`

Path-condition regions and the ASIP frontier tree.

### `Region` — abstract base

Implementations:

* ``EmptyRegion`` — SMT-proved empty; mass exactly 0.
* ``UnconstrainedRegion`` — the full input space; mass exactly 1.
* ``AxisAlignedBox(bounds, formula)`` — Cartesian product of
  per-variable intervals; closed-form mass with zero variance.
* ``GeneralRegion(base, formula, smt)`` — axis-aligned envelope +
  SMT predicate; mass via importance sampling.

Common API: ``formula``, ``is_axis_aligned``,
``mass(distribution, smt, rng, n_mc=1000) -> tuple[float, float]``,
``sample(distribution, smt, rng, n) -> SampleBatch``,
``contains(x) -> bool``.

### `build_region(formula, distribution, smt) -> Region`

Dispatcher; returns the most-specific region kind for ``formula``.

### `Frontier(distribution, smt, n_mc_for_mass=1000)`

Tree of :class:`FrontierNode`s. Maintains the partition invariant
(see [`algorithm.md`](algorithm.md) §5).

Key methods:

* ``leaves() / open_leaves() / closed_leaves() / all_nodes()``
* ``open_mass() -> float`` — :math:`W_{\text{open}}`.
* ``total_leaf_mass() -> float`` — should be ≈ 1.
* ``compute_mu_hat() -> tuple[float, float]`` — ``(mu_hat,
  variance)``.
* ``find_leaf_for(x) -> FrontierNode``.
* ``ensure_mass(node, rng) -> None``.
* ``refine(node, clause, rng) -> list[FrontierNode]`` — splits a
  leaf and proportionally distributes its mass.
* ``try_close(node, min_samples) -> bool`` — applies the closure
  rule (sample-based + SMT-shortcut).
* ``add_observation(node, branch_sequence, phi_value, path_clauses)``.

### `FrontierNode`

Dataclass. Fields and derived quantities:

* ``region``, ``status``, ``parent``, ``children``, ``depth``,
  ``refining_clause``
* ``w_hat``, ``w_var``, ``mass_computed``
* ``n_samples``, ``n_hits``, ``observed_sequences``,
  ``observed_phis``, ``observed_paths``
* ``mu_hat``, ``mu_var``, ``mu_mean_var``,
  ``variance_contribution``, ``is_leaf``, ``formula``

### `Status` (enum)

``OPEN``, ``CLOSED_TRUE``, ``CLOSED_FALSE``, ``EMPTY``,
``DIVERGED`` (reserved; see source for the current semantics).

### `SampleBatch`

Dataclass: ``inputs: dict[str, np.ndarray]``, ``n: int``,
``rejection_ratio: float | None``. Iterate per-sample via
``iter_assignments()``.

## 5. Concolic execution — :mod:`dise.concolic`

### `SymbolicInt(concrete, expr, tracer)`

Concolic integer: arithmetic builds new symbolic values; comparisons
and ``bool()`` record a :class:`BranchRecord` on the tracer and
return a plain Python ``bool``. Hashed by ``id`` for safe dict
use.

### `BranchRecord`

Frozen dataclass: ``clause_taken``, ``clause_alt``.

### `Tracer(smt, max_branches=10_000)`

Accumulates branch records; raises an internal ``_BranchLimit`` when
``max_branches`` is reached.

### `ConcolicResult`

Output of :func:`run_concolic`: ``output``, ``phi_value``,
``path_condition``, ``inputs``, ``terminated``.

### `run_concolic(program, inputs, property_fn, smt, max_branches=10_000) -> ConcolicResult`

Run ``program`` once under concolic tracing. The property is
applied to the (still-symbolic) output, so comparisons inside
``property_fn`` are recorded.

## 6. Sampler — :mod:`dise.sampler`

### `Sampler` — abstract base

Implementations:

* ``RejectionSampler(max_attempts_per_sample=200)`` — closed-form
  for axis-aligned regions; rejection from the envelope for general
  regions. Returns partial batches on rare events.
* ``IntegerLatticeMHSampler(n_burn_in=200, thin=5, sigma_scale=0.15,
  init_attempts=2000)`` — Metropolis-Hastings on the integer lattice
  for rare-event general regions.

Both expose ``sample(region, distribution, smt, rng, n) -> SampleBatch``.

## 7. Estimator — :mod:`dise.estimator`

### `compute_estimator_state(frontier, delta, method="wilson") -> EstimatorState`

Aggregates the frontier into an :class:`EstimatorState`. Methods:

* ``"wilson"`` (default) — Bonferroni-Wilson; tightest at fixed
  sample counts.
* ``"anytime"`` — time-uniform Wilson via Bonferroni-in-time; sound
  under ASIP's adaptive stopping (recommended for ATVA-grade
  certificates).
* ``"bernstein"`` — classical Bernstein on total estimator variance.
* ``"empirical-bernstein"`` — Maurer–Pontil 2009.

### `EstimatorState`

Dataclass with ``mu_hat``, ``variance``, ``eps_stat``, ``eps_mass``,
``W_open``, ``delta``, ``interval``, ``n_total_samples``,
``n_leaves``, ``n_open_leaves``, ``n_closed_leaves``.

### Half-width primitives

* ``wilson_halfwidth_for_leaf(n, h, delta)`` — fixed-n Wilson.
* ``wilson_halfwidth_anytime(n, h, delta)`` — time-uniform Wilson.
* ``bernstein_halfwidth(variance, delta, per_sample_bound=1.0)``.
* ``empirical_bernstein_halfwidth_mp(empirical_variance, n, delta,
  range_bound=1.0)``.

## 8. Scheduler — :mod:`dise.scheduler`

### `ASIPScheduler(program, distribution, property_fn, smt, config, rng, sampler=None)`

The main driver. Construct via the dataclass-based
:class:`SchedulerConfig` and call ``scheduler.run() -> SchedulerResult``.

### `SchedulerConfig`

Tuning knobs (see docstring for the four termination conditions):

* ``epsilon``, ``delta``
* ``budget_samples`` (``int | None``), ``budget_seconds`` (``float | None``)
* ``min_gain_per_cost``
* ``method`` — see :func:`compute_estimator_state`
* ``bootstrap_samples``, ``batch_size``
* ``refinement_cost_in_samples``, ``max_refinement_depth``
* ``n_mass_samples`` — IS batch size for general-region mass
* ``smt_timeout_ms``, ``closure_min_samples``, ``max_concolic_branches``
* ``verbose``

### `SchedulerResult`

``final_estimator: EstimatorState``, ``iterations: list[IterationLog]``,
``samples_used``, ``refinements_done``, ``smt_calls``,
``terminated_reason``, ``frontier``.

### `IterationLog`

Per-iteration record: ``iter_idx``, ``action_kind``, ``leaf_depth``,
``samples_used_after``, ``mu_hat``, ``eps_stat``, ``W_open``,
``interval``, ``n_leaves``, ``n_open_leaves``.

## 9. Baselines — :mod:`dise.baselines`

### `Baseline` ★ — abstract base + `BaselineResult` ★

Each baseline implements ``run(program, distribution, property_fn,
budget, delta, seed) -> BaselineResult``. The result has ``name``,
``mu_hat``, ``interval``, ``samples_used``, ``wall_clock_s``,
``delta``, and method-specific ``extras``.

### Concrete baselines ★

* ``PlainMonteCarlo()`` — vanilla MC with a Wilson interval.
* ``StratifiedRandomMC(n_strata=16)`` — random hash-bucket
  stratification with Bonferroni per-bucket Wilson.
* ``DiSEBaseline(**estimate_kwargs)`` — adapter exposing
  :func:`dise.estimate` through the :class:`Baseline` protocol.
  Accepts any keyword forwarded to :func:`estimate`.

## 10. Experiment runner — :mod:`dise.experiment`

### `run_experiment(...) -> ExperimentReport`

Cartesian product over ``methods × seeds`` for one benchmark.

```python
run_experiment(
    benchmark_name, description, program, distribution, property_fn,
    methods,                     # Iterable[Baseline]
    budget=5000, delta=0.05,
    seeds=range(5),
    mc_samples=20_000, mc_seed=12_345, skip_mc=False,
) -> ExperimentReport
```

### `ExperimentReport`

``benchmark``, ``description``, ``mc_truth``, ``mc_se``,
``runs: list[RunResult]``, ``aggregates: list[MethodAggregate]``.
Serialize via :func:`save_report` / :func:`load_report`.

### `RunResult`

One row of the report (per method × seed):
``benchmark``, ``method``, ``seed``, ``budget``, ``delta``,
``mu_hat``, ``interval``, ``half_width``, ``samples_used``,
``wall_clock_s``, ``mc_truth``, ``interval_contains_truth``,
``error_vs_truth``, ``extras``.

### `MethodAggregate`

Per-method summary across seeds: ``n_seeds``, ``median_mu_hat``,
``median_half_width``, ``median_samples``, ``median_wall_clock_s``,
``coverage``, ``median_error_vs_truth``, ``iqr_half_width``,
``iqr_samples``.

### Helpers

* ``ground_truth_mc(program, distribution, property_fn, n_samples,
  seed=12_345) -> tuple[float, float]`` — plain MC reference.
* ``run_method(method, ...) -> RunResult`` — single-method driver.
* ``default_methods(budget, bootstrap=200, batch_size=50, epsilon=0.05,
  n_strata=16) -> list[Baseline]`` — the canonical
  ``[plain_mc, stratified_random, dise]`` set.
* ``save_report(report, path) / load_report(path)`` — JSON I/O.

## 11. Benchmarks — :mod:`dise.benchmarks`

### `Benchmark`

Dataclass bundling ``name``, ``description``, ``program``,
``distribution``, ``property_fn``, ``suggested_budget``,
``suggested_bootstrap``, ``suggested_batch_size``,
``closed_form_mu``, ``notes``, ``metadata``.

### Registry helpers

* ``register(factory) -> factory`` — decorator. Idempotent re-
  registration is allowed (handles the ``python -m dise.benchmarks.foo``
  vs ``benchmarks.foo`` double-load case).
* ``list_benchmarks() -> list[str]`` — sorted names of all registered.
* ``get_benchmark(name) -> Benchmark`` — fresh instance.

### Bundled benchmarks (run ``dise list`` for the live roster)

| Name                                          | Module                          |
|-----------------------------------------------|---------------------------------|
| ``coin_machine_U(1,9999)``                    | ``dise.benchmarks.coin_machine``|
| ``gcd_steps_le_5_BG(p=0.1,N=100)``            | ``dise.benchmarks.gcd_geometric``|
| ``modpow_fits_in_4b_m=37``                    | ``dise.benchmarks.modular_exp`` |
| ``miller_rabin_w=2_BG(p=0.05,N=200)``         | ``dise.benchmarks.miller_rabin``|
| ``popcount_w6`` / ``parity_w6`` / ``log2_w6`` | ``dise.benchmarks.bitvector_kernels``|
| ``collatz_le_30_BG(p=0.05,N=200)``            | ``dise.benchmarks.collatz``     |
| ``sieve_primality_U(2,200)``                  | ``dise.benchmarks.sieve_primality``|
| ``integer_sqrt_correct_U(1,1023)``            | ``dise.benchmarks.integer_sqrt``|
| ``sparse_trie_depth_le_3_U(0,63)``            | ``dise.benchmarks.sparse_trie_depth``|
| ``assertion_overflow_mul_w=8_U(1,31)``        | ``dise.benchmarks.assertion_overflow``|

Each benchmark module exposes a ``main()`` so it can be run as
``python -m dise.benchmarks.<name> [flags]`` — see [`cli-reference.md`](cli-reference.md).

## 12. Hypothesis integration — :mod:`dise.integrations.hypothesis`

Requires ``hypothesis`` (soft-imported; install with ``pip install
hypothesis``). See [`hypothesis-integration.md`](hypothesis-integration.md)
for the framing.

```python
from dise.integrations.hypothesis import (
    auto_from_strategy,       # SearchStrategy → Distribution
    from_integers,            # explicit Uniform(lo, hi) constructor
    from_sampled_from,        # explicit Uniform from consecutive ints
    estimate_from_strategy,   # single-strategy entry point
    estimate_from_strategies, # multi-strategy entry point
)
```

Tier-1 supported strategies:

* ``st.integers(min_value, max_value)`` → :class:`Uniform`.
* ``st.sampled_from(consecutive_ints)`` → :class:`Uniform`.

For other strategies, construct a DiSE :class:`Distribution`
manually and call :func:`dise.estimate` directly.

## 13. Plotting — :mod:`dise.plot`

Imported lazily by the CLI (``dise plot ...``); requires
``matplotlib``. Helper exposing ``run(args)`` with two ``--kind``
modes:

* ``compare`` — 3-panel bar chart (half-width, samples, coverage).
* ``convergence`` — log-scale ``(samples, half_width)`` per seed.
