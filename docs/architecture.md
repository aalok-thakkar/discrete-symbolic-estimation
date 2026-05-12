# Architecture

DiSE is a small Python package; this document maps the layers and the
data flow.

## 1. Layer diagram

```
                       dise.cli  (the `dise` command)
                                  │
                                  ▼
              dise.experiment.run_experiment ─ dise.baselines
                                  │
                                  ▼
                    dise.estimator.api.estimate
                                  │
                                  ▼
                     dise.scheduler.ASIPScheduler
            ┌─────────────────────┼─────────────────────┐
            ▼                     ▼                     ▼
       dise.regions          dise.concolic         dise.estimator
       (Region kinds,        (SymbolicInt,         (EstimatorState,
        Frontier,             run_concolic)         Wilson / Bernstein /
        FrontierNode)                                empirical-Bernstein)
            │                     │                     │
            └─────────┬───────────┴────────┬────────────┘
                      ▼                    ▼
                 dise.smt          dise.distributions
                 (SMTBackend +     (Distribution ABC,
                  Mock + Z3 +       ProductDistribution,
                  CachedBackend)    Geometric, …)
                                  │
                                  ▼
                            dise.sampler
                       (RejectionSampler,
                        IntegerLatticeMHSampler)
```

## 2. Module responsibilities

| Module                | Responsibility                                                                                    |
|-----------------------|---------------------------------------------------------------------------------------------------|
| `dise.distributions`  | Univariate + product discrete distributions; closed-form PMF/CDF/mass; truncated sampling.        |
| `dise.smt`            | Backend abstraction (`SMTBackend`); `Z3Backend`, `MockBackend`, `CachedBackend` memoizing facade. |
| `dise.regions`        | `Region` kinds; `build_region` dispatcher; `Frontier` + `FrontierNode` with proportional split.  |
| `dise.concolic`       | `SymbolicInt`, `run_concolic`; property-induced branches captured.                                |
| `dise.sampler`        | `RejectionSampler` and integer-lattice MH `IntegerLatticeMHSampler`.                              |
| `dise.estimator`      | Theorem 1 variance assembly; Wilson / Bernstein / MP-empirical-Bernstein half-widths; `EstimatorState`. |
| `dise.scheduler`      | ASIP main loop; allocate/refine action selection; SMT call accounting.                            |
| `dise.estimator.api`  | User-facing `estimate()` and `EstimationResult`.                                                  |
| `dise.baselines`      | `PlainMonteCarlo`, `StratifiedRandomMC`, `DiSEBaseline` adapter (common protocol).                |
| `dise.experiment`     | Multi-seed runner; JSON report; per-method aggregates (median + IQR).                             |
| `dise.cli`            | `dise list / run / compare / experiment / plot / version`.                                        |
| `dise.plot`           | Matplotlib helpers for `dise plot`.                                                               |
| `dise.benchmarks.*`   | Concrete benchmark instances + registry.                                                          |

## 3. Key data types

* **`Region`** (`dise.regions._base`). Abstract base. Concrete:
  `EmptyRegion`, `UnconstrainedRegion`, `AxisAlignedBox`, `GeneralRegion`.
  Each exposes `formula`, `mass`, `sample`, `contains`,
  `is_axis_aligned`.
* **`Frontier` / `FrontierNode`** (`dise.regions._frontier`). Tree.
  `refine` proportionally splits parent mass; `try_close` runs the
  symbolic shortcut when path clauses are available.
* **`SampleBatch`** (`dise.regions._base`). Constrained-sample
  container; may be *partial* (signaling rare-event regions).
* **`SymbolicInt`** (`dise.concolic`). Concolic integer; arithmetic
  builds new symbolic values, comparisons record `BranchRecord`s.
* **`EstimatorState`** (`dise.estimator`). Anytime snapshot of
  `(mu_hat, interval, eps_stat, eps_mass, W_open, …)`.
* **`SchedulerConfig`** (`dise.scheduler`). Algorithmic knobs:
  `epsilon`, `delta`, `budget_samples`, `bootstrap_samples`,
  `batch_size`, `refinement_cost_in_samples`,
  `max_refinement_depth`, `closure_min_samples`, etc.
* **`Baseline`** (`dise.baselines`). Common protocol for
  comparators. `BaselineResult` is the row type in experiment reports.
* **`Benchmark`** (`dise.benchmarks._base`). Bundles
  `program`, `distribution`, `property_fn`, and metadata.

## 4. Key invariants

| #   | Invariant                                                                          | Enforced by                                       |
|-----|------------------------------------------------------------------------------------|---------------------------------------------------|
| I1  | `is_satisfiable` is *sound* (never wrong, may say `unknown`).                      | `Z3Backend`, `MockBackend` contracts.             |
| I2  | `mu_var` is never $0$ on a finite OPEN leaf (Wilson smoothing).                    | `FrontierNode.mu_var`.                            |
| I3  | Closure requires either (a) sample-path-determinism *and* `phi` agreement, or     | `Frontier.try_close`.                             |
|     |   (b) SMT proof that `F_pi ∧ ¬path` is `unsat`.                                    |                                                   |
| I4  | A partial `SampleBatch` is *information*, not an error (rare-event regions).      | `RejectionSampler`, `GeneralRegion.sample`.       |
| I5  | The certified interval is two-sided, honest, and clipped to $[0, 1]$.              | `compute_estimator_state`.                        |
| I6  | $\sum_\pi w_\pi = 1$ for axis-aligned frontiers; for general frontiers $\le$ IS    | `Frontier.refine`'s proportional-split branch.    |
|     |   noise concentrated at the parent's mass.                                         |                                                   |

## 5. Cross-cutting concerns

### Logging

Currently print-based at the CLI layer. Library code is silent by
default. Future work: route through `logging.getLogger("dise")` with a
verbosity flag.

### Reproducibility

* Every entry point accepts `seed`.
* `estimate()` constructs a `numpy.random.Generator(PCG64(seed))` and
  uses *only* that source.
* `MC ground-truth` uses a fixed independent seed (12 345 by default
  in the experiment runner) so the reference is reproducible.

### Performance

* `CachedBackend` wraps any `SMTBackend` to memoize `is_satisfiable`,
  `evaluate`, `is_axis_aligned`, `project_to_variable`,
  `extract_var_bound`, `free_vars`. Cache stats (`hit_rate`) are
  exposed via `CacheStats`.
* Closed-form mass on axis-aligned boxes avoids most SMT calls.
* Mass-conservative proportional split caps the number of IS batches
  at $O(\text{refinements})$.

## 6. File layout

```
src/dise/
├── __init__.py             # public API surface
├── cli.py                  # `dise` CLI
├── plot.py                 # matplotlib helpers (loaded lazily)
├── distributions/
│   └── __init__.py
├── smt/
│   ├── __init__.py         # default_backend, has_z3
│   ├── base.py             # SMTBackend, Clause, SatResult
│   ├── mock.py             # MockBackend + MockExpr
│   ├── cache.py            # CachedBackend, CacheStats
│   └── z3_backend.py       # Z3Backend
├── regions/
│   ├── __init__.py
│   ├── _base.py            # Region ABC, Status, SampleBatch
│   ├── _concrete.py        # EmptyRegion, ..., build_region
│   └── _frontier.py        # Frontier, FrontierNode, proportional split
├── concolic/
│   └── __init__.py         # SymbolicInt, run_concolic
├── sampler/
│   └── __init__.py         # RejectionSampler, IntegerLatticeMHSampler
├── estimator/
│   ├── __init__.py         # Bernstein / MP-EB / Wilson + compute_estimator_state
│   └── api.py              # estimate() + EstimationResult
├── scheduler/
│   └── __init__.py         # ASIPScheduler + variance-aware refinement
├── baselines/
│   └── __init__.py         # PlainMonteCarlo, StratifiedRandomMC, DiSEBaseline
├── experiment/
│   └── __init__.py         # multi-seed runner, JSON reports, aggregates
└── benchmarks/             # registered benchmark suite
    ├── __init__.py
    ├── _base.py            # Benchmark protocol + registry
    ├── _common.py          # argparser + helpers
    ├── bitvector_kernels.py
    ├── collatz.py
    ├── gcd_geometric.py
    ├── integer_sqrt.py
    ├── miller_rabin.py
    ├── modular_exp.py
    ├── sieve_primality.py
    └── sparse_trie_depth.py
```
