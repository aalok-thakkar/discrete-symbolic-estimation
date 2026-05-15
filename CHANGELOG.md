# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added (Waudby-Smith & Ramdas betting half-width)

- **``method="betting"``** on :func:`dise.estimate`,
  :func:`dise.failure_probability`, :class:`SchedulerConfig`, and the
  ``dise run`` CLI. Implements the predictable-plug-in
  empirical-Bernstein (PrPl-EB) anytime-valid confidence sequence from
  Waudby-Smith & Ramdas (2024) Theorem 2. Closed-form, variance-
  adaptive, anytime-valid, and **strictly tighter than the existing
  ``"anytime"`` Bonferroni-in-time Wilson construction** in
  low-variance regimes — no :math:`\pi^2/6` inflation. Recommended
  setting for ATVA-grade certificates under ASIP's adaptive schedule.
- **``prpl_eb_halfwidth_anytime``** and **``prpl_eb_center``** in
  :mod:`dise.estimator`. The half-width function consumes a Bernoulli
  observation sequence (which the frontier already records per leaf as
  ``observed_phis``).

### Added (strict-unknown closure)

- **`strict_unknown` knob** on :class:`SchedulerConfig`, :func:`dise.estimate`,
  and :func:`dise.failure_probability`. When `True`, the closure rule
  refuses to close a leaf whose symbolic path-determinism check returns
  `"unknown"`; the leaf stays open and contributes to `W_open`. This is
  the strict-soundness setting and eliminates the ~1 % closure-bias that
  `MockBackend` and hard arithmetic can otherwise induce. Default `False`
  preserves legacy behavior. Resolves item 6 of the previous open-extensions
  list (`docs/limitations.md` §7).

### Added (project-wide audit-driven refactor)

- **CLI**: ``dise run`` now accepts ``--method
  {wilson,anytime,bernstein,empirical-bernstein}``, ``dise compare``
  and ``dise experiment`` accept ``--epsilon``. The ``method``
  kwarg is plumbed end-to-end through :class:`SchedulerConfig`,
  :func:`dise.estimate`, and :func:`dise.failure_probability`.
- **Docs**: three new reference documents — [`docs/tutorial.md`](docs/tutorial.md)
  (worked walkthrough), [`docs/api-reference.md`](docs/api-reference.md)
  (consolidated Python-API reference), and
  [`docs/cli-reference.md`](docs/cli-reference.md) (subcommand-by-
  subcommand). README's documentation table updated to surface them.

### Changed

- **Public API surface**: dead :class:`Clause` class removed from
  :mod:`dise.smt`; :class:`Status.DIVERGED` retained but documented
  as reserved-for-future-use.
- **Deduplicated** ``ground_truth_mc``: the version in
  :mod:`dise.benchmarks._common` now re-exports from
  :mod:`dise.experiment` (single source of truth).
- **Stale counts fixed**: ``dise list`` now reports 12 benchmarks (was
  10/11/"eleven" in various docs); test suite is 290+ items (was
  220/250/251 across docs).
- **Documentation**: ``evaluation.md`` benchmark table now includes
  the ``coin_machine`` row; ``architecture.md`` file-layout + module
  table now lists ``benchmarks/coin_machine``,
  ``benchmarks/assertion_overflow``, ``integrations/`` and
  ``integrations/hypothesis``.
- **Sparse docstrings polished** across :mod:`dise.regions._base`,
  :mod:`dise.regions._concrete`, :mod:`dise.integrations`,
  :mod:`dise.benchmarks._common`, plus :class:`Baseline`,
  :class:`DiSEBaseline`, :func:`save_report`, :func:`load_report`,
  :func:`ground_truth_mc`, :func:`run_method`, :func:`default_methods`.
- **Citation fixes**: ``related-work.md`` corrects "McBook" → "Owen
  (2013)"; the internal inverse-normal-CDF approximation is now
  correctly attributed to Acklam (2003).
- **Type-hint restoration** on every concrete
  :class:`~dise.regions.Region` subclass's ``mass`` / ``sample``
  methods (had been elided from the ABC's signature).
- ``failure_probability`` docstring now carries a ``.. warning::``
  block about the unbounded default ``budget=None``.

### Added (anytime-valid + intro example + Hypothesis bridge)
- **Anytime-valid Wilson bound.** New
  `dise.estimator.wilson_halfwidth_anytime` and
  `compute_estimator_state(..., method="anytime")`. The bound is
  time-uniform via Bonferroni-in-time on the Basel-normalized
  $\delta_n = 6\delta/(\pi^2 n^2)$, so it remains valid under
  data-dependent stopping and adaptive sample sizes — the recommended
  setting for ATVA-style certificates under ASIP's adaptive schedule.
- **Theorem 2 rewritten with explicit assumptions** (filtration,
  stopping time, A1–A4) and an anytime-valid statement.
  ``docs/algorithm.md`` §13 ("Statistical correctness under adaptive
  choices") explicitly addresses adaptive sample sizes, optional
  stopping, partition dependence, and refinement-decision correlation.
- **Bibliography of anytime-valid concentration:** Robbins 1970,
  Howard et al. 2021, Howard–Ramdas 2022, Waudby-Smith–Ramdas 2024 in
  ``docs/related-work.md``.
- **`coin_machine` intro benchmark.** A three-region branching program
  with a rare bug — the canonical pedagogical example for ASIP.
  Plain MC needs ~10^5 samples; DiSE certifies $\mu = 99/9999$ to
  half-width 0 in ~200 concolic runs.
- **`dise.integrations.hypothesis` adapter.** Tier-1 conversion of
  `hypothesis.strategies.SearchStrategy` instances to DiSE
  `Distribution`s (`from_integers`, `from_sampled_from`,
  `auto_from_strategy`) plus `estimate_from_strategy` /
  `estimate_from_strategies` entry points. Documented in
  ``docs/hypothesis-integration.md`` together with the research
  framing ("operational property-based testing").

### Added (previous release — optional budget + assertion API)
- **Optional sample budget.** `SchedulerConfig.budget_samples` and
  `estimate(budget=...)` now accept `None` to disable the sample cap.
  The algorithm runs until `epsilon_reached` (the *primary* stopping
  condition, always active).
- **Wall-clock cap.** New `budget_seconds` knob in `SchedulerConfig`,
  `estimate()`, and the CLI (`--budget-seconds`). Disabled by default.
- **Diminishing-returns floor.** New `min_gain_per_cost` knob: the
  scheduler declares `no_actions_available` when the best candidate's
  gain-per-cost falls below this threshold.
- **`dise.failure_probability(...)` API.** Convenience wrapper for the
  classical assertion-violation framing: estimate
  `Pr_D[program raises AssertionError]` (or any user-specified
  exception class via `catch=`). The wrapper installs the appropriate
  try/except instrumentation and delegates to `estimate`.
- **New `assertion_overflow_mul_w=8_U(1,31)` benchmark.** A canonical
  assertion-violation kernel: integer-multiplication overflow check
  under uniform inputs.
- **New `terminated_reason='time_exhausted'`** for the wall-clock cap.
- **CLI flags `--no-budget`, `--budget-seconds`, `--min-gain-per-cost`**
  on `dise run` and per-benchmark scripts.
- **`docs/related-work.md`** — bibliography and positioning vs. PSE,
  PRISM-style model checkers, and statistical concentration work.
- **`ARTIFACT.md`** — artifact-evaluation guide mapped to the
  Functional / Reusable / Available criteria.
- **`CHANGELOG.md`** — this file.

### Changed
- `docs/algorithm.md` reorganized to put the two framings (output
  property vs. assertion violation) up front, with the
  `failure_probability` wrapper explicitly derived as a special case
  in §12.
- README quickstart now shows *two* examples — one of each framing —
  and documents the optional budget.

## [0.1.0] — 2026-05-12

### Added
- Initial release. Distribution-aware reliability estimation via
  ASIP (Adaptive Symbolic Importance Partitioning):
  - Scheduler with mass-conservative refinement, variance-aware clause
    selection, and a Wilson / Bernstein / Maurer-Pontil empirical-
    Bernstein interval menu.
  - Concolic execution with property-induced branch capture.
  - SMT abstraction (Z3 + MockBackend + CachedBackend).
  - RejectionSampler and IntegerLatticeMHSampler.
  - Baselines (PlainMC, StratifiedRandomMC) for comparison.
  - Initially 10 benchmarks (extended to 12 in subsequent unreleased
    work; see the *Added* section above). The 0.1.0 set: GCD, modpow,
    Miller-Rabin, bitvector kernels (popcount / parity / log2),
    Collatz, sieve primality, integer sqrt, sparse trie.
  - Multi-seed experiment runner with JSON reports + matplotlib
    plotting.
  - Top-level `dise` CLI (list / run / compare / experiment / plot).
  - Dockerfile + `scripts/reproduce.sh` for hermetic reproduction.
  - 251 tests (unit + Hypothesis property tests + CLI smoke tests).
    Subsequent unreleased work brings the suite to 290+ tests.
  - GitHub Actions CI workflow.
  - Full proofs of variance / coverage / closure theorems in
    `docs/algorithm.md`.
