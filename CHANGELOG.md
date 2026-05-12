# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
  - 10 benchmarks: GCD, modpow, Miller-Rabin, bitvector kernels,
    Collatz, sieve primality, integer sqrt, sparse trie.
  - Multi-seed experiment runner with JSON reports + matplotlib
    plotting.
  - Top-level `dise` CLI (list / run / compare / experiment / plot).
  - Dockerfile + `scripts/reproduce.sh` for hermetic reproduction.
  - 251 tests (unit + Hypothesis property tests + CLI smoke tests).
  - GitHub Actions CI workflow.
  - Full proofs of variance / coverage / closure theorems in
    `docs/algorithm.md`.
