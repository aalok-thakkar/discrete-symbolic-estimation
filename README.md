# DiSE — Discrete Symbolic Estimation

[![python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![CI](https://github.com/aalok-thakkar/discrete-symbolic-estimation/actions/workflows/ci.yml/badge.svg)](https://github.com/aalok-thakkar/discrete-symbolic-estimation/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**DiSE** is a research prototype implementing **distribution-aware
reliability estimation** for deterministic integer / bitvector programs.

Given a program $P$, a discrete distribution $D$ over its inputs, a
Boolean property $\varphi$, a target accuracy $(\varepsilon, \delta)$
and (optionally) a sample budget, DiSE returns an estimate $\hat\mu$ of
$\Pr_{x \sim D}[\varphi(P(x)) = 1]$ together with a **certified
two-sided half-width** $\varepsilon_{\text{stat}} + W_{\text{open}}$
such that

$$
\Pr\big[\,|\hat\mu - \mu| \le \varepsilon_{\text{stat}} + W_{\text{open}}\,\big]
\;\ge\; 1 - \delta.
$$

The algorithm — **ASIP**, *Adaptive Symbolic Importance Partitioning* —
maintains a frontier of path-condition regions and at every step chooses
between *allocating more samples* to an open region or *SMT-refining*
it on a branch predicate. Closed-form mass on axis-aligned LIA regions
is the variance-reduction lever; SMT-driven refinement is the partition
driver.

## Two framings

DiSE accommodates two equivalent ways of phrasing a reliability question:

1. **Output property.** `property_fn(P(x))` is any Boolean predicate on
   the program's output (e.g. `steps <= k`, `result fits in w bits`).
   This is the general entry point :func:`dise.estimate`.

2. **Assertion violation.** Given a program with `assert` statements,
   estimate the failure probability $\Pr_D[P \text{ violates some
   assertion}]$ — the classical formal-verification framing. Use
   :func:`dise.failure_probability`, which instruments the program to
   convert `AssertionError` (or any user-specified exception) into a
   Boolean failure marker, then delegates to :func:`dise.estimate`.

The output-property framing strictly subsumes assertion-violation; the
wrapper is provided for ergonomics.

## Highlights

* **Anytime certified intervals.** Every iteration produces a valid
  $(1 - \delta)$-coverage interval. ``method="anytime"`` uses a
  time-uniform Wilson bound so the interval is sound under
  data-dependent stopping and adaptive sample sizes (see
  [`docs/algorithm.md`](docs/algorithm.md) §13). Termination reasons
  (`epsilon_reached`, `budget_exhausted`, `time_exhausted`,
  `no_actions_available`) are reported transparently.
* **Optional budget.** The algorithm is *budget-neutral by design*:
  termination on $\varepsilon$ is primary, the sample budget is an
  optional safety net. Pass `budget=None` for soundness-mode runs;
  `budget_seconds` provides an alternative wall-clock cap.
* **Mass-conservative refinement.** Children of any leaf partition the
  parent's mass *exactly* (modulo Wilson-smoothed IS noise on general
  regions).
* **Multiple half-width regimes.** Per-leaf Wilson (default,
  practical), classical Bernstein (conservative, soundness-only),
  Maurer–Pontil empirical-Bernstein (tighter for high-data leaves).
* **Sound SMT shortcut for closure.** Validates path-determinism
  symbolically before closing a leaf. The brief's "skip on
  `unknown`" rule is implemented exactly.
* **SMT caching.** A `CachedBackend` wrapper memoizes repeated formulas
  — typical 40–60 % hit rate on the headline benchmark.
* **MCMC fallback.** `IntegerLatticeMHSampler` provides
  Metropolis-Hastings sampling for rare-event general regions where
  rejection sampling's acceptance is too low.
* **Built-in baselines** (`PlainMonteCarlo`, `StratifiedRandomMC`)
  and an **experiment runner** with multi-seed aggregation, JSON
  reports, and matplotlib plotting.

## Installation

```bash
uv sync                                       # or: pip install -e ".[dev,plot]"
uv run dise list                              # 12 registered benchmarks
uv run pytest                                 # 290+ tests
```

See [`INSTALL.md`](INSTALL.md) for details, Docker instructions, and
troubleshooting. For artifact-evaluation reviewers see
[`ARTIFACT.md`](ARTIFACT.md).

## Quickstart — pedagogical intro example

A three-region branching program with a rare bug, designed so the
partition refinement is *visible*:

```python
from dise import estimate, Uniform

def coin_machine(x: int) -> int:
    if x < 10:           return 0     # safe region A
    if x < 100:          return 1     # always-flag region B
    if x % 1000 == 0:    return 1     # rare bug (mass ≈ 0.09%)
    return 0                          # safe region C-ok

result = estimate(
    program=coin_machine,
    distribution={"x": Uniform(1, 9999)},
    property_fn=lambda y: y == 1,
    epsilon=0.005, delta=0.05,
)
print(result)
# EstimationResult(mu_hat=0.0099, interval=[0.0099, 0.0099], samples=220, ...)
```

The truth is exactly $99/9999 \approx 0.0099$ (regions B and the
``x % 1000 == 0`` slice). Plain MC needs $\Theta(1/\mu) \approx 10^5$
samples to certify the rare bug at the same half-width; DiSE refines
into four leaves (A, B, C-bug, C-ok) and certifies the answer in a
few hundred concolic runs.

## Quickstart — operational reliability (the running benchmark)

```python
from dise import estimate, BoundedGeometric

def gcd_with_steps(a: int, b: int) -> int:
    steps = 0
    while b != 0:
        a, b = b, a % b
        steps += 1
    return steps

result = estimate(
    program=gcd_with_steps,
    distribution={
        "a": BoundedGeometric(p=0.1, N=100),
        "b": BoundedGeometric(p=0.1, N=100),
    },
    property_fn=lambda steps: steps <= 5,
    epsilon=0.05, delta=0.05,
    # budget defaults to 10_000; pass `budget=None` to remove the cap.
)
print(result)
# EstimationResult(mu_hat=0.9893, interval=[0.9612, 1.0000], samples=..., ...)
```

## Quickstart — distribution-aware property-based testing

Pair DiSE with Hypothesis strategies to certify failure rates of a
property-based test under operational input distributions:

```python
import hypothesis.strategies as st
from dise.integrations.hypothesis import estimate_from_strategies

result = estimate_from_strategies(
    strategies={"a": st.integers(min_value=1, max_value=31),
                "b": st.integers(min_value=1, max_value=31)},
    property_fn=lambda a, b: a * b < (1 << 8),     # 8-bit overflow check
    epsilon=0.05, delta=0.05,
)
print(result.mu_hat, result.interval)
```

See [`docs/hypothesis-integration.md`](docs/hypothesis-integration.md)
for the rationale and the strategy ↔ symbolic-region research direction.

## Quickstart — assertion-violation framing

```python
from dise import failure_probability, Uniform

def safe_mul(a: int, b: int) -> int:
    s = a * b
    assert s < (1 << 8), "8-bit overflow"
    return s

result = failure_probability(
    program=safe_mul,
    distribution={"a": Uniform(1, 31), "b": Uniform(1, 31)},
    epsilon=0.05, delta=0.05,
    # default: budget=None — run until epsilon_reached.
)
print(result.mu_hat)             # certified overflow probability ≈ 0.40
print(result.interval)           # [lo, hi]
```

## CLI

```bash
# Run DiSE on one registered benchmark
dise run gcd_steps_le_5_BG\(p=0.1,N=100\) --budget 5000 --json-out out.json

# Disable the sample cap (soundness mode); rely on epsilon for termination
dise run integer_sqrt_correct_U\(1,1023\) --no-budget --epsilon 0.01

# Cap wall-clock at 30 seconds
dise run miller_rabin_w=2_BG\(p=0.05,N=200\) --budget-seconds 30

# Compare DiSE against PlainMC + StratifiedRandomMC, 5 seeds
dise compare gcd_steps_le_5_BG\(p=0.1,N=100\) \
    --budget 5000 --n-seeds 5 --json-out compare.json

# Run the full benchmark suite, write JSON reports + summary
dise experiment --budget 5000 --n-seeds 3 --out-dir results/

# Render figures
dise plot --report results/gcd_steps_le_5_BG\(p=0.1,N=100\).json \
    --out fig/gcd.png --kind compare
```

Full reproduction is a single command:

```bash
scripts/reproduce.sh                  # canonical configuration
QUICK=1 scripts/reproduce.sh          # fast smoke
```

## Documentation

| Document                                                          | What's there                                                                                |
|-------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| [`docs/tutorial.md`](docs/tutorial.md)                            | First-time-user walkthrough using the ``coin_machine`` example.                             |
| [`docs/algorithm.md`](docs/algorithm.md)                          | Problem statement, ASIP pseudocode, proofs of Theorems 1–3, complexity, anytime semantics.  |
| [`docs/architecture.md`](docs/architecture.md)                    | Module diagram, key data types, invariants, file layout.                                    |
| [`docs/api-reference.md`](docs/api-reference.md)                  | Consolidated Python-API reference, module by module.                                        |
| [`docs/cli-reference.md`](docs/cli-reference.md)                  | Every ``dise`` subcommand and its flags.                                                    |
| [`docs/evaluation.md`](docs/evaluation.md)                        | Experimental methodology: comparators, benchmarks, metrics, soundness verification.         |
| [`docs/related-work.md`](docs/related-work.md)                    | Bibliography and positioning vs. PSE, probabilistic model checkers, sampling-based MC.      |
| [`docs/hypothesis-integration.md`](docs/hypothesis-integration.md)| DiSE × Hypothesis: distribution-aware property-based testing.                               |
| [`docs/limitations.md`](docs/limitations.md)                      | What DiSE does *not* support; open extensions ranked by impact.                             |
| [`INSTALL.md`](INSTALL.md)                                        | Install via ``uv``, ``pip``, or Docker.                                                     |
| [`EXPERIMENTS.md`](EXPERIMENTS.md)                                | Exact reproduction commands for every paper table and figure.                               |
| [`ARTIFACT.md`](ARTIFACT.md)                                      | Artifact-evaluation checklist (functional / reusable / available).                          |
| [`CHANGELOG.md`](CHANGELOG.md)                                    | Per-release notes.                                                                          |

## Citation

See [`CITATION.cff`](CITATION.cff). BibTeX:

```bibtex
@software{dise2026,
  title  = {DiSE: Discrete Symbolic Estimation},
  author = {Thakkar, Aalok},
  year   = {2026},
  url    = {https://github.com/aalok-thakkar/discrete-symbolic-estimation},
}
```

## License

MIT. See [`LICENSE`](LICENSE).
