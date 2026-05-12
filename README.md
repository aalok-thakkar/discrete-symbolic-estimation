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
  $(1 - \delta)$-coverage interval. Termination reasons
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
uv run dise list                              # 11 registered benchmarks
uv run pytest                                 # ~250+ tests
```

See [`INSTALL.md`](INSTALL.md) for details, Docker instructions, and
troubleshooting. For artifact-evaluation reviewers see
[`ARTIFACT.md`](ARTIFACT.md).

## Quickstart — output-property framing

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

| Document                                       | What's there                                                                                |
|------------------------------------------------|---------------------------------------------------------------------------------------------|
| [`docs/algorithm.md`](docs/algorithm.md)       | Problem statement, ASIP pseudocode, proofs of Theorems 1–3, complexity, anytime semantics.  |
| [`docs/architecture.md`](docs/architecture.md) | Module diagram, key data types, invariants, file layout.                                    |
| [`docs/evaluation.md`](docs/evaluation.md)     | Experimental methodology: comparators, benchmarks, metrics, soundness verification.         |
| [`docs/related-work.md`](docs/related-work.md) | Bibliography and positioning vs. PSE, probabilistic model checkers, sampling-based MC.     |
| [`docs/limitations.md`](docs/limitations.md)   | What DiSE does *not* support; open extensions ranked by impact.                             |
| [`INSTALL.md`](INSTALL.md)                     | Install via `uv`, `pip`, or Docker.                                                         |
| [`EXPERIMENTS.md`](EXPERIMENTS.md)             | Exact reproduction commands for every paper table and figure.                                |
| [`ARTIFACT.md`](ARTIFACT.md)                   | Artifact-evaluation checklist (functional / reusable / available).                          |
| [`CHANGELOG.md`](CHANGELOG.md)                 | Per-release notes.                                                                          |

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
