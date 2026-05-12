# DiSE — Discrete Symbolic Estimation

[![python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![license: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

DiSE is a research prototype implementing **distribution-aware
reliability estimation** for deterministic integer / bitvector
programs under structured discrete operational distributions.

Given a program $P$, a discrete distribution $D$ over its inputs, a
Boolean property $\phi$, a target accuracy $(\varepsilon, \delta)$ and a
sample budget $B$, DiSE returns an estimate $\hat\mu$ of
$\Pr_{x \sim D}[\phi(P(x)) = 1]$ together with a **certified
two-sided half-width** $\varepsilon_{\text{total}} = \varepsilon_{\text{stat}} + W_{\text{open}}$
such that

$$
\Pr\big[\,|\hat\mu - \mu| \le \varepsilon_{\text{total}}\,\big] \;\ge\; 1 - \delta.
$$

The algorithm — **ASIP**, Adaptive Symbolic Importance Partitioning —
maintains a frontier of path-condition regions and chooses, at each
step, between *allocating more samples* to an open region or
*SMT-refining* it on a branch predicate. Closed-form mass on
axis-aligned LIA regions is the variance-reduction lever; SMT-driven
refinement is the partition-driver.

## Highlights

* **Anytime certified intervals.** Every iteration of the run
  produces a valid $(1 - \delta)$-coverage interval; termination
  reasons (`epsilon_reached`, `budget_exhausted`,
  `no_actions_available`) are reported transparently.
* **Mass-conservative refinement.** Children of any leaf partition
  the parent's mass *exactly* (modulo Wilson-smoothed IS noise on
  general regions).
* **Multiple half-width regimes.** Per-leaf Wilson (default,
  practical), classical Bernstein (conservative, soundness-only),
  Maurer–Pontil empirical-Bernstein (tighter for high-data leaves).
* **SMT caching.** A `CachedBackend` wrapper memoizes repeated
  formulas — typical 40–60 % hit rate on the headline benchmark.
* **MCMC fallback.** `IntegerLatticeMHSampler` provides
  Metropolis-Hastings sampling for rare-event general regions
  where rejection sampling's acceptance is too low.
* **Built-in baselines** (`PlainMonteCarlo`, `StratifiedRandomMC`)
  and an **experiment runner** with multi-seed aggregation, JSON
  reports, and matplotlib plotting.

## Installation

```bash
uv sync                                       # or: pip install -e ".[dev,plot]"
uv run dise list                              # 10 registered benchmarks
uv run pytest                                 # full unit-test suite
```

See [`INSTALL.md`](INSTALL.md) for details, Docker instructions, and
troubleshooting.

## Quickstart (Python API)

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
    epsilon=0.05, delta=0.05, budget=5000,
)
print(result)
# EstimationResult(mu_hat=0.9893, interval=[0.9612, 1.0000], samples=5000, ...)
```

## Quickstart (CLI)

```bash
# Run DiSE on one registered benchmark
dise run gcd_steps_le_5_BG\(p=0.1,N=100\) --budget 5000 --json-out out.json

# Compare DiSE against PlainMC + StratifiedRandomMC, 5 seeds
dise compare gcd_steps_le_5_BG\(p=0.1,N=100\) \
    --budget 5000 --n-seeds 5 --json-out compare.json

# Run the full benchmark suite, write JSON reports + summary
dise experiment --budget 5000 --n-seeds 3 --out-dir results/

# Render figures
dise plot --report results/gcd_steps_le_5_BG\(p=0.1,N=100\).json \
    --out fig/gcd.png --kind compare
```

Reproducing every paper experiment is a single command:

```bash
scripts/reproduce.sh
```

## Documentation

| Document                                       | What's there                                                                                |
|------------------------------------------------|---------------------------------------------------------------------------------------------|
| [`docs/algorithm.md`](docs/algorithm.md)       | Problem statement, ASIP pseudocode, Theorem 1 (variance), Theorem 2 (interval), Theorem 3 (closure). |
| [`docs/architecture.md`](docs/architecture.md) | Module diagram, key data types, invariants, file layout.                                    |
| [`docs/evaluation.md`](docs/evaluation.md)     | Experimental methodology: comparators, benchmarks, metrics, soundness verification.         |
| [`docs/limitations.md`](docs/limitations.md)   | What DiSE does *not* support; open extensions ranked by impact.                             |
| [`INSTALL.md`](INSTALL.md)                     | Install via `uv`, `pip`, or Docker.                                                         |
| [`EXPERIMENTS.md`](EXPERIMENTS.md)             | Exact reproduction commands for every paper table and figure.                                |

## Cite

See [`CITATION.cff`](CITATION.cff). BibTeX:

```bibtex
@software{dise2026,
  title  = {DiSE: Discrete Symbolic Estimation},
  author = {Thakkar, Aalok},
  year   = {2026},
}
```

## License

MIT. See [`LICENSE`](LICENSE).
