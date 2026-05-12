# Artifact evaluation guide

This document maps the DiSE artifact onto the standard
**Functional / Reusable / Available** criteria used by the ATVA, SAS,
NFM, and TACAS artifact-evaluation committees.

## Summary

| Badge        | Status | Where to verify                                                  |
|--------------|--------|------------------------------------------------------------------|
| **Available**  | ✅ | Public GitHub repository (MIT-licensed), Docker image, archived release. |
| **Functional** | ✅ | `pytest` (≥ 250 tests) + the `dise` CLI sanity commands pass.   |
| **Reusable**   | ✅ | Public Python API (`estimate`, `failure_probability`), benchmark protocol, JSON I/O. |

## 1. Available

* **Source**: <https://github.com/aalok-thakkar/discrete-symbolic-estimation>
* **License**: MIT (see [`LICENSE`](LICENSE)).
* **Long-term archive**: tag `v0.1.0` (and any subsequent releases).
* **Container**: `docker build -t dise .` (see [`Dockerfile`](Dockerfile)).

## 2. Functional

Reproduce the artifact's core claims in a clean environment:

```bash
# 1. Install
uv sync
uv pip install -e ".[dev,plot]"

# 2. Type-check + lint
uv run mypy src/dise
uv run ruff check src/dise tests/

# 3. Full test suite
uv run pytest

# 4. CLI smoke tests
uv run dise version
uv run dise list                # 11 registered benchmarks
uv run dise compare 'integer_sqrt_correct_U(1,1023)' \
    --budget 400 --n-seeds 2 --mc-samples 1000
```

Each command should exit with status 0. The `pytest` run prints
`<N> passed`. The `dise compare` invocation prints a three-row table
(`plain_mc`, `stratified_random`, `dise`).

For the headline result from the brief — the GCD step-count
benchmark — a single run is:

```bash
uv run dise run 'gcd_steps_le_5_BG(p=0.1,N=100)' \
    --epsilon 0.05 --budget 5000 --cache-smt --backend z3
```

Expected: `terminated_reason='epsilon_reached'` and the certified
interval contains the MC ground truth (≈ 0.989).

For the assertion-violation framing:

```bash
uv run dise run 'assertion_overflow_mul_w=8_U(1,31)' \
    --epsilon 0.05 --no-budget --backend mock
```

Expected: `mu_hat ≈ 0.40` (the 8-bit overflow probability for
$a, b \sim \mathrm{Uniform}(1, 31)$), `terminated_reason='epsilon_reached'`.

## 3. Reusable

The artifact is structured as a Python package; reuse takes three forms.

### 3.1 Library API

```python
from dise import estimate, failure_probability, Uniform, BoundedGeometric

# Output-property framing
result = estimate(
    program=my_program,
    distribution={"x": Uniform(1, 100)},
    property_fn=lambda y: y < 50,
    epsilon=0.05, delta=0.05,
    budget=None,   # run until epsilon_reached
)

# Assertion-violation framing
result = failure_probability(
    program=my_program_with_asserts,
    distribution={"a": BoundedGeometric(p=0.1, N=100)},
    epsilon=0.05, delta=0.05,
)
```

Every exported name is documented in [`docs/architecture.md`](docs/architecture.md).

### 3.2 Benchmark protocol

New benchmarks register themselves via a decorator:

```python
# benchmarks/my_kernel.py
from dise.benchmarks._base import Benchmark, register
from dise.distributions import Uniform

@register
def my_kernel() -> Benchmark:
    return Benchmark(
        name="my_kernel_u100",
        description="...",
        program=lambda x: ...,
        distribution={"x": Uniform(1, 100)},
        property_fn=lambda y: y > 0,
        suggested_budget=5000,
    )
```

After import, the benchmark appears in `dise list` and is runnable
via `dise run my_kernel_u100`.

### 3.3 Baseline / experiment infrastructure

```python
from dise.experiment import run_experiment, default_methods

report = run_experiment(
    benchmark_name="my_kernel_u100",
    description="...",
    program=my_program,
    distribution=my_distribution,
    property_fn=my_property,
    methods=default_methods(budget=5000),
    seeds=range(5),
)
print(report.aggregates)  # one MethodAggregate per comparator
```

JSON-serialize with `dise.experiment.save_report`; plot with
`dise plot --report ... --kind compare`.

## 4. Reproducing the paper's experiments

See [`EXPERIMENTS.md`](EXPERIMENTS.md). The one-command path is:

```bash
scripts/reproduce.sh                # canonical: 3 seeds, budget 5000
QUICK=1 scripts/reproduce.sh        # fast: 2 seeds, budget 500
```

Output: per-benchmark JSON reports + a `summary.json` in `results/`,
plus PNG plots in `fig/` (when `matplotlib` is installed).

## 5. Hardware and software requirements

* **CPU**: any x86-64 / ARM64 with ≥ 1 GiB RAM. All experiments are
  single-threaded; wall-clock numbers in the paper are from a
  consumer laptop (M-series Mac).
* **OS**: Linux / macOS / Windows.
* **Python**: 3.10, 3.11, or 3.12 (tested in CI).
* **Memory**: ≤ 500 MiB for any single benchmark.
* **External dependencies**: `numpy`, `scipy`, `z3-solver`,
  `matplotlib` (optional, for plotting). All pip-installable; pinned
  in `uv.lock`.

## 6. Known limitations

See [`docs/limitations.md`](docs/limitations.md). Of particular note:

* `MockBackend` only handles axis-aligned arithmetic; the
  path-determinism shortcut for closure degrades to a sample-based
  heuristic and may introduce ≈ 1 % bias on hard benchmarks.
* Rare-event regions (acceptance < 1 % under any natural envelope) are
  handled by `IntegerLatticeMHSampler`, whose mixing is heuristic and
  not theoretically tied to a $\delta$-certified bound.

## 7. Contact

Issues and questions: <https://github.com/aalok-thakkar/discrete-symbolic-estimation/issues>.
