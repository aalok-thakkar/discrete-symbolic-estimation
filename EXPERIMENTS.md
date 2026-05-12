# Reproducing the paper's experiments

> All commands below assume you have followed [`INSTALL.md`](INSTALL.md)
> and `dise list` succeeds. Wall-clock estimates are for a laptop;
> Z3-backed runs are roughly 2× to 4× slower than `MockBackend`.

## Table 1 — Comparison vs. baselines (single benchmark)

```bash
dise compare gcd_steps_le_5_BG\(p=0.1,N=100\) \
    --budget 5000 --n-seeds 5 --mc-samples 20000 \
    --json-out results/gcd_le_5.json
```

Replace `gcd_steps_le_5_BG(p=0.1,N=100)` with any name from
`dise list`. The resulting table prints `mu_hat ± median half-width`
per `(method, benchmark)`; the JSON includes per-seed rows.

## Table 2 — Full suite

```bash
dise experiment --budget 5000 --n-seeds 3 --mc-samples 20000 \
    --out-dir results/
```

This iterates over every registered benchmark and writes
`results/<benchmark>.json` plus `results/summary.json`. Console output
is a Markdown-friendly table.

To restrict to a subset:

```bash
dise experiment --benchmarks \
    gcd_steps_le_5_BG\(p=0.1,N=100\) \
    integer_sqrt_correct_U\(1,1023\) \
    miller_rabin_w=2_BG\(p=0.05,N=200\) \
    --budget 5000 --n-seeds 3
```

## Figures 1–N — `dise plot`

```bash
dise plot --report results/gcd_le_5.json --out fig/gcd_le_5.png --kind compare
dise plot --report results/gcd_le_5.json --out fig/convergence.png --kind convergence
```

`--kind compare` produces a 3-panel bar chart (half-width, samples,
coverage) per method. `--kind convergence` plots per-seed
``(samples, half_width)`` curves.

## End-to-end reproduction

The shell wrapper bundles everything:

```bash
scripts/reproduce.sh
```

It runs the experiment suite with the canonical defaults (budget 5000,
3 seeds, full benchmark list), then emits `results/*.png` if
`matplotlib` is installed. Total wall-clock on a laptop is ≈ 30–60
minutes (Z3 backend) or ≈ 5–10 minutes (`--backend mock`, less
accurate but fast).

## Hyperparameter ablations

To vary one knob at a time:

```bash
for budget in 1000 2000 5000 10000; do
    dise compare gcd_steps_le_5_BG\(p=0.1,N=100\) \
        --budget "$budget" --n-seeds 5 \
        --json-out "results/budget_${budget}.json"
done
```

The JSON reports include `samples_used` per seed, enabling
*budget-vs-half-width* curves to be drawn from the same data.

## Reproducing the headline acceptance commands from the brief

```bash
# Brief acceptance check 1: converges with terminated_reason='epsilon_reached'
dise run gcd_steps_le_5_BG\(p=0.1,N=100\) --epsilon 0.05 --budget 2000 \
    --backend z3 --cache-smt

# Brief acceptance check 2: half-width < 0.05 at budget 5000
dise run gcd_steps_le_5_BG\(p=0.1,N=100\) --epsilon 0.05 --budget 5000 \
    --backend z3 --cache-smt
```

Both produce an `EstimationResult` line on stdout that includes
`half_width`, `terminated_reason`, and `samples_used`. Pass
``--json-out result.json`` to keep the structured record.

## Seeding policy

* `--seed S` controls DiSE's RNG.
* The MC ground truth uses a fixed independent seed (12 345) so it is
  reproducible.
* In `dise compare` and `dise experiment`, seeds enumerate
  `range(n_seeds)`. If you need a different range, the Python API
  (`dise.experiment.run_experiment`) accepts any iterable.

## Wall-clock and SMT-cache

Use `--cache-smt` (single-run CLI) to enable the `CachedBackend`
wrapper. Cache statistics are available programmatically:

```python
from dise.smt import CachedBackend, Z3Backend
backend = CachedBackend(Z3Backend())
# ... run DiSE with `backend=...` ...
print(backend.stats.hit_rate)
```

On the GCD running example with budget 5000, the cache hit rate is
typically 40–60 %.
