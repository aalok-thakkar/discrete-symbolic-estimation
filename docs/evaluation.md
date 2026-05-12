# Evaluation methodology

We evaluate DiSE against two baselines on a suite of ten integer
benchmarks. This document spells out exactly how the experiments are
constructed; the full reproduction commands are in
[`EXPERIMENTS.md`](../EXPERIMENTS.md).

## 1. Comparators

| Method                  | Class                                  | Notes                                                                                |
|-------------------------|----------------------------------------|--------------------------------------------------------------------------------------|
| `plain_mc`              | `dise.baselines.PlainMonteCarlo`       | Vanilla MC with a Wilson-score certified interval at the same `delta`.               |
| `stratified_random`     | `dise.baselines.StratifiedRandomMC`    | 16 random hash buckets (Bonferroni-corrected). No symbolic guidance.                 |
| `dise`                  | `dise.baselines.DiSEBaseline`          | ASIP with Wilson half-widths and the SMT closure shortcut (default).                 |

All three methods are exposed through the same
[`Baseline`](../src/dise/baselines/__init__.py) protocol so the
experiment runner can iterate uniformly.

## 2. Benchmarks

| Name                              | Description                                                    | Key knob          |
|-----------------------------------|----------------------------------------------------------------|-------------------|
| `gcd_steps_le_5_BG(p=0.1,N=100)`  | Euclidean GCD step count ≤ 5; the brief's running example.    | `k` (threshold)   |
| `modpow_fits_in_4b_m=37`          | Modular exponentiation result fits in `w` bits.                | `w`, `m`          |
| `miller_rabin_w=2_BG(p=0.05,N=200)` | Miller-Rabin (witness 2) deems `n` probably-prime.            | `witness`, `N`    |
| `popcount_w6`, `parity_w6`, `log2_w6` | Hacker's-Delight kernels; correctness property (`mu = 1`). | `w`               |
| `collatz_le_30_BG(p=0.05,N=200)`  | Collatz trajectory length ≤ 30 (irregular control flow).       | `k`, `N`          |
| `sieve_primality_U(2,200)`        | Trial-division primality.                                       | `N`               |
| `integer_sqrt_correct_U(1,1023)`  | `isqrt(n)**2 <= n < (isqrt(n)+1)**2` (specification check).   | `N`               |
| `sparse_trie_depth_le_3_U(0,63)`  | Max-depth of a 4-ary trie after two insertions.                | `N`, `k`          |

Run `dise list` for the always-up-to-date roster.

## 3. Metrics

For every $(\text{method}, \text{benchmark}, \text{seed})$ triple,
the experiment runner records:

* **`mu_hat`** — point estimate.
* **`interval = [lo, hi]`** — certified $(1 - \delta)$-coverage interval.
* **`half_width = (hi - lo) / 2`** — *the* headline metric.
* **`samples_used`** — actual concolic / MC calls.
* **`wall_clock_s`** — perf_counter measurement.
* **`interval_contains_truth`** — whether the MC ground truth lies in
  the certified interval. Aggregated across seeds, this is the
  empirical *coverage*; for sound methods at confidence $1 - \delta$
  we expect it to be $\ge 1 - \delta$.

Aggregated per `(method, benchmark)`:

* **`median_mu_hat`**, **`median_half_width`**, **`median_samples`**,
  **`median_wall_clock_s`** — robust to outliers.
* **`iqr_half_width`**, **`iqr_samples`** — variability across seeds.
* **`coverage`** — empirical, see above.

## 4. MC ground truth

For each benchmark, the experiment runner computes a high-budget plain
Monte-Carlo estimate (`mc_samples = 20 000` by default) using a fixed
independent seed (`12 345`), so the reference is reproducible and the
same across methods.

This MC estimate has its own standard error
$\mathrm{se}_{\mathrm{MC}} = \sqrt{\hat\mu_{\mathrm{MC}} (1 - \hat\mu_{\mathrm{MC}}) / n_{\mathrm{MC}}}$.
We do *not* report this as a certified interval — it is a regression
check.

## 5. Comparison style

Two question forms guide our reporting:

* **At fixed budget, who gets the tightest certified interval?**
  Compares `median_half_width` across methods at matched `budget`.
* **At what budget does each method reach a target $\varepsilon$?**
  Compares `median_samples` across methods constrained by
  `terminated_reason == "epsilon_reached"`.

DiSE's structural-variance-reduction wins are most visible in benchmarks
where the operational distribution concentrates on a path-deterministic
region with weight $\approx 1$ (e.g. `gcd_steps_le_10_*`,
`integer_sqrt_correct_*`). On flat distributions over many program
paths (e.g. `popcount_w6` under uniform), the benefit narrows because
each leaf demands its own constrained-sample budget.

## 6. Soundness verification

For every method × seed, we record
``interval_contains_truth``. Over $n$ seeds × $b$ benchmarks, the
empirical coverage rate should be $\ge 1 - \delta$ asymptotically. A
single seed × benchmark giving the wrong answer is *expected*
($\delta = 0.05$); the aggregate column shows the rate.

When DiSE's coverage rate falls below the target, the most likely
explanations (in decreasing order of probability) are:

1. The SMT backend is `MockBackend`, so the path-determinism shortcut
   degrades to a sample-based heuristic; documented in
   [`limitations.md`](limitations.md).
2. `closure_min_samples` is too small for the rare-minority-path
   regime of a particular benchmark; raise it.
3. The benchmark's program raises before tracing terminates (rare).

## 7. Reproducibility

Two routes:

1. **Single command** — see [`scripts/reproduce.sh`](../scripts/reproduce.sh).
2. **Manual** — see [`EXPERIMENTS.md`](../EXPERIMENTS.md). The
   experiment runner writes per-benchmark JSON reports plus a
   `summary.json`; tables in the paper come from these files via
   `dise plot`.

All seeds are derived from a single `--seed` flag; the experiment
runner enumerates `range(n_seeds)`. Wall-clock measurements are
machine-dependent and reported here for indication only.
