# CLI reference

The top-level ``dise`` command is registered as a console script by
``pip install -e .`` / ``uv sync``. It is a thin wrapper around the
:mod:`dise.estimator.api`, :mod:`dise.experiment`, and
:mod:`dise.plot` modules — anything reachable through the CLI is
also reachable through the Python API.

```
dise <subcommand> [options]
```

Run ``dise --version`` or ``dise version`` for the package version.

Every subcommand reads its arguments from ``argparse``; ``-h`` /
``--help`` on any subcommand prints the full flag list with defaults.
This document is the authoritative reference; the per-subcommand
``--help`` is the short-form alias.

## ``dise list``

Print every registered benchmark with its one-line description.

```bash
dise list
```

Example output (truncated):

```
  assertion_overflow_mul_w=8_U(1,31)  — Pr[a*b overflows 8-bit unsigned] …
  bitvector_log2_w6                   — log2(x) ≥ -1 for x in [0, 2^6)
  …
```

## ``dise run BENCHMARK [...]``

Run DiSE on one registered benchmark with a fixed seed. This is the
single-shot equivalent of :func:`dise.estimate`.

| Flag                    | Default      | Notes                                                                  |
|-------------------------|--------------|------------------------------------------------------------------------|
| ``--epsilon``           | ``0.05``     | Target certified half-width.                                           |
| ``--delta``             | ``0.05``     | Confidence parameter.                                                  |
| ``--budget``            | ``5000``     | Sample cap (concolic runs).                                            |
| ``--no-budget``         | off          | Disables the sample cap (algorithm runs until ``epsilon_reached``).    |
| ``--budget-seconds``    | ``None``     | Optional wall-clock cap in seconds.                                    |
| ``--min-gain-per-cost`` | ``0.0``      | Diminishing-returns floor.                                             |
| ``--method``            | ``wilson``   | ``wilson`` / ``anytime`` / ``bernstein`` / ``empirical-bernstein``.    |
| ``--bootstrap``         | ``200``      | Initial samples drawn at the root before adaptive action selection.    |
| ``--batch-size``        | ``50``       | Samples per allocate action.                                           |
| ``--seed``              | ``0``        | RNG seed.                                                              |
| ``--backend``           | ``auto``     | ``auto`` / ``z3`` / ``mock``.                                          |
| ``--cache-smt``         | off          | Wrap the backend in :class:`~dise.smt.CachedBackend`.                 |
| ``--mc-samples``        | ``10_000``   | MC ground-truth sample count.                                          |
| ``--skip-mc``           | off          | Don't compute the MC reference (saves time, no coverage check).        |
| ``--json-out PATH``     | ``None``     | If set, write a JSON summary to ``PATH`` in addition to stderr/stdout. |

The progress report goes to **stderr**; a one-line JSON summary goes
to **stdout** so ``dise run … | jq …`` works while a human still
sees the report.

Examples:

```bash
# Headline reproduction: GCD k=5 at the canonical budget.
dise run 'gcd_steps_le_5_BG(p=0.1,N=100)' --budget 5000 --json-out out.json

# Soundness-mode run: no sample cap, 60 s wall-clock cap,
# anytime-valid interval (set in the program via the API; the CLI
# always uses the SchedulerConfig default `method`).
dise run 'integer_sqrt_correct_U(1,1023)' --no-budget --budget-seconds 60 --epsilon 0.005

# Quick-and-conservative: Mock backend, small budget.
dise run 'coin_machine_U(1,9999)' --backend mock --budget 500
```

## ``dise compare BENCHMARK [...]``

Run DiSE *and* baseline methods on one benchmark over multiple
seeds, then print a comparison table.

| Flag                | Default      | Notes                                                  |
|---------------------|--------------|--------------------------------------------------------|
| ``--epsilon``       | ``0.05``     | Target half-width for DiSE.                            |
| ``--budget``        | ``5000``     | Per-run sample cap (applies to all methods).           |
| ``--bootstrap``     | ``200``      | DiSE bootstrap samples.                                |
| ``--batch-size``    | ``50``       | DiSE allocate batch size.                              |
| ``--delta``         | ``0.05``     | Confidence parameter.                                  |
| ``--n-seeds``       | ``5``        | Repetitions per method.                                |
| ``--mc-samples``    | ``20_000``   | MC ground-truth size.                                  |
| ``--skip-mc``       | off          | Skip the MC reference (no coverage column).            |
| ``--json-out PATH`` | ``None``     | Write the full per-seed report as JSON.                |

Output is a table with one row per method, columns: ``mu_hat``,
``half_width``, ``samples``, ``coverage``, ``wall(s)``. The
``coverage`` column is the fraction of seeds whose certified
interval contained the MC truth — for sound methods at confidence
``1 - delta`` we expect ``coverage ≥ 1 - delta`` asymptotically.

```bash
dise compare 'gcd_steps_le_5_BG(p=0.1,N=100)' \
    --budget 5000 --n-seeds 5 --json-out compare.json
```

## ``dise experiment [BENCHMARKS...]``

Run :func:`dise.experiment.run_experiment` over a subset (default:
*all*) of registered benchmarks and write per-benchmark JSON reports
plus a flat ``summary.json``.

| Flag                | Default     | Notes                                                                       |
|---------------------|-------------|-----------------------------------------------------------------------------|
| ``--benchmarks``    | (all)       | Space-separated subset of benchmark names.                                  |
| ``--epsilon``       | ``0.05``    | Target half-width.                                                          |
| ``--budget``        | ``5000``    | Per-run sample cap.                                                         |
| ``--bootstrap``     | ``200``     | DiSE bootstrap.                                                             |
| ``--batch-size``    | ``50``      | DiSE allocate batch.                                                        |
| ``--delta``         | ``0.05``    | Confidence parameter.                                                       |
| ``--n-seeds``       | ``3``       | Seeds per (benchmark, method).                                              |
| ``--mc-samples``    | ``20_000``  | MC ground-truth size.                                                       |
| ``--skip-mc``       | off         | Skip the MC reference.                                                      |
| ``--out-dir PATH``  | ``results`` | Output directory for per-benchmark JSON and ``summary.json``.               |

After completion, the console prints a 2-D table with one row per
benchmark and one column per method.

```bash
# Full suite.
dise experiment --budget 5000 --n-seeds 3 --out-dir results/

# Subset.
dise experiment --benchmarks 'gcd_steps_le_5_BG(p=0.1,N=100)' \
                              'integer_sqrt_correct_U(1,1023)' \
                --budget 5000 --n-seeds 5
```

## ``dise plot --report PATH --out FIG.png --kind KIND``

Render a figure from a JSON report (produced by ``dise compare`` /
``dise experiment``). Requires ``matplotlib`` — install via
``pip install '.[plot]'``.

| ``--kind``        | Figure                                                                                   |
|-------------------|------------------------------------------------------------------------------------------|
| ``compare``       | Three-panel bar chart: median half-width, median samples used, empirical coverage (%).  |
| ``convergence``   | Per-seed ``(samples, half-width)`` curves on a log-y axis.                              |

```bash
dise plot --report results/coin_machine_U_1_9999_.json \
          --out fig/coin_machine.png --kind compare
```

## ``dise version``

Print the installed package version.

```bash
dise version
# dise 0.1.0
```

## Per-benchmark scripts (``python -m dise.benchmarks.<name>``)

Every benchmark module also exposes a ``main()`` so it can be run
directly with all of the shared flags (``--epsilon``, ``--delta``,
``--budget``, ``--no-budget``, ``--budget-seconds``,
``--min-gain-per-cost``, ``--bootstrap``, ``--batch-size``,
``--seed``, ``--backend``, ``--cache-smt``, ``--mc-samples``,
``--skip-mc``, ``--json-out``) plus benchmark-specific knobs (e.g.
``--p``, ``--N``, ``--k``):

```bash
# Equivalent to `dise run 'gcd_steps_le_5_BG(p=0.1,N=100)'` but
# allows overriding p, N, k.
python -m dise.benchmarks.gcd_geometric --p 0.05 --N 200 --k 7 --budget 5000

# Run the pedagogical intro example.
python -m dise.benchmarks.coin_machine --N 9999 --budget 2000
```

The shared argparser is documented in
:mod:`dise.benchmarks._common`; see also
[`api-reference.md`](api-reference.md) §11.

## Stdout vs stderr policy

All ``dise`` subcommands write **human-readable** progress and
tables to **stderr** and **machine-readable** one-line JSON to
**stdout**. To capture both into one file:

```bash
dise run 'coin_machine_U(1,9999)' > out.txt 2>&1
```

Pipe-friendly:

```bash
dise run 'coin_machine_U(1,9999)' | jq '.mu_hat'
```
