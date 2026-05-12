"""``dise`` command-line interface.

Subcommands:

* ``dise list``           — list registered benchmarks.
* ``dise run``            — run DiSE on a registered benchmark.
* ``dise compare``        — run DiSE + baselines on one benchmark, multi-seed.
* ``dise experiment``     — run a full experiment suite, write JSON report.
* ``dise plot``           — render figures from a JSON report (requires matplotlib).
* ``dise version``        — print package version.

This entry point is wired via ``[project.scripts]`` in ``pyproject.toml``
so it is usable as plain ``dise`` after installation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from . import __version__


def _cmd_list(_args: argparse.Namespace) -> int:
    from dise.benchmarks import get_benchmark, list_benchmarks

    names = list_benchmarks()
    width = max(len(n) for n in names)
    for n in names:
        b = get_benchmark(n)
        print(f"  {n:<{width}}  — {b.description}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from dise.benchmarks import get_benchmark
    from dise.benchmarks._common import run_and_print

    bench = get_benchmark(args.benchmark)
    # Pass-through args from `run`.
    namespace = argparse.Namespace(
        epsilon=args.epsilon,
        delta=args.delta,
        budget=args.budget,
        bootstrap=args.bootstrap,
        batch_size=args.batch_size,
        seed=args.seed,
        backend=args.backend,
        cache_smt=args.cache_smt,
        mc_samples=args.mc_samples,
        skip_mc=args.skip_mc,
        json_out=args.json_out,
    )
    run_and_print(bench, namespace)
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from dise.benchmarks import get_benchmark

    from .experiment import default_methods, run_experiment, save_report

    bench = get_benchmark(args.benchmark)
    methods = default_methods(budget=args.budget, bootstrap=args.bootstrap, batch_size=args.batch_size)
    report = run_experiment(
        benchmark_name=bench.name,
        description=bench.description,
        program=bench.program,
        distribution=bench.distribution,
        property_fn=bench.property_fn,
        methods=methods,
        budget=args.budget,
        delta=args.delta,
        seeds=range(args.n_seeds),
        mc_samples=args.mc_samples,
        skip_mc=args.skip_mc,
    )
    _print_compare_table(report)
    if args.json_out:
        save_report(report, args.json_out)
        print(f"# wrote {args.json_out}", file=sys.stderr)
    return 0


def _cmd_experiment(args: argparse.Namespace) -> int:
    from dise.benchmarks import get_benchmark, list_benchmarks

    from .experiment import default_methods, run_experiment, save_report

    bench_names = args.benchmarks if args.benchmarks else list_benchmarks()
    os.makedirs(args.out_dir, exist_ok=True)
    summary: list[dict[str, Any]] = []
    for name in bench_names:
        bench = get_benchmark(name)
        print(f"# running {name} ...", file=sys.stderr)
        methods = default_methods(
            budget=args.budget, bootstrap=args.bootstrap, batch_size=args.batch_size
        )
        report = run_experiment(
            benchmark_name=bench.name,
            description=bench.description,
            program=bench.program,
            distribution=bench.distribution,
            property_fn=bench.property_fn,
            methods=methods,
            budget=args.budget,
            delta=args.delta,
            seeds=range(args.n_seeds),
            mc_samples=args.mc_samples,
            skip_mc=args.skip_mc,
        )
        safe = name.replace("/", "_").replace(" ", "_")
        out = os.path.join(args.out_dir, f"{safe}.json")
        save_report(report, out)
        for agg in report.aggregates:
            summary.append({
                "benchmark": agg.benchmark,
                "method": agg.method,
                "median_mu_hat": agg.median_mu_hat,
                "median_half_width": agg.median_half_width,
                "median_samples": agg.median_samples,
                "coverage": agg.coverage,
                "median_wall_clock_s": agg.median_wall_clock_s,
            })
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"# wrote {summary_path} (and per-benchmark JSON in {args.out_dir})",
          file=sys.stderr)
    _print_summary(summary)
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    try:
        from . import plot
    except ImportError as e:
        print(f"plotting requires matplotlib: {e}", file=sys.stderr)
        return 2
    return plot.run(args)


def _cmd_version(_args: argparse.Namespace) -> int:
    print(f"dise {__version__}")
    return 0


def _print_compare_table(report) -> None:
    print(f"# benchmark: {report.benchmark}")
    print(f"# {report.description}")
    if report.mc_truth is not None:
        print(f"# mc_truth: mu_MC = {report.mc_truth:.4f}  (n={int(1 / (report.mc_se**2 + 1e-12)) if report.mc_se else 0})")
    header = f"{'method':<24} {'mu_hat':>8} {'half_w':>9} {'samples':>9} {'cov.':>6} {'wall(s)':>9}"
    print(header)
    print("-" * len(header))
    for agg in report.aggregates:
        cov = f"{100.0 * agg.coverage:5.1f}%" if agg.coverage is not None else "  —  "
        print(
            f"{agg.method:<24} {agg.median_mu_hat:>8.4f} {agg.median_half_width:>9.4f} "
            f"{agg.median_samples:>9d} {cov:>6} {agg.median_wall_clock_s:>9.3f}"
        )


def _print_summary(summary: list[dict[str, Any]]) -> None:
    # Tabulate per (benchmark, method)
    benches = sorted({s["benchmark"] for s in summary})
    methods = sorted({s["method"] for s in summary})
    print()
    print(f"{'benchmark':<40} " + " ".join(f"{m:>16}" for m in methods))
    for b in benches:
        cells = []
        for m in methods:
            row = next((s for s in summary if s["benchmark"] == b and s["method"] == m), None)
            if row is None:
                cells.append(f"{'—':>16}")
            else:
                cells.append(
                    f"{row['median_mu_hat']:.3f}±{row['median_half_width']:.3f}"
                )
        print(f"{b:<40} " + " ".join(f"{c:>16}" for c in cells))


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dise",
        description="DiSE: distribution-aware reliability estimation.",
    )
    p.add_argument(
        "--version", action="version", version=f"dise {__version__}"
    )
    sub = p.add_subparsers(dest="command", required=True)

    # list
    sub.add_parser("list", help="list registered benchmarks")

    # version
    sub.add_parser("version", help="print version")

    # run
    pr = sub.add_parser("run", help="run DiSE on a registered benchmark")
    pr.add_argument("benchmark", type=str)
    pr.add_argument("--epsilon", type=float, default=0.05)
    pr.add_argument("--delta", type=float, default=0.05)
    pr.add_argument("--budget", type=int, default=5000)
    pr.add_argument("--bootstrap", type=int, default=200)
    pr.add_argument("--batch-size", type=int, default=50)
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--backend", choices=["auto", "z3", "mock"], default="auto")
    pr.add_argument("--cache-smt", action="store_true")
    pr.add_argument("--mc-samples", type=int, default=10_000)
    pr.add_argument("--skip-mc", action="store_true")
    pr.add_argument("--json-out", type=str, default=None)

    # compare
    pc = sub.add_parser(
        "compare",
        help="run DiSE + baselines on one benchmark with N seeds",
    )
    pc.add_argument("benchmark", type=str)
    pc.add_argument("--budget", type=int, default=5000)
    pc.add_argument("--bootstrap", type=int, default=200)
    pc.add_argument("--batch-size", type=int, default=50)
    pc.add_argument("--delta", type=float, default=0.05)
    pc.add_argument("--n-seeds", type=int, default=5)
    pc.add_argument("--mc-samples", type=int, default=20_000)
    pc.add_argument("--skip-mc", action="store_true")
    pc.add_argument("--json-out", type=str, default=None)

    # experiment
    pe = sub.add_parser(
        "experiment",
        help="run the full benchmark suite, write JSON reports + summary",
    )
    pe.add_argument(
        "--benchmarks",
        nargs="*",
        default=None,
        help="subset of benchmark names (default: all registered)",
    )
    pe.add_argument("--budget", type=int, default=5000)
    pe.add_argument("--bootstrap", type=int, default=200)
    pe.add_argument("--batch-size", type=int, default=50)
    pe.add_argument("--delta", type=float, default=0.05)
    pe.add_argument("--n-seeds", type=int, default=3)
    pe.add_argument("--mc-samples", type=int, default=20_000)
    pe.add_argument("--skip-mc", action="store_true")
    pe.add_argument(
        "--out-dir",
        type=str,
        default="results",
        help="directory to write per-benchmark JSON + summary.json",
    )

    # plot
    pp = sub.add_parser(
        "plot",
        help="render figures from a JSON report (requires matplotlib)",
    )
    pp.add_argument("--report", type=str, required=True)
    pp.add_argument("--out", type=str, required=True)
    pp.add_argument("--kind", choices=["compare", "convergence"], default="compare")

    return p


_DISPATCH = {
    "list": _cmd_list,
    "version": _cmd_version,
    "run": _cmd_run,
    "compare": _cmd_compare,
    "experiment": _cmd_experiment,
    "plot": _cmd_plot,
}


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    fn = _DISPATCH[args.command]
    return int(fn(args))


if __name__ == "__main__":
    sys.exit(main())
