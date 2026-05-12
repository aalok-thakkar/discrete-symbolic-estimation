"""Smoke tests for the ``dise`` CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from dise import __version__


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "dise.cli", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_cli_version() -> None:
    result = _run(["version"])
    assert result.returncode == 0
    assert __version__ in result.stdout


def test_cli_list() -> None:
    result = _run(["list"])
    assert result.returncode == 0
    assert "gcd_steps_le_5" in result.stdout
    assert "miller_rabin" in result.stdout


def test_cli_run_smoke(tmp_path: Path) -> None:
    out = tmp_path / "result.json"
    result = _run([
        "run",
        "integer_sqrt_correct_U(1,1023)",
        "--budget", "200",
        "--bootstrap", "50",
        "--batch-size", "20",
        "--mc-samples", "300",
        "--skip-mc",
        "--backend", "mock",
        "--json-out", str(out),
    ])
    assert result.returncode == 0, result.stderr
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["benchmark"].startswith("integer_sqrt_correct")
    assert "mu_hat" in payload["dise_result"]


def test_cli_compare_smoke(tmp_path: Path) -> None:
    out = tmp_path / "compare.json"
    result = _run([
        "compare",
        "integer_sqrt_correct_U(1,1023)",
        "--budget", "200",
        "--n-seeds", "2",
        "--mc-samples", "300",
        "--json-out", str(out),
    ])
    assert result.returncode == 0, result.stderr
    assert out.exists()
    payload = json.loads(out.read_text())
    methods = {a["method"] for a in payload["aggregates"]}
    assert methods == {"plain_mc", "stratified_random", "dise"}


def test_cli_experiment_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "results"
    result = _run([
        "experiment",
        "--benchmarks", "integer_sqrt_correct_U(1,1023)",
        "--budget", "200",
        "--n-seeds", "2",
        "--mc-samples", "300",
        "--out-dir", str(out_dir),
    ])
    assert result.returncode == 0, result.stderr
    assert (out_dir / "summary.json").exists()
    # One per-benchmark JSON file should have been written.
    json_files = list(out_dir.glob("*.json"))
    assert len(json_files) >= 2  # summary + per-benchmark


def test_cli_plot_smoke(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    # First produce a JSON report.
    report = tmp_path / "report.json"
    res = _run([
        "compare",
        "integer_sqrt_correct_U(1,1023)",
        "--budget", "150",
        "--n-seeds", "2",
        "--mc-samples", "200",
        "--json-out", str(report),
    ])
    assert res.returncode == 0, res.stderr
    out_png = tmp_path / "fig.png"
    res = _run([
        "plot",
        "--report", str(report),
        "--out", str(out_png),
        "--kind", "compare",
    ])
    assert res.returncode == 0, res.stderr
    assert out_png.exists()
    assert out_png.stat().st_size > 1024  # non-trivial PNG
