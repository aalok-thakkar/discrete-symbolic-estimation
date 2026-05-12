# Installation

DiSE is a Python package distributed in source form. It supports
Python 3.10+ and runs on Linux, macOS, and Windows.

## 1. Install via `uv` (recommended)

[`uv`](https://docs.astral.sh/uv/) is a fast Python package manager
and project tool. After cloning the repository:

```bash
cd discrete-symbolic-estimation
uv sync                               # installs runtime + dev deps
uv pip install -e ".[dev,plot]"       # editable + matplotlib for `dise plot`
uv run dise --version
uv run pytest                         # 220+ tests, ~1 min on a laptop
```

The `dise` command is registered as a script entry point and is
runnable as plain `dise <subcommand>` after `uv sync`.

## 2. Install via `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,plot]"
dise --version
pytest
```

## 3. Install via Docker

A pinned reproducible image is provided. From the repository root:

```bash
docker build -t dise .
docker run --rm dise dise list
docker run --rm dise dise compare gcd_steps_le_5_BG\(p=0.1,N=100\) --budget 500 --n-seeds 2 --mc-samples 1000
```

To rerun the full experiment suite inside the container and copy the
JSON reports out:

```bash
docker run --rm -v "$PWD/results:/work/results" dise scripts/reproduce.sh
```

## 4. Optional dependencies

| Group   | Purpose                                                          |
|---------|------------------------------------------------------------------|
| `dev`   | `pytest`, `pytest-cov`, `mypy`, `ruff`, `hypothesis`             |
| `plot`  | `matplotlib` — required for `dise plot`                          |

The Z3 SMT backend is included as a *required* dependency
(`z3-solver`). If for some reason it is unavailable, DiSE falls back
to the `MockBackend`; see [`docs/limitations.md`](docs/limitations.md)
for caveats.

## 5. Verifying the install

A green install means all four of the following commands succeed:

```bash
uv run dise version
uv run dise list                                         # 10 benchmarks
uv run pytest                                            # full unit-test suite
uv run dise compare integer_sqrt_correct_U\(1,1023\) \
  --budget 400 --n-seeds 2 --mc-samples 1000             # ~30 s
```

If `pytest` reports 220 passing tests and `dise compare` prints a
three-row table (`plain_mc`, `stratified_random`, `dise`), the install
is correct.

## 6. Troubleshooting

* **`Z3Backend is None` or import errors.** Make sure `z3-solver`
  installed via the dependency pin (it ships pre-built wheels for
  most platforms). On ARM Linux you may need
  `pip install z3-solver --no-binary :all:` and a working `cmake`.
* **`matplotlib` missing.** Install the optional `plot` extra
  (`pip install '.[plot]'` or `uv pip install .[plot]`).
* **Slow Z3 runs (> 30 s).** Use `--cache-smt` (single-run CLI) or
  `--backend mock` for axis-aligned benchmarks.
