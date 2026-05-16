"""Bulk-fetch SV-COMP C benchmarks from the sosy-lab/sv-benchmarks repo.

Pulls .c files from categories that fit DiSE's input model (integer
programs, no heap/concurrency/floats), capped at a per-category limit.
Files larger than ``MAX_BYTES`` are skipped — DiSE's transpiler is for
small/medium programs, and very large SV-COMP files almost always use
features we can't handle.

Usage:
  uv run python experiments/fetch_svcomp.py
  PER_CATEGORY=100 uv run python experiments/fetch_svcomp.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "experiments" / "svcomp_samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Categories under sosy-lab/sv-benchmarks/c/ that contain mostly
# integer / bitvector programs without heap or concurrency.  Each entry
# is a relative path under c/.
CATEGORIES = [
    "loops-crafted-1",
    "loop-acceleration",
    "loop-invariants",
    "loop-simple",
    "loops",
    "loops-zilu",
    "loop-zilu",            # alternate spelling, harmless if 404
    "bitvector",
    "bitvector-loops",
    "bitvector-regression",
    "loop-new",
    "nla-digbench",
    "nla-digbench-scaling",
    "openssl",              # selective; many use pointers
    "ssh-simplified",
    "eca-programs",
    "termination-crafted",
    "termination-restricted-15",
    "verifythis",
]

MAX_BYTES = int(os.environ.get("MAX_BYTES", 8000))
PER_CATEGORY = int(os.environ.get("PER_CATEGORY", 60))
RAW_BASE = "https://raw.githubusercontent.com/sosy-lab/sv-benchmarks/master/c"
API_BASE = "https://api.github.com/repos/sosy-lab/sv-benchmarks/contents/c"


def _get_json(url: str) -> list:
    req = Request(url, headers={"Accept": "application/vnd.github+json"})
    try:
        with urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except (HTTPError, URLError) as e:
        return []


def _get_text(url: str) -> bytes | None:
    try:
        with urlopen(url, timeout=30) as r:
            return r.read()
    except (HTTPError, URLError):
        return None


def main() -> int:
    print(f"# bulk-fetching SV-COMP programs")
    print(f"# max bytes per file: {MAX_BYTES}, per-category limit: {PER_CATEGORY}")
    print(f"# output: {OUT_DIR}")
    print()
    total_listed = 0
    total_fetched = 0
    for cat in CATEGORIES:
        entries = _get_json(f"{API_BASE}/{cat}")
        if not entries:
            print(f"  {cat:35s} (no entries / 404)")
            continue
        candidates = [
            e for e in entries
            if e.get("type") == "file"
            and e.get("name", "").endswith(".c")
            and e.get("size", 1e9) <= MAX_BYTES
        ]
        candidates = candidates[:PER_CATEGORY]
        total_listed += len(candidates)
        fetched = 0
        for e in candidates:
            name = e["name"]
            target = OUT_DIR / name
            if target.exists():
                continue
            url = f"{RAW_BASE}/{cat}/{name}"
            body = _get_text(url)
            if body is None:
                continue
            target.write_bytes(body)
            fetched += 1
            time.sleep(0.02)  # be polite to the CDN
        total_fetched += fetched
        print(f"  {cat:35s} listed={len(candidates):3d}  fetched={fetched:3d}")
    print()
    print(f"# total listed: {total_listed}")
    print(f"# total fetched (new this run): {total_fetched}")
    print(f"# total in {OUT_DIR.name}/: {sum(1 for _ in OUT_DIR.glob('*.c'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
