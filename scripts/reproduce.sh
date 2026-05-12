#!/usr/bin/env bash
# Reproduce the canonical experiment suite documented in EXPERIMENTS.md.
#
# Usage:
#   scripts/reproduce.sh                    # full suite, 3 seeds, budget 5000
#   QUICK=1 scripts/reproduce.sh            # fast smoke (budget 500, 2 seeds)
#   N_SEEDS=5 BUDGET=10000 scripts/reproduce.sh
#
# Outputs:
#   results/<benchmark>.json   — per-benchmark report
#   results/summary.json       — flat table of (benchmark × method) aggregates
#   fig/<benchmark>.png        — compare-style 3-panel plot, if matplotlib is present
#   fig/convergence.png        — convergence plot for the headline benchmark

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-results}"
FIG_DIR="${FIG_DIR:-fig}"
N_SEEDS="${N_SEEDS:-3}"
BUDGET="${BUDGET:-5000}"
MC_SAMPLES="${MC_SAMPLES:-20000}"

if [[ -n "${QUICK:-}" ]]; then
    N_SEEDS=2
    BUDGET=500
    MC_SAMPLES=2000
fi

mkdir -p "$OUT_DIR" "$FIG_DIR"

echo "==> running full experiment suite"
echo "    out_dir=$OUT_DIR  n_seeds=$N_SEEDS  budget=$BUDGET  mc_samples=$MC_SAMPLES"
dise experiment \
    --budget "$BUDGET" \
    --n-seeds "$N_SEEDS" \
    --mc-samples "$MC_SAMPLES" \
    --out-dir "$OUT_DIR" \
    "$@"

echo ""
echo "==> rendering figures (if matplotlib is installed)"
for report in "$OUT_DIR"/*.json; do
    [[ "$report" == *"summary.json" ]] && continue
    base="$(basename "$report" .json)"
    # Strip parens so the file name is path-safe.
    safe="${base//\(/_}"
    safe="${safe//\)/_}"
    dise plot --report "$report" --out "$FIG_DIR/$safe.png" --kind compare \
        || echo "  (skipped $report — matplotlib not installed?)"
done

# Convergence plot on the headline benchmark, if present.
HEADLINE="$OUT_DIR/gcd_steps_le_5_BG(p=0.1,N=100).json"
if [[ -f "$HEADLINE" ]]; then
    dise plot --report "$HEADLINE" --out "$FIG_DIR/convergence.png" --kind convergence \
        || true
fi

echo ""
echo "==> done. Reports in $OUT_DIR/; figures in $FIG_DIR/."
