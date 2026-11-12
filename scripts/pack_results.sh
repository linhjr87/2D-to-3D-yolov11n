#!/usr/bin/env bash
# Pack everything needed to analyse the runs elsewhere, without weights:
#   results/                                   evaluation outputs (metrics.json, compare.json, confusion matrices)
#   runs/*/{args.yaml,env.json,env_resume_*.json,results.csv,*.png}   per-run records and curves
#   runs/*.log                                 training logs
# Output: results_<git short sha>_<UTC date>.tar.gz in the repository root (or in OUT_DIR if set).
# Weights (*.pt) are deliberately not included: upload them separately if they are needed.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_DIR="${RUNS_DIR:-runs}"
RESULTS_DIR="${RESULTS_DIR:-results}"
OUT_DIR="${OUT_DIR:-$REPO_DIR}"
cd "$REPO_DIR"

sha="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
out="$OUT_DIR/results_${sha}_$(date -u +%Y%m%d).tar.gz"

[ -d "$RESULTS_DIR" ] || echo "pack_results: warning: $RESULTS_DIR/ does not exist (run the evaluation first)" >&2

# Build the file list with find so that spaces in names are safe (NUL-separated) and *.pt is excluded.
{
    [ ! -d "$RESULTS_DIR" ] || find "$RESULTS_DIR" -type f ! -name '*.pt' -print0
    [ ! -d "$RUNS_DIR" ] || find "$RUNS_DIR" -mindepth 1 -maxdepth 1 -type f -name '*.log' -print0
    [ ! -d "$RUNS_DIR" ] || find "$RUNS_DIR" -mindepth 2 -maxdepth 2 -type f \
        \( -name args.yaml -o -name env.json -o -name 'env_resume_*.json' -o -name results.csv -o -name '*.png' \) -print0
} > "$OUT_DIR/.pack_list.$$"
trap 'rm -f "$OUT_DIR/.pack_list.$$"' EXIT

[ -s "$OUT_DIR/.pack_list.$$" ] || { echo "pack_results: nothing to pack" >&2; exit 1; }
tar --null -czf "$out" -T "$OUT_DIR/.pack_list.$$"
echo "pack_results: wrote $out ($(du -h "$out" | cut -f1)), $(tar -tzf "$out" | wc -l) files"
