#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if [[ "${YES:-0}" != "1" ]]; then
  confirm_or_exit "Launch the AgentCollusion synthetic grid (4 schedules x 5 seeds, then 2 x 20 reruns) on the configured endpoint? This incurs API/GPU costs."
fi

bash "$REPO_ROOT/scripts/check_env.sh" ac
require_python

cd "$AC_DIR"

has_results() {
  find "$1" -name run.json -print -quit 2>/dev/null | grep -q .
}

run_grid() {
  local output_root="$1" configs="$2" seeds="$3" loss_rate="$4"
  local skip_analysis="$5" analysis_prefix="$6"

  if has_results "$AC_DIR/$output_root/enabled" && ! is_force; then
    section "Skipping $output_root (results exist; FORCE=1 to rerun)"
    return 0
  fi

  if is_force && [[ -d "$AC_DIR/$output_root" ]]; then
    section "FORCE=1: removing $output_root"
    rm -rf "$AC_DIR/$output_root"
  fi

  section "Running $output_root"
  CONFIGS="$configs" \
  SEEDS="$seeds" \
  LOSS_RATES="$loss_rate" \
  OUTPUT_ROOT="$output_root" \
  MAX_PARALLEL="${AC_MAX_PARALLEL:-10}" \
  SYNTHETIC_ROUNDS="${AC_SYNTHETIC_ROUNDS:-3}" \
  CHAR_LIMIT="${AC_CHAR_LIMIT:-200}" \
  SKIP_ANALYSIS="$skip_analysis" \
  ANALYSIS_PREFIX="$analysis_prefix" \
  PYTHON="$PYTHON" \
  MODEL="${MODEL:-openai/qwen36-synthetic}" \
  bash deploy/run_synthetic_grid.sh
}

run_grid \
  "results/synthetic/grid" \
  "${AC_SCHEDULES:-all-false all-true mixed-40 mixed-80}" \
  "${AC_SEEDS:-7 8 9 10 11}" \
  "${AC_LOSS_RATE:-0.10}" \
  "0" \
  "grid_enabled"

run_grid \
  "results/synthetic/grid_rerun10" \
  "all-false mixed-40" \
  "${AC_RERUN_SEEDS:-12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31}" \
  "${AC_LOSS_RATE:-0.10}" \
  "1" \
  "grid_rerun10"

if ! has_results "$AC_DIR/results/synthetic/grid_rerun10/enabled"; then
  die "Rerun grid produced no run.json files."
fi

section "Analyzing rerun grid"
mkdir -p "$AC_DIR/analysis/results"
"$PYTHON" analysis/synthetic_metrics.py \
  --runs "results/synthetic/grid_rerun10/enabled/**/run.json" \
  --metrics-out "analysis/results/grid_rerun10_metrics.json" \
  --episodes-out "analysis/results/grid_rerun10_episodes.csv" \
  --annotations-out "analysis/results/grid_rerun10_annotations.jsonl"

section "AgentCollusion grid complete"
