#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODE="${1:-full}"

check_python_modules() {
  require_python
  "$PYTHON" - <<'PY' || die "Required Python modules are missing. Run scripts/setup.sh."
import sys
for module in ("numpy", "matplotlib"):
    __import__(module)
PY
}

check_modal_credentials() {
  require_modal
  if [[ -n "${MODAL_TOKEN_ID:-}" && -n "${MODAL_TOKEN_SECRET:-}" ]]; then
    return 0
  fi
  if [[ -n "${MODAL_PROFILE:-}" ]]; then
    return 0
  fi
  if modal_cmd token current >/dev/null 2>&1; then
    return 0
  fi
  die "No Modal credentials. Run 'modal setup' or set MODAL_TOKEN_ID/MODAL_TOKEN_SECRET/MODAL_PROFILE in .env."
}

check_ac_endpoint() {
  require_var OPENAI_API_BASE
  require_var OPENAI_API_KEY
  require_var MODEL
}

check_v4_results() {
  local campaign
  for campaign in "$RPS_V4_QWEN" "$RPS_V4_GLM" "$RPS_V4_MIXTRAL"; do
    require_dir "$CNI_DIR/results/llm_rps_v2/$campaign/games"
  done
}

check_fresh_results() {
  local model arm campaign
  for model in $FRESH_MODELS; do
    for arm in $FRESH_ARMS; do
      campaign="${FRESH_PREFIX}_${model}_${arm}"
      require_dir "$CNI_DIR/results/llm_rps_v2/$campaign/games"
    done
  done
}

check_gcg_rps_results() {
  require_file "$CNI_DIR/results_paper/data/gcg_qwen_pilot/qwen32b_chronological_bang5_runs_0_4.json"
  require_file "$CNI_DIR/results_paper/data/gcg_qwen_pilot/qwen32b_chronological_bang5_runs_5_9.json"
}

check_has_run_json() {
  local root="$1" label="$2"
  if ! find -L "$root" -name run.json -print -quit 2>/dev/null | grep -q .; then
    die "No run.json found for $label under $root"
  fi
}

check_ac_grid_results() {
  local schedule seed dir
  for schedule in $AC_SCHEDULES; do
    for seed in $AC_SEEDS; do
      dir="$AC_DIR/results/synthetic/grid/enabled/loss010/$schedule/seed$seed"
      check_has_run_json "$dir" "grid $schedule/seed$seed"
    done
  done
  for seed in $AC_RERUN_SEEDS; do
    dir="$AC_DIR/results/synthetic/grid_rerun10/enabled/loss010/all-false/seed$seed"
    check_has_run_json "$dir" "rerun all-false/seed$seed"
  done
  # Mixed-40 reruns are useful for the descriptive table but not required by the figures.
  for seed in $AC_RERUN_SEEDS; do
    dir="$AC_DIR/results/synthetic/grid_rerun10/enabled/loss010/mixed-40/seed$seed"
    if ! find -L "$dir" -name run.json -print -quit 2>/dev/null | grep -q .; then
      warn "No mixed-40 rerun found for seed $seed (not required for the figures)."
      break
    fi
  done
}

check_ac_gcg_results() {
  require_file "$AC_DIR/analysis/results/gcg_poc/result.json"
  check_has_run_json "$AC_DIR/results/synthetic/gcg_poc/suffix" "AC GCG baseline"
  check_has_run_json "$AC_DIR/results/synthetic/gcg_poc/suffix_v2" "AC GCG suffix"
}

case "$MODE" in
  full)
    section "Checking full pipeline prerequisites"
    check_python_modules
    check_modal_credentials
    check_ac_endpoint
    ;;
  rps)
    section "Checking RPS pipeline prerequisites"
    check_python_modules
    check_modal_credentials
    ;;
  ac)
    section "Checking AgentCollusion pipeline prerequisites"
    check_python_modules
    check_modal_credentials
    check_ac_endpoint
    ;;
  figures)
    section "Checking existing results for figure rebuilding"
    check_python_modules
    check_v4_results
    check_fresh_results
    check_gcg_rps_results
    check_ac_grid_results
    check_ac_gcg_results
    ;;
  *)
    die "Unknown mode '$MODE'. Use: full, rps, ac, figures."
    ;;
esac

section "Environment check OK ($MODE)"
