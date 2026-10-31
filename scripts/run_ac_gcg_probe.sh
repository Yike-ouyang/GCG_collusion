#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if [[ "${YES:-0}" != "1" ]]; then
  confirm_or_exit "Launch the AgentCollusion GCG probe on a Modal H100? This incurs GPU costs."
fi

bash "$REPO_ROOT/scripts/check_env.sh" ac
require_python
require_modal

cd "$AC_DIR"

source_run_dir="$AC_DIR/results/synthetic/grid/enabled/loss010/all-false"
if ! find "$source_run_dir" -name run.json -print -quit 2>/dev/null | grep -q .; then
  die "Missing the all-false grid run used by the GCG probe. Run run_initial_runs.sh first."
fi

if [[ -f "$AC_DIR/analysis/results/gcg_poc/result.json" ]] && ! is_force; then
  section "Skipping GCG probe (result.json exists; FORCE=1 to rerun)"
  exit 0
fi

if is_force; then
  section "FORCE=1: removing previous GCG probe output"
  rm -rf "$AC_DIR/analysis/results/gcg_poc"
fi

section "Running GCG probe"
modal_cmd run deploy/gcg_poc_modal.py \
  --episode "${AC_GCG_EPISODE:-3}" \
  --agent "${AC_GCG_AGENT:-alice}" \
  --target "${AC_GCG_TARGET:-reject}"

require_file "$AC_DIR/analysis/results/gcg_poc/result.json"
section "AgentCollusion GCG probe complete"
