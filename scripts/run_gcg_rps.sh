#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if [[ "${YES:-0}" != "1" ]]; then
  confirm_or_exit "Launch the chronological GCG optimisation for RPS batches '${GCG_RPS_BATCHES}' on Modal? This incurs GPU costs and can take hours."
fi

bash "$REPO_ROOT/scripts/check_env.sh" rps
require_python
require_modal
require_dir "$CNI_DIR/results/llm_rps_v2/$RPS_V4_QWEN/games"

cd "$CNI_DIR"
export PYTHONPATH="$CNI_DIR/src:$CNI_DIR/experiments"

for batch in $GCG_RPS_BATCHES; do
  first="${batch%_*}"
  last="${batch#*_}"
  name="qwen32b_chronological_bang5_runs_${first}_${last}"
  local_json="$CNI_DIR/results_paper/data/gcg_qwen_pilot/${name}.json"
  bundle_dir="/tmp/gcg-chronological-${first}-${last}"

  if [[ -f "$local_json" ]] && ! is_force; then
    section "Skipping $name (local checkpoint exists; FORCE=1 to rerun)"
    continue
  fi

  if is_force; then
    section "FORCE=1: removing local and remote checkpoints for $name"
    rm -f "$local_json" "$CNI_DIR/results_paper/data/gcg_qwen_pilot/${name}.md"
    modal_cmd volume rm netinf-v2 "/results/gcg_qwen_pilot/${name}.json" >/dev/null 2>&1 || true
    modal_cmd volume rm netinf-v2 "/results/gcg_qwen_pilot/${name}.md" >/dev/null 2>&1 || true
  fi

  section "Preparing prompt bundle for runs ${first}-${last}"
  rm -rf "$bundle_dir"
  "$PYTHON" coalition_metrics/04_gcg_suffix/run_suffix_pilot.py \
    --prepare-only \
    --campaign "$CNI_DIR/results/llm_rps_v2/$RPS_V4_QWEN" \
    --runs $(seq "$first" "$last") \
    --rounds $(seq 1 "${RPS_ROUNDS:-60}") \
    --out "$bundle_dir"

  require_file "$bundle_dir/cases_and_prompts.json"

  section "Running chronological GCG for runs ${first}-${last}"
  modal_cmd run cloud/gcg_qwen_pilot.py \
    --mode chronological \
    --train-bundle "$bundle_dir/cases_and_prompts.json" \
    --output-name "$name" \
    --initial-suffix "${GCG_INITIAL_SUFFIX:-!!!!!}" \
    --block-size "${GCG_BLOCK_SIZE:-10}" \
    --candidate-count "${GCG_CANDIDATE_COUNT:-4}" \
    --individual-weight "${GCG_INDIVIDUAL_WEIGHT:-0.25}"

done

section "Fetching GCG checkpoints"
bash "$REPO_ROOT/scripts/fetch_results.sh" gcg-rps

section "RPS GCG checkpoints complete"
