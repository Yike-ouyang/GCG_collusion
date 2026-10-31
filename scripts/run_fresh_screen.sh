#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if [[ "${YES:-0}" != "1" ]]; then
  confirm_or_exit "Launch the fresh suffix screen (3 models x 4 arms x 3 games x 10 rounds) on Modal? This incurs GPU costs."
fi

bash "$REPO_ROOT/scripts/check_env.sh" rps
require_python
require_modal

cd "$CNI_DIR"

export STUDY_PREFIX="$FRESH_PREFIX"
export RUNS="$FRESH_RUNS"
export ROUNDS="$FRESH_ROUNDS"
export SEED_OFFSET="$FRESH_SEED_OFFSET"
export WORKERS="$FRESH_WORKERS"
export CHUNK="$FRESH_CHUNK"
export DETACH=0

for model in $FRESH_MODELS; do
  for arm in $FRESH_ARMS; do
    campaign="${FRESH_PREFIX}_${model}_${arm}"
    local_dir="$CNI_DIR/results/llm_rps_v2/$campaign"

    if [[ -d "$local_dir/games" ]] && ! is_force; then
      section "Skipping $campaign (local results exist; FORCE=1 to rerun)"
      continue
    fi

    if is_force; then
      section "FORCE=1: removing local and remote campaign $campaign"
      rm -rf "$local_dir"
      modal_cmd volume rm -r netinf-v2 "/results/llm_rps_v2/$campaign" >/dev/null 2>&1 || true
    fi

    section "Launching $campaign"
    bash coalition_metrics/05_gcg_suffix_evaluation/fresh_game_study/run_campaign.sh launch "$model" "$arm"
  done
done

section "Fetching and analysing fresh screen"
bash coalition_metrics/05_gcg_suffix_evaluation/fresh_game_study/fetch_and_analyze.sh all $FRESH_ARMS

section "Fresh suffix screen complete"
