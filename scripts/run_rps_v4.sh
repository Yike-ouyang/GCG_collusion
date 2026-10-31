#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if [[ "${YES:-0}" != "1" ]]; then
  confirm_or_exit "Launch the three RPS v4 campaigns (20 runs x 60 rounds each) on Modal? This incurs GPU costs."
fi

bash "$REPO_ROOT/scripts/check_env.sh" rps

cd "$CNI_DIR"
export PYTHONPATH="$CNI_DIR/src:$CNI_DIR/experiments"

if [[ "${SKIP_DEPLOY:-0}" != "1" ]]; then
  section "Deploying Modal apps"
  NETINF_APP="${NETINF_APP:-netinf-v2}" modal_cmd deploy cloud/v2_modal.py
  NETINF_APP="${NETINF_APP_MIXTRAL:-netinf-v2-mixtral}" modal_cmd deploy cloud/v2_modal.py
else
  warn "SKIP_DEPLOY=1: using existing Modal deployments."
fi

run_one() {
  local model="$1" campaign="$2" app="$3"
  local local_dir="$CNI_DIR/results/llm_rps_v2/$campaign"

  if [[ -d "$local_dir/games" ]] && ! is_force; then
    section "Skipping $campaign (local results exist; FORCE=1 to rerun)"
    return 0
  fi

  if is_force; then
    section "FORCE=1: removing local and remote campaign $campaign"
    rm -rf "$local_dir"
    modal_cmd volume rm -r netinf-v2 "/results/llm_rps_v2/$campaign" >/dev/null 2>&1 || true
  fi

  section "Launching $campaign ($model)"
  NETINF_APP="$app" modal_cmd run cloud/v2_modal.py::main \
    --design same_model \
    --model "$model" \
    --runs "${RPS_RUNS:-20}" \
    --rounds "${RPS_ROUNDS:-60}" \
    --workers "${RPS_WORKERS:-34}" \
    --prompts "${RPS_PROMPT:-prompts/pot_plan_v4.json}" \
    --tag "${RPS_TAG:-big-v4}" \
    --campaign "$campaign"

  section "Fetching $campaign"
  mkdir -p "$CNI_DIR/results/llm_rps_v2"
  modal_cmd volume get --force netinf-v2 \
    "/results/llm_rps_v2/$campaign" "$CNI_DIR/results/llm_rps_v2"
}

run_one qwen32b "$RPS_V4_QWEN" "${NETINF_APP:-netinf-v2}"
run_one glm32b "$RPS_V4_GLM" "${NETINF_APP:-netinf-v2}"
run_one mixtral8x7b "$RPS_V4_MIXTRAL" "${NETINF_APP_MIXTRAL:-netinf-v2-mixtral}"

section "RPS v4 campaigns complete"
