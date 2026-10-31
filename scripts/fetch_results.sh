#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODE="${1:-all}"
require_modal

fetch_v4() {
  local campaign
  mkdir -p "$CNI_DIR/results/llm_rps_v2"
  for campaign in "$RPS_V4_QWEN" "$RPS_V4_GLM" "$RPS_V4_MIXTRAL"; do
    section "Fetching $campaign"
    modal_cmd volume get --force netinf-v2 \
      "/results/llm_rps_v2/${campaign}" "$CNI_DIR/results/llm_rps_v2"
  done
}

fetch_fresh() {
  local model arm campaign
  mkdir -p "$CNI_DIR/results/llm_rps_v2"
  for model in $FRESH_MODELS; do
    for arm in $FRESH_ARMS; do
      campaign="${FRESH_PREFIX}_${model}_${arm}"
      section "Fetching $campaign"
      modal_cmd volume get --force netinf-v2 \
        "/results/llm_rps_v2/${campaign}" "$CNI_DIR/results/llm_rps_v2"
    done
  done
}

fetch_gcg_rps() {
  local batch first last name out
  out="$CNI_DIR/results_paper/data/gcg_qwen_pilot"
  mkdir -p "$out"
  for batch in $GCG_RPS_BATCHES; do
    first="${batch%_*}"
    last="${batch#*_}"
    name="qwen32b_chronological_bang5_runs_${first}_${last}"
    section "Fetching $name"
    modal_cmd volume get --force netinf-v2 \
      "/results/gcg_qwen_pilot/${name}.json" "$out"
    modal_cmd volume get --force netinf-v2 \
      "/results/gcg_qwen_pilot/${name}.md" "$out"
  done
}

case "$MODE" in
  v4) fetch_v4 ;;
  fresh) fetch_fresh ;;
  gcg-rps) fetch_gcg_rps ;;
  all)
    fetch_v4
    fetch_fresh
    fetch_gcg_rps
    ;;
  *) die "Unknown fetch mode '$MODE'. Use: v4, fresh, gcg-rps, all." ;;
esac
