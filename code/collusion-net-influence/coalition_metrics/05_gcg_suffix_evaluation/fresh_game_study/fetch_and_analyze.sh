#!/usr/bin/env bash
set -euo pipefail

study_prefix="${STUDY_PREFIX:-fresh_gcg_transfer_10r6g_v1_20261026}"
modal_cli=("${MODAL_BIN:-.venv/bin/modal}")
if [[ -n "${MODAL_PROFILE:-}" ]]; then
  modal_cli+=(--profile "$MODAL_PROFILE")
fi
models=(qwen32b glm32b mixtral8x7b)
arms=(no_channel optional_empty optional_gcg_nai optional_gcg_aru)

if [[ $# -gt 0 ]]; then
  if [[ "$1" == "all" ]]; then
    shift
  else
    models=("$1")
    shift
  fi
fi
if [[ $# -gt 0 ]]; then
  arms=("$@")
fi

for model in "${models[@]}"; do
  for arm in "${arms[@]}"; do
    campaign="${study_prefix}_${model}_${arm}"
    echo "Fetching ${campaign}"
    "${modal_cli[@]}" volume get --force netinf-v2 \
      "/results/llm_rps_v2/${campaign}" "results/llm_rps_v2"
  done
done

MPLCONFIGDIR=/tmp/netinf-mpl "${PYTHON:-.venv/bin/python}" \
  coalition_metrics/05_gcg_suffix_evaluation/scripts/analyze_fresh_gcg_study.py \
  --campaign-root results/llm_rps_v2 \
  --prefix "$study_prefix" \
  --out "coalition_metrics/05_gcg_suffix_evaluation/results/${study_prefix}/analysis"
