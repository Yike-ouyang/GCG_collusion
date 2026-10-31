#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: bash $0 estimate|launch MODEL ARM" >&2
  echo "Models: qwen32b glm32b mixtral8x7b" >&2
  echo "Arms: no_channel optional_empty optional_gcg_nai optional_gcg_aru optional_initial_bang5 optional_random_placebo optional_counterinstruction" >&2
  exit 2
fi

mode="$1"
model="$2"
arm="$3"
study_prefix="${STUDY_PREFIX:-fresh_gcg_transfer_10r6g_v1_20261026}"
seed_offset="${SEED_OFFSET:-20261027}"
runs="${RUNS:-6}"
rounds="${ROUNDS:-10}"
workers="${WORKERS:-10}"
chunk="${CHUNK:-2}"
modal_cli=("${MODAL_BIN:-.venv/bin/modal}")
if [[ -n "${MODAL_PROFILE:-}" ]]; then
  modal_cli+=(--profile "$MODAL_PROFILE")
fi

detach_flag=(--detach)
if [[ "${DETACH:-1}" == "0" ]]; then
  detach_flag=()
fi

# The selected GCG candidates are the strongest same-stage recurrences in the
# latest pilot: ない at block 1 and ある at block 2.
case "$arm" in
  no_channel)
    condition="no_channel"
    prompt="prompts/pot_plan_v4.json"
    ;;
  optional_empty)
    condition="channel_optional"
    prompt="prompts/fresh_gcg_transfer/optional_empty.json"
    ;;
  optional_gcg_nai)
    condition="channel_optional"
    prompt="prompts/fresh_gcg_transfer/optional_gcg_nai.json"
    ;;
  optional_gcg_aru)
    condition="channel_optional"
    prompt="prompts/fresh_gcg_transfer/optional_gcg_aru.json"
    ;;
  optional_initial_bang5)
    condition="channel_optional"
    prompt="prompts/fresh_gcg_transfer/optional_initial_bang5.json"
    ;;
  optional_random_placebo)
    condition="channel_optional"
    prompt="prompts/fresh_gcg_transfer/optional_random_placebo.json"
    ;;
  optional_counterinstruction)
    condition="channel_optional"
    prompt="prompts/fresh_gcg_transfer/optional_counterinstruction.json"
    ;;
  *) echo "Unknown arm: $arm" >&2; exit 2 ;;
esac

case "$model" in
  qwen32b)
    hf_model="Qwen/Qwen2.5-32B-Instruct"
    gpu_price="3.95"
    modal_app="netinf-v2"
    ;;
  glm32b)
    hf_model="zai-org/GLM-4-32B-0414"
    gpu_price="3.95"
    modal_app="netinf-v2"
    ;;
  mixtral8x7b)
    hf_model="mistralai/Mixtral-8x7B-Instruct-v0.1"
    gpu_price="4.54"
    modal_app="netinf-v2-mixtral"
    ;;
  *) echo "Unknown model: $model" >&2; exit 2 ;;
esac

campaign="${study_prefix}_${model}_${arm}"

if [[ "$mode" == "estimate" ]]; then
  PYTHONPATH=src:experiments "${PYTHON:-.venv/bin/python}" experiments/llm_rps_v2.py \
    --estimate --design same_model \
    --agents "${hf_model}@http://estimate.invalid/v1,${hf_model}@http://estimate.invalid/v1,${hf_model}@http://estimate.invalid/v1" \
    --runs "$runs" --rounds "$rounds" --workers "$workers" \
    --prompts "$prompt" --conditions "$condition" \
    --seed-offset "$seed_offset" --gpu-price "$gpu_price" --tokens-per-sec "${TOKENS_PER_SEC:-300}"
  exit 0
fi

if [[ "$mode" != "launch" ]]; then
  echo "Mode must be 'estimate' or 'launch'." >&2
  exit 2
fi

NETINF_APP="$modal_app" "${modal_cli[@]}" run "${detach_flag[@]}" cloud/v2_modal.py::main \
  --design same_model --model "$model" --conditions "$condition" \
  --runs "$runs" --rounds "$rounds" --workers "$workers" --chunk "$chunk" \
  --prompts "$prompt" --campaign "$campaign" \
  --extra "--seed-offset $seed_offset"
