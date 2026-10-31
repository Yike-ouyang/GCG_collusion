#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if [[ "${YES:-0}" != "1" ]]; then
  confirm_or_exit "Launch the AgentCollusion GCG reinjection runs (baseline + suffix) on the configured endpoint? This incurs API/GPU costs."
fi

bash "$REPO_ROOT/scripts/check_env.sh" ac
require_python
require_file "$AC_DIR/analysis/results/gcg_poc/result.json"

cd "$AC_DIR"

SUFFIX="$("$PYTHON" -c "import json; print(json.load(open('analysis/results/gcg_poc/result.json'))['suffix'])")"
if [[ -z "$SUFFIX" ]]; then
  die "Could not read the suffix from analysis/results/gcg_poc/result.json"
fi

common=(
  --experiment-mode synthetic-certification
  --alice-model "${MODEL:-openai/qwen36-synthetic}"
  --bob-model "${MODEL:-openai/qwen36-synthetic}"
  --alice-reasoning-effort "${REASONING_EFFORT:-medium}"
  --bob-reasoning-effort "${REASONING_EFFORT:-medium}"
  --alice-temperature "${TEMPERATURE:-1.0}"
  --bob-temperature "${TEMPERATURE:-1.0}"
  --alice-max-output-tokens "${MAX_OUTPUT_TOKENS:-8192}"
  --bob-max-output-tokens "${MAX_OUTPUT_TOKENS:-8192}"
  --synthetic-schedule all-false
  --synthetic-seed "${AC_GCG_SEED:-10}"
  --synthetic-loss-rate "${AC_GCG_LOSS_RATE:-0.10}"
  --synthetic-rounds "${AC_GCG_ROUNDS:-3}"
  --communication enabled
  --no-preflight
  --repeats 1
)

run_baseline() {
  local out="$AC_DIR/results/synthetic/gcg_poc/suffix"
  if find "$out" -name run.json -print -quit 2>/dev/null | grep -q . && ! is_force; then
    section "Skipping baseline reinjection (run.json exists; FORCE=1 to rerun)"
    return 0
  fi
  if is_force; then rm -rf "$out"; fi
  section "Running baseline reinjection (no suffix)"
  "$PYTHON" -m experiments "${common[@]}" --output-dir "$out"
}

run_suffix() {
  local out="$AC_DIR/results/synthetic/gcg_poc/suffix_v2"
  if find "$out" -name run.json -print -quit 2>/dev/null | grep -q . && ! is_force; then
    section "Skipping suffix reinjection (run.json exists; FORCE=1 to rerun)"
    return 0
  fi
  if is_force; then rm -rf "$out"; fi
  section "Running suffix reinjection"
  "$PYTHON" -m experiments "${common[@]}" \
    --synthetic-verdict-suffix "$SUFFIX" \
    --synthetic-verdict-suffix-agent "${AC_GCG_AGENT:-alice}" \
    --output-dir "$out"
}

run_baseline
run_suffix

section "AgentCollusion GCG reinjection complete"
