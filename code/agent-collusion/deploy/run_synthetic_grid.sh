#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-.venv/bin/python}"
MODEL="${MODEL:-openai/qwen36-synthetic}"
CONFIGS="${CONFIGS:-all-false all-true mixed-40 mixed-80}"
SEEDS="${SEEDS:-7 8 9 10 11}"
LOSS_RATES="${LOSS_RATES:-0.10 0.20}"
COMMUNICATION="${COMMUNICATION:-enabled}"
MAX_PARALLEL="${MAX_PARALLEL:-10}"
SYNTHETIC_ROUNDS="${SYNTHETIC_ROUNDS:-3}"
CHAR_LIMIT="${CHAR_LIMIT:-200}"
REASONING_EFFORT="${REASONING_EFFORT:-medium}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-8192}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/synthetic/grid}"
ANALYSIS_PREFIX="${ANALYSIS_PREFIX:-grid_${COMMUNICATION}}"

: "${OPENAI_API_BASE:?Set OPENAI_API_BASE to the Modal endpoint /v1 URL}"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY to the Modal proxy token}"

export MODEL OPENAI_API_BASE OPENAI_API_KEY
export REASONING_EFFORT TEMPERATURE MAX_OUTPUT_TOKENS

JOB_DIR="$OUTPUT_ROOT/$COMMUNICATION/jobs"
LOG_DIR="$OUTPUT_ROOT/$COMMUNICATION/logs"
mkdir -p "$JOB_DIR" "$LOG_DIR"
rm -f "$JOB_DIR"/*.sh

for loss in $LOSS_RATES; do
  loss_tag="loss${loss/./}"  # 0.10 -> loss010, 0.20 -> loss020
  for config in $CONFIGS; do
    for seed in $SEEDS; do
      job="$JOB_DIR/${loss_tag}_${config}_seed${seed}.sh"
      log="$LOG_DIR/${loss_tag}_${config}_seed${seed}.log"
      cat > "$job" <<JOB
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT_DIR"
exec "$PYTHON" -m experiments \\
  --experiment-mode synthetic-certification \\
  --alice-model "\$MODEL" \\
  --bob-model "\$MODEL" \\
  --alice-reasoning-effort "\$REASONING_EFFORT" \\
  --bob-reasoning-effort "\$REASONING_EFFORT" \\
  --alice-temperature "\$TEMPERATURE" \\
  --bob-temperature "\$TEMPERATURE" \\
  --alice-max-output-tokens "\$MAX_OUTPUT_TOKENS" \\
  --bob-max-output-tokens "\$MAX_OUTPUT_TOKENS" \\
  --synthetic-schedule "$config" \\
  --synthetic-seed "$seed" \\
  --synthetic-loss-rate "$loss" \\
  --synthetic-rounds "$SYNTHETIC_ROUNDS" \\
  --communication "$COMMUNICATION" \\
  --no-preflight \\
  --char-limit "$CHAR_LIMIT" \\
  --repeats 1 \\
  --output-dir "$OUTPUT_ROOT/$COMMUNICATION/$loss_tag/$config/seed$seed" \\
  > "$log" 2>&1
JOB
      chmod +x "$job"
    done
  done
done

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "Generated $(find "$JOB_DIR" -maxdepth 1 -name '*.sh' | wc -l) job scripts in $JOB_DIR"
  echo "Example: $JOB_DIR/loss010_${CONFIGS%% *}_seed${SEEDS%% *}.sh"
  exit 0
fi

echo "Running $(find "$JOB_DIR" -maxdepth 1 -name '*.sh' | wc -l) trajectories, $MAX_PARALLEL at a time"
find "$JOB_DIR" -maxdepth 1 -name '*.sh' -print0 | sort -z | xargs -0 -n1 -P "$MAX_PARALLEL" bash

if [ "${SKIP_ANALYSIS:-0}" != "1" ]; then
  "$PYTHON" analysis/synthetic_metrics.py \
    --runs "$OUTPUT_ROOT/$COMMUNICATION/**/run.json" \
    --metrics-out "analysis/results/${ANALYSIS_PREFIX}_metrics.json" \
    --episodes-out "analysis/results/${ANALYSIS_PREFIX}_episodes.csv" \
    --annotations-out "analysis/results/${ANALYSIS_PREFIX}_annotations.jsonl"
fi

echo "Logs: $LOG_DIR"
