#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

bash "$REPO_ROOT/scripts/check_env.sh" figures
require_python

section "Building RPS panel data"
cd "$CNI_DIR"
export PYTHONPATH="$CNI_DIR/src:$CNI_DIR/experiments"

"$PYTHON" experiments/plot_pair_advantage_cumulative.py \
  "$CNI_DIR/results/llm_rps_v2/$RPS_V4_QWEN" \
  "$CNI_DIR/results/llm_rps_v2/$RPS_V4_GLM" \
  "$CNI_DIR/results/llm_rps_v2/$RPS_V4_MIXTRAL"
"$PYTHON" results_paper/scripts/plot_mi_curves.py
"$PYTHON" results_paper/scripts/plot_delta_mi_heatmaps.py
"$PYTHON" results_paper/scripts/plot_chronological_gcg_loss.py
"$PYTHON" results_paper/scripts/plot_fresh_pair_advantage.py
"$PYTHON" results_paper/scripts/plot_fresh_pair_advantage.py --reference v4_truncated
"$PYTHON" results_paper/scripts/plot_fresh_suffix_heatmap.py
"$PYTHON" results_paper/scripts/summarize_fresh_vs_old.py

section "Building AgentCollusion panel data"
cd "$AC_DIR"
"$PYTHON" scientific_figures/episode/plot_loss010_scores.py --out-dir "$PANEL_DIR"
"$PYTHON" scientific_figures/episode/plot_loss010_gcg_validation.py --out-dir "$PANEL_DIR"
"$PYTHON" analysis/gcg_poc/plot_loss_descent.py --output "$PANEL_DIR/gcg_loss_descent"

section "Composing the five article figures"
cd "$REPO_ROOT"
"$PYTHON" -m src.figure_composition.make_figures --figure all

section "Figure build complete"
echo "Final PDFs:"
find "$FIGURE_DIR" -maxdepth 1 -name '*.pdf' -print | sort
