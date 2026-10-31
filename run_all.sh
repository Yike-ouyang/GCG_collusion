#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/scripts/_common.sh"

MODE="full"
DO_RENDER="${RENDER_ARTICLE:-1}"

usage() {
  cat <<'USAGE'
Usage: ./run_all.sh [options]

Options:
  --figures-only   Rebuild figures and article from existing results (no campaigns).
  --rps-only       Run RPS initial + RPS GCG + fresh screen, then build figures if AC data exists.
  --ac-only        Run AgentCollusion grid + GCG probe + reinjection, then build figures if RPS data exists.
  --no-render      Do not render article.pdf.
  --yes            Skip confirmation prompts.
  --force          Rerun campaigns even if outputs exist.
  --skip-deploy    Do not redeploy Modal apps (RPS stages).
  --dry-run        Print the planned stages and exit.
  -h, --help       Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --figures-only) MODE="figures" ;;
    --rps-only) MODE="rps" ;;
    --ac-only) MODE="ac" ;;
    --no-render) DO_RENDER=0 ;;
    --yes) export YES=1 ;;
    --force) export FORCE=1 ;;
    --skip-deploy) export SKIP_DEPLOY=1 ;;
    --dry-run) export DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
  shift
done

export RENDER_ARTICLE="$DO_RENDER"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  cat <<DRYRUN
Planned mode: $MODE
Repository:   $REPO_ROOT
Python:       $PYTHON
RPS results:  $CNI_DIR/results/llm_rps_v2
AC results:   $AC_DIR/results/synthetic
Figures:      $FIGURE_DIR
Render:       $RENDER_ARTICLE
Stages:
DRYRUN
  case "$MODE" in
    figures)
      echo "  scripts/build_figures.sh"
      echo "  scripts/render_article.sh"
      ;;
    rps)
      echo "  scripts/run_rps_v4.sh"
      echo "  scripts/run_gcg_rps.sh"
      echo "  scripts/run_fresh_screen.sh"
      echo "  scripts/build_figures.sh (if AC data exists)"
      echo "  scripts/render_article.sh"
      ;;
    ac)
      echo "  scripts/run_ac_grid.sh"
      echo "  scripts/run_ac_gcg_probe.sh"
      echo "  scripts/run_ac_gcg_reinjection.sh"
      echo "  scripts/build_figures.sh (if RPS data exists)"
      echo "  scripts/render_article.sh"
      ;;
    full)
      echo "  run_initial_runs.sh"
      echo "  run_gcg_probes.sh"
      echo "  run_verifications.sh"
      echo "  scripts/build_figures.sh"
      echo "  scripts/render_article.sh"
      ;;
  esac
  exit 0
fi

case "$MODE" in
  figures)
    bash "$REPO_ROOT/scripts/build_figures.sh"
    ;;
  rps)
    if [[ "${YES:-0}" != "1" ]]; then
      confirm_or_exit "Run the RPS pipeline (initial campaigns, GCG, fresh screen)? This launches paid jobs."
      export YES=1
    fi
    bash "$REPO_ROOT/scripts/check_env.sh" rps
    bash "$REPO_ROOT/scripts/run_rps_v4.sh"
    bash "$REPO_ROOT/scripts/run_gcg_rps.sh"
    bash "$REPO_ROOT/scripts/run_fresh_screen.sh"
    if bash "$REPO_ROOT/scripts/check_env.sh" figures >/dev/null 2>&1; then
      bash "$REPO_ROOT/scripts/build_figures.sh"
    else
      warn "AC results are incomplete; skipping figure composition."
    fi
    ;;
  ac)
    if [[ "${YES:-0}" != "1" ]]; then
      confirm_or_exit "Run the AgentCollusion pipeline (grid, GCG probe, reinjection)? This launches paid jobs."
      export YES=1
    fi
    bash "$REPO_ROOT/scripts/check_env.sh" ac
    bash "$REPO_ROOT/scripts/run_ac_grid.sh"
    bash "$REPO_ROOT/scripts/run_ac_gcg_probe.sh"
    bash "$REPO_ROOT/scripts/run_ac_gcg_reinjection.sh"
    if bash "$REPO_ROOT/scripts/check_env.sh" figures >/dev/null 2>&1; then
      bash "$REPO_ROOT/scripts/build_figures.sh"
    else
      warn "RPS results are incomplete; skipping figure composition."
    fi
    ;;
  full)
    if [[ "${YES:-0}" != "1" ]]; then
      confirm_or_exit "Run the full reproduction pipeline (all initial campaigns, GCG probes, verifications, figures, article)? This launches paid jobs and can take hours."
      export YES=1
    fi
    bash "$REPO_ROOT/run_initial_runs.sh"
    bash "$REPO_ROOT/run_gcg_probes.sh"
    bash "$REPO_ROOT/run_verifications.sh"
    bash "$REPO_ROOT/scripts/build_figures.sh"
    ;;
  *)
    die "Unknown mode: $MODE"
    ;;
esac

if [[ "$RENDER_ARTICLE" == "1" ]]; then
  bash "$REPO_ROOT/scripts/render_article.sh"
else
  warn "RENDER_ARTICLE=0: article.pdf was not rebuilt."
fi

section "Reproduction complete ($MODE)"
