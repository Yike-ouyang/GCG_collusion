#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/scripts/_common.sh"

if [[ "${YES:-0}" != "1" ]]; then
  confirm_or_exit "Run all initial datasets (RPS v4 campaigns + AgentCollusion synthetic grid)? This launches paid jobs."
  export YES=1
fi

bash "$REPO_ROOT/scripts/check_env.sh" full
bash "$REPO_ROOT/scripts/run_rps_v4.sh"
bash "$REPO_ROOT/scripts/run_ac_grid.sh"

section "Initial runs complete"
