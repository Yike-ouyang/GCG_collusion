#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/scripts/_common.sh"

if [[ "${YES:-0}" != "1" ]]; then
  confirm_or_exit "Run the verifications (fresh RPS suffix screen + AgentCollusion GCG reinjection)? This launches paid jobs."
  export YES=1
fi

bash "$REPO_ROOT/scripts/check_env.sh" full
bash "$REPO_ROOT/scripts/run_fresh_screen.sh"
bash "$REPO_ROOT/scripts/run_ac_gcg_reinjection.sh"

section "Verifications complete"
