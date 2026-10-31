#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f ".env" ]]; then
  cp .env.example .env
  echo "Created .env from .env.example. Edit it before running campaigns."
fi

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Creating .venv with python3..."
  python3 -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

# shellcheck disable=SC1091
source scripts/_common.sh

.venv/bin/python - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit(f"Python >= 3.11 required, found {sys.version.split()[0]}")
for module in ("numpy", "matplotlib", "modal", "litellm", "dotenv"):
    __import__(module)
print("Python dependencies OK")
PY

if ! command -v quarto >/dev/null 2>&1; then
  warn "Quarto is not installed; article rendering will be skipped. Install it from https://quarto.org/."
fi

if [[ -n "${NETINF_HF_SECRET:-}" && -n "${HF_TOKEN:-}" ]]; then
  echo "NETINF_HF_SECRET=$NETINF_HF_SECRET is set."
  echo "Create the Modal secret once if needed:"
  echo "  $MODAL_BIN secret create $NETINF_HF_SECRET HF_TOKEN=\$HF_TOKEN"
fi

section "Setup complete"
echo "Repository: $REPO_ROOT"
echo "Python:     $PYTHON"
echo "Figures:    $FIGURE_DIR"
echo "Panels:     $PANEL_DIR"
echo "Edit .env, then run ./run_initial_runs.sh, ./run_gcg_probes.sh, ./run_verifications.sh, or ./run_all.sh."
