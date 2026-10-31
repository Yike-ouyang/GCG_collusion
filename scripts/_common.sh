#!/usr/bin/env bash
# Shared environment and path helpers. Source this file from every stage script.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

PYTHON="${PYTHON:-.venv/bin/python}"
MODAL_BIN="${MODAL_BIN:-.venv/bin/modal}"
if [[ "$PYTHON" != /* ]]; then PYTHON="$REPO_ROOT/$PYTHON"; fi
if [[ "$MODAL_BIN" != /* ]]; then MODAL_BIN="$REPO_ROOT/$MODAL_BIN"; fi

export REPO_ROOT
export PYTHON
export MODAL_BIN
export CNI_DIR="$REPO_ROOT/code/collusion-net-influence"
export AC_DIR="$REPO_ROOT/code/agent-collusion"
export FIGURE_DIR="${FIGURE_DIR:-$REPO_ROOT/figures}"
export PANEL_DIR="${PANEL_DIR:-$FIGURE_DIR/panels}"
export CNI_ROOT="$CNI_DIR"
export AC_ROOT="$AC_DIR"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/netinf-mpl}"

mkdir -p "$FIGURE_DIR" "$PANEL_DIR" "$MPLCONFIGDIR"

# The original RPS figure scripts write to ROOT/figures_to_keep.
if [[ -L "$CNI_DIR/figures_to_keep" ]]; then
  rm -f "$CNI_DIR/figures_to_keep"
fi
if [[ ! -e "$CNI_DIR/figures_to_keep" ]]; then
  ln -s "$PANEL_DIR" "$CNI_DIR/figures_to_keep"
fi

# AgentCollusion's CLI calls load_dotenv(repo_root / ".env").
# Never overwrite a real .env; only replace our symlink.
if [[ -L "$AC_DIR/.env" ]]; then
  rm -f "$AC_DIR/.env"
fi
if [[ ! -e "$AC_DIR/.env" ]]; then
  ln -s "$REPO_ROOT/.env" "$AC_DIR/.env"
fi

section() {
  printf '\n==> %s\n' "$*"
}

warn() {
  printf 'WARNING: %s\n' "$*" >&2
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_var() {
  local name="$1"
  [[ -n "${!name:-}" ]] || die "$name is empty. Set it in .env."
}

require_file() {
  local path="$1"
  [[ -e "$path" ]] || die "Missing required path: $path"
}

require_dir() {
  local path="$1"
  [[ -d "$path" ]] || die "Missing required directory: $path"
}

require_python() {
  [[ -x "$PYTHON" ]] || die "Python not found at $PYTHON. Run scripts/setup.sh."
}

require_modal() {
  [[ -x "$MODAL_BIN" ]] || die "Modal CLI not found at $MODAL_BIN. Run scripts/setup.sh."
}

modal_cmd() {
  require_modal
  if [[ -n "${MODAL_PROFILE:-}" ]]; then
    "$MODAL_BIN" --profile "$MODAL_PROFILE" "$@"
  else
    "$MODAL_BIN" "$@"
  fi
}

confirm_or_exit() {
  local message="$1"
  if [[ "${YES:-0}" == "1" ]]; then
    return 0
  fi
  printf '%s [y/N] ' "$message"
  read -r answer
  case "$answer" in
    y|Y|yes|YES) ;;
    *) die "Aborted." ;;
  esac
}

is_force() {
  [[ "${FORCE:-0}" == "1" ]]
}
