#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if ! command -v quarto >/dev/null 2>&1; then
  die "Quarto is not installed. Install it from https://quarto.org/ or skip with RENDER_ARTICLE=0."
fi

cd "$REPO_ROOT"
section "Rendering article.pdf"
quarto render article.qmd --to pdf
rm -f "$REPO_ROOT/article.tex"

require_file "$REPO_ROOT/article.pdf"
section "Article rendered: $REPO_ROOT/article.pdf"
