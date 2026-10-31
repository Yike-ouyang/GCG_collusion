"""Shared paths for the reproducible article pipeline."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "code"

CNI_ROOT = Path(
    os.environ.get("CNI_ROOT", CODE_ROOT / "collusion-net-influence")
).resolve()
AC_ROOT = Path(
    os.environ.get("AC_ROOT", CODE_ROOT / "agent-collusion")
).resolve()

RPS_RESULTS = CNI_ROOT / "results" / "llm_rps_v2"
CHRONO_DATA = CNI_ROOT / "results_paper" / "data" / "gcg_qwen_pilot"
AC_RESULTS = AC_ROOT / "results" / "synthetic"
AC_ANALYSIS = AC_ROOT / "analysis" / "results"

FIGURE_DIR = Path(os.environ.get("FIGURE_DIR", REPO_ROOT / "figures")).resolve()
PANEL_DIR = Path(os.environ.get("PANEL_DIR", FIGURE_DIR / "panels")).resolve()
ARTICLE_PDF = REPO_ROOT / "article.pdf"
