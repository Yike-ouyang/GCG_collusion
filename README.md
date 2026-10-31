# GCG-collusion — reproducible article pipeline

This repository contains the code needed to reproduce the experiments, figure data
and article PDF for:

> Can GCG Optimise Token Suffixes for Collusion Mitigation in LLM Agents?

This project is part of the [Apart Research AI Collusion Research sprint](https://apartresearch.com/sprints/ai-collusion-research-sprint-2026-10-23-to-2026-10-25).

## What can be reproduced

| Stage | What it runs | Main outputs |
|---|---|---|
| Initial runs | RPS v4 campaigns (Qwen2.5-32B, GLM-4-32B, Mixtral-8×7B); AgentCollusion synthetic grid | `code/collusion-net-influence/results/llm_rps_v2/`; `code/agent-collusion/results/synthetic/` |
| GCG probes | Chronological GCG on RPS; AgentCollusion GCG probe | `results_paper/data/gcg_qwen_pilot/`; `analysis/results/gcg_poc/result.json` |
| Verifications | Fresh RPS suffix screen; AgentCollusion GCG reinjection | fresh campaign folders; `results/synthetic/gcg_poc/suffix{,_v2}` |
| Figures | Panel scripts + article composition | PDFs in `figures/` |
| Article | Quarto render | `article.pdf` |



## Repository layout

```text
GCG_collusion/
├── article.qmd / article.pdf
├── figures/                      # final figure PDFs only
├── code/
│   ├── collusion-net-influence/  # RPS experiments + figure-data scripts
│   └── agent-collusion/          # AgentCollusion experiments + figure scripts
├── src/figure_composition/       # five-figure composition
├── scripts/                      # stage scripts
├── run_initial_runs.sh
├── run_gcg_probes.sh
├── run_verifications.sh
├── run_all.sh
├── .env.example
└── requirements.txt
```

## Prerequisites

- Linux, Python >= 3.11 (3.12 recommended)
- [Quarto](https://quarto.org/) for `article.pdf`
- A Modal account for the RPS/GCG campaigns
- An OpenAI-compatible endpoint for AgentCollusion (`OPENAI_API_BASE`, `OPENAI_API_KEY`, `MODEL`)
- Optional Hugging Face token for gated models

## Setup

```bash
cp .env.example .env
# edit .env: Modal credentials, endpoint, HF token, run parameters
./scripts/setup.sh
```

`setup.sh` creates `.venv`, installs `requirements.txt`, creates the figure/panel
directories and the `figures_to_keep` / `.env` symlinks used by the inherited code.

### Modal authentication

Either run once:

```bash
.venv/bin/modal setup
```

or set `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in `.env`.
If you use a named profile, set `MODAL_PROFILE`.

If `NETINF_HF_SECRET` is set, create the Modal secret once:

```bash
.venv/bin/modal secret create "$NETINF_HF_SECRET" HF_TOKEN="$HF_TOKEN"
```

## Entry points

### 1. Initial runs

```bash
./run_initial_runs.sh
```

Runs:
- `scripts/run_rps_v4.sh`: the three RPS v4 campaigns with the exact campaign names
  used by the figures.
- `scripts/run_ac_grid.sh`: the 4 × 5 AgentCollusion grid and the 2 × 20 reruns.

### 2. GCG probes

```bash
./run_gcg_probes.sh
```

Runs:
- `scripts/run_gcg_rps.sh`: chronological GCG batches `0_4` and `5_9`.
- `scripts/run_ac_gcg_probe.sh`: AgentCollusion GCG probe.

### 3. Verifications

```bash
./run_verifications.sh
```

Runs:
- `scripts/run_fresh_screen.sh`: 3 models × 4 arms × 3 games × 10 rounds.
- `scripts/run_ac_gcg_reinjection.sh`: baseline and suffix reinjection runs.

### 4. Everything

```bash
./run_all.sh
```

Runs the three stages above, then `scripts/build_figures.sh` and
`scripts/render_article.sh`.

Useful options:

```bash
./run_all.sh --figures-only   # rebuild figures + article from existing results
./run_all.sh --rps-only       # RPS stages only
./run_all.sh --ac-only        # AgentCollusion stages only
./run_all.sh --no-render      # skip article.pdf
./run_all.sh --dry-run        # print the planned stages
./run_all.sh --yes            # no confirmation prompt
./run_all.sh --force          # rerun existing campaigns
./run_all.sh --skip-deploy    # reuse existing Modal deployments
```

## Cost and time warning

The full pipeline launches paid GPU/API jobs and can take hours. The scripts print
the planned work and ask for confirmation unless `--yes` is passed. Use
`./run_all.sh --dry-run` first.

The three RPS v4 campaigns are 20 runs × 60 rounds each. The fresh screen is
3 models × 4 arms × 3 games × 10 rounds. AgentCollusion adds the synthetic grid,
the GCG probe and two reinjection runs.

## Outputs

Tracked:
- `figures/fig1_rps_ac_overview.pdf`
- `figures/fig2_gcg_optimisation.pdf`
- `figures/fig3_suffix_transfer.pdf`
- `figures/figS1_mi_curves.pdf`
- `figures/figS2_pairwise_heatmaps.pdf`
- `article.pdf`

Ignored:
- all `results/`, `analysis/results/`, `results_paper/data/`, logs
- `figures/panels/`, PNG previews
- `article.tex`, `.quarto/`
- `.env`, `.venv/`

## Figure-data mapping

| Article figure | Composition source | Original data scripts |
|---|---|---|
| Figure 1 | `src/figure_composition/make_figures.py` | `code/collusion-net-influence/{experiments,results_paper/scripts}` + `code/agent-collusion/scientific_figures` |
| Figure 2 | same | RPS chronological loss + AgentCollusion GCG/validation |
| Figure 3 | same | fresh screen + fresh suffix heatmap |
| Figure S1/S2 | same | RPS MI curves + pairwise heatmaps |

## Notes

- The AgentCollusion endpoint is not deployed by this repo. Set it in `.env`.
- The RPS prompts and fresh-screen prompt JSONs are inputs and are tracked.
- The original source repositories are preserved under `code/` to keep their imports
  and relative paths intact.
- The complete source history is available in the upstream repositories; this repo
  is the article-reproduction artifact.
