"""v2 campaigns on Modal: one vLLM deployment per model, per-seat routing, GPU benchmark (app `netinf-v2`).

Prerequisites (once):
    modal secret create huggingface HF_TOKEN=hf_...      # read token; only for gated meta-llama
    # (without Llama: deploy with NETINF_HF_SECRET= and use granite8b as the third family)
    # accept on huggingface.co: meta-llama/Llama-3.1-8B-Instruct (Llama 3.1 licence, manual approval)

1. Benchmark GPUs and smoke-test the models (5 rounds, all 5 conditions in parallel, one GPU per run):
    modal run cloud/v2_modal.py::bench --models qwen7b,ministral8b,granite8b,olmo7b --gpus A10G,L4 --max-gpus 8
   Prints, per model x GPU: cold start, throughput, cost per 30-round game, format failures, truncations,
   logprob coverage, the projected longest prompt and the max_model_len to deploy with.

2. Deploy the model servers (scale to zero when idle; the campaign keeps one warm container per model):
    NETINF_GPU=L4 NETINF_MAX_MODEL_LEN=12288 modal deploy cloud/v2_modal.py

3. Run a campaign (detached: keeps running with the laptop closed):
    modal run --detach cloud/v2_modal.py::main --design same_model --model qwen7b --runs 6 --rounds 30
    modal run --detach cloud/v2_modal.py::main --design mixed_models --lineup qwen7b,llama8b,ministral8b --runs 6 --rounds 30
   `--dry-run` prints the estimate and launches nothing.

4. Fetch and analyse:
    modal volume get netinf-v2 /results/llm_rps_v2/<campaign> ./results/llm_rps_v2/
    python experiments/analyze_v2.py results/llm_rps_v2/<same campaign> results/llm_rps_v2/<mixed campaign>
"""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import sys
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import modal

VOLUME_NAME = "netinf-v2"  # results and logs of every app go to the same volume
REPO, VOL, HF_CACHE = Path("/repo"), Path("/vol"), Path("/root/.cache/huggingface")
ROOT = Path(__file__).resolve().parents[1]
MISTRAL_FORMAT = "--tokenizer-mode mistral --config-format mistral --load-format mistral"

# Pinned weights: the revision is passed to vLLM and logged with every call.
MODELS = {
    "qwen7b": {"hf": "Qwen/Qwen2.5-7B-Instruct", "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
               "licence": "Apache-2.0", "gated": "no", "vllm": ""},
    "llama8b": {"hf": "meta-llama/Llama-3.1-8B-Instruct", "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
                "licence": "Llama 3.1 Community Licence", "gated": "yes, manual approval by Meta", "vllm": ""},
    "ministral8b": {"hf": "mistralai/Ministral-8B-Instruct-2410", "revision": "2f494a194c5b980dfb9772cb92d26cbb671fce5a",
                    "licence": "Mistral Research Licence (non-commercial research use)", "gated": "no",
                    "vllm": MISTRAL_FORMAT},
    "mistral7b": {"hf": "mistralai/Mistral-7B-Instruct-v0.3", "revision": "c170c708c41dac9275d15a8fff4eca08d52bab71",
                  "licence": "Apache-2.0", "gated": "no", "vllm": MISTRAL_FORMAT},
    # ungated stand-ins for Llama while Meta's approval is pending (third model family)
    "granite8b": {"hf": "ibm-granite/granite-3.3-8b-instruct", "revision": "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b",
                  "licence": "Apache-2.0", "gated": "no", "vllm": ""},
    "olmo7b": {"hf": "allenai/OLMo-2-1124-7B-Instruct", "revision": "470b1fba1ae01581f270116362ee4aa1b97f4c84",
               "licence": "Apache-2.0", "gated": "no", "vllm": ""},
    # >= 32B, one H100 each, FP8 weights (same quantisation for all three), 16k context for 60-round debriefs
    "qwen32b": {"hf": "Qwen/Qwen2.5-32B-Instruct", "revision": "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
                "licence": "Apache-2.0", "gated": "no", "vllm": "--quantization fp8", "gpu": "H100", "max_len": 16384},
    "glm32b": {"hf": "zai-org/GLM-4-32B-0414", "revision": "077b5c2f5c43bd3239fd605a0600229e8facbd4a",
               "licence": "MIT", "gated": "no", "vllm": "--quantization fp8", "gpu": "H100", "max_len": 16384},
    # Mixtral on an H200 (141 GB) with the same online FP8 quantisation as Qwen and GLM. Tried and failed on an
    # 80 GB H100: online FP8 (bf16 weights staged first, out of memory) and the pre-quantised RedHatAI FP8 checkpoint
    # (vLLM 0.9.1 has no CUTLASS FP8-MoE kernel image for it: "no kernel image is available").
    "mixtral8x7b": {"hf": "mistralai/Mixtral-8x7B-Instruct-v0.1", "revision": "eba92302a2861cdc0098cc54bc9f17cb2c47eb61",
                    "licence": "Apache-2.0", "gated": "no", "vllm": "--quantization fp8", "gpu": "H200", "max_len": 16384},
    "yi34b": {"hf": "01-ai/Yi-1.5-34B-Chat-16K", "revision": "ff74452e11f0f749ab872dc19b1dd3813c25c4d8",
              "licence": "Apache-2.0", "gated": "no", "vllm": "--quantization fp8", "gpu": "H100", "max_len": 16384},
}
GPU_PRICE = {"A10G": 1.10, "L4": 0.80, "A100-40GB": 2.10, "A100-80GB": 2.50, "L40S": 1.95, "H100": 3.95, "H200": 4.54,
             "B200": 6.25}  # $/h, modal.com 2026-10-25

# Read on your machine at `modal run` / `modal deploy` time, e.g. `NETINF_GPU=L4 modal deploy cloud/v2_modal.py`,
# and baked into the images below: the container must see the same values, or it rebuilds different functions
# (a missing NETINF_HF_SECRET in the container made bench_one crash-loop).
SETTINGS = {
    "NETINF_GPU": os.environ.get("NETINF_GPU", "A10G"),
    "NETINF_MAX_MODEL_LEN": os.environ.get("NETINF_MAX_MODEL_LEN", "12288"),
    "NETINF_CONCURRENCY": os.environ.get("NETINF_CONCURRENCY", "32"),  # requests one container takes at once
    "NETINF_MAX_CONTAINERS": os.environ.get("NETINF_MAX_CONTAINERS", "1"),  # GPUs per model at most (quota: 10)
    # HF token secret, only needed for gated models (Llama); empty = no secret
    "NETINF_HF_SECRET": os.environ.get("NETINF_HF_SECRET", ""),
    # a second deployment (e.g. netinf-v2-mixtral) can be added without redeploying the servers of a running campaign
    "NETINF_APP": os.environ.get("NETINF_APP", "netinf-v2"),
}
APP_NAME = SETTINGS["NETINF_APP"]
GPU = SETTINGS["NETINF_GPU"]
MAX_MODEL_LEN = int(SETTINGS["NETINF_MAX_MODEL_LEN"])
CONCURRENCY = int(SETTINGS["NETINF_CONCURRENCY"])
MAX_CONTAINERS = int(SETTINGS["NETINF_MAX_CONTAINERS"])
HF_SECRET = SETTINGS["NETINF_HF_SECRET"]

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_cache = modal.Volume.from_name("netinf-hf-cache", create_if_missing=True)  # weights download once, faster cold starts
hf_secrets = [modal.Secret.from_name(HF_SECRET)] if HF_SECRET else []


def _with_repo(image: modal.Image) -> modal.Image:
    image = image.env(SETTINGS)  # same configuration inside the container as on your machine
    for sub in ("src", "experiments", "prompts"):
        image = image.add_local_dir(str(ROOT / sub), remote_path=f"/repo/{sub}", copy=True,
                                    ignore=["__pycache__", "*.pyc"])
    return image


gpu_image = _with_repo(modal.Image.from_registry(
    "vllm/vllm-openai:v0.9.1", setup_dockerfile_commands=["RUN ln -sf $(which python3) /usr/local/bin/python"])
    .entrypoint([]).pip_install("matplotlib>=3.8"))
cpu_image = _with_repo(modal.Image.debian_slim(python_version="3.11").pip_install("numpy>=1.26", "matplotlib>=3.8"))


def vllm_command(key: str, max_model_len: int, host: str = "0.0.0.0") -> str:
    m = MODELS[key]
    return (f"python3 -m vllm.entrypoints.openai.api_server --model {m['hf']} --revision {m['revision']} "
            f"--served-model-name {m['hf']} --max-model-len {max_model_len} --enable-prefix-caching "
            # same sampling defaults for every model: only temperature (1.0, sent per request) differs from vLLM's
            f"--generation-config vllm --max-logprobs 5 --seed 0 --gpu-memory-utilization 0.90 "
            f"--host {host} --port 8000 {m['vllm']}")


def server(key: str) -> dict:
    """Function options of one model server: its own GPU (big models) or NETINF_GPU (7-8B models)."""
    return dict(image=gpu_image, gpu=MODELS[key].get("gpu", GPU), volumes={str(HF_CACHE): hf_cache},
                secrets=hf_secrets, timeout=24 * 3600, scaledown_window=300, min_containers=0,
                max_containers=MAX_CONTAINERS)


def max_len(key: str) -> int:
    return MODELS[key].get("max_len", MAX_MODEL_LEN)


@app.function(**server("qwen7b"))
@modal.concurrent(max_inputs=CONCURRENCY)
@modal.web_server(port=8000, startup_timeout=1800)
def serve_qwen7b():
    subprocess.Popen(vllm_command("qwen7b", max_len("qwen7b")), shell=True)


@app.function(**server("llama8b"))
@modal.concurrent(max_inputs=CONCURRENCY)
@modal.web_server(port=8000, startup_timeout=1800)
def serve_llama8b():
    subprocess.Popen(vllm_command("llama8b", max_len("llama8b")), shell=True)


@app.function(**server("ministral8b"))
@modal.concurrent(max_inputs=CONCURRENCY)
@modal.web_server(port=8000, startup_timeout=1800)
def serve_ministral8b():
    subprocess.Popen(vllm_command("ministral8b", max_len("ministral8b")), shell=True)


@app.function(**server("mistral7b"))
@modal.concurrent(max_inputs=CONCURRENCY)
@modal.web_server(port=8000, startup_timeout=1800)
def serve_mistral7b():
    subprocess.Popen(vllm_command("mistral7b", max_len("mistral7b")), shell=True)


@app.function(**server("granite8b"))
@modal.concurrent(max_inputs=CONCURRENCY)
@modal.web_server(port=8000, startup_timeout=1800)
def serve_granite8b():
    subprocess.Popen(vllm_command("granite8b", max_len("granite8b")), shell=True)


@app.function(**server("olmo7b"))
@modal.concurrent(max_inputs=CONCURRENCY)
@modal.web_server(port=8000, startup_timeout=1800)
def serve_olmo7b():
    subprocess.Popen(vllm_command("olmo7b", max_len("olmo7b")), shell=True)


@app.function(**server("qwen32b"))
@modal.concurrent(max_inputs=CONCURRENCY)
@modal.web_server(port=8000, startup_timeout=2400)
def serve_qwen32b():
    subprocess.Popen(vllm_command("qwen32b", max_len("qwen32b")), shell=True)


@app.function(**server("glm32b"))
@modal.concurrent(max_inputs=CONCURRENCY)
@modal.web_server(port=8000, startup_timeout=2400)
def serve_glm32b():
    subprocess.Popen(vllm_command("glm32b", max_len("glm32b")), shell=True)


@app.function(**server("mixtral8x7b"))
@modal.concurrent(max_inputs=CONCURRENCY)
@modal.web_server(port=8000, startup_timeout=2400)
def serve_mixtral8x7b():
    subprocess.Popen(vllm_command("mixtral8x7b", max_len("mixtral8x7b")), shell=True)


@app.function(**server("yi34b"))
@modal.concurrent(max_inputs=CONCURRENCY)
@modal.web_server(port=8000, startup_timeout=2400)
def serve_yi34b():
    subprocess.Popen(vllm_command("yi34b", max_len("yi34b")), shell=True)


def log(fh, message: str) -> None:
    line = f"[{datetime.now(timezone.utc):%H:%M:%S}] {message}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def run_logged(cmd: list[str], log_path: Path, env: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", buffering=1) as fh:
        log(fh, "$ " + shlex.join(cmd))
        proc = subprocess.Popen(cmd, cwd=str(REPO), env=env, text=True, bufsize=1,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout:
            log(fh, line.rstrip("\n"))
        if proc.wait() != 0:
            raise RuntimeError(f"command failed with exit code {proc.returncode}")


def wait_ready(base_url: str, timeout: float = 1800, proc: subprocess.Popen | None = None) -> float:
    """Seconds until GET {base_url}/models answers (the cold start, when the container was down)."""
    start = time.time()
    while time.time() - start < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"vLLM exited with code {proc.returncode}")
        try:
            with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=30) as r:
                if r.status == 200:
                    return time.time() - start
        except Exception:
            time.sleep(5)
    raise TimeoutError(f"{base_url} not ready after {timeout:.0f}s")


def env_for(log_dir: Path) -> dict:
    return {**os.environ, "NETINF_TRACE_DIR": str(log_dir / "llm_calls"), "PYTHONPATH": "/repo/src:/repo/experiments",
            "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg"}


# ---------- 1. benchmark + smoke test ----------

@app.function(image=gpu_image, gpu="A10G", volumes={str(HF_CACHE): hf_cache, "/vol": volume}, secrets=hf_secrets,
              timeout=2 * 3600)
def bench_one(key: str, gpu: str, rounds: int, prompts: str) -> dict:
    """Start vLLM in this container, play one `rounds`-round game per condition (5 in parallel), measure."""
    tag = f"bench_{key}_{gpu}_{datetime.now():%Y%m%d-%H%M%S}"
    log_dir = VOL / "logs" / tag
    log_dir.mkdir(parents=True, exist_ok=True)
    report = {"model": key, "gpu": gpu, "hf": MODELS[key]["hf"], "revision": MODELS[key]["revision"]}
    vllm_log = open(log_dir / "vllm.log", "ab", buffering=0)
    t0 = time.time()
    for attempt in (1, 2):  # a second try for transient failures (a dropped connection while downloading weights)
        proc = subprocess.Popen(vllm_command(key, 16384, host="127.0.0.1"), shell=True, stdout=vllm_log,
                                stderr=subprocess.STDOUT)
        try:
            report["cold_start_s"] = round(wait_ready("http://127.0.0.1:8000/v1", proc=proc))
            report["loaded"], report["attempts"] = True, attempt
            break
        except Exception as e:
            proc.kill()
            hf_cache.commit()  # keep what was downloaded
            if attempt == 2:
                return {**report, "loaded": False, "error": repr(e)[:300], "cold_start_s": round(time.time() - t0)}
            time.sleep(30)
    hf_cache.commit()
    out = Path("/tmp/bench")
    m = MODELS[key]
    t1 = time.time()
    try:
        run_logged([sys.executable, "experiments/llm_rps_v2.py", "--design", "same_model", "--model", m["hf"],
                    "--base-url", "http://127.0.0.1:8000/v1", "--revision", m["revision"], "--runs", "1",
                    "--rounds", str(rounds), "--workers", "5", "--perms", "70", "--prompts", prompts,
                    "--out", str(out), "--campaign", tag, "--cache", "/tmp/bench.sqlite"],
                   log_dir / "games.log", env_for(log_dir))
    finally:
        proc.terminate()
    wall = time.time() - t1
    games = [json.loads(f.read_text()) for f in sorted((out / tag / "games").glob("*.json"))]
    fails, maxp, tokens, dists, moves = {}, {}, {"prompt": 0, "completion": 0}, 0, 0
    for g in games:
        for k, v in g["failures"].items():
            fails[k] = fails.get(k, 0) + v
        for k, v in g["max_prompt_tokens"].items():
            maxp[k] = max(maxp.get(k, 0), v)
        for k in tokens:
            tokens[k] += g["token_usage"].get(k, 0)
        for r in g["rounds"]:
            moves += len(r["move_dist"])
            dists += sum(d is not None for d in r["move_dist"].values())
    # project the debrief prompt (the longest one) to a 30-round game: it grows by one outcome line and one
    # round of chat per round; chars -> tokens with this game's own ratio
    proj = 0
    for g in games:
        for d in g["debriefs"].values():
            per_round = sum(len(r["outcome"]) + 1 + sum(len(f"[round {r['round']}] {c['from']}: {c['text']}") + 1
                                                          for c in r["chat"]) for r in g["rounds"]) / len(g["rounds"])
            chars30 = d["prompt_chars"] + (30 - len(g["rounds"])) * per_round
            proj = max(proj, d["prompt_tokens"] / d["prompt_chars"] * chars30)
    longest = max([proj] + list(maxp.values()))
    # understanding of the rules, from the reflections: false "X beats Y" statements and wrong stated points
    sys.path.insert(0, "/repo/src")  # this function runs in the container, outside the games' PYTHONPATH
    from netinf.understanding import payoff_misconceptions, rule_statements
    claims = wrong = acc_n = acc_err = texts = misconceptions = 0
    for g in games:
        for r in g["rounds"]:
            said = [x["raw"] for x in (r.get("reflections") or {}).values()] + [m["text"] for m in r["chat"]]
            for text in said:
                c, w = rule_statements(text)
                claims, wrong = claims + c, wrong + w
                texts += 1
                misconceptions += bool(payoff_misconceptions(text))
        for stats in g.get("reflection_metrics", {}).values():
            n = stats["accounting_error"]["n"]
            acc_n += n
            acc_err += stats["accounting_error"]["rate"] * n if n else 0
    report.update(games=len(games), wall_s=round(wall), output_tokens=tokens["completion"],
                  tokens_per_s=round(tokens["completion"] / wall, 1) if wall else None,
                  calls_failures=fails, max_prompt_tokens=maxp, projected_debrief_tokens_30_rounds=round(proj),
                  # longest prompt + 20%, plus the largest answer (debrief 400 tokens), rounded up to 256
                  max_model_len=int(math.ceil((1.2 * longest + 400) / 256) * 256),
                  logprob_coverage=round(dists / moves, 3) if moves else None,
                  rule_statements=claims, false_rule_rate=round(wrong / claims, 3) if claims else None,
                  payoff_misconception_rate=round(misconceptions / texts, 3) if texts else None,
                  accounting_error_rate=round(acc_err / acc_n, 3) if acc_n else None, prompts=prompts)
    price = GPU_PRICE.get(gpu, float("nan"))
    # cost of one 30-round game at the benchmark's concurrency: the 5 games ran in parallel for `wall` seconds
    report["usd_per_30_round_game"] = round(wall * (30 / rounds) / len(games) * price / 3600, 4) if games else None
    (log_dir / "report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.local_entrypoint()
def bench(models: str = "qwen7b,ministral8b,granite8b,olmo7b", gpus: str = "A10G,L4", rounds: int = 5,
          prompts: str = "prompts/pot_plan_v2_chat.json", max_gpus: int = 8):
    """Runs at most `max_gpus` model x GPU tests at once (one GPU each), in waves."""
    jobs = [(k, g) for k in models.split(",") for g in gpus.split(",")]
    for k, g in jobs:
        if k not in MODELS or g not in GPU_PRICE:
            raise SystemExit(f"unknown model {k} or GPU {g}")
    reports = []
    for i in range(0, len(jobs), max_gpus):
        wave = jobs[i:i + max_gpus]
        print(f"wave {i // max_gpus + 1}: {len(wave)} GPUs: {wave}", flush=True)
        calls = [(k, g, bench_one.with_options(gpu=g).spawn(k, g, rounds, prompts)) for k, g in wave]
        for k, g, c in calls:
            try:
                reports.append(c.get())
            except Exception as e:
                reports.append({"model": k, "gpu": g, "loaded": False, "error": repr(e)[:300]})
    head = ("model", "gpu", "loaded", "cold start s", "tok/s", "$ / 30-round game", "move fail", "untagged", "plan fail",
            "truncated", "logprobs", "false 'X beats Y'", "wrong payoff claims", "wrong points", "longest prompt (30 r.)",
            "max_model_len")
    print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
    for r in reports:
        f = r.get("calls_failures", {})
        trunc = sum(v for k, v in f.items() if k.startswith("truncated"))
        print(f"| {r['model']} | {r['gpu']} | {r.get('loaded')} | {r.get('cold_start_s', '-')} | {r.get('tokens_per_s', '-')} | "
              f"{r.get('usd_per_30_round_game', '-')} | {f.get('move', 0)} | {f.get('move_untagged', 0)} | {f.get('plan', 0)} | "
              f"{trunc} | {r.get('logprob_coverage', '-')} | {r.get('false_rule_rate', '-')} ({r.get('rule_statements', 0)}) | "
              f"{r.get('payoff_misconception_rate', '-')} | "
              f"{r.get('accounting_error_rate', '-')} | {r.get('projected_debrief_tokens_30_rounds', '-')} | "
              f"{r.get('max_model_len', '-')} |" + (f" error: {r['error']}" if r.get("error") else ""))
    Path("results/llm_rps_v2").mkdir(parents=True, exist_ok=True)
    out = Path(f"results/llm_rps_v2/bench_{datetime.now():%Y%m%d-%H%M}.json")
    out.write_text(json.dumps(reports, indent=2))
    print(f"\nreports saved to {out}")


# ---------- 3. campaign ----------

@app.function(image=cpu_image, timeout=24 * 3600, volumes={"/vol": volume})
def run_campaign(campaign: str, keys: list[str], design: str, game_args: list[str], runs: int, chunk: int) -> str:
    log_dir = VOL / "logs" / campaign
    log_dir.mkdir(parents=True, exist_ok=True)
    orch = open(log_dir / "orchestrator.log", "a", buffering=1)
    servers = {k: modal.Function.from_name(APP_NAME, f"serve_{k}") for k in dict.fromkeys(keys)}
    manifest = {"campaign": campaign, "design": design, "models": {k: MODELS[k] for k in servers},
                "started": datetime.now(timezone.utc).isoformat(), "game_args": game_args}
    try:
        for k, fn in servers.items():
            fn.autoscaler.update(min_containers=1)  # one warm container per model for the whole campaign
        urls = {}
        for k, fn in servers.items():
            url = fn.get_web_url()
            if not url:
                raise RuntimeError(f"serve_{k} has no URL: deploy first (modal deploy cloud/v2_modal.py)")
            urls[k] = url.rstrip("/") + "/v1"
        manifest["cold_start_s"] = {k: round(wait_ready(u)) for k, u in urls.items()}
        log(orch, f"servers ready: {manifest['cold_start_s']}")
        spec = {k: f"{MODELS[k]['hf']}@{urls[k]}#{MODELS[k]['revision']}" for k in keys}
        routing = (["--lineup", ",".join(spec[k] for k in keys)] if design == "mixed_models"
                   else ["--agents", ",".join([spec[keys[0]]] * 3)])  # per-seat routing; same model in every seat
        out = VOL / "results" / "llm_rps_v2"
        for i, start in enumerate(range(0, runs, chunk)):
            n = min(chunk, runs - start)
            run_logged([sys.executable, "experiments/llm_rps_v2.py", "--design", design, *routing, *game_args,
                        "--runs", str(n), "--run-offset", str(start), "--out", str(out), "--campaign", campaign,
                        "--cache", str(VOL / "cache" / f"llm_v2_{campaign}.sqlite")],
                       log_dir / f"games_chunk_{i:02d}.log", env_for(log_dir))
            volume.commit()  # finished games survive a later failure; relaunching with --campaign resumes
            log(orch, f"chunk {i} committed")
        run_logged([sys.executable, "experiments/analyze_v2.py", str(out / campaign)], log_dir / "analysis.log",
                   env_for(log_dir))
        manifest["status"] = "done"
        return str(out / campaign)
    except Exception:
        log(orch, traceback.format_exc())
        manifest["status"] = "failed"
        raise
    finally:
        for fn in servers.values():
            fn.autoscaler.update(min_containers=0)  # scale to zero after the 5-minute idle window
        manifest["ended"] = datetime.now(timezone.utc).isoformat()
        (log_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        orch.close()
        volume.commit()


@app.local_entrypoint()
def main(design: str = "same_model", model: str = "qwen7b", lineup: str = "qwen7b,llama8b,ministral8b",
         conditions: str = "", runs: int = 6, rounds: int = 30, workers: int = 30, chunk: int = 6,
         prompts: str = "prompts/pot_plan_v2_chat.json", extra: str = "", tag: str = "", campaign: str = "",
         gpu: str = "A10G", tokens_per_sec: float = 300.0, dry_run: bool = False):
    keys = lineup.split(",") if design == "mixed_models" else [model]
    if unknown := [k for k in keys if k not in MODELS]:
        raise SystemExit(f"unknown model keys {unknown}; known: {list(MODELS)}")
    if design == "mixed_models" and len(keys) != 3:
        raise SystemExit("mixed_models needs three models in --lineup")
    game_args = ["--rounds", str(rounds), "--workers", str(workers), "--prompts", prompts, *shlex.split(extra)]
    if conditions:
        game_args += ["--conditions", conditions]
    fake = ",".join(f"{MODELS[k]['hf']}@http://x/v1" for k in keys)
    est = subprocess.run([sys.executable, str(ROOT / "experiments" / "llm_rps_v2.py"), "--estimate", "--design", design,
                          "--lineup" if design == "mixed_models" else "--agents", fake if design == "mixed_models" else ",".join([fake] * 3),
                          "--runs", str(runs), "--tokens-per-sec", str(tokens_per_sec),
                          "--gpu-price", str(sum(GPU_PRICE[MODELS[k].get("gpu", gpu)] for k in keys) / len(keys)),
                          *game_args], capture_output=True, text=True, cwd=str(ROOT),
                         env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'experiments'}"})
    if est.returncode != 0:
        raise SystemExit(est.stderr)
    print(est.stdout)
    if dry_run:
        return
    label = "mixed" if design == "mixed_models" else model
    campaign = campaign or f"{datetime.now():%Y%m%d-%H%M}_{design}_{label}" + (f"_{tag}" if tag else "")
    print(f"launching campaign {campaign} on models {keys}", flush=True)
    print(run_campaign.remote(campaign, keys, design, game_args, runs, chunk))
