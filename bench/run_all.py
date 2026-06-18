"""Run the full benchmark suite in one command, reproducibly.

Preflights the inference servers, records a run manifest (so a results set is
traceable to the exact host / models / GPU rate that produced it), then runs
every benchmark in order and regenerates all charts. Steps whose server
dependency is down are skipped with a warning rather than failing the whole run.

Servers (both are the mini-llm-gateway app with different BACKEND):
    BACKEND=ollama uvicorn main:app --port 8000     # :8000  -> /ollama/*
    BACKEND=mlx    uvicorn main:app --port 8001     # :8001  -> /mlx/*

Usage:
    python bench/run_all.py                # full suite
    python bench/run_all.py --quick        # small params, for a smoke test
    GPU_HOURLY_RATE=1.20 python bench/run_all.py
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from cost_calculator import get_gpu_hourly_rate

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
MANIFEST_PATH = RESULTS_DIR / "MANIFEST.json"

OLLAMA_GATEWAY = "http://127.0.0.1:8000"
MLX_GATEWAY = "http://127.0.0.1:8001"
OLLAMA_API = "http://127.0.0.1:11434"

# Reasoning model for the reasoning-tax step; the step is skipped (not failed)
# if it isn't pulled. A reasoning fine-tune of the qwen2.5-7B baseline base.
REASONING_MODEL = "deepseek-r1:7b"

PYTHON = sys.executable


# --- preflight ------------------------------------------------------------

def service_up(url: str, path: str = "/healthz", timeout: float = 2.0) -> bool:
    try:
        r = httpx.get(url + path, timeout=timeout)
        return r.status_code < 500
    except Exception:
        return False


def preflight() -> Dict[str, bool]:
    print("Preflight:")
    model_names = [m["name"] for m in ollama_models()]
    services = {
        "ollama_gateway": service_up(OLLAMA_GATEWAY),
        "mlx_gateway": service_up(MLX_GATEWAY),
        "ollama_daemon": service_up(OLLAMA_API, path="/api/tags"),
        "reasoning_model": REASONING_MODEL in model_names,
    }
    for name, up in services.items():
        print(f"  {'OK  ' if up else 'DOWN'}  {name}")
    if not services["reasoning_model"]:
        print(f"\n  (optional) Pull the reasoning model for the reasoning-tax step:")
        print(f"    ollama pull {REASONING_MODEL}")
    if not services["ollama_gateway"]:
        print("\n  Start the Ollama gateway:")
        print("    cd ../mini-llm-gateway && BACKEND=ollama uvicorn main:app --port 8000")
    if not services["mlx_gateway"]:
        print("\n  (optional) Start the MLX gateway for backend-comparison steps:")
        print("    cd ../mini-llm-gateway && BACKEND=mlx uvicorn main:app --port 8001")
    print()
    return services


# --- manifest -------------------------------------------------------------

def ollama_models() -> List[Dict[str, str]]:
    try:
        r = httpx.get(OLLAMA_API + "/api/tags", timeout=5.0)
        return [
            {"name": m["name"], "digest": m.get("digest", "")[:12],
             "size": m.get("size")}
            for m in r.json().get("models", [])
        ]
    except Exception:
        return []


def ollama_version() -> Optional[str]:
    try:
        return httpx.get(OLLAMA_API + "/api/version", timeout=5.0).json().get("version")
    except Exception:
        return None


def write_manifest(services: Dict[str, bool], quick: bool) -> None:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "ollama_version": ollama_version(),
        "gpu_hourly_rate_usd": get_gpu_hourly_rate(),
        "quick_mode": quick,
        "services_up": services,
        "ollama_models": ollama_models(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote manifest -> {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    print(f"  host={manifest['host']} | {manifest['processor']} | "
          f"rate=${manifest['gpu_hourly_rate_usd']:.2f}/hr | "
          f"ollama={manifest['ollama_version']}\n")


# --- steps ----------------------------------------------------------------

def build_steps(quick: bool) -> List[Dict]:
    """Each step: name, requires (services that must be up), cmd."""
    def b(script: str, *extra: str) -> List[str]:
        return [PYTHON, str(REPO_ROOT / "bench" / script), *extra]

    if quick:
        return [
            {"name": "device/model sweep", "requires": ["ollama_gateway"],
             "cmd": b("load.py", "--models", "llama3.2:1b", "--prompts",
                      "Say hello in one word", "--devices", "gpu",
                      "--concurrency", "1", "2", "--requests", "4")},
            {"name": "ollama vs mlx", "requires": ["ollama_gateway", "mlx_gateway"],
             "cmd": b("compare.py", "--concurrency", "1", "4", "--requests", "4")},
            {"name": "mlx batch sweep", "requires": ["mlx_gateway"],
             "cmd": b("bench_mlx.py", "--concurrency", "1", "--requests", "4")},
            {"name": "difficulty invariance", "requires": ["ollama_gateway"],
             "cmd": b("difficulty_invariance.py", "--repeats", "1",
                      "--max-tokens", "64", "--no-charts")},
            {"name": "reasoning tax", "requires": ["ollama_daemon", "reasoning_model"],
             "cmd": b("reasoning_tax.py", "--repeats", "1",
                      "--max-tokens", "512", "--no-charts")},
            {"name": "context scaling", "requires": ["ollama_gateway"],
             "cmd": b("context_scaling.py", "--lengths", "256", "1024",
                      "--repeats", "1", "--max-tokens", "32", "--no-charts")},
            {"name": "prefill/decode cost", "requires": [],
             "cmd": b("prefill_decode_cost.py")},
            {"name": "regenerate charts", "requires": [],
             "cmd": b("../charts/generate_charts.py")},
        ]

    return [
        {"name": "device/model sweep", "requires": ["ollama_gateway"],
         "cmd": b("load.py")},
        {"name": "ollama vs mlx", "requires": ["ollama_gateway", "mlx_gateway"],
         "cmd": b("compare.py")},
        {"name": "mlx batch sweep", "requires": ["mlx_gateway"],
         "cmd": b("bench_mlx.py")},
        {"name": "difficulty invariance", "requires": ["ollama_gateway"],
         "cmd": b("difficulty_invariance.py")},
        {"name": "reasoning tax", "requires": ["ollama_daemon", "reasoning_model"],
         "cmd": b("reasoning_tax.py")},
        {"name": "context scaling", "requires": ["ollama_gateway"],
         "cmd": b("context_scaling.py")},
        {"name": "prefill/decode cost", "requires": [],
         "cmd": b("prefill_decode_cost.py")},
        {"name": "regenerate charts", "requires": [],
         "cmd": b("../charts/generate_charts.py")},
    ]


def run_step(step: Dict, available: Dict[str, bool]) -> str:
    """Return one of: ok, failed, skipped."""
    missing = [s for s in step["requires"] if not available.get(s)]
    if missing:
        print(f"\n=== SKIP  {step['name']}  (needs {', '.join(missing)}) ===")
        return "skipped"
    print(f"\n=== RUN   {step['name']} ===", flush=True)
    t0 = time.time()
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    result = subprocess.run(step["cmd"], cwd=REPO_ROOT, env=env)
    dt = time.time() - t0
    if result.returncode == 0:
        print(f"--- ok ({dt:.0f}s)")
        return "ok"
    print(f"--- FAILED (exit {result.returncode}, {dt:.0f}s)")
    return "failed"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full benchmark suite")
    parser.add_argument("--quick", action="store_true",
                        help="Small params for a fast smoke test")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preflight, write manifest, and print the plan without running steps")
    args = parser.parse_args()

    print(f"\n{'='*60}\n LLM Serving Benchmark Suite\n{'='*60}\n")
    services = preflight()
    write_manifest(services, args.quick)

    steps = build_steps(args.quick)

    if args.dry_run:
        print("Plan (--dry-run, nothing executed):")
        for step in steps:
            missing = [s for s in step["requires"] if not services.get(s)]
            state = f"SKIP (needs {', '.join(missing)})" if missing else "run"
            print(f"  [{state}]  {step['name']}")
            print(f"           {' '.join(step['cmd'])}")
        return 0

    outcomes: Dict[str, str] = {}
    suite_t0 = time.time()
    for step in steps:
        outcomes[step["name"]] = run_step(step, services)

    print(f"\n{'='*60}\n Summary  ({time.time()-suite_t0:.0f}s total)\n{'='*60}")
    for name, outcome in outcomes.items():
        mark = {"ok": "OK  ", "failed": "FAIL", "skipped": "skip"}[outcome]
        print(f"  {mark}  {name}")

    failed = [n for n, o in outcomes.items() if o == "failed"]
    if failed:
        print(f"\n{len(failed)} step(s) failed.")
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
