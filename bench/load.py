"""Model x prompt x device x concurrency sweep, measured against Ollama directly.

This benchmark talks to Ollama's native API (:11434) rather than the gateway,
because its purpose is a *hardware* comparison -- CPU vs GPU (Metal) -- and that
is controlled by Ollama's ``num_gpu`` option, which the gateway abstracts away:

    CPU : options={"num_gpu": 0}     -> force all layers onto CPU
    GPU : options={}                 -> Metal default (all layers on GPU)

Token counts come from Ollama's authoritative ``eval_count`` (output tokens) and
``prompt_eval_count`` (input tokens), not a word-count estimate. Every result row
carries ``cost_per_million_tokens`` via the shared cost_calculator.

Defaults are deliberately modest: CPU inference of a 7B model is slow, and the
charts (generate_charts.py) only consume the concurrency=1 rows, so a deep
concurrency sweep here would be wasted wall-clock. Override with flags as needed.

Usage:
    python bench/load.py
    python bench/load.py --models qwen2.5:7b --devices cpu gpu --requests 24
    python bench/load.py --concurrency 1 4 8 --max-tokens 256
"""

import argparse
import asyncio
import csv
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from cost_calculator import cost_per_million_tokens, get_gpu_hourly_rate

OLLAMA_URL = "http://127.0.0.1:11434"
MODELS = ["llama3.2:1b", "qwen2.5:7b"]
PROMPTS = [
    "Say hello in one word",
    "Summarize the plot of The Matrix in a single sentence",
    "Translate 'Hello' to Spanish",
    "Write a short Python function that computes factorial",
]
DEVICES = ["cpu", "gpu"]
CONCURRENCY_LEVELS = [1, 4]
REQUESTS_PER_LEVEL = 16
MAX_NEW_TOKENS = 128          # cap output so CPU 7B runs stay bounded
RESULTS_DIR = Path("results")
CSV_PATH = RESULTS_DIR / "results.csv"
PROMPT_DIR = RESULTS_DIR / "by_prompt"
DEVICE_DIR = RESULTS_DIR / "by_device"


def slugify(value: str) -> str:
    return "_".join(
        part for part in "".join(ch if ch.isalnum() else "_" for ch in value.lower()).split("_") if part
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def format_number(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.4f}"


def build_options(device: str, max_tokens: int) -> Dict[str, Any]:
    options: Dict[str, Any] = {"num_predict": max_tokens}
    if device == "cpu":
        options["num_gpu"] = 0      # 0 GPU layers => pure CPU inference
    return options


async def send_request(
    client: httpx.AsyncClient, model: str, prompt: str, device: str, max_tokens: int
) -> Dict[str, Any]:
    start = time.time()
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": build_options(device, max_tokens),
    }
    try:
        response = await client.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=300.0)
        response.raise_for_status()
        elapsed = time.time() - start
        data = response.json()
        return {
            "latency": elapsed,
            "tokens": int(data.get("eval_count", 0)),          # output tokens
            "input_tokens": int(data.get("prompt_eval_count", 0)),
            "success": True,
            "error": None,
        }
    except Exception as exc:
        return {"latency": None, "tokens": 0, "input_tokens": 0, "success": False, "error": str(exc)}


async def benchmark_model_at_concurrency(
    model: str, prompt: str, device: str, concurrency: int, num_requests: int, max_tokens: int,
) -> List[Dict[str, Any]]:
    async with httpx.AsyncClient() as client:
        results: List[Dict[str, Any]] = []
        for i in range(0, num_requests, concurrency):
            chunk = [
                send_request(client, model, prompt, device, max_tokens)
                for _ in range(min(concurrency, num_requests - i))
            ]
            results.extend(await asyncio.gather(*chunk))
        return results


async def prewarm_models(models: Iterable[str], devices: Iterable[str], max_tokens: int) -> None:
    print("\n" + "=" * 72 + "\nPrewarming models...")
    async with httpx.AsyncClient() as client:
        for model in models:
            for device in devices:
                print(f"  Prewarming {model} on device {device}")
                try:
                    r = await client.post(
                        f"{OLLAMA_URL}/api/generate",
                        json={"model": model, "prompt": "count to 5", "stream": False,
                              "options": build_options(device, 16)},
                        timeout=300.0,
                    )
                    r.raise_for_status()
                except Exception as exc:
                    print(f"    Warning: prewarm failed for {model}/{device}: {exc}")
    print("Prewarm complete.\n" + "=" * 72)


def percentile(values: List[float], fraction: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = max(0, min(len(sorted_values) - 1, math.ceil(fraction * len(sorted_values)) - 1))
    return sorted_values[idx]


async def run_benchmark(
    prompts: Iterable[str], models: Iterable[str], devices: Iterable[str],
    concurrency_levels: Iterable[int], requests_per_level: int, max_tokens: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    await prewarm_models(models, devices, max_tokens)

    for prompt in prompts:
        for model in models:
            for device in devices:
                for concurrency in concurrency_levels:
                    print("\n" + "=" * 72)
                    print(f"Prompt: {prompt}")
                    print(f"Model: {model} | Device: {device} | Concurrency: {concurrency} | Requests: {requests_per_level}")
                    print("=" * 72)

                    request_results = await benchmark_model_at_concurrency(
                        model, prompt, device, concurrency, requests_per_level, max_tokens,
                    )

                    successful = [r for r in request_results if r["success"]]
                    failed = [r for r in request_results if not r["success"]]
                    latencies = [r["latency"] for r in successful if r["latency"] is not None]
                    tokens = sum(r["tokens"] for r in successful)
                    duration = sum(latencies) if latencies else 0.0

                    tps = tokens / duration if duration else None
                    rate = get_gpu_hourly_rate()
                    row = {
                        "prompt": prompt,
                        "model": model,
                        "device": device,
                        "concurrency": concurrency,
                        "requested": requests_per_level,
                        "success_count": len(successful),
                        "failure_count": len(failed),
                        "avg_latency": sum(latencies) / len(latencies) if latencies else None,
                        "min_latency": min(latencies) if latencies else None,
                        "p50_latency": percentile(latencies, 0.50) if latencies else None,
                        "p95_latency": percentile(latencies, 0.95) if latencies else None,
                        "p99_latency": percentile(latencies, 0.99) if latencies else None,
                        "max_latency": max(latencies) if latencies else None,
                        "total_tokens": tokens,
                        "avg_tokens": tokens / len(successful) if successful else None,
                        "tokens_per_sec": tps,
                        "requests_per_sec": len(successful) / duration if duration else None,
                        "duration": duration,
                        "cost_per_million_tokens": cost_per_million_tokens(rate, tps) if tps else None,
                    }

                    if failed:
                        print(f"  {len(failed)} failed requests (e.g. {failed[0]['error']})")
                    print(
                        f"  success: {row['success_count']} | avg: {format_number(row['avg_latency'])}s | "
                        f"p95: {format_number(row['p95_latency'])}s | tok/s: {format_number(row['tokens_per_sec'])} | "
                        f"${format_number(row['cost_per_million_tokens'])}/M"
                    )
                    rows.append(row)
    return rows


FIELDNAMES = [
    "prompt", "model", "device", "concurrency", "requested",
    "success_count", "failure_count", "avg_latency", "min_latency",
    "p50_latency", "p95_latency", "p99_latency", "max_latency",
    "total_tokens", "avg_tokens", "tokens_per_sec", "requests_per_sec",
    "duration", "cost_per_million_tokens",
]


def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})
    print(f"\nResults saved to {path}")


def save_grouped_csv(rows: List[Dict[str, Any]], group_key: str, output_dir: Path) -> None:
    ensure_dir(output_dir)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(group_key, "unknown")), []).append(row)
    for group_value, group_rows in grouped.items():
        save_csv(group_rows, output_dir / f"{slugify(group_key)}_{slugify(group_value)}.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM serving benchmark sweep (Ollama native)")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--prompts", nargs="+", default=PROMPTS)
    parser.add_argument("--devices", nargs="+", default=DEVICES, choices=["cpu", "gpu"])
    parser.add_argument("--concurrency", nargs="+", type=int, default=CONCURRENCY_LEVELS)
    parser.add_argument("--requests", type=int, default=REQUESTS_PER_LEVEL)
    parser.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Device/model sweep (Ollama native) | GPU_HOURLY_RATE=${get_gpu_hourly_rate():.2f}/hr | "
          f"max_tokens={args.max_tokens}")
    rows = asyncio.run(
        run_benchmark(
            prompts=args.prompts, models=args.models, devices=args.devices,
            concurrency_levels=args.concurrency, requests_per_level=args.requests,
            max_tokens=args.max_tokens,
        )
    )
    save_csv(rows, CSV_PATH)
    save_grouped_csv(rows, "prompt", PROMPT_DIR)
    save_grouped_csv(rows, "device", DEVICE_DIR)


if __name__ == "__main__":
    main()
