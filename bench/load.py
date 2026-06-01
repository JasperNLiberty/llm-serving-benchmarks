import argparse
import asyncio
import csv
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import httpx

BASE_URL = "http://127.0.0.1:8000"
MODELS = ["llama3.2:1b", "qwen2.5:7b"]
PROMPTS = [
    "Say hello in one word",
    "Summarize the plot of The Matrix in a single sentence",
    "Translate 'Hello' to Spanish",
    "Write a short Python function that computes factorial",
]
DEVICES = ["cpu", "gpu"]
CONCURRENCY_LEVELS = [1, 2, 4, 8]
REQUESTS_PER_LEVEL = 10
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


def estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def extract_token_count(response: httpx.Response) -> int:
    try:
        data = response.json()
    except ValueError:
        return estimate_tokens(response.text)

    if isinstance(data, dict):
        usage = data.get("usage")
        if isinstance(usage, dict):
            for key in ("total_tokens", "tokens", "prompt_tokens", "completion_tokens"):
                if isinstance(usage.get(key), (int, float)):
                    return int(usage[key])

        token_usage = data.get("token_usage")
        if isinstance(token_usage, dict):
            if isinstance(token_usage.get("total"), (int, float)):
                return int(token_usage["total"])

        if "choices" in data and isinstance(data["choices"], list):
            texts = []
            for choice in data["choices"]:
                if isinstance(choice, dict):
                    message = choice.get("message")
                    if isinstance(message, dict) and isinstance(message.get("content"), str):
                        texts.append(message["content"])
                    elif isinstance(choice.get("text"), str):
                        texts.append(choice["text"])
            if texts:
                return estimate_tokens(" ".join(texts))

        if isinstance(data.get("text"), str):
            return estimate_tokens(data["text"])

    return estimate_tokens(response.text)


def build_payload(prompt: str, model: str, device: Optional[str]) -> Dict[str, Any]:
    payload = {"message": prompt, "model": model}
    if device:
        payload["device"] = device
        payload["options"] = {"device": device}
    return payload


async def send_request(
    client: httpx.AsyncClient, model: str, prompt: str, device: Optional[str]
) -> Dict[str, Any]:
    start = time.time()
    payload = build_payload(prompt, model, device)

    try:
        response = await client.post(
            f"{BASE_URL}/ollama/chat",
            json=payload,
            timeout=120.0,
        )
        response.raise_for_status()

        elapsed = time.time() - start
        tokens = extract_token_count(response)
        return {
            "latency": elapsed,
            "tokens": tokens,
            "success": True,
            "error": None,
        }
    except Exception as exc:
        return {
            "latency": None,
            "tokens": 0,
            "success": False,
            "error": str(exc),
        }


async def benchmark_model_at_concurrency(
    model: str,
    prompt: str,
    device: Optional[str],
    concurrency: int,
    num_requests: int,
) -> List[Dict[str, Any]]:
    async with httpx.AsyncClient() as client:
        results: List[Dict[str, Any]] = []
        for i in range(0, num_requests, concurrency):
            chunk = [
                send_request(client, model, prompt, device)
                for _ in range(min(concurrency, num_requests - i))
            ]
            completed = await asyncio.gather(*chunk)
            results.extend(completed)
        return results


def percentile(values: List[float], fraction: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = max(0, min(len(sorted_values) - 1, math.ceil(fraction * len(sorted_values)) - 1))
    return sorted_values[idx]


async def run_benchmark(
    prompts: Iterable[str],
    models: Iterable[str],
    devices: Iterable[str],
    concurrency_levels: Iterable[int],
    requests_per_level: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for prompt in prompts:
        for model in models:
            for device in devices:
                for concurrency in concurrency_levels:
                    print("\n" + "=" * 72)
                    print(f"Prompt: {prompt}")
                    print(f"Model: {model} | Device: {device} | Concurrency: {concurrency} | Requests: {requests_per_level}")
                    print("" + "=" * 72)

                    request_results = await benchmark_model_at_concurrency(
                        model,
                        prompt,
                        device,
                        concurrency,
                        requests_per_level,
                    )

                    successful = [r for r in request_results if r["success"]]
                    failed = [r for r in request_results if not r["success"]]
                    latencies = [r["latency"] for r in successful if r["latency"] is not None]
                    tokens = sum(r["tokens"] for r in successful)
                    duration = sum(latencies) if latencies else 0.0

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
                        "tokens_per_sec": tokens / duration if duration else None,
                        "requests_per_sec": len(successful) / duration if duration else None,
                        "duration": duration,
                    }

                    if failed:
                        print(f"  {len(failed)} failed requests")
                    print(
                        f"  success: {row['success_count']} | avg: {format_number(row['avg_latency'])}s | "
                        f"p95: {format_number(row['p95_latency'])}s | tokens/sec: {format_number(row['tokens_per_sec'])}"
                    )
                    rows.append(row)

    return rows


def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "prompt",
        "model",
        "device",
        "concurrency",
        "requested",
        "success_count",
        "failure_count",
        "avg_latency",
        "min_latency",
        "p50_latency",
        "p95_latency",
        "p99_latency",
        "max_latency",
        "total_tokens",
        "avg_tokens",
        "tokens_per_sec",
        "requests_per_sec",
        "duration",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"\n✓ Results saved to {path}")


def save_grouped_csv(rows: List[Dict[str, Any]], group_key: str, output_dir: Path) -> None:
    ensure_dir(output_dir)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        group_value = row.get(group_key, "unknown")
        grouped.setdefault(str(group_value), []).append(row)

    for group_value, group_rows in grouped.items():
        filename = output_dir / f"{slugify(group_key)}_{slugify(group_value)}.csv"
        save_csv(group_rows, filename)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM serving benchmark sweep")
    parser.add_argument("--models", nargs="+", default=MODELS, help="Models to benchmark")
    parser.add_argument("--prompts", nargs="+", default=PROMPTS, help="Prompts to benchmark")
    parser.add_argument("--devices", nargs="+", default=DEVICES, help="Inference devices to test")
    parser.add_argument("--concurrency", nargs="+", type=int, default=CONCURRENCY_LEVELS, help="Concurrency levels to test")
    parser.add_argument("--requests", type=int, default=REQUESTS_PER_LEVEL, help="Requests per concurrency level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = asyncio.run(
        run_benchmark(
            prompts=args.prompts,
            models=args.models,
            devices=args.devices,
            concurrency_levels=args.concurrency,
            requests_per_level=args.requests,
        )
    )

    save_csv(rows, CSV_PATH)
    save_grouped_csv(rows, "prompt", PROMPT_DIR)
    save_grouped_csv(rows, "device", DEVICE_DIR)


if __name__ == "__main__":
    main()
