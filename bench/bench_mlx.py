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

BASE_URL = "http://127.0.0.1:8001"
MODELS = ["qwen2.5:7b"]
PROMPTS = [
    "Say hello in one word",
    # "Summarize the plot of The Matrix in a single sentence",
    # "Translate 'Hello' to Spanish",
    # "Write a short Python function that computes factorial",
]
CONCURRENCY_LEVELS = [8, 16, 32]
REQUESTS_PER_LEVEL = 32
RESULTS_DIR = Path("results/mlx")
CSV_PATH = RESULTS_DIR / "results_mlx.csv"
PROMPT_DIR = RESULTS_DIR / "by_prompt"
BATCH_DIR = RESULTS_DIR / "by_batch_size"


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


def build_payload(prompt: str, model: str) -> Dict[str, Any]:
    return {"prompt": prompt, "model": model}


async def send_request(
    client: httpx.AsyncClient, model: str, prompt: str
) -> Dict[str, Any]:
    start = time.time()
    payload = build_payload(prompt, model)

    try:
        response = await client.post(
            f"{BASE_URL}/mlx/chat",
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
    concurrency: int,
    num_requests: int,
) -> List[Dict[str, Any]]:
    async with httpx.AsyncClient() as client:
        results: List[Dict[str, Any]] = []
        for i in range(0, num_requests, concurrency):
            chunk = [
                send_request(client, model, prompt)
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


async def prewarm_model(model: str, prompt: str = "count to 10") -> None:
    print("\n" + "=" * 72)
    print("Prewarming MLX model...")
    async with httpx.AsyncClient() as client:
        print(f"  Prewarming {model}")
        payload = build_payload(prompt, model)
        try:
            response = await client.post(
                f"{BASE_URL}/mlx/chat",
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
        except Exception as exc:
            print(f"    Warning: prewarm failed: {exc}")
    print("Prewarm complete.")
    print("=" * 72)


async def run_benchmark(
    prompts: Iterable[str],
    model: str,
    concurrency_levels: Iterable[int],
    requests_per_level: int,
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    # prewarm the selected MLX model to load it into memory
    await prewarm_model(model)

    for prompt in prompts:
        for concurrency in concurrency_levels:
            print("\n" + "=" * 72)
            print(f"Prompt: {prompt}")
            print(f"Model: {model} | Concurrency: {concurrency} | Batch Size: {batch_size} | Requests: {requests_per_level}")
            print("=" * 72)

            request_results = await benchmark_model_at_concurrency(
                model,
                prompt,
                concurrency,
                requests_per_level,
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
                "batch_size": batch_size,
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
                print(f"  {len(failed)} failed requests")
            print(
                f"  success: {row['success_count']} | avg: {format_number(row['avg_latency'])}s | "
                f"p95: {format_number(row['p95_latency'])}s | tokens/sec: {format_number(row['tokens_per_sec'])} | "
                f"req/sec: {format_number(row['requests_per_sec'])}"
            )
            rows.append(row)

    return rows


def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "prompt",
        "model",
        "batch_size",
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
        "cost_per_million_tokens",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"\n✓ MLX results saved to {path}")


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
    parser = argparse.ArgumentParser(description="MLX serving benchmark sweep")
    parser.add_argument("--model", default=MODELS[0], help="Model to benchmark")
    parser.add_argument("--prompts", nargs="+", default=PROMPTS, help="Prompts to benchmark")
    parser.add_argument("--concurrency", nargs="+", type=int, default=CONCURRENCY_LEVELS, help="Concurrency levels to test")
    parser.add_argument("--requests", type=int, default=REQUESTS_PER_LEVEL, help="Requests per concurrency level")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for MLX")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = asyncio.run(
        run_benchmark(
            prompts=args.prompts,
            model=args.model,
            concurrency_levels=args.concurrency,
            requests_per_level=args.requests,
            batch_size=args.batch_size,
        )
    )

    save_csv(rows, CSV_PATH)
    save_grouped_csv(rows, "prompt", PROMPT_DIR)
    save_grouped_csv(rows, "batch_size", BATCH_DIR)


if __name__ == "__main__":
    main()