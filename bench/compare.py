import argparse
import asyncio
import csv
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from cost_calculator import cost_per_million_tokens, get_gpu_hourly_rate

OLLAMA_URL = "http://127.0.0.1:8000"
MLX_URL = "http://127.0.0.1:8001"

PROMPTS = [
    "What is deep learning?",
    # "Summarize the plot of The Matrix in a single sentence",
    # "Translate 'Hello' to Spanish",
    # "Write a short Python function that computes factorial",
]

CONCURRENCY_LEVELS = [1, 4, 8, 16]
REQUESTS_PER_LEVEL = 16
RESULTS_DIR = Path("results/comparison")


def get_csv_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RESULTS_DIR / f"ollama_vs_mlx_{timestamp}.csv"


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
        if isinstance(data.get("response"), str):
            return estimate_tokens(data["response"])
        if isinstance(data.get("text"), str):
            return estimate_tokens(data["text"])

    return estimate_tokens(response.text)


async def send_request(
    client: httpx.AsyncClient, base_url: str, backend: str, model: str, prompt: str, max_tokens: int = 50
) -> Dict[str, Any]:
    start = time.time()
    
    if backend == "ollama":
        endpoint = f"{base_url}/ollama/chat"
        payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens}
    else:  # mlx
        endpoint = f"{base_url}/mlx/chat"
        payload = {"prompt": prompt, "model": model, "max_tokens": max_tokens}

    try:
        response = await client.post(
            endpoint,
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

async def benchmark_backend(
    backend: str,
    base_url: str,
    model: str,
    prompt: str,
    concurrency: int,
    num_requests: int,
) -> List[Dict[str, Any]]:
    async with httpx.AsyncClient() as client:
        results: List[Dict[str, Any]] = []
        for i in range(0, num_requests, concurrency):
            chunk = [
                send_request(client, base_url, backend, model, prompt)
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


async def prewarm(backend: str, base_url: str, model: str, prompt: str = "count to 10") -> None:
    async with httpx.AsyncClient() as client:
        try:
            await send_request(client, base_url, backend, model, prompt)
        except Exception as exc:
            print(f"Warning: prewarm failed for {backend}: {exc}")


async def run_comparison(
    prompts: List[str],
    model: str,
    concurrency_levels: List[int],
    requests_per_level: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    # Prewarm both backends
    print("\nPrewarming Ollama...")
    await prewarm("ollama", OLLAMA_URL, model)
    print("Prewarming MLX...")
    await prewarm("mlx", MLX_URL, model)

    for prompt in prompts:
        for concurrency in concurrency_levels:
            print("\n" + "=" * 80)
            print(f"Prompt: {prompt}")
            print(f"Concurrency: {concurrency} | Requests: {requests_per_level}")
            print("=" * 80)

            # Benchmark Ollama
            print(f"  Benchmarking Ollama...")
            ollama_results = await benchmark_backend(
                "ollama",
                OLLAMA_URL,
                model,
                prompt,
                concurrency,
                requests_per_level,
            )

            # Benchmark MLX
            print(f"  Benchmarking MLX...")
            mlx_results = await benchmark_backend(
                "mlx",
                MLX_URL,
                model,
                prompt,
                concurrency,
                requests_per_level,
            )

            # Process Ollama results
            ollama_successful = [r for r in ollama_results if r["success"]]
            ollama_latencies = [r["latency"] for r in ollama_successful if r["latency"] is not None]
            ollama_tokens = sum(r["tokens"] for r in ollama_successful)
            ollama_duration = sum(ollama_latencies) if ollama_latencies else 0.0

            # Process MLX results
            mlx_successful = [r for r in mlx_results if r["success"]]
            mlx_latencies = [r["latency"] for r in mlx_successful if r["latency"] is not None]
            mlx_tokens = sum(r["tokens"] for r in mlx_successful)
            mlx_duration = sum(mlx_latencies) if mlx_latencies else 0.0

            # Create rows
            rate = get_gpu_hourly_rate()
            ollama_tps = ollama_tokens / ollama_duration if ollama_duration else None
            mlx_tps = mlx_tokens / mlx_duration if mlx_duration else None

            ollama_row = {
                "prompt": prompt,
                "backend": "ollama",
                "model": model,
                "concurrency": concurrency,
                "requested": requests_per_level,
                "success_count": len(ollama_successful),
                "failure_count": len(ollama_results) - len(ollama_successful),
                "avg_latency": sum(ollama_latencies) / len(ollama_latencies) if ollama_latencies else None,
                "min_latency": min(ollama_latencies) if ollama_latencies else None,
                "p50_latency": percentile(ollama_latencies, 0.50) if ollama_latencies else None,
                "p95_latency": percentile(ollama_latencies, 0.95) if ollama_latencies else None,
                "p99_latency": percentile(ollama_latencies, 0.99) if ollama_latencies else None,
                "max_latency": max(ollama_latencies) if ollama_latencies else None,
                "total_tokens": ollama_tokens,
                "avg_tokens": ollama_tokens / len(ollama_successful) if ollama_successful else None,
                "tokens_per_sec": ollama_tps,
                "requests_per_sec": len(ollama_successful) / ollama_duration if ollama_duration else None,
                "duration": ollama_duration,
                "cost_per_million_tokens": cost_per_million_tokens(rate, ollama_tps) if ollama_tps else None,
            }

            mlx_row = {
                "prompt": prompt,
                "backend": "mlx",
                "model": model,
                "concurrency": concurrency,
                "requested": requests_per_level,
                "success_count": len(mlx_successful),
                "failure_count": len(mlx_results) - len(mlx_successful),
                "avg_latency": sum(mlx_latencies) / len(mlx_latencies) if mlx_latencies else None,
                "min_latency": min(mlx_latencies) if mlx_latencies else None,
                "p50_latency": percentile(mlx_latencies, 0.50) if mlx_latencies else None,
                "p95_latency": percentile(mlx_latencies, 0.95) if mlx_latencies else None,
                "p99_latency": percentile(mlx_latencies, 0.99) if mlx_latencies else None,
                "max_latency": max(mlx_latencies) if mlx_latencies else None,
                "total_tokens": mlx_tokens,
                "avg_tokens": mlx_tokens / len(mlx_successful) if mlx_successful else None,
                "tokens_per_sec": mlx_tps,
                "requests_per_sec": len(mlx_successful) / mlx_duration if mlx_duration else None,
                "duration": mlx_duration,
                "cost_per_million_tokens": cost_per_million_tokens(rate, mlx_tps) if mlx_tps else None,
            }

            rows.extend([ollama_row, mlx_row])

            # Print summary
            print(f"  Ollama: avg={format_number(ollama_row['avg_latency'])}s | "
                  f"p95={format_number(ollama_row['p95_latency'])}s | "
                  f"tokens/sec={format_number(ollama_row['tokens_per_sec'])}")
            print(f"  MLX:    avg={format_number(mlx_row['avg_latency'])}s | "
                  f"p95={format_number(mlx_row['p95_latency'])}s | "
                  f"tokens/sec={format_number(mlx_row['tokens_per_sec'])}")

    return rows


def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "prompt",
        "backend",
        "model",
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

    print(f"\n✓ Comparison results saved to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Ollama vs MLX inference")
    parser.add_argument("--model", default="qwen2.5:7b", help="Model to benchmark")
    parser.add_argument("--prompts", nargs="+", default=PROMPTS, help="Prompts to benchmark")
    parser.add_argument("--concurrency", nargs="+", type=int, default=CONCURRENCY_LEVELS, help="Concurrency levels to test")
    parser.add_argument("--requests", type=int, default=REQUESTS_PER_LEVEL, help="Requests per concurrency level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = asyncio.run(
        run_comparison(
            prompts=args.prompts,
            model=args.model,
            concurrency_levels=args.concurrency,
            requests_per_level=args.requests,
        )
    )

    csv_path = get_csv_path()
    save_csv(rows, csv_path)


if __name__ == "__main__":
    main()