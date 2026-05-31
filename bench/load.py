import asyncio
import csv
import time
import math
from pathlib import Path

import httpx
from typing import List, Dict, Optional

BASE_URL = "http://127.0.0.1:8000"
MODELS = ["llama3.2:1b", "qwen2.5:7b"]
CONCURRENCY_LEVELS = [1, 2, 4, 8]
REQUESTS_PER_LEVEL = 10
PROMPT = "Say hello in one word"
RESULTS_DIR = Path("results")
CSV_PATH = RESULTS_DIR / "results.csv"
PNG_PATH = RESULTS_DIR / "benchmark_results.png"


async def send_request(client: httpx.AsyncClient, model: str) -> Optional[float]:
    """Send one request, return latency in seconds."""
    start = time.time()
    try:
        response = await client.post(
            f"{BASE_URL}/ollama/chat",
            json={"message": PROMPT, "model": model},
            timeout=120.0,
        )
        response.raise_for_status()
        elapsed = time.time() - start
        return elapsed
    except Exception as e:
        print(f"Error: {e}")
        return None


async def benchmark_model_at_concurrency(
    model: str, concurrency: int, num_requests: int
) -> List[float]:
    """Run concurrent requests and return list of latencies."""
    async with httpx.AsyncClient() as client:
        tasks = [
            send_request(client, model)
            for _ in range(num_requests)
        ]
        # Limit concurrency by chunking
        results = []
        for i in range(0, len(tasks), concurrency):
            chunk = tasks[i : i + concurrency]
            latencies = await asyncio.gather(*chunk)
            results.extend([l for l in latencies if l is not None])
        return results


async def run_benchmark():
    """Run full benchmark sweep."""
    results = {}
    
    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"Testing model: {model}")
        print(f"{'='*60}")
        
        results[model] = {}
        
        for concurrency in CONCURRENCY_LEVELS:
            print(f"  Concurrency: {concurrency} | Requests: {REQUESTS_PER_LEVEL}", end="")
            latencies = await benchmark_model_at_concurrency(
                model, concurrency, REQUESTS_PER_LEVEL
            )
            
            if not latencies:
                print(" | FAILED (no successful requests)")
                continue
            
            avg = sum(latencies) / len(latencies)
            # Safe p95 calculation: pick the ceil 95th percentile index
            sorted_lat = sorted(latencies)
            idx = max(0, min(len(sorted_lat) - 1, math.ceil(0.95 * len(sorted_lat)) - 1))
            p95 = sorted_lat[idx]
            max_lat = max(latencies)
            
            results[model][concurrency] = {
                "latencies": latencies,
                "avg": avg,
                "p95": p95,
                "max": max_lat,
            }
            
            print(f" | avg: {avg:.2f}s | p95: {p95:.2f}s | max: {max_lat:.2f}s")
    
    return results


def ensure_results_dir(path: Path) -> None:
    """Create the parent results directory if it does not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_results(results: Dict, output_path: Path = PNG_PATH):
    """Plot latency vs concurrency for each model."""
    if not results:
        print("No results to plot.")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        return

    ensure_results_dir(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Average latency
    ax = axes[0]
    for model in results:
        concurrencies = sorted(results[model].keys())
        avgs = [results[model][c]["avg"] for c in concurrencies]
        ax.plot(concurrencies, avgs, marker="o", label=model, linewidth=2)

    ax.set_xlabel("Concurrency (requests in parallel)")
    ax.set_ylabel("Latency (seconds)")
    ax.set_title("Average Latency vs Concurrency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: P95 latency (tail latency)
    ax = axes[1]
    for model in results:
        concurrencies = sorted(results[model].keys())
        p95s = [results[model][c]["p95"] for c in concurrencies]
        ax.plot(concurrencies, p95s, marker="s", label=model, linewidth=2)

    ax.set_xlabel("Concurrency (requests in parallel)")
    ax.set_ylabel("P95 Latency (seconds)")
    ax.set_title("Tail Latency (P95) vs Concurrency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\n✓ Plot saved to {output_path}")


def save_csv(results: Dict, path: Path = CSV_PATH):
    """Save results to CSV."""
    if not results:
        print("No results to save.")
        return

    ensure_results_dir(path)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concurrency", "model", "avg", "p95", "max"])
        for model in results:
            for concurrency in sorted(results[model].keys()):
                r = results[model][concurrency]
                writer.writerow([
                    concurrency,
                    model,
                    round(r["avg"], 3),
                    round(r["p95"], 3),
                    round(r["max"], 3),
                ])
    print(f"\n✓ Results saved to {path}")


if __name__ == "__main__":
    results = asyncio.run(run_benchmark())
    save_csv(results)
    plot_results(results)
