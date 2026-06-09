"""
Does human difficulty affect per-token latency?

Hypothesis: for a standard (non-reasoning) dense model, the time to produce a
token is fixed by the architecture, not by how "hard" the prompt is for a human.
So inter-token latency should be ~flat across prompts of wildly different human
difficulty, and total time should be governed by token *count* alone.

This script tests that by streaming from the gateway's /ollama/chat/stream
endpoint and measuring, per prompt category:
  - TTFT (time to first token)         -- prefill cost, after prewarm
  - decode latency (ms / output token) -- the flat-vs-difficulty signal
  - tokens_per_sec (eval-based)        -- authoritative, from Ollama timings
  - $/M tokens                         -- the portfolio metric

Methodology: prewarm first (cold-start load would swamp TTFT), randomize trial
order (thermal/ordering bias), repeat each prompt N times, report variance.

Usage:
    python bench/difficulty_invariance.py
    python bench/difficulty_invariance.py --model qwen2.5:7b --repeats 5 --max-tokens 256
"""

import argparse
import csv
import json
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from cost_calculator import cost_per_million_tokens, get_gpu_hourly_rate

GATEWAY_URL = "http://localhost:8000"
STREAM_PATH = "/ollama/chat/stream"
RESULTS_DIR = Path("results/difficulty")
CHARTS_DIR = Path("charts")

# Prompts span a spectrum of human difficulty while each targets a comparable
# output length (~40 lines). The point is to vary difficulty, not length --
# per-token metrics are length-normalized regardless.
PROMPTS: Dict[str, str] = {
    "trivial": "List 40 random common English nouns, one per line.",
    "recall": "List the first 40 prime numbers, one per line.",
    "hard_structured": "List 40 English words that are palindromes, one per line.",
    "reasoning": (
        "Solve step by step: a tank fills at 8 liters/min and drains at 3 "
        "liters/min. Starting empty, show the water level after each minute "
        "for 40 minutes, one line per minute."
    ),
    "creative": "Write a 40-line poem about the changing seasons, one line per line.",
}

# Rough human-difficulty ordering, used only for stable chart x-axis ordering.
CATEGORY_ORDER = ["trivial", "recall", "creative", "reasoning", "hard_structured"]


def stream_one(
    client: httpx.Client, url: str, model: str, prompt: str, max_tokens: int
) -> Tuple[List[float], Optional[Dict[str, Any]]]:
    """Run one streamed request. Return (per-delta timestamps, final summary)."""
    delta_ts: List[float] = []
    summary: Optional[Dict[str, Any]] = None
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens}
    with client.stream("POST", url, json=payload, timeout=300.0) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("done"):
                summary = obj
            elif "error" in obj:
                raise RuntimeError(obj["error"])
            else:
                delta_ts.append(obj["t"])
    return delta_ts, summary


def inter_token_gaps_ms(delta_ts: List[float]) -> List[float]:
    """Wall-clock gaps between consecutive emitted tokens, in milliseconds."""
    return [(b - a) * 1000.0 for a, b in zip(delta_ts, delta_ts[1:])]


def percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    import math

    k = max(0, min(len(s) - 1, math.ceil(pct / 100 * len(s)) - 1))
    return s[k]


def build_row(
    model: str, category: str, prompt: str, trial: int,
    delta_ts: List[float], summary: Dict[str, Any], rate: float,
) -> Dict[str, Any]:
    gaps = inter_token_gaps_ms(delta_ts)
    tps = summary.get("tokens_per_sec") or 0.0
    return {
        "model": model,
        "category": category,
        "prompt": prompt,
        "trial": trial,
        "input_tokens": summary.get("input_tokens"),
        "output_tokens": summary.get("output_tokens"),
        "deltas_emitted": len(delta_ts),
        "ttft_s": summary.get("ttft"),
        "elapsed_s": summary.get("elapsed"),
        "tokens_per_sec": tps,
        # Two views of per-token cost: server eval-based, and client wall-clock.
        "eval_ms_per_token": (1000.0 / tps) if tps else None,
        "decode_ms_per_token": statistics.mean(gaps) if gaps else None,
        "decode_p50_ms": percentile(gaps, 50),
        "decode_p95_ms": percentile(gaps, 95),
        "cost_usd": summary.get("cost_usd"),
        "cost_per_million_tokens": cost_per_million_tokens(rate, tps) if tps else None,
    }


def prewarm(client: httpx.Client, url: str, model: str) -> None:
    print(f"Prewarming {model} (discarding cold-start load time)...")
    try:
        _, summary = stream_one(client, url, model, "count to 5", 16)
        if summary:
            print(f"  warm. ttft now ~{summary.get('ttft'):.3f}s")
    except Exception as exc:
        print(f"  Warning: prewarm failed: {exc}")


def run(
    model: str, repeats: int, max_tokens: int, url: str, cooldown: float
) -> List[Dict[str, Any]]:
    rate = get_gpu_hourly_rate()
    rows: List[Dict[str, Any]] = []

    # Build a randomized schedule of (category, trial) units so thermal drift
    # and ordering don't correlate with any one category.
    schedule: List[Tuple[str, int]] = [
        (cat, t) for cat in PROMPTS for t in range(repeats)
    ]
    random.shuffle(schedule)

    with httpx.Client() as client:
        prewarm(client, url, model)
        total = len(schedule)
        for i, (category, trial) in enumerate(schedule, 1):
            prompt = PROMPTS[category]
            print(f"[{i}/{total}] {category} (trial {trial})...", end=" ", flush=True)
            try:
                delta_ts, summary = stream_one(client, url, model, prompt, max_tokens)
                if not summary:
                    print("no summary, skipping")
                    continue
                row = build_row(model, category, prompt, trial, delta_ts, summary, rate)
                rows.append(row)
                print(
                    f"out={row['output_tokens']}tok "
                    f"ttft={row['ttft_s']:.3f}s "
                    f"decode={row['decode_ms_per_token']:.1f}ms/tok "
                    f"${row['cost_per_million_tokens']:.2f}/M"
                )
            except Exception as exc:
                print(f"FAILED: {exc}")
            if cooldown:
                time.sleep(cooldown)
    return rows


def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "category", "prompt", "trial",
        "input_tokens", "output_tokens", "deltas_emitted",
        "ttft_s", "elapsed_s", "tokens_per_sec",
        "eval_ms_per_token", "decode_ms_per_token", "decode_p50_ms", "decode_p95_ms",
        "cost_usd", "cost_per_million_tokens",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"\nSaved {len(rows)} rows to {path}")


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Aggregate per category: mean/stdev of the per-token + cost metrics."""
    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)

    agg: Dict[str, Dict[str, float]] = {}
    for cat, group in by_cat.items():
        decode = [r["decode_ms_per_token"] for r in group if r["decode_ms_per_token"]]
        ttft = [r["ttft_s"] for r in group if r["ttft_s"] is not None]
        cost = [r["cost_per_million_tokens"] for r in group if r["cost_per_million_tokens"]]
        agg[cat] = {
            "decode_ms_mean": statistics.mean(decode) if decode else 0.0,
            "decode_ms_std": statistics.pstdev(decode) if len(decode) > 1 else 0.0,
            "ttft_mean": statistics.mean(ttft) if ttft else 0.0,
            "cost_mean": statistics.mean(cost) if cost else 0.0,
            "n": len(group),
        }
    return agg


def make_charts(rows: List[Dict[str, Any]], model: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping charts")
        return

    agg = summarize(rows)
    cats = [c for c in CATEGORY_ORDER if c in agg]
    rate = get_gpu_hourly_rate()

    decode_means = [agg[c]["decode_ms_mean"] for c in cats]
    decode_stds = [agg[c]["decode_ms_std"] for c in cats]
    ttft_means = [agg[c]["ttft_mean"] for c in cats]
    cost_means = [agg[c]["cost_mean"] for c in cats]

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # Headline: decode latency per token vs difficulty, with variance bars.
    # A flat line across categories is the thesis.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(cats, decode_means, yerr=decode_stds, capsize=5, color="#3F51B5", width=0.6)
    if decode_means:
        baseline = statistics.mean(decode_means)
        ax.axhline(baseline, ls="--", color="#888", label=f"mean = {baseline:.1f} ms/tok")
        ax.legend()
    ax.set_title(f"Decode latency vs human difficulty — {model}\n(flat = difficulty doesn't change per-token cost)", fontsize=11)
    ax.set_ylabel("ms / output token")
    ax.set_xlabel("prompt category (left=easy, right=hard for humans)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = CHARTS_DIR / "difficulty_decode_latency.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")

    # Supporting: TTFT and $/M tokens by category.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(cats, ttft_means, color="#009688", width=0.6)
    ax1.set_title("Time to first token (prefill, warmed)")
    ax1.set_ylabel("seconds")
    ax1.tick_params(axis="x", rotation=20)
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(cats, cost_means, color="#FF5722", width=0.6)
    ax2.set_title(rf"\$/M tokens  (GPU rate = \${rate:.2f}/hr)")
    ax2.set_ylabel("$ / M tokens")
    ax2.tick_params(axis="x", rotation=20)
    ax2.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Difficulty invariance — supporting metrics — {model}", fontweight="bold")
    fig.tight_layout()
    p = CHARTS_DIR / "difficulty_ttft_and_cost.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test difficulty-invariance of per-token latency")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--repeats", type=int, default=5, help="Trials per prompt category")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--url", default=GATEWAY_URL + STREAM_PATH)
    parser.add_argument("--cooldown", type=float, default=0.0, help="Seconds to sleep between requests")
    parser.add_argument("--no-charts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Difficulty-invariance benchmark | model={args.model} | "
          f"repeats={args.repeats} | max_tokens={args.max_tokens} | "
          f"GPU_HOURLY_RATE=${get_gpu_hourly_rate():.2f}/hr\n")

    rows = run(args.model, args.repeats, args.max_tokens, args.url, args.cooldown)
    if not rows:
        print("No successful runs; nothing to save.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_csv(rows, RESULTS_DIR / f"difficulty_invariance_{timestamp}.csv")

    print("\nPer-category summary (decode ms/token ± std | ttft s | $/M tokens):")
    agg = summarize(rows)
    for cat in [c for c in CATEGORY_ORDER if c in agg]:
        a = agg[cat]
        print(f"  {cat:16s} {a['decode_ms_mean']:6.1f} ± {a['decode_ms_std']:4.1f} ms | "
              f"{a['ttft_mean']:.3f} s | ${a['cost_mean']:.2f}/M  (n={a['n']})")

    if not args.no_charts:
        print("\nCharts:")
        make_charts(rows, args.model)


if __name__ == "__main__":
    main()
