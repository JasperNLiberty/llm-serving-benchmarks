"""How does context length change inference cost?

Holds output length fixed and sweeps the *input* (prompt) length, so the only
variable is how much context the model must prefill and cache. Three effects
are expected, and this script measures or models each:

  1. TTFT rises with context      -- prefill is O(context): the whole prompt is
     processed before the first token, so time-to-first-token grows with input.
  2. Decode slows gently       -- each output token is one forward pass over the
     cache; per-token decode is roughly context-agnostic until the growing KV
     cache saturates memory bandwidth, after which it creeps up.
  3. KV cache grows linearly      -- modelled analytically from the qwen2.5-7B
     architecture (GQA: only 4 KV heads), because Ollama does not expose live
     cache size on Metal. This is the memory that caps how much context (and how
     many concurrent requests) a GPU can hold.

The cost story: at a *fixed output length*, $/request still climbs with input
length, entirely because of prefill. Long-context features are not free.

Streams from the gateway's /ollama/chat/stream endpoint (same contract as
difficulty_invariance.py): per-token frames then a final summary with ttft,
token counts, tokens_per_sec, elapsed, and cost.

Usage:
    python bench/context_scaling.py
    python bench/context_scaling.py --model qwen2.5:7b --repeats 3 --max-tokens 128
    python bench/context_scaling.py --lengths 256 1024 4096 8192
"""

import argparse
import csv
import json
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
RESULTS_DIR = Path("results/context")
CHARTS_DIR = Path("charts")

# qwen2.5-7B architecture (from `ollama show qwen2.5:7b` model_info). GQA keeps
# the KV cache small: 4 KV heads instead of 28 attention heads is a 7x saving.
KV_CONFIG = {
    "n_layers": 28,
    "n_kv_heads": 4,
    "head_dim": 128,      # embedding_length 3584 / 28 attention heads
    "dtype_bytes": 2,     # fp16 KV cache (Ollama default on Metal)
}
MAX_CONTEXT = 32768       # qwen2.5:7b trained context length

DEFAULT_LENGTHS = [256, 512, 1024, 2048, 4096, 8192]

# A neutral filler paragraph repeated to reach a target prompt length. Content is
# irrelevant -- only the token count matters -- but readable text tokenizes like
# real input rather than degenerate repetition.
FILLER_SENTENCE = (
    "The inference server processes each request by first reading the prompt "
    "into the key-value cache and then generating output tokens one at a time. "
)
INSTRUCTION = "\n\nIn one sentence, state what the text above is about."


def kv_cache_bytes(seq_len: int) -> int:
    """Analytical KV cache size in bytes for a given sequence length.

    cache = 2 (K and V) * n_layers * n_kv_heads * head_dim * seq_len * dtype_bytes
    """
    c = KV_CONFIG
    return 2 * c["n_layers"] * c["n_kv_heads"] * c["head_dim"] * seq_len * c["dtype_bytes"]


def kv_cache_mib(seq_len: int) -> float:
    return kv_cache_bytes(seq_len) / (1024 * 1024)


def build_prompt(target_tokens: int) -> str:
    """Build a prompt of roughly ``target_tokens`` tokens.

    ~0.75 tokens/word for English, so aim for ~target_tokens words of filler.
    The exact length is approximate; the server reports the true input_tokens
    and all metrics are plotted against that measured value, not the target.
    """
    words_needed = int(target_tokens / 0.75)
    words_per_sentence = len(FILLER_SENTENCE.split())
    repeats = max(1, words_needed // words_per_sentence)
    return (FILLER_SENTENCE * repeats) + INSTRUCTION


def stream_one(
    client: httpx.Client, url: str, model: str, prompt: str, max_tokens: int
) -> Tuple[List[float], Optional[Dict[str, Any]]]:
    """Run one streamed request. Return (per-delta timestamps, final summary)."""
    delta_ts: List[float] = []
    summary: Optional[Dict[str, Any]] = None
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens}
    with client.stream("POST", url, json=payload, timeout=600.0) as response:
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
    return [(b - a) * 1000.0 for a, b in zip(delta_ts, delta_ts[1:])]


def build_row(
    model: str, target_len: int, trial: int,
    delta_ts: List[float], summary: Dict[str, Any], rate: float,
) -> Dict[str, Any]:
    gaps = inter_token_gaps_ms(delta_ts)
    tps = summary.get("tokens_per_sec") or 0.0
    input_tokens = summary.get("input_tokens") or 0
    return {
        "model": model,
        "target_input_tokens": target_len,
        "trial": trial,
        "input_tokens": input_tokens,
        "output_tokens": summary.get("output_tokens"),
        "ttft_s": summary.get("ttft"),
        "elapsed_s": summary.get("elapsed"),
        "tokens_per_sec": tps,
        "decode_ms_per_token": statistics.mean(gaps) if gaps else None,
        "cost_usd": summary.get("cost_usd"),
        "cost_per_million_tokens": cost_per_million_tokens(rate, tps) if tps else None,
        # Analytical KV-cache footprint at the measured prompt length.
        "kv_cache_mib": round(kv_cache_mib(input_tokens), 3) if input_tokens else None,
    }


def prewarm(client: httpx.Client, url: str, model: str) -> None:
    print(f"Prewarming {model} (discarding cold-start load time)...")
    try:
        _, summary = stream_one(client, url, model, "count to 5", 16)
        if summary:
            print(f"  warm. ttft now ~{summary.get('ttft'):.3f}s")
    except Exception as exc:
        print(f"  Warning: prewarm failed: {exc}")


def run(model: str, lengths: List[int], repeats: int, max_tokens: int,
        url: str) -> List[Dict[str, Any]]:
    rate = get_gpu_hourly_rate()
    rows: List[Dict[str, Any]] = []
    with httpx.Client() as client:
        prewarm(client, url, model)
        total = len(lengths) * repeats
        i = 0
        for target_len in lengths:
            prompt = build_prompt(target_len)
            for trial in range(repeats):
                i += 1
                print(f"[{i}/{total}] ~{target_len} input tok (trial {trial})...",
                      end=" ", flush=True)
                try:
                    delta_ts, summary = stream_one(client, url, model, prompt, max_tokens)
                    if not summary:
                        print("no summary, skipping")
                        continue
                    row = build_row(model, target_len, trial, delta_ts, summary, rate)
                    rows.append(row)
                    print(f"in={row['input_tokens']}tok "
                          f"ttft={row['ttft_s']:.3f}s "
                          f"decode={row['decode_ms_per_token']:.1f}ms/tok "
                          f"kv={row['kv_cache_mib']:.0f}MiB "
                          f"${row['cost_per_million_tokens']:.2f}/M")
                except Exception as exc:
                    print(f"FAILED: {exc}")
    return rows


def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "target_input_tokens", "trial", "input_tokens", "output_tokens",
        "ttft_s", "elapsed_s", "tokens_per_sec", "decode_ms_per_token",
        "cost_usd", "cost_per_million_tokens", "kv_cache_mib",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"\nSaved {len(rows)} rows to {path}")


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """Aggregate per target length: mean ttft, decode ms, tps, cost, kv."""
    by_len: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        by_len.setdefault(r["target_input_tokens"], []).append(r)

    out = []
    for target_len in sorted(by_len):
        group = by_len[target_len]
        def mean(key):
            vals = [g[key] for g in group if g.get(key) is not None]
            return statistics.mean(vals) if vals else 0.0
        out.append({
            "target_input_tokens": target_len,
            "input_tokens": mean("input_tokens"),
            "ttft_s": mean("ttft_s"),
            "decode_ms_per_token": mean("decode_ms_per_token"),
            "tokens_per_sec": mean("tokens_per_sec"),
            "cost_per_million_tokens": mean("cost_per_million_tokens"),
            "kv_cache_mib": mean("kv_cache_mib"),
            "n": len(group),
        })
    return out


def make_charts(agg: List[Dict[str, float]], model: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping charts")
        return

    x = [a["input_tokens"] for a in agg]
    ttft = [a["ttft_s"] for a in agg]
    decode = [a["decode_ms_per_token"] for a in agg]
    cost = [a["cost_per_million_tokens"] for a in agg]
    kv = [a["kv_cache_mib"] for a in agg]
    rate = get_gpu_hourly_rate()

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # Chart 1: TTFT (prefill grows) vs decode latency (flat) — the headline.
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1 = "#009688"
    ax1.plot(x, ttft, marker="o", color=color1, label="TTFT (prefill)")
    ax1.set_xlabel("input context length (tokens)")
    ax1.set_ylabel("time to first token (s)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "#FF5722"
    ax2.plot(x, decode, marker="s", color=color2, label="decode latency")
    ax2.set_ylabel("decode latency (ms / output token)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, max(decode) * 1.6 if decode else 1)

    ax1.set_title(f"Prefill explodes with context; decode creeps up as KV cache fills — {model}\n"
                  f"(output length fixed; only input length varies)", fontsize=11)
    fig.tight_layout()
    p = CHARTS_DIR / "context_ttft_vs_decode.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")

    # Chart 2: KV cache growth (analytical) + $/request driver.
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1 = "#3F51B5"
    ax1.plot(x, kv, marker="o", color=color1)
    ax1.set_xlabel("input context length (tokens)")
    ax1.set_ylabel("KV cache (MiB, analytical)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    # Annotate the full-context ceiling.
    full = kv_cache_mib(MAX_CONTEXT)
    ax1.axhline(full, ls="--", color="#999")
    ax1.text(x[0], full, f"  full {MAX_CONTEXT//1024}k context = {full:.0f} MiB",
             va="bottom", fontsize=8, color="#666")

    ax2 = ax1.twinx()
    color2 = "#E91E63"
    ax2.plot(x, cost, marker="s", color=color2)
    ax2.set_ylabel("$ / M tokens", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title(f"KV cache and cost vs context length — {model}\n"
                  f"(KV cache analytical from arch; GPU rate = ${rate:.2f}/hr)", fontsize=11)
    fig.tight_layout()
    p = CHARTS_DIR / "context_kvcache_and_cost.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Context-length scaling benchmark")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS,
                        help="Target input lengths in tokens")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Output tokens (held fixed to isolate prefill)")
    parser.add_argument("--url", default=GATEWAY_URL + STREAM_PATH)
    parser.add_argument("--no-charts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Context-scaling benchmark | model={args.model} | "
          f"lengths={args.lengths} | repeats={args.repeats} | "
          f"max_tokens={args.max_tokens} | "
          f"GPU_HOURLY_RATE=${get_gpu_hourly_rate():.2f}/hr\n")

    rows = run(args.model, args.lengths, args.repeats, args.max_tokens, args.url)
    if not rows:
        print("No successful runs; nothing to save.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_csv(rows, RESULTS_DIR / f"context_scaling_{timestamp}.csv")

    agg = summarize(rows)
    print("\nPer-length summary (input tok | ttft | decode ms/tok | tps | $/M | KV MiB):")
    for a in agg:
        print(f"  ~{a['target_input_tokens']:>5} -> in={a['input_tokens']:6.0f} | "
              f"ttft={a['ttft_s']:.3f}s | {a['decode_ms_per_token']:5.1f} ms | "
              f"{a['tokens_per_sec']:5.1f} tps | ${a['cost_per_million_tokens']:.2f}/M | "
              f"{a['kv_cache_mib']:6.1f} MiB  (n={a['n']})")

    if not args.no_charts:
        print("\nCharts:")
        make_charts(agg, args.model)


if __name__ == "__main__":
    main()
