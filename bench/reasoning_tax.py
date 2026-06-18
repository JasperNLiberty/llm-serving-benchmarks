"""
The reasoning tax: what do thinking models actually cost?

Difficulty invariance (see difficulty_invariance.py) showed that for a standard
*non-reasoning* model, per-token decode latency is flat across human difficulty,
so cost tracks token COUNT, not "hardness." A reasoning model is expected to
break that: per-token price stays flat (same architecture), but it emits a
variable number of hidden *thinking* tokens that grows with difficulty — so its
cost and latency scale with difficulty THROUGH the token count.

This benchmark tests that directly. It runs a reasoning model and a non-reasoning
baseline of the same base architecture (deepseek-r1:7b is a reasoning fine-tune
of the qwen2.5-7B base) across a graded difficulty ladder of checkable problems,
and measures per tier:
  - thinking-token count          -- the difficulty signal (reasoning only)
  - latency split: prefill / thinking / answer phases
  - $/request and $/M tokens
  - accuracy and $/correct-answer -- the honest metric when quality differs
  - the reasoning tax = cost(reasoning) / cost(baseline)

It hits Ollama's /api/chat directly (not the gateway) because the thinking
stream is exposed there: each chunk carries either a `message.thinking` delta or
a `message.content` delta, which is what lets us separate the two phases.

Prereq:
    ollama pull deepseek-r1:7b      # the reasoning model (~4.7 GB)
    # baseline qwen2.5:7b is assumed already present

Usage:
    python bench/reasoning_tax.py
    python bench/reasoning_tax.py --repeats 5 --max-tokens 4096
    python bench/reasoning_tax.py --reasoning-model deepseek-r1:8b
"""

import argparse
import csv
import json
import random
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from cost_calculator import (
    cost_from_gpu_seconds,
    cost_per_million_tokens,
    cost_per_request,
    get_gpu_hourly_rate,
)

OLLAMA_URL = "http://localhost:11434"
RESULTS_DIR = Path("results/reasoning")
CHARTS_DIR = Path("charts")

# A graded ladder of problems with a single checkable numeric answer, ordered
# from trivial to multi-step. Same answer format ("just the number") so grading
# is a clean numeric match and output length is comparable for the baseline.
TIERS: List[Tuple[str, str, str]] = [
    # (tier, prompt, expected answer)
    ("t1_trivial", "What is 7 + 6? Give only the final number.", "13"),
    ("t2_easy", "A book costs $12. You buy 3 of them. What is the total cost in "
                "dollars? Give only the final number.", "36"),
    ("t3_medium", "A train travels 60 km in 1.5 hours. What is its average speed "
                  "in km/h? Give only the final number.", "40"),
    ("t4_hard", "Natalia sold clips to 48 friends in April, then sold half as "
                "many clips in May. How many clips did she sell altogether? "
                "Give only the final number.", "72"),
    ("t5_very_hard", "A store had 120 apples. It sold one third of them in the "
                     "morning, then one quarter of the remaining apples in the "
                     "afternoon. How many apples are left? Give only the final "
                     "number.", "60"),
]
TIER_ORDER = [t[0] for t in TIERS]
EXPECTED = {t[0]: t[2] for t in TIERS}
PROMPT = {t[0]: t[1] for t in TIERS}


def graded(answer_text: str, expected: str) -> bool:
    """True if the last integer in the answer equals the expected answer.

    Robust to "The answer is 72." and thousands separators; we compare the final
    number the model emits, which is the convention for these "just the number"
    prompts.
    """
    nums = re.findall(r"-?\d+", answer_text.replace(",", ""))
    return bool(nums) and nums[-1] == str(expected)


def stream_chat(
    client: httpx.Client, model: str, prompt: str, max_tokens: int, think: bool
) -> Dict[str, Any]:
    """One streamed /api/chat request, splitting thinking vs answer.

    Returns timestamps and text for each phase plus Ollama's final summary
    (eval_count, prompt_eval_count, eval_duration). Token counts per phase are
    the number of streamed deltas (Ollama streams ~one token per chunk).
    """
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {"num_predict": max_tokens, "temperature": 0},
    }
    if think:
        payload["think"] = True

    t0 = time.perf_counter()
    ttft: Optional[float] = None
    think_ts: List[float] = []
    ans_ts: List[float] = []
    thinking_text: List[str] = []
    answer_text: List[str] = []
    summary: Optional[Dict[str, Any]] = None

    with client.stream("POST", OLLAMA_URL + "/api/chat", json=payload, timeout=900.0) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("error"):
                raise RuntimeError(obj["error"])
            now = time.perf_counter() - t0
            msg = obj.get("message") or {}
            th = msg.get("thinking")
            ct = msg.get("content")
            if th:
                if ttft is None:
                    ttft = now
                think_ts.append(now)
                thinking_text.append(th)
            if ct:
                if ttft is None:
                    ttft = now
                ans_ts.append(now)
                answer_text.append(ct)
            if obj.get("done"):
                summary = obj
    end = time.perf_counter() - t0
    return {
        "ttft": ttft,
        "end": end,
        "think_ts": think_ts,
        "ans_ts": ans_ts,
        "thinking": "".join(thinking_text),
        "answer": "".join(answer_text),
        "summary": summary or {},
    }


def build_row(
    model: str, is_reasoning: bool, tier: str, trial: int,
    res: Dict[str, Any], rate: float,
) -> Dict[str, Any]:
    s = res["summary"]
    ttft = res["ttft"] or 0.0
    end = res["end"]
    think_ts = res["think_ts"]
    ans_ts = res["ans_ts"]

    # Phase boundaries. Prefill = time to first token. Thinking runs from the
    # first token to the first answer token; answer runs from there to the end.
    answer_start = ans_ts[0] if ans_ts else end
    prefill_s = ttft
    thinking_s = max(0.0, answer_start - ttft) if think_ts else 0.0
    answer_s = max(0.0, end - answer_start)

    out_tokens = s.get("eval_count") or (len(think_ts) + len(ans_ts))
    in_tokens = s.get("prompt_eval_count") or 0
    eval_dur_ns = s.get("eval_duration") or 0
    tps = out_tokens / (eval_dur_ns / 1e9) if eval_dur_ns > 0 else 0.0

    correct = graded(res["answer"], EXPECTED[tier])

    return {
        "model": model,
        "is_reasoning": is_reasoning,
        "tier": tier,
        "trial": trial,
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "thinking_tokens": len(think_ts),
        "answer_tokens": len(ans_ts),
        "ttft_s": round(ttft, 4),
        "thinking_s": round(thinking_s, 4),
        "answer_s": round(answer_s, 4),
        "elapsed_s": round(end, 4),
        "tokens_per_sec": round(tps, 2),
        "cost_usd": cost_per_request(rate, tps, in_tokens, out_tokens) if tps else 0.0,
        "cost_per_million_tokens": cost_per_million_tokens(rate, tps) if tps else 0.0,
        # Time-based dollar split of the generation phases.
        "prefill_cost_usd": cost_from_gpu_seconds(rate, prefill_s),
        "thinking_cost_usd": cost_from_gpu_seconds(rate, thinking_s),
        "answer_cost_usd": cost_from_gpu_seconds(rate, answer_s),
        "correct": int(correct),
    }


def ollama_models(client: httpx.Client) -> List[str]:
    try:
        tags = client.get(OLLAMA_URL + "/api/tags", timeout=10).json()
        return [m["name"] for m in tags.get("models", [])]
    except Exception:
        return []


def prewarm(client: httpx.Client, model: str, think: bool) -> None:
    print(f"Prewarming {model}...")
    try:
        stream_chat(client, model, "What is 1 + 1? Give only the number.", 64, think)
        print("  warm.")
    except Exception as exc:
        print(f"  Warning: prewarm failed: {exc}")


def run_model(
    client: httpx.Client, model: str, is_reasoning: bool,
    repeats: int, max_tokens: int, rate: float, cooldown: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    schedule = [(tier, t) for tier in TIER_ORDER for t in range(repeats)]
    random.shuffle(schedule)  # decorrelate thermal drift from any tier
    prewarm(client, model, is_reasoning)
    total = len(schedule)
    for i, (tier, trial) in enumerate(schedule, 1):
        tag = "reason" if is_reasoning else "base"
        print(f"[{tag} {i}/{total}] {tier} (trial {trial})...", end=" ", flush=True)
        try:
            res = stream_chat(client, model, PROMPT[tier], max_tokens, is_reasoning)
            row = build_row(model, is_reasoning, tier, trial, res, rate)
            rows.append(row)
            print(f"think={row['thinking_tokens']}tok ans={row['answer_tokens']}tok "
                  f"{row['elapsed_s']:.1f}s ${row['cost_usd']*1000:.4f}/k "
                  f"{'OK' if row['correct'] else 'WRONG'}")
        except Exception as exc:
            print(f"FAILED: {exc}")
        if cooldown:
            time.sleep(cooldown)
    return rows


FIELDNAMES = [
    "model", "is_reasoning", "tier", "trial",
    "input_tokens", "output_tokens", "thinking_tokens", "answer_tokens",
    "ttft_s", "thinking_s", "answer_s", "elapsed_s", "tokens_per_sec",
    "cost_usd", "cost_per_million_tokens",
    "prefill_cost_usd", "thinking_cost_usd", "answer_cost_usd", "correct",
]


def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})
    print(f"\nSaved {len(rows)} rows to {path}")


def agg_by_tier(rows: List[Dict[str, Any]], is_reasoning: bool) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for tier in TIER_ORDER:
        g = [r for r in rows if r["tier"] == tier and r["is_reasoning"] == is_reasoning]
        if not g:
            continue
        out[tier] = {
            "thinking_tokens": statistics.mean(r["thinking_tokens"] for r in g),
            "output_tokens": statistics.mean(r["output_tokens"] for r in g),
            "cost_usd": statistics.mean(r["cost_usd"] for r in g),
            "prefill_cost": statistics.mean(r["prefill_cost_usd"] for r in g),
            "thinking_cost": statistics.mean(r["thinking_cost_usd"] for r in g),
            "answer_cost": statistics.mean(r["answer_cost_usd"] for r in g),
            "accuracy": statistics.mean(r["correct"] for r in g),
            "n": len(g),
        }
    return out


def report(rows: List[Dict[str, Any]]) -> None:
    base = agg_by_tier(rows, False)
    reason = agg_by_tier(rows, True)
    print("\n=== Reasoning tax by difficulty tier ===")
    print(f"{'tier':14s} {'think tok':>9s} {'base $/req':>11s} {'reason $/req':>12s} "
          f"{'tax x':>6s} {'base acc':>8s} {'reason acc':>10s}")
    for tier in TIER_ORDER:
        b = base.get(tier)
        r = reason.get(tier)
        if not b or not r:
            continue
        tax = (r["cost_usd"] / b["cost_usd"]) if b["cost_usd"] else float("nan")
        print(f"{tier:14s} {r['thinking_tokens']:9.0f} "
              f"${b['cost_usd']*1000:10.4f}k ${r['cost_usd']*1000:11.4f}k "
              f"{tax:5.1f}x {b['accuracy']*100:7.0f}% {r['accuracy']*100:9.0f}%")
    # $/correct-answer (the honest metric when quality differs).
    print("\n$/correct-answer (cost per *correct* response; ∞ if accuracy 0):")
    for tier in TIER_ORDER:
        b, r = base.get(tier), reason.get(tier)
        if not b or not r:
            continue
        bc = b["cost_usd"] / b["accuracy"] if b["accuracy"] else float("inf")
        rc = r["cost_usd"] / r["accuracy"] if r["accuracy"] else float("inf")
        print(f"  {tier:14s} base ${bc*1000:8.4f}/k   reason ${rc*1000:8.4f}/k")


def make_charts(rows: List[Dict[str, Any]], reasoning_model: str, baseline_model: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping charts")
        return
    import numpy as np

    base = agg_by_tier(rows, False)
    reason = agg_by_tier(rows, True)
    tiers = [t for t in TIER_ORDER if t in base and t in reason]
    if not tiers:
        print("not enough data for charts")
        return
    rate = get_gpu_hourly_rate()
    x = np.arange(len(tiers))
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Tokens vs difficulty: reasoning thinking-tokens climb; baseline flat.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.2, [reason[t]["thinking_tokens"] for t in tiers], 0.4,
           label=f"{reasoning_model} thinking tokens", color="#7E57C2")
    ax.bar(x + 0.2, [base[t]["output_tokens"] for t in tiers], 0.4,
           label=f"{baseline_model} output tokens", color="#26A69A")
    ax.set_xticks(x); ax.set_xticklabels(tiers, rotation=15)
    ax.set_ylabel("tokens")
    ax.set_title("Tokens generated vs difficulty — reasoning breaks difficulty invariance\n"
                 "(thinking-token count grows with difficulty; baseline output stays flat)",
                 fontsize=11)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = CHARTS_DIR / "reasoning_tokens_vs_difficulty.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  saved {p}")

    # 2) Cost per request: reasoning vs baseline, with the tax annotated.
    fig, ax = plt.subplots(figsize=(9, 5))
    bcost = [base[t]["cost_usd"] * 1000 for t in tiers]
    rcost = [reason[t]["cost_usd"] * 1000 for t in tiers]
    ax.bar(x - 0.2, bcost, 0.4, label=baseline_model, color="#26A69A")
    ax.bar(x + 0.2, rcost, 0.4, label=reasoning_model, color="#7E57C2")
    for i, t in enumerate(tiers):
        if base[t]["cost_usd"]:
            tax = reason[t]["cost_usd"] / base[t]["cost_usd"]
            ax.text(x[i] + 0.2, rcost[i], f"{tax:.0f}x", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(tiers, rotation=15)
    ax.set_ylabel("$ per request (×10⁻³)")
    ax.set_title(rf"The reasoning tax: \$/request vs difficulty  (GPU \${rate:.2f}/hr)"
                 "\n(label = reasoning cost ÷ baseline cost)", fontsize=11)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = CHARTS_DIR / "reasoning_tax_cost.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  saved {p}")

    # 3) Latency decomposition (reasoning): prefill / thinking / answer dollars.
    fig, ax = plt.subplots(figsize=(9, 5))
    pf = [reason[t]["prefill_cost"] * 1000 for t in tiers]
    th = [reason[t]["thinking_cost"] * 1000 for t in tiers]
    an = [reason[t]["answer_cost"] * 1000 for t in tiers]
    ax.bar(tiers, pf, label="prefill", color="#90CAF9")
    ax.bar(tiers, th, bottom=pf, label="thinking", color="#7E57C2")
    ax.bar(tiers, an, bottom=[p_ + t_ for p_, t_ in zip(pf, th)], label="answer", color="#FFB74D")
    ax.set_ylabel("$ per request (×10⁻³)")
    ax.set_title(f"Where the dollars go — {reasoning_model}\n"
                 "(prefill negligible; thinking + a verbose worked answer both grow with difficulty)",
                 fontsize=10)
    ax.tick_params(axis="x", rotation=15)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = CHARTS_DIR / "reasoning_latency_decomposition.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  saved {p}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Measure the reasoning tax vs difficulty")
    ap.add_argument("--reasoning-model", default="deepseek-r1:7b")
    ap.add_argument("--baseline-model", default="qwen2.5:7b")
    ap.add_argument("--repeats", type=int, default=3, help="trials per tier per model")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="hard cap on output; reasoning can be long on hard tiers")
    ap.add_argument("--cooldown", type=float, default=0.0)
    ap.add_argument("--no-charts", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rate = get_gpu_hourly_rate()
    print(f"Reasoning-tax benchmark | reasoning={args.reasoning_model} | "
          f"baseline={args.baseline_model} | repeats={args.repeats} | "
          f"max_tokens={args.max_tokens} | GPU=${rate:.2f}/hr\n")

    with httpx.Client() as client:
        available = ollama_models(client)
        if not available:
            sys.exit("Ollama not reachable at " + OLLAMA_URL + " — is `ollama serve` running?")
        for m in (args.reasoning_model, args.baseline_model):
            if m not in available:
                sys.exit(f"Model '{m}' not found in Ollama. Pull it first:\n  ollama pull {m}")

        rows: List[Dict[str, Any]] = []
        rows += run_model(client, args.reasoning_model, True,
                          args.repeats, args.max_tokens, rate, args.cooldown)
        rows += run_model(client, args.baseline_model, False,
                          args.repeats, args.max_tokens, rate, args.cooldown)

    if not rows:
        print("No successful runs; nothing to save.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_csv(rows, RESULTS_DIR / f"reasoning_tax_{ts}.csv")
    report(rows)
    if not args.no_charts:
        print("\nCharts:")
        make_charts(rows, args.reasoning_model, args.baseline_model)


if __name__ == "__main__":
    main()
