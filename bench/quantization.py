"""Quantization sweep: what does precision cost?

Compares the same model at two quantization levels — qwen2.5:7b (Q4_K_M, ~4.7 GB)
vs qwen2.5:7b-instruct-q8_0 (Q8, ~8.1 GB) — on the same hardware. Q8 keeps more
bits per weight, so it must move ~1.7x more memory per token: expect lower
throughput and higher $/token, in exchange for being closer to full precision.

Measures, per model:
  - throughput (tok/s) and $/M tokens, on medium-length generations
  - accuracy on a small numeric ladder (a light quality check, not a full eval)
  - on-disk size (a proxy for VRAM footprint)

    ollama pull qwen2.5:7b-instruct-q8_0
    python bench/quantization.py
"""

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from cost_calculator import cost_per_million_tokens, get_gpu_hourly_rate

OLLAMA = "http://localhost:11434"
CHARTS_DIR = Path("charts")
RESULTS_DIR = Path("results/quantization")

# Longer prompts to get a real throughput read.
THROUGHPUT_PROMPTS = [
    "Explain how a CPU cache works, in about 150 words.",
    "Describe the water cycle step by step.",
    "Write a short story about a lighthouse keeper, ~150 words.",
]
# Numeric ladder for the light quality check (same answers regardless of quant).
QUALITY = [
    ("A train travels 60 km in 1.5 hours. Average speed in km/h? Just the number.", "40"),
    ("Natalia sold clips to 48 friends, then half as many next month. Total? Just the number.", "72"),
    ("A store had 120 apples. Sold 1/3, then 1/4 of the rest. How many left? Just the number.", "60"),
    ("What is 17 * 23? Just the number.", "391"),
]
SIZE_GB = {"qwen2.5:7b": 4.7, "qwen2.5:7b-instruct-q8_0": 8.1}


def gen(client, model, prompt, max_tokens):
    r = client.post(f"{OLLAMA}/api/generate", json={
        "model": model, "prompt": prompt, "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0}}, timeout=300)
    d = r.json()
    if "error" in d:
        raise RuntimeError(d["error"])
    out, dur = d.get("eval_count", 0), d.get("eval_duration", 0)
    tps = out / (dur / 1e9) if dur > 0 else 0.0
    return d.get("response", ""), out, tps


def last_int(s):
    nums = re.findall(r"-?\d+", s.replace(",", ""))
    return nums[-1] if nums else None


def run_model(client, model, rate) -> Dict:
    print(f"\n{model}:")
    # warm up
    gen(client, model, "hi", 8)
    tps_list = []
    for p in THROUGHPUT_PROMPTS:
        _, out, tps = gen(client, model, p, 256)
        tps_list.append(tps)
        print(f"  throughput: {out}tok @ {tps:.1f} tok/s")
    correct = 0
    for prompt, ans in QUALITY:
        resp, _, _ = gen(client, model, prompt, 256)
        ok = last_int(resp) == ans
        correct += ok
        print(f"  quality: {'OK ' if ok else 'XX '} expected {ans}, got {last_int(resp)}")
    mean_tps = statistics.mean(tps_list)
    return {
        "model": model,
        "tok_per_sec": mean_tps,
        "cost_per_million_tokens": cost_per_million_tokens(rate, mean_tps),
        "accuracy": correct / len(QUALITY),
        "size_gb": SIZE_GB.get(model, 0),
    }


def make_chart(rows: List[Dict]):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return
    labels = ["Q4\n(qwen2.5:7b)", "Q8\n(q8_0)"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    tps = [r["tok_per_sec"] for r in rows]
    cost = [r["cost_per_million_tokens"] for r in rows]
    size = [r["size_gb"] for r in rows]
    for ax, data, title, color, unit in [
        (axes[0], tps, "Throughput (tok/s)\nhigher = better", "#26A69A", ""),
        (axes[1], cost, "$/M tokens\nlower = cheaper", "#FF7043", "$"),
        (axes[2], size, "On-disk size (GB)\n≈ VRAM footprint", "#5C6BC0", ""),
    ]:
        ax.bar(labels, data, color=color, width=0.6)
        ax.set_title(title, fontsize=10)
        for i, v in enumerate(data):
            ax.text(i, v, f"{unit}{v:.1f}" if unit or v < 100 else f"{v:.0f}", ha="center", va="bottom")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Quantization: Q4 vs Q8 (same model, same hardware)", fontweight="bold")
    fig.tight_layout()
    CHARTS_DIR.mkdir(exist_ok=True)
    p = CHARTS_DIR / "quantization_q4_vs_q8.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); print(f"\nsaved {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q4", default="qwen2.5:7b")
    ap.add_argument("--q8", default="qwen2.5:7b-instruct-q8_0")
    args = ap.parse_args()
    rate = get_gpu_hourly_rate()

    with httpx.Client() as client:
        tags = [m["name"] for m in client.get(f"{OLLAMA}/api/tags", timeout=10).json().get("models", [])]
        for m in (args.q4, args.q8):
            if m not in tags:
                sys.exit(f"Model {m} not pulled. Run: ollama pull {m}")
        rows = [run_model(client, args.q4, rate), run_model(client, args.q8, rate)]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "quantization.json").write_text(json.dumps(rows, indent=2))
    print("\n=== Q4 vs Q8 ===")
    for r in rows:
        print(f"  {r['model']:32s} {r['tok_per_sec']:6.1f} tok/s  "
              f"${r['cost_per_million_tokens']:6.2f}/M  acc {r['accuracy']*100:3.0f}%  {r['size_gb']}GB")
    q4, q8 = rows
    print(f"\nQ8 is {q4['tok_per_sec']/q8['tok_per_sec']:.2f}x slower and "
          f"{q8['cost_per_million_tokens']/q4['cost_per_million_tokens']:.2f}x the $/token of Q4, "
          f"for {q8['size_gb']/q4['size_gb']:.2f}x the memory.")
    make_chart(rows)


if __name__ == "__main__":
    main()
