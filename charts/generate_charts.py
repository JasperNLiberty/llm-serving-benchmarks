"""
Generate and save benchmark charts to the charts/ directory.

Reads from results/ CSVs, computes $/M tokens via GPU_HOURLY_RATE (default $0.80/hr),
and writes PNG files that can be committed and shared without re-running benchmarks.

Usage:
    python charts/generate_charts.py
    GPU_HOURLY_RATE=1.20 python charts/generate_charts.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "bench"))
from cost_calculator import cost_per_million_tokens, get_gpu_hourly_rate

CHARTS_DIR = Path(__file__).parent
RESULTS_DIR = Path(__file__).parent.parent / "results"

GPU_RATE = get_gpu_hourly_rate()

COLORS = {"ollama": "#2196F3", "mlx": "#FF9800", "cpu": "#9E9E9E", "gpu": "#4CAF50"}
MODEL_COLORS = {"llama3.2:1b": "#7C4DFF", "qwen2.5:7b": "#F44336"}


def add_cost(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cost_per_million_tokens"] = df["tokens_per_sec"].apply(
        lambda tps: cost_per_million_tokens(GPU_RATE, tps) if pd.notna(tps) and tps > 0 else None
    )
    return df


def save(fig: plt.Figure, name: str) -> None:
    path = CHARTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(Path(__file__).parent.parent)}")


def chart_ollama_vs_mlx() -> None:
    csv_files = sorted((RESULTS_DIR / "comparison").glob("ollama_vs_mlx_*.csv"))
    if not csv_files:
        print("  skipping ollama_vs_mlx charts: no comparison CSV found")
        return

    df = add_cost(pd.read_csv(csv_files[-1]))
    group = df.groupby(["backend", "concurrency"], as_index=False).agg(
        tokens_per_sec=("tokens_per_sec", "mean"),
        cost_per_million_tokens=("cost_per_million_tokens", "mean"),
        avg_latency=("avg_latency", "mean"),
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Ollama vs MLX — qwen2.5:7b on Apple M1 Max", fontsize=13, fontweight="bold")

    for backend, grp in group.groupby("backend"):
        color = COLORS.get(backend, "#999")
        ax1.plot(grp["concurrency"], grp["tokens_per_sec"], marker="o", label=backend, color=color)
        ax2.plot(grp["concurrency"], grp["cost_per_million_tokens"], marker="o", label=backend, color=color)

    ax1.set_title("Throughput")
    ax1.set_xlabel("Concurrency")
    ax1.set_ylabel("Tokens / sec")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title(f"Cost  (GPU rate = ${GPU_RATE:.2f}/hr)")
    ax2.set_xlabel("Concurrency")
    ax2.set_ylabel("$ / M tokens")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save(fig, "ollama_vs_mlx_throughput_and_cost.png")

    # Standalone $/M tokens bar chart at concurrency=1
    c1 = group[group["concurrency"] == 1].copy()
    fig2, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        c1["backend"],
        c1["cost_per_million_tokens"],
        color=[COLORS.get(b, "#999") for b in c1["backend"]],
        width=0.4,
    )
    for bar, val in zip(bars, c1["cost_per_million_tokens"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, f"${val:.2f}", ha="center", va="bottom", fontsize=11)
    ax.set_title(f"$/M tokens by Backend — qwen2.5:7b, concurrency=1\n(GPU rate = ${GPU_RATE:.2f}/hr)", fontsize=11)
    ax.set_ylabel("$ / M tokens")
    ax.set_ylim(0, c1["cost_per_million_tokens"].max() * 1.3)
    ax.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    save(fig2, "cost_per_million_tokens_by_backend.png")


def chart_model_size() -> None:
    cpu_path = RESULTS_DIR / "by_device" / "device_cpu.csv"
    gpu_path = RESULTS_DIR / "by_device" / "device_gpu.csv"
    if not cpu_path.exists() or not gpu_path.exists():
        print("  skipping model-size charts: by_device CSVs not found")
        return

    df = add_cost(pd.concat([pd.read_csv(cpu_path), pd.read_csv(gpu_path)]))

    # Aggregate by model + device across all prompts, at concurrency=1 for a clean single-variable comparison
    c1 = df[df["concurrency"] == 1].groupby(["model", "device"], as_index=False).agg(
        tokens_per_sec=("tokens_per_sec", "mean"),
        cost_per_million_tokens=("cost_per_million_tokens", "mean"),
    )

    models = c1["model"].unique()
    devices = c1["device"].unique()
    x = range(len(models))
    width = 0.35

    # Throughput + cost side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Throughput & Cost by Model Size — Apple M1 Max, concurrency=1", fontsize=13, fontweight="bold")

    for i, device in enumerate(devices):
        subset = c1[c1["device"] == device]
        offset = (i - len(devices) / 2 + 0.5) * width
        tps_vals = [subset[subset["model"] == m]["tokens_per_sec"].values[0] if len(subset[subset["model"] == m]) else 0 for m in models]
        cost_vals = [subset[subset["model"] == m]["cost_per_million_tokens"].values[0] if len(subset[subset["model"] == m]) else 0 for m in models]
        color = COLORS.get(device, "#999")
        ax1.bar([xi + offset for xi in x], tps_vals, width, label=device, color=color)
        ax2.bar([xi + offset for xi in x], cost_vals, width, label=device, color=color)

    ax1.set_title("Throughput")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(models, rotation=10)
    ax1.set_ylabel("Tokens / sec")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    ax2.set_title(f"Cost  (GPU rate = ${GPU_RATE:.2f}/hr)")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(models, rotation=10)
    ax2.set_ylabel("$ / M tokens")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save(fig, "throughput_and_cost_vs_model_size.png")

    # Standalone $/M tokens by model — headline visual (GPU device only for clean comparison)
    gpu_only = c1[c1["device"] == "gpu"].copy()
    fig3, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        gpu_only["model"],
        gpu_only["cost_per_million_tokens"],
        color=[MODEL_COLORS.get(m, "#999") for m in gpu_only["model"]],
        width=0.4,
    )
    for bar, val in zip(bars, gpu_only["cost_per_million_tokens"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"${val:.2f}", ha="center", va="bottom", fontsize=11)
    ax.set_title(f"$/M tokens by Model Size — GPU (Metal), concurrency=1\n(GPU rate = ${GPU_RATE:.2f}/hr)", fontsize=11)
    ax.set_ylabel("$ / M tokens")
    ax.set_ylim(0, gpu_only["cost_per_million_tokens"].max() * 1.3)
    ax.grid(axis="y", alpha=0.3)
    fig3.tight_layout()
    save(fig3, "cost_per_million_tokens_by_model.png")


if __name__ == "__main__":
    print(f"Generating charts (GPU_HOURLY_RATE=${GPU_RATE:.2f}/hr)...")
    chart_ollama_vs_mlx()
    chart_model_size()
    print("Done.")
