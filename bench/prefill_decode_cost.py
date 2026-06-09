"""Decompose request cost into prefill dollars vs decode dollars.

A request holds the GPU for its full wall-clock duration, and the GPU is billed
by time, so the dollar cost of a request splits cleanly at time-to-first-token:

    prefill cost  = rate/3600 * ttft_s                  (whole prompt, one pass)
    decode  cost  = rate/3600 * (elapsed_s - ttft_s)    (output tokens, one by one)

The two phases have very different per-token economics, which this script makes
explicit:

  - Prefill is compute-bound and parallel: it ingests every input token in a
    single forward pass, so its cost-per-*input*-token is tiny.
  - Decode is memory-bound and sequential: one output token per forward pass,
    so its cost-per-*output*-token is much higher.

This reads the difficulty-invariance CSVs (the runs that capture per-request
ttft_s and elapsed_s), aggregates per prompt category, prints a table, writes a
decomposition CSV, and saves a stacked-bar chart.

Usage:
    python bench/prefill_decode_cost.py
    python bench/prefill_decode_cost.py results/difficulty/difficulty_invariance_*.csv
    GPU_HOURLY_RATE=1.20 python bench/prefill_decode_cost.py
"""

import sys
from pathlib import Path
from typing import List

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from cost_calculator import get_gpu_hourly_rate, prefill_decode_cost_split

RESULTS_DIR = Path("results/difficulty")
CHARTS_DIR = Path("charts")
OUT_CSV = RESULTS_DIR / "prefill_decode_cost.csv"
REQUIRED = {"ttft_s", "elapsed_s", "input_tokens", "output_tokens"}
# Same human-difficulty ordering used by the difficulty-invariance chart.
CATEGORY_ORDER = ["trivial", "recall", "creative", "reasoning", "hard_structured"]


def discover(args: List[str]) -> List[Path]:
    if args:
        return [Path(a) for a in args]
    return sorted(RESULTS_DIR.glob("difficulty_invariance_*.csv"))


def decompose(df: pd.DataFrame, rate: float) -> pd.DataFrame:
    """Add per-request prefill/decode dollar columns to a difficulty dataframe."""
    splits = df.apply(
        lambda r: prefill_decode_cost_split(rate, r["ttft_s"], r["elapsed_s"]),
        axis=1,
    )
    df = df.copy()
    df["prefill_cost_usd"] = [s["prefill_cost_usd"] for s in splits]
    df["decode_cost_usd"] = [s["decode_cost_usd"] for s in splits]
    df["request_cost_usd"] = [s["total_cost_usd"] for s in splits]
    df["prefill_fraction"] = [s["prefill_fraction"] for s in splits]
    # Per-phase, per-token cost surfaces the compute- vs memory-bound asymmetry.
    df["prefill_cost_per_input_token"] = df["prefill_cost_usd"] / df["input_tokens"].where(df["input_tokens"] > 0)
    df["decode_cost_per_output_token"] = df["decode_cost_usd"] / df["output_tokens"].where(df["output_tokens"] > 0)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("category", as_index=False).agg(
        n=("category", "size"),
        prefill_cost_usd=("prefill_cost_usd", "mean"),
        decode_cost_usd=("decode_cost_usd", "mean"),
        request_cost_usd=("request_cost_usd", "mean"),
        prefill_fraction=("prefill_fraction", "mean"),
        prefill_cost_per_input_token=("prefill_cost_per_input_token", "mean"),
        decode_cost_per_output_token=("decode_cost_per_output_token", "mean"),
    )
    order = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    agg["__o"] = agg["category"].map(lambda c: order.get(c, len(order)))
    return agg.sort_values("__o").drop(columns="__o").reset_index(drop=True)


def print_table(agg: pd.DataFrame, rate: float) -> None:
    print(f"\nPrefill vs decode cost per request  (GPU rate = ${rate:.2f}/hr)\n")
    header = (f"  {'category':16s} {'n':>3s} {'prefill$':>11s} {'decode$':>11s} "
              f"{'total$':>11s} {'prefill%':>8s} {'$/in-tok':>11s} {'$/out-tok':>11s}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, r in agg.iterrows():
        print(f"  {r['category']:16s} {int(r['n']):3d} "
              f"{r['prefill_cost_usd']:11.2e} {r['decode_cost_usd']:11.2e} "
              f"{r['request_cost_usd']:11.2e} {r['prefill_fraction']*100:7.1f}% "
              f"{r['prefill_cost_per_input_token']:11.2e} {r['decode_cost_per_output_token']:11.2e}")
    # Headline asymmetry, averaged across categories.
    pin = agg["prefill_cost_per_input_token"].mean()
    pout = agg["decode_cost_per_output_token"].mean()
    if pin > 0:
        print(f"\n  A decode (output) token costs ~{pout / pin:.0f}x a prefill (input) token "
              f"of GPU time on this setup.")


def make_chart(agg: pd.DataFrame, rate: float) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping chart")
        return

    cats = agg["category"].tolist()
    prefill = agg["prefill_cost_usd"] * 1e6  # show in micro-dollars for readability
    decode = agg["decode_cost_usd"] * 1e6

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(cats, decode, label="decode", color="#FF5722")
    ax.bar(cats, prefill, bottom=decode, label="prefill", color="#009688")
    for i, (_, r) in enumerate(agg.iterrows()):
        ax.text(i, (r["request_cost_usd"]) * 1e6, f"{r['prefill_fraction']*100:.0f}%\nprefill",
                ha="center", va="bottom", fontsize=8, color="#444")
    ax.set_title(f"Per-request cost: prefill vs decode — qwen2.5:7b\n"
                 f"(GPU rate = ${rate:.2f}/hr; bar label = prefill share)", fontsize=11)
    ax.set_ylabel(r"cost per request  ($ \times 10^{-6}$)")
    ax.set_xlabel("prompt category (left=easy, right=hard for humans)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.margins(y=0.15)
    fig.tight_layout()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    path = CHARTS_DIR / "prefill_vs_decode_cost.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  saved {path}")


def main() -> None:
    rate = get_gpu_hourly_rate()
    paths = discover(sys.argv[1:])
    if not paths:
        print(f"No difficulty CSVs found in {RESULTS_DIR}/. Run difficulty_invariance.py first.")
        return

    frames = []
    for p in paths:
        if not p.exists():
            print(f"  {p}: not found, skipping")
            continue
        df = pd.read_csv(p)
        missing = REQUIRED - set(df.columns)
        if missing:
            print(f"  {p}: missing columns {sorted(missing)}, skipping")
            continue
        frames.append(decompose(df, rate))

    if not frames:
        print("No usable difficulty CSVs (need ttft_s + elapsed_s).")
        return

    df = pd.concat(frames, ignore_index=True)
    agg = summarize(df)
    print_table(agg, rate)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(OUT_CSV, index=False)
    print(f"\n  saved {OUT_CSV}  ({len(df)} requests across {len(agg)} categories)")

    make_chart(agg, rate)


if __name__ == "__main__":
    main()
