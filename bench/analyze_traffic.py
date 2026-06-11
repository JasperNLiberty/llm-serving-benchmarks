"""Analyze a traffic-sim run: recover the load pattern and show its cost/latency cost.

Reads the JSONL emitted by traffic_sim.py and produces, per run, a 4-panel figure:

  1. Arrival rate over time      -- recovered from the logs; proves the log captures
                                    the traffic shape.
  2. Latency p50/p95/p99 vs load -- tail latency blowing up at peak while the mean
                                    stays calm is the headline production lesson.
  3. Offered load vs latency     -- the queueing knee / throughput-collapse point.
  4. Effective vs nominal $/token-- cost at real utilization vs the spec-sheet number.

Also prints a Little's Law sanity check (L ~= lambda * W).

Usage:
    python bench/analyze_traffic.py                       # latest run of each pattern
    python bench/analyze_traffic.py results/traffic/ramp_20260610_xxxx.jsonl
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from cost_calculator import cost_per_million_tokens, get_gpu_hourly_rate

TRAFFIC_DIR = Path("results/traffic")
CHARTS_DIR = Path("charts")
BIN_S = 5.0   # time-bin width for rate/latency-over-time panels


def discover(args: List[str]) -> List[Path]:
    if args:
        return [Path(a) for a in args]
    # latest run per pattern
    latest: Dict[str, Path] = {}
    for p in sorted(TRAFFIC_DIR.glob("*.jsonl")):
        pattern = p.stem.split("_")[0]
        latest[pattern] = p   # sorted -> last wins = newest timestamp
    return list(latest.values())


def load(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def percentile_series(df: pd.DataFrame, value: str, bin_col: str, pct: float) -> pd.Series:
    return df.groupby(bin_col)[value].quantile(pct / 100)


def littles_law(df: pd.DataFrame) -> str:
    ok = df[df["status"] == "ok"]
    if ok.empty:
        return "Little's Law: no successful requests"
    completion = ok["dispatch_offset_s"] + ok["latency_s"]
    window = completion.max() - ok["dispatch_offset_s"].min()
    lam = len(ok) / window if window > 0 else 0.0          # throughput, req/s
    W = ok["latency_s"].mean()                              # mean time in system
    L_pred = lam * W                                        # predicted avg in-flight
    L_meas = df["in_flight_at_dispatch"].mean()             # measured (PASTA)
    return (f"Little's Law: L = lambda x W -> {lam:.2f} req/s x {W:.2f}s = "
            f"{L_pred:.2f} predicted vs {L_meas:.2f} measured avg in-flight")


def analyze_one(path: Path, rate: float) -> Dict:
    df = load(path)
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    pattern = meta.get("pattern", path.stem.split("_")[0])

    df["bin"] = (df["dispatch_offset_s"] // BIN_S) * BIN_S
    ok = df[df["status"] == "ok"].copy()

    # Per-bin arrival rate (req/s) and latency percentiles (ok only).
    rate_by_bin = df.groupby("bin").size() / BIN_S
    p50 = percentile_series(ok, "latency_s", "bin", 50)
    p95 = percentile_series(ok, "latency_s", "bin", 95)
    p99 = percentile_series(ok, "latency_s", "bin", 99)

    # Nominal vs effective $/M tokens -- the utilization (idle-GPU) story.
    #   nominal   = cost per token while the GPU is actively decoding (spec sheet),
    #               from the busy single-stream throughput.
    #   effective = you rent the GPU for the whole window but only serve tokens part
    #               of the time, so cost/token = (rate x window) / tokens_served.
    # effective / nominal is the under-utilization multiplier ("$/token is Nx spec").
    nominal_cost = cost_per_million_tokens(rate, ok["tokens_per_sec"].median()) if not ok.empty else 0
    if not ok.empty:
        completion = ok["dispatch_offset_s"] + ok["latency_s"]
        window_s = max(1e-6, completion.max() - ok["dispatch_offset_s"].min())
        total_tokens = ok["output_tokens"].fillna(0).sum()
        gpu_cost_window = rate / 3600 * window_s
        effective_cost = (gpu_cost_window / total_tokens * 1e6) if total_tokens else 0
        # Per-bin effective cost over time: idle bins serve few tokens -> dear tokens.
        tokens_by_bin = ok.groupby("bin")["output_tokens"].sum()
        eff_by_bin = (rate / 3600 * BIN_S) / tokens_by_bin * 1e6
        eff_by_bin = eff_by_bin[tokens_by_bin > 0]
    else:
        effective_cost, eff_by_bin = 0, pd.Series(dtype=float)

    _make_chart(pattern, rate_by_bin, p50, p95, p99, ok, eff_by_bin, nominal_cost, rate)

    return {
        "pattern": pattern,
        "n": len(df),
        "errors": int((df["status"] != "ok").sum()),
        "p50": ok["latency_s"].median() if not ok.empty else None,
        "p95": ok["latency_s"].quantile(0.95) if not ok.empty else None,
        "p99": ok["latency_s"].quantile(0.99) if not ok.empty else None,
        "peak_inflight": int(df["in_flight_at_dispatch"].max()),
        "nominal_cost_per_m": round(nominal_cost, 2),
        "effective_cost_per_m": round(effective_cost, 2),
        "littles": littles_law(df),
    }


def _make_chart(pattern, rate_by_bin, p50, p95, p99, ok, eff_by_bin, nominal, rate) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping chart")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Traffic pattern: {pattern}  (model=llama3.2:1b, "
                 f"GPU rate=${rate:.2f}/hr)", fontsize=13, fontweight="bold")

    # 1. Arrival rate over time
    ax = axes[0, 0]
    ax.plot(rate_by_bin.index, rate_by_bin.values, marker="o", color="#2196F3")
    ax.set_title("1. Arrival rate over time (recovered from logs)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("requests / s"); ax.grid(True, alpha=0.3)

    # 2. Latency percentiles over time + load
    ax = axes[0, 1]
    ax.plot(p50.index, p50.values, marker="o", label="p50", color="#4CAF50")
    ax.plot(p95.index, p95.values, marker="s", label="p95", color="#FF9800")
    ax.plot(p99.index, p99.values, marker="^", label="p99", color="#F44336")
    ax.set_title("2. Latency percentiles over time")
    ax.set_xlabel("time (s)"); ax.set_ylabel("latency (s)"); ax.legend(); ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(rate_by_bin.index, rate_by_bin.values, ls="--", color="#2196F3", alpha=0.5)
    ax2.set_ylabel("req/s (dashed)", color="#2196F3")

    # 3. Offered load vs latency (the knee)
    ax = axes[1, 0]
    ax.scatter(ok["in_flight_at_dispatch"], ok["latency_s"], alpha=0.4, s=18, color="#7C4DFF")
    knee = ok.groupby("in_flight_at_dispatch")["latency_s"].median()
    ax.plot(knee.index, knee.values, color="#311B92", marker="o", label="median")
    ax.set_title("3. Offered load vs latency (queueing knee)")
    ax.set_xlabel("in-flight requests at dispatch"); ax.set_ylabel("latency (s)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 4. Effective $/M over time vs nominal -- the idle-GPU penalty.
    ax = axes[1, 1]
    ax.plot(eff_by_bin.index, eff_by_bin.values, marker="o", color="#E91E63",
            label="effective (GPU rented / tokens served)")
    ax.axhline(nominal, ls="--", color="#555",
               label=f"nominal (busy decode) = ${nominal:.1f}/M")
    ax.set_title("4. Effective vs nominal $/M tokens over time\n"
                 "(gap = paying for idle GPU; log scale)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("$ / M tokens (log)")
    ax.set_yscale("log")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    out = CHARTS_DIR / f"traffic_{pattern}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def main() -> None:
    rate = get_gpu_hourly_rate()
    paths = discover(sys.argv[1:])
    if not paths:
        print(f"No traffic runs found in {TRAFFIC_DIR}/. Run traffic_sim.py first.")
        return

    summaries = []
    for path in paths:
        if not path.exists():
            print(f"  {path}: not found"); continue
        print(f"\n=== {path.name} ===")
        s = analyze_one(path, rate)
        print(f"  {s['n']} reqs, {s['errors']} errors | "
              f"p50={s['p50']:.2f}s p95={s['p95']:.2f}s p99={s['p99']:.2f}s | "
              f"peak in-flight={s['peak_inflight']}")
        print(f"  nominal ${s['nominal_cost_per_m']}/M vs effective "
              f"${s['effective_cost_per_m']}/M tokens")
        print(f"  {s['littles']}")
        summaries.append(s)

    if summaries:
        print("\n=== cross-pattern summary ===")
        print(pd.DataFrame(summaries)[
            ["pattern", "n", "errors", "p50", "p95", "p99", "peak_inflight",
             "nominal_cost_per_m", "effective_cost_per_m"]
        ].to_string(index=False))


if __name__ == "__main__":
    main()
