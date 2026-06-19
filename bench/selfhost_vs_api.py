"""Self-hosted vs API crossover: at what traffic does owning a GPU beat paying per token?

Two cost structures:
  - API:        pay per token, $0 when idle  -> cost is linear in volume.
  - Self-hosted: rent a GPU by the hour, pay whether busy or not -> cost is a
                 step function (one more GPU each time you exhaust capacity), and
                 its *effective* $/token depends entirely on UTILIZATION.

The punchline (and the link to examples/02): self-hosting only wins once you push
enough steady traffic to keep the GPU busy. At low utilization the idle GPU tax
shoves the crossover far to the right. This script plots monthly cost vs traffic
for an API and for self-hosting at two utilizations, and prints the crossovers.

    python bench/selfhost_vs_api.py
"""

import argparse
import math
from pathlib import Path

CHARTS_DIR = Path("charts")
HOURS_PER_MONTH = 720
SECONDS_PER_MONTH = HOURS_PER_MONTH * 3600


def self_hosted_monthly(tokens_per_month, gpu_hourly, throughput_tok_s, utilization):
    """Step-function cost: ceil(load / effective-capacity) GPUs × hourly × 720."""
    capacity = throughput_tok_s * SECONDS_PER_MONTH * utilization  # tokens/gpu/month
    gpus = max(1, math.ceil(tokens_per_month / capacity))
    return gpus * gpu_hourly * HOURS_PER_MONTH


def first_crossover(api_price, gpu_hourly, throughput_tok_s, utilization, hi=1e12):
    """Smallest tokens/month where self-hosted <= API (1-GPU regime)."""
    # 1-GPU self-hosted is flat = gpu_hourly*720; API = V*price. Cross at V*=flat/price.
    flat = gpu_hourly * HOURS_PER_MONTH
    return flat / api_price


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-hourly", type=float, default=0.80)
    ap.add_argument("--throughput", type=float, default=2500.0,
                    help="served tokens/sec per GPU (batched). Assumption — note it.")
    ap.add_argument("--api-price-m", type=float, default=0.30,
                    help="API price $/M tokens (a 7-8B-class hosted endpoint)")
    args = ap.parse_args()

    api_price = args.api_price_m / 1e6  # $/token
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy needed for the chart")
        return

    # x-axis: tokens/day, log scale
    tok_day = np.logspace(6, 11, 200)            # 1M .. 100B tokens/day
    tok_month = tok_day * 30

    api = tok_month * api_price
    sh_full = [self_hosted_monthly(v, args.gpu_hourly, args.throughput, 1.0) for v in tok_month]
    sh_real = [self_hosted_monthly(v, args.gpu_hourly, args.throughput, 0.40) for v in tok_month]

    # One crossover: a single GPU's flat monthly cost equals the API bill. (It's
    # the same for both utilizations — a 1-GPU bill doesn't depend on how busy it
    # is.) Utilization instead decides how soon you must buy the *next* GPU.
    x_cross = (args.gpu_hourly * HOURS_PER_MONTH / api_price) / 30  # tokens/day

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.loglog(tok_day, api, label=f"API (${args.api_price_m:.2f}/M, pay-per-token)", lw=2.5, color="#26A69A")
    ax.loglog(tok_day, sh_full, label="self-hosted @ 100% utilization", lw=2, color="#7E57C2")
    ax.loglog(tok_day, sh_real, label="self-hosted @ 40% utilization (realistic)", lw=2, ls="--", color="#EF5350")
    ax.axvline(x_cross, color="#555", alpha=0.5, ls=":")
    ax.annotate(f"crossover\n{x_cross/1e6:.0f}M tok/day", xy=(x_cross, api[np.argmin(np.abs(tok_day - x_cross))]),
                fontsize=9, color="#333", ha="center", va="bottom")
    ax.set_xlabel("traffic (tokens / day)")
    ax.set_ylabel("cost ($ / month)")
    ax.set_title(f"Self-hosted vs API  (GPU ${args.gpu_hourly:.2f}/hr, {args.throughput:.0f} tok/s served)\n"
                 "left of crossover the API is cheaper; low utilization makes self-hosting step up sooner", fontsize=10)
    ax.legend(); ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    CHARTS_DIR.mkdir(exist_ok=True)
    p = CHARTS_DIR / "selfhost_vs_api.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); print(f"saved {p}")

    print(f"\nAssumptions: GPU ${args.gpu_hourly}/hr, {args.throughput:.0f} tok/s served, API ${args.api_price_m}/M")
    print(f"Crossover (1 GPU vs API): {x_cross/1e6:.0f}M tokens/day "
          f"(~{x_cross/256/1e3:.0f}k requests/day at 256 tok each)")
    print("Below it the API is cheaper — a whole GPU is wasteful for trickle traffic.")
    print("Above it self-hosting wins on flat cost, BUT low utilization forces extra GPUs")
    print("sooner (the 40% curve steps above the 100% curve), eroding the advantage.")


if __name__ == "__main__":
    main()
