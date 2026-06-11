"""Open-loop traffic simulator: drive the gateway with a time-varying request stream.

Unlike the closed-loop sweeps (which only send the next request after the previous
finishes), this fires requests on a schedule *regardless* of how many are still in
flight -- so a queue can actually build up when arrivals outrun the server, which is
exactly what makes the load/latency/cost dynamics interesting.

Arrivals follow a non-homogeneous Poisson process: at time t the instantaneous rate
is lambda(t), and inter-arrival gaps are Exp(lambda(t)). Four patterns:

  poisson : constant rate            -- baseline; clean Little's Law check
  diurnal : sine "day" cycle         -- trough -> peak -> trough; capacity planning
  ramp    : linearly rising rate     -- find the throughput-collapse knee
  bursty  : quiet baseline + spikes  -- stress queueing and tail latency

Each request hits the gateway's non-streaming /ollama/chat (which returns token
counts and cost and is queued behind the gateway's concurrency semaphore), carries
an X-Request-ID that the gateway echoes into its own JSON log (so client and server
logs join), and is recorded as one JSON line in results/traffic/<pattern>_<ts>.jsonl.

Defaults use llama3.2:1b with short outputs: the point is traffic dynamics, and the
1B's higher serving capacity lets the arrival rate actually cross the system's knee
within a few minutes (a 7B on M1 saturates near ~0.15 req/s).

Usage:
    python bench/traffic_sim.py --pattern diurnal --duration 180 --peak-rate 4
    python bench/traffic_sim.py --pattern ramp --duration 150 --base-rate 0.5 --peak-rate 6
    python bench/traffic_sim.py --pattern bursty --duration 180 --base-rate 0.5 --peak-rate 6
    python bench/traffic_sim.py --pattern poisson --duration 120 --base-rate 2
"""

import argparse
import asyncio
import json
import math
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from cost_calculator import get_gpu_hourly_rate

GATEWAY_URL = "http://127.0.0.1:8000/ollama/chat"
RESULTS_DIR = Path("results/traffic")

# Short, varied prompts -- output length is capped low so the server can sustain a
# few req/s and the arrival rate can actually exceed capacity.
PROMPTS = [
    "Name a color.",
    "Give one tip for better sleep.",
    "Translate 'good morning' to French.",
    "What is 17 + 25?",
    "Suggest a name for a cat.",
    "Write a haiku about rain.",
]


def rate_at(t: float, args: argparse.Namespace) -> float:
    """Instantaneous arrival rate lambda(t), in requests/second."""
    base, peak, dur = args.base_rate, args.peak_rate, args.duration
    if args.pattern == "poisson":
        return base
    if args.pattern == "diurnal":
        # one "day" == duration; trough at the ends, single peak in the middle.
        return base + (peak - base) * (0.5 - 0.5 * math.cos(2 * math.pi * t / dur))
    if args.pattern == "ramp":
        return base + (peak - base) * (t / dur)
    if args.pattern == "bursty":
        in_burst = (t % args.burst_period) < args.burst_len
        return peak if in_burst else base
    raise ValueError(args.pattern)


def generate_arrivals(args: argparse.Namespace) -> List[float]:
    """Arrival offsets (s from start) via thinning-free NHPP: gap ~ Exp(lambda(t))."""
    arrivals: List[float] = []
    t = 0.0
    while t < args.duration:
        rate = max(1e-6, rate_at(t, args))
        t += random.expovariate(rate)
        if t < args.duration:
            arrivals.append(t)
    return arrivals


async def do_request(
    client: httpx.AsyncClient, prompt: str, model: str, max_tokens: int,
    arrival_offset: float, start: float, state: Dict[str, int], rows: List[Dict[str, Any]],
) -> None:
    request_id = uuid.uuid4().hex[:12]
    state["in_flight"] += 1
    in_flight_now = state["in_flight"]          # offered concurrency at dispatch
    dispatch = time.perf_counter() - start
    row: Dict[str, Any] = {
        "request_id": request_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "arrival_offset_s": round(arrival_offset, 4),
        "dispatch_offset_s": round(dispatch, 4),
        "in_flight_at_dispatch": in_flight_now,
        "model": model,
    }
    try:
        r = await client.post(
            GATEWAY_URL,
            json={"model": model, "prompt": prompt, "max_tokens": max_tokens},
            headers={"X-Request-ID": request_id},
            timeout=180.0,
        )
        latency = time.perf_counter() - start - dispatch
        r.raise_for_status()
        data = r.json()
        row.update({
            "status": "ok",
            "http_status": r.status_code,
            "latency_s": round(latency, 4),
            "input_tokens": data.get("input_tokens"),
            "output_tokens": data.get("output_tokens"),
            "tokens_per_sec": data.get("tokens_per_sec"),
            "cost_usd": data.get("cost_usd"),
        })
    except Exception as exc:
        latency = time.perf_counter() - start - dispatch
        row.update({
            "status": "error",
            "http_status": getattr(getattr(exc, "response", None), "status_code", None),
            "latency_s": round(latency, 4),
            "error": str(exc)[:200],
        })
    finally:
        state["in_flight"] -= 1
    rows.append(row)


async def run(args: argparse.Namespace) -> List[Dict[str, Any]]:
    arrivals = generate_arrivals(args)
    print(f"Scheduled {len(arrivals)} arrivals over {args.duration}s "
          f"(pattern={args.pattern}, base={args.base_rate}, peak={args.peak_rate} req/s)")
    if not arrivals:
        return []

    rows: List[Dict[str, Any]] = []
    state = {"in_flight": 0}
    tasks: set = set()
    next_report = 10.0

    async with httpx.AsyncClient() as client:
        # Prewarm so the first request doesn't eat cold-start load time.
        try:
            await client.post(GATEWAY_URL, json={"model": args.model, "prompt": "hi",
                              "max_tokens": 4}, timeout=180.0)
        except Exception as exc:
            print(f"  prewarm warning: {exc}")

        start = time.perf_counter()
        for arrival in arrivals:
            now = time.perf_counter() - start
            if arrival > now:
                await asyncio.sleep(arrival - now)
            prompt = random.choice(PROMPTS)
            task = asyncio.create_task(
                do_request(client, prompt, args.model, args.max_tokens,
                           arrival, start, state, rows)
            )
            tasks.add(task)
            task.add_done_callback(tasks.discard)

            elapsed = time.perf_counter() - start
            if elapsed >= next_report:
                print(f"  t={elapsed:5.0f}s | sent={len(rows)+len(tasks)} | "
                      f"in_flight={state['in_flight']} | done={len(rows)}")
                next_report += 10.0

        print(f"  all arrivals dispatched; draining {len(tasks)} in-flight requests...")
        if tasks:
            await asyncio.gather(*tasks)
    return rows


def save(rows: List[Dict[str, Any]], args: argparse.Namespace) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{args.pattern}_{ts}.jsonl"
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    # Sidecar metadata so the run is self-describing for analysis.
    meta = {
        "pattern": args.pattern, "duration_s": args.duration,
        "base_rate": args.base_rate, "peak_rate": args.peak_rate,
        "burst_period": args.burst_period, "burst_len": args.burst_len,
        "model": args.model, "max_tokens": args.max_tokens,
        "gpu_hourly_rate_usd": get_gpu_hourly_rate(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_requests": len(rows),
    }
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return path


def summarize(rows: List[Dict[str, Any]]) -> None:
    ok = [r for r in rows if r["status"] == "ok"]
    errs = len(rows) - len(ok)
    lats = sorted(r["latency_s"] for r in ok)
    if not lats:
        print(f"\n{len(rows)} requests, {errs} errors, no successful latencies.")
        return

    def pct(p):
        return lats[min(len(lats) - 1, int(p / 100 * len(lats)))]
    max_inflight = max((r["in_flight_at_dispatch"] for r in rows), default=0)
    cost = sum(r.get("cost_usd") or 0 for r in ok)
    print(f"\nRequests: {len(rows)} ({errs} errors) | "
          f"latency p50={pct(50):.2f}s p95={pct(95):.2f}s p99={pct(99):.2f}s | "
          f"peak in-flight={max_inflight} | total cost=${cost:.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Open-loop traffic simulator")
    p.add_argument("--pattern", default="diurnal",
                   choices=["poisson", "diurnal", "ramp", "bursty"])
    p.add_argument("--duration", type=float, default=180.0, help="Run length in seconds")
    p.add_argument("--base-rate", type=float, default=0.5, help="Baseline/trough req/s")
    p.add_argument("--peak-rate", type=float, default=4.0, help="Peak req/s")
    p.add_argument("--burst-period", type=float, default=30.0, help="Bursty: seconds between bursts")
    p.add_argument("--burst-len", type=float, default=8.0, help="Bursty: burst window length (s)")
    p.add_argument("--model", default="llama3.2:1b")
    p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    print(f"Traffic sim | pattern={args.pattern} | model={args.model} | "
          f"GPU_HOURLY_RATE=${get_gpu_hourly_rate():.2f}/hr\n")
    rows = asyncio.run(run(args))
    if not rows:
        print("No requests sent.")
        return
    path = save(rows, args)
    summarize(rows)
    print(f"\nSaved {len(rows)} request records to {path}")


if __name__ == "__main__":
    main()
