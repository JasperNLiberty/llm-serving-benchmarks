# Experiment 02 — Traffic Load vs Fixed Capacity

## The one variable
We change **how hard we drive a fixed-capacity server** — the same gateway (concurrency cap = 2, `qwen2.5:7b`, $0.80/hr) under **under-capacity** load (1 concurrent, below the cap) vs **over-capacity** load (8 concurrent, well past the cap). Hardware and model are identical; only the arrival pressure changes.

## Question
When you hold capacity fixed and vary load, what happens to **latency** and to **cost per token** — and can you make both good at once?

## Hypothesis / prediction
You **cannot** optimize latency and utilization simultaneously:
1. **Under-capacity** → the GPU is mostly idle between requests. Latency is great (no queue), but you still pay for the idle silicon, so **effective $/token is high** (the idle-GPU tax).
2. **Over-capacity** → the queue runs away past the concurrency knee: **tail latency explodes**, in-flight pins at the cap, queue depth climbs. But the GPU runs hot, so **effective $/token is low**.

So the configuration with the *best* latency has the *worst* cost, and vice-versa. That trade-off **is** the capacity-planning decision.

## How it runs
- **Rigorous:** `bench/traffic_sim.py` already drove four realistic arrival shapes (poisson/bursty/diurnal/ramp) through the gateway → `bench/analyze_traffic.py` (charts + Little's Law). Those charts are the controlled evidence.
- **Live contrast (two pinned runs):**
  ```sh
  # under-capacity (below the cap):
  python observability/loadgen.py --model qwen2.5:7b --concurrency 1 --duration 70
  # over-capacity (overwhelm the 2 slots):
  python observability/loadgen.py --model qwen2.5:7b --concurrency 8 --duration 70
  ```

## How the data is metered (the trail to the screenshot)
`loadgen` → gateway **`/ollama/chat`** (queued behind the concurrency scheduler) → Ollama. The gateway records latency histograms, `chat_in_flight`, `chat_queue_depth`, cost, and exposes `gpu_slots_utilization` at scrape → Prometheus → Grafana panels **latency percentiles (7)**, **in-flight & queue (10)**, **slot utilization (6)**, and **effective $/M = nominal ÷ utilization (5)**.

## Screenshots I will capture, and what each decides
| Screenshot | Shows | Supports hypothesis if… | Disproves if… |
|---|---|---|---|
| `02a-panel-7.png` / `02b-panel-7.png` | latency p50/p95/p99, under vs over | under = flat low; over = p99 blows up | both similar |
| `02b-panel-10.png` | in-flight & queue, over-capacity | in-flight pinned at 2, queue climbing | queue stays 0 |
| `02a-panel-5.png` / `02b-panel-5.png` | effective $/M, under vs over | under **high**, over **low** (inverse of latency) | both equal |
| `02a-panel-6.png` / `02b-panel-6.png` | utilization, under vs over | under low, over ~100% | both equal |
| `traffic_*.png` (bench) | the four arrival shapes' tail latency + effective $/M | best-latency pattern = worst cost | no tension |
