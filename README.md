# LLM Serving Benchmarks

Benchmarking suite for local LLM inference on Apple M1 Max (Metal/MPS). Measures throughput, latency, and — most importantly — **cost (`$/M tokens`)** across models, backends, prompts, and concurrency levels.

## Primary metric: $/M tokens

Anyone can publish a tokens/sec chart. The question that actually matters is: *what does this configuration cost, and when does the cheaper option win?* Every result in this repo pairs throughput with a dollar number so that question has an answer.

Cost is computed from a configurable GPU hourly rate via the `GPU_HOURLY_RATE` environment variable (default: **$0.80/hr**, representing an A10G-class cloud GPU or amortized M1 Max machine cost):

```
$/M tokens = (GPU_HOURLY_RATE / 3600) / tokens_per_sec × 1,000,000
```

The exact rate is swappable — set `GPU_HOURLY_RATE=X` before any benchmark run to model a different instance type. The cost logic lives in [`bench/cost_calculator.py`](bench/cost_calculator.py).

Every saved CSV under `results/` carries a `cost_per_million_tokens` column. CSVs collected before cost instrumentation existed were brought up to schema with [`bench/backfill_cost.py`](bench/backfill_cost.py), which recomputes cost deterministically from each row's `tokens_per_sec` — identical to what a fresh run would have written:

```sh
python bench/backfill_cost.py            # idempotent; skips files that already have cost
```

## Key findings

- **Ollama vs MLX (qwen2.5:7b):** Ollama handles concurrent requests ~3–4× more efficiently than MLX at concurrency=1, translating directly to lower cost. MLX cost grows steeply with concurrency due to its lack of batching.
- **Model size (llama3.2:1b vs qwen2.5:7b):** The 1b model costs **$10.40/M tokens** vs **$15.70/M tokens** for the 7b — a 34% premium for 7× the parameters. Whether that quality delta justifies the cost depends on the task.
- **Prefill vs decode (qwen2.5:7b):** For ~250-token generations, prefill is only **~5–7%** of per-request GPU cost — decode dominates. Per token, though, a **decode (output) token costs ~6× a prefill (input) token** of GPU time, because prefill ingests the whole prompt in one parallel compute-bound pass while decode emits one token per memory-bound pass. This flips for short completions over long prompts, where prefill dominates.

## Charts

| Chart | Description |
|---|---|
| ![throughput and cost](charts/ollama_vs_mlx_throughput_and_cost.png) | Throughput and $/M tokens vs concurrency, Ollama vs MLX |
| ![cost by backend](charts/cost_per_million_tokens_by_backend.png) | $/M tokens by backend at concurrency=1 |
| ![throughput and cost by model](charts/throughput_and_cost_vs_model_size.png) | Throughput and $/M tokens by model size, CPU vs GPU |
| ![cost by model](charts/cost_per_million_tokens_by_model.png) | $/M tokens by model size (GPU/Metal, concurrency=1) |
| ![prefill vs decode cost](charts/prefill_vs_decode_cost.png) | Per-request cost split into prefill vs decode dollars, by prompt category |

Regenerate all charts from saved results:

```sh
python charts/generate_charts.py
# or with a custom rate:
GPU_HOURLY_RATE=1.20 python charts/generate_charts.py
```

## Setup

```sh
pip install -r requirements.txt
```

## Run Benchmarks

**Ollama + device sweep** (requires [mini-llm-gateway](https://github.com/noslack/mini-llm-gateway) on port 8000):

```sh
python bench/load.py
```

Sweeps models × prompts × CPU/GPU × concurrency levels. Outputs CSVs to `results/`.

**Ollama vs MLX comparison** (gateway on port 8000, MLX server on port 8001):

```sh
python bench/compare.py
```

**MLX-only batch size sweep:**

```sh
python bench/bench_mlx.py
```

**Prefill vs decode cost decomposition** (reads the difficulty-invariance CSVs; no server needed):

```sh
python bench/prefill_decode_cost.py
```

Splits each request's GPU cost at time-to-first-token into prefill vs decode dollars, reports the per-phase per-token asymmetry, and writes `results/difficulty/prefill_decode_cost.csv` plus the chart above.

All scripts respect `GPU_HOURLY_RATE` and append `cost_per_million_tokens` to every result row.

## Benchmarking methodology

- **Warm-up:** each model/backend is pre-warmed with a throwaway request before timing begins
- **Multiple trials:** each configuration runs `REQUESTS_PER_LEVEL` requests (default 16–48) and reports mean, p50, p95, p99 latency
- **Variance reported:** latency distributions are captured, not just a single number
- **Variable isolation:** each sweep varies one dimension at a time (model, device, concurrency, backend)

## Platform

Apple M1 Max, macOS. Metal/MPS only — no CUDA, no vLLM. Backend: Ollama (OpenAI-compatible API).
