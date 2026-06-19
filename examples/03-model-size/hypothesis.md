# Experiment 03 — Model Size (1B vs 7B)

## The one variable
We change **model size** — `llama3.2:1b` (1B params) vs `qwen2.5:7b` (7B params, 7× larger). Same hardware, same GPU rate, same prompts, same single-stream serving through the gateway. Only the parameter count changes.

## Question
What does the bigger model actually cost per token, and is the premium proportional to its size?

## Hypothesis / prediction
A bigger model is slower per token (more weights to move through memory each step), so its **$/token is higher** — but **not** proportionally to the parameter count. Decode is memory-bandwidth-bound, so 7× the parameters should cost only a low single-digit multiple in $/token, not 7×. Prediction: roughly **~2× cost for 7× the params**. The decision this informs: *don't reach for the big model by default — use the smallest model that clears the quality bar.*

## How it runs
- **Rigorous:** `bench/load.py` swept both models (×device×concurrency); charts already produced.
- **Live contrast (two pinned runs):**
  ```sh
  python observability/loadgen.py --model llama3.2:1b  --concurrency 2 --duration 60
  python observability/loadgen.py --model qwen2.5:7b   --concurrency 2 --duration 60
  ```

## How the data is metered (the trail to the screenshot)
`loadgen` → gateway **`/ollama/chat`** → Ollama. The gateway computes `$/M tokens` from the observed throughput at $0.80/hr and exposes throughput + cost → Prometheus → Grafana **live $/M (panel 1)** and **throughput tokens/sec (panel 8)**.

## Screenshots I will capture, and what each decides
| Screenshot | Shows | Supports hypothesis if… | Disproves if… |
|---|---|---|---|
| `03-cost_per_million_tokens_by_model.png` (bench) | $/M, 1B vs 7B | 7B ≈ ~2× 1B (not 7×) | 7B ≈ 7× |
| `03-throughput_and_cost_vs_model_size.png` (bench) | throughput + cost vs size | 1B much faster/cheaper | similar |
| `03a-panel-1.png` (1B) / `03b-panel-1.png` (7B) | live $/M per model | 7B panel clearly higher | identical |
| `03a-panel-8.png` / `03b-panel-8.png` | live throughput per model | 1B higher tok/s | identical |
