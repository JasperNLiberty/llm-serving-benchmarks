# Experiment 05 — Quantization (Q4 vs Q8)

## The one variable
We change **quantization precision** of the *same* model — `qwen2.5:7b` at **Q4_K_M** (~4.7 GB) vs `qwen2.5:7b-instruct-q8_0` at **Q8** (~8.1 GB). Same architecture, same weights, same hardware; only the number of bits per weight changes.

## Question
What does precision cost? Is the higher-precision model worth its extra memory and slowdown?

## Hypothesis / prediction
Decode is memory-bandwidth-bound — the GPU re-reads the weights every token. Q8 stores ~1.7× more bytes per weight than Q4, so it must move ~1.7× more memory per token. Prediction:
1. **Q8 is slower** (lower tok/s) and therefore **higher $/token**, roughly in proportion to the size ratio.
2. **Q8 needs ~1.7× the memory** (VRAM footprint ≈ on-disk size) — which also means fewer concurrent sessions / smaller KV-cache headroom per GPU.
3. **Quality**: Q8 sits closer to full precision, but for many tasks the gap to Q4 is small. So the decision is *use the lowest quant that still clears the quality bar* — usually Q4.

## How it runs
- `python bench/quantization.py` — runs both models on medium-length generations (throughput → $/M) and a small numeric ladder (a light quality check), reports tok/s, $/M, accuracy, and on-disk size, and charts them.

## How the data is metered
`bench/quantization.py` → Ollama `/api/generate` → `eval_count` / `eval_duration` give authoritative output-token throughput → `$/M tokens` at $0.80/hr via the shared `cost_calculator`. On-disk size is a proxy for VRAM footprint.

## Screenshots, and what each decides
| Screenshot | Shows | Supports hypothesis if… | Disproves if… |
|---|---|---|---|
| `05-quantization_q4_vs_q8.png` | tok/s, $/M, size — Q4 vs Q8 | Q8 slower + pricier + larger; accuracy similar | Q8 free / much more accurate |
