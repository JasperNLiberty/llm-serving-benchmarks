# Experiment 04 — Context Length (Prompt Size)

## The one variable
We change **input/context length** — from ~400 tokens to ~12,000 tokens — while holding the **output length fixed** and the model fixed (`qwen2.5:7b`). Only how much you put *into* the prompt changes (the RAG / long-context knob).

## Question
How does cost scale as you stuff more context (retrieved documents, history) into the prompt?

## Hypothesis / prediction
Long context is expensive in ways a tokens/sec chart hides:
1. **Prefill is O(context)** — time-to-first-token grows ~linearly with prompt length.
2. The **KV cache grows with context**, so even *decode* slows as memory bandwidth saturates.
3. So **$/token climbs with context length** even at fixed output — and most of the cost is invisible up front (TTFT) and in memory (KV cache), not in the visible answer.

## How it runs
- `bench/context_scaling.py` sweeps prompt length through the gateway, measuring TTFT, decode ms/token, KV-cache size (analytical, from the model config), and `$/M tokens` at each length. (This is a per-request structural property, best shown by the controlled sweep rather than a live dashboard panel.)

## How the data is metered (the trail to the screenshot)
`context_scaling.py` → gateway **`/ollama/chat/stream`** (so TTFT is measured at the first token) → Ollama. TTFT and decode latency come from the stream; token counts are Ollama's authoritative `eval_count` / `prompt_eval_count`; KV-cache size is computed from qwen2.5-7B's architecture (28 layers, 4 KV heads via GQA, head_dim 128, fp16 → 56 KiB/token); `$/M` from throughput at $0.80/hr.

## Screenshots, and what each decides
| Screenshot | Shows | Supports hypothesis if… | Disproves if… |
|---|---|---|---|
| `04-context_ttft_vs_decode.png` | TTFT and decode ms/token vs context | TTFT grows steeply; decode also rises | both flat |
| `04-context_kvcache_and_cost.png` | KV-cache size and $/M vs context | both climb with context | flat |
