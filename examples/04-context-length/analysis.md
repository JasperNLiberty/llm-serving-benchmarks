# Experiment 04 — Analysis

**Run:** `bench/context_scaling.py`, `qwen2.5:7b`, output length fixed, GPU rate $0.80/hr. Input length swept 406 → 11,872 tokens.

## Verdict
> **Supported.** Holding output fixed and only growing the prompt, a ~12k-token context costs **2.6× more per token** than a ~400-token one ($23.6 vs $9.1 /M). Prefill (TTFT) grows **~18×** (1.2 s → 22.8 s), decode itself slows **2.5×** as the KV cache balloons to **650 MiB** — none of which shows up in a tokens/sec number.

## Evidence

### Prefill explodes, decode creeps up
![ttft vs decode](screenshots/04-context_ttft_vs_decode.png)

**What you're looking at:** time-to-first-token (prefill) and decode latency per output token, vs prompt length.
**What it means:** prefill is O(context) — it must process every prompt token in one pass before emitting anything, so TTFT grows ~linearly (1.2 s → 22.8 s, ~18×). Decode is *not* perfectly flat either: it slows ~2.5× (42.8 → 110 ms/token) as the growing KV cache saturates memory bandwidth.

### KV cache and cost both climb
![kv cache and cost](screenshots/04-context_kvcache_and_cost.png)

**What you're looking at:** KV-cache size (analytical) and `$/M tokens` vs context length.
**What it means:** at 56 KiB/token the cache reaches **650 MiB** by ~12k tokens (and would need **~1.75 GiB** at a full 32k context) — per request. Cost per token rises 2.6× over the same range. The expense is mostly upfront latency and memory, invisible in the output.

## Numbers
| input tokens | TTFT | decode ms/tok | KV cache | $/M tokens |
|---:|---:|---:|---:|---:|
| 406 | 1.22 s | 42.8 | 22 MiB | $9.13 |
| 1,498 | 2.88 s | 47.9 | 82 MiB | $10.24 |
| 2,980 | 6.62 s | 54.6 | 163 MiB | $11.66 |
| 5,944 | 19.58 s | 72.6 | 325 MiB | $16.04 |
| 11,872 | 22.83 s | 110.0 | 649 MiB | $23.58 |

## Caveats
- KV-cache size is computed from the model config (Ollama doesn't expose live cache size on Metal), not a measured allocation.
- Single M1; the *structure* (prefill O(context), KV growth, cost climb) transfers; absolute throughput does not.
