# Experiment 04 — The Business Story

*A hypothetical to make the numbers matter. Fictional company; the data is real (see [analysis.md](analysis.md)).*

## The company
**Lexor** builds an AI legal-research assistant. Its RAG pipeline retrieves passages from case law and stuffs them into the prompt. The product team's instinct: *"retrieve more — more context means better answers."* They currently pad every query to ~12k tokens "to be safe."

## The decision on the table
**How much context should we retrieve per query?** More feels strictly better for quality. What does it cost?

## What the experiment tells them
Context is not free, and the cost is mostly hidden. Padding to ~12k tokens vs a tight ~400-token prompt:
- **2.6× the cost per token** (and you're paying it on every query),
- **~18× the time-to-first-token** (1.2 s → 22.8 s) — a latency wall users feel immediately,
- **650 MiB of KV cache per request** — which, multiplied by concurrent users, is what actually caps how many sessions a GPU can hold.

![ttft vs decode](screenshots/04-context_ttft_vs_decode.png)

The "be safe, retrieve more" default is quietly the most expensive line in the inference bill *and* the worst thing for perceived latency.

## The recommendation
> **Treat context length as a budget, not a free dial.** Rerank and trim retrieved passages to the few that matter instead of padding to a fixed large size. Cutting typical context from ~12k to ~3k tokens roughly **halves $/token and cuts TTFT several-fold**, and slashes per-request KV-cache pressure — which raises how many concurrent sessions each GPU can serve. Reserve big contexts for the queries that genuinely need them.

## The one-sentence executive summary
> Every token of retrieved context is paid for on every request in dollars, first-token latency, and GPU memory — so aggressive retrieval trimming (rerank to the few passages that matter) is usually a larger, safer cost win than buying more GPUs.
