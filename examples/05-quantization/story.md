# Experiment 05 — The Business Story

*A hypothetical to make the numbers matter. Fictional company; the data is real (see [analysis.md](analysis.md)).*

## The company
**Nimbus** serves a 7B assistant to paying users on a fixed fleet of single-GPU boxes. They deploy at **Q8** "for quality." Lately they're hitting an out-of-memory wall at peak — too many concurrent sessions for the VRAM — and the proposed fix is to **buy more GPUs.**

## The decision on the table
**Buy more GPUs, or change quantization?** The team assumes Q8 is the safe, fast choice and that capacity means more hardware.

## What the experiment tells them
The instinct that Q8 is the "faster/safer" choice doesn't hold: Q8 is **no faster than Q4** (actually ~6% *slower*) and costs essentially the same per token. What Q8 *actually* costs them is **1.72× the VRAM** — and VRAM is exactly the wall they're hitting.

![Q4 vs Q8](screenshots/05-quantization_q4_vs_q8.png)

Switching the deployment to **Q4 frees ~40% of the model's memory footprint** at ~the same throughput. That reclaimed VRAM is what lets a single GPU hold **more concurrent sessions** (and more KV-cache for longer contexts) — the capacity they were about to buy hardware for. The only thing they give up is whatever marginal quality Q8 provides, which they should *measure on their own task* before assuming it matters.

## The recommendation
> **Before buying GPUs, drop Q8 → Q4 and measure quality on your real workload.** Q4 costs ~the same per token but uses ~40% less memory, directly raising how many concurrent users each existing GPU can serve. Reserve Q8 (or fp16) for tasks where a proper eval shows the precision actually moves your quality metric. "Higher precision for safety" is a memory bill you may be paying for nothing.

## The one-sentence executive summary
> On this hardware Q8 buys you almost no speed but costs 1.7× the VRAM — so when you hit a memory wall, the first lever is **quantize down to Q4 (after a quality check), not buy more GPUs**, because the win quantization offers is capacity, not throughput.
