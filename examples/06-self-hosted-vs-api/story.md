# Experiment 06 — The Business Story

*A hypothetical to make the numbers matter. Fictional company; the model is real (see [analysis.md](analysis.md)).*

## The company
**Quill** is a Series A startup with an AI writing assistant. They're on a hosted API and serving **~5M tokens/day**. The bill is creeping up, and an engineer pitches the board: *"we're paying retail per token — let's buy our own GPUs and cut the bill."* It sounds obviously right.

## The decision on the table
**Build or buy?** Self-hosting *feels* cheaper because you stop paying the API's margin. Is it, at their scale?

## What the experiment tells them
At 5M tokens/day, Quill is **an order of magnitude below the ~64M tok/day crossover.** Self-hosting would mean renting a GPU that sits **>90% idle** — and you pay for idle silicon all the same. They'd *increase* their bill, not cut it, and add ops burden on top.

![crossover](screenshots/06-selfhost_vs_api.png)

The crossover isn't a single number, it's a *condition*: self-hosting wins only once you have **both** enough volume **and** steady-enough traffic to keep the GPU busy. Quill has neither yet — their traffic is spiky and small. The API's "$0 when idle" is exactly the right cost shape for them.

## The recommendation
> **Stay on the API until you cross ~50–100M tokens/day of steady traffic.** At Quill's volume, self-hosting a GPU would be mostly idle and cost *more*, plus ops overhead. Revisit when sustained volume approaches the crossover **and** utilization can be kept high (batching, autoscaling). Until then, the API's pay-per-token shape is a feature, not a markup — and the engineering effort is better spent on the product than on babysitting GPUs.

## The one-sentence executive summary
> Self-hosting only beats an API once you have enough *steady* traffic to keep a GPU busy (here ~64M tokens/day) — below that, "buy our own GPUs" raises the bill and adds ops load, because you pay for the idle GPU the API never charged you for.
