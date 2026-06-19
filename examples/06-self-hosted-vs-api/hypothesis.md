# Experiment 06 — Self-Hosted vs API (Build vs Buy)

## The one variable
We change **how you procure inference** — rent a GPU and self-host, or pay a hosted **API** per token — for the same workload. Everything else (the model class, the traffic) is held fixed; only the cost *structure* changes.

## Question
At what traffic volume does owning a GPU become cheaper than paying an API per token?

## Hypothesis / prediction
The two have different cost shapes, so there's a crossover:
- **API** = pay-per-token → cost is **linear** in volume, **$0 when idle**.
- **Self-hosted** = rent the GPU by the hour whether busy or not → cost is a **step function** (one more GPU each time you exhaust capacity), and its *effective* $/token depends entirely on **utilization**.

Prediction: the **API wins at low or spiky volume** (a whole GPU is wasteful for trickle traffic — the idle-GPU tax from [experiment 02](../02-traffic-and-capacity/)), and **self-hosting wins only above a high, steady volume** where you can keep the GPU busy. Low utilization shoves the practical crossover further out.

## How it runs
- `python bench/selfhost_vs_api.py` — models monthly cost vs traffic for an API and for self-hosting at 100% and 40% utilization, finds the crossover, and plots it. (This is a procurement/modeling question, not a live-load one, so the evidence is the crossover chart, parameterized by the $/token figures our own benchmarks produce.)

## How the data is metered (inputs to the model)
The self-hosted side is grounded in this repo's measured economics: a GPU hourly rate ($0.80/hr) and an achievable served throughput → $/token (the same `cost_calculator` used everywhere here). The API side is a representative hosted price for a 7–8B-class endpoint ($0.30/M tokens). Utilization is the lever that ties back to the traffic/capacity experiment.

## Screenshots, and what each decides
| Screenshot | Shows | Supports hypothesis if… | Disproves if… |
|---|---|---|---|
| `06-selfhost_vs_api.png` | monthly cost vs tokens/day: API vs self-hosted @ 100% / 40% util | API cheaper left of a crossover; self-hosted curves step up, 40% sooner | no crossover / lines parallel |
