# Experiment 01 — Reasoning vs Non-Reasoning Model

## The one variable
We change **whether the model reasons** — `deepseek-r1:7b` (a reasoning fine-tune) vs `qwen2.5:7b` (its non-reasoning base). Same parameter count, same base architecture, same hardware, same GPU rate ($0.80/hr), same prompts. The *only* difference is the hidden "thinking" step.

## Question
What does turning on reasoning actually cost, how does that cost scale with problem difficulty, and when is it worth paying?

## Hypothesis / prediction
A reasoning model **breaks difficulty invariance**. For a normal model, per-token cost is flat and a hard question costs the same per token as an easy one (it just answers). The reasoning model should instead:
1. keep the same *per-token* rate (~42 tok/s — same architecture), but
2. emit a growing pile of hidden **thinking tokens** as difficulty rises,
3. so **$/request scales up with difficulty** (a "reasoning tax"), and
4. that tax is **pure waste on easy problems** both models get right — justified only when the task is hard enough that the cheap model *fails*.

A large fraction of the tokens billed should be invisible thinking.

## How it runs
- **Comparison (rigorous):** `python bench/reasoning_tax.py` — both models × a graded easy→hard ladder, 3 trials each, measures thinking-token count, the per-phase dollar split, accuracy, and `$/correct-answer`. Produces the matplotlib charts.
- **Live serving (observability):** drive reasoning traffic *through the gateway* so it's metered end-to-end:
  ```sh
  # from mini-llm-gateway/, stack already up:
  python observability/loadgen.py --reasoning --duration 90 --concurrency 3
  ```

## How the data is metered (the trail to the screenshot)
`loadgen --reasoning` → gateway **`POST /ollama/think/stream`** → Ollama `/api/chat` with `think=true`. The gateway tags each streamed delta as thinking or answer and records `thinking_tokens_total` / `answer_tokens_total` (exact counts) plus the usual cost metrics → Prometheus scrapes `/metrics` every 5 s → Grafana panels **"Reasoning: thinking vs answer tokens/sec"** (panel 11) and **"thinking-token share"** (panel 12), with cost on panel 1 and queueing on panel 10.

## Screenshots I will capture, and what each decides
| Screenshot | Shows | Supports hypothesis if… | Disproves if… |
|---|---|---|---|
| `01-reasoning_tax_cost.png` (bench) | $/request, reasoning vs baseline, by tier | reasoning bars tower over baseline and the ratio grows with difficulty | bars are similar / ratio flat |
| `01-reasoning_tokens_vs_difficulty.png` (bench) | thinking tokens vs tier vs baseline output | thinking climbs with difficulty, baseline flat | thinking flat |
| `01-panel-12.png` (Grafana) | live thinking-token share | a large share (~⅓+) is hidden thinking | share ≈ 0 |
| `01-panel-11.png` (Grafana) | thinking vs answer tokens/sec, live | both streams present, thinking substantial | no thinking stream |
| `01-panel-1.png` (Grafana) | live $/M under reasoning load | elevated vs baseline's ~$3.6/M | identical |
| `01-dashboard-full.png` | the whole board under reasoning load | cost + reasoning panels all populated | — |
