# Experiment NN — <title>

## The one variable
We change **<X>** and hold everything else constant (<list the controls: model, GPU rate, concurrency cap, prompt set, …>).

## Question
<The single question this experiment answers.>

## Hypothesis / prediction
<What we expect to see, and why — the mechanism. Be specific enough to be wrong.>

## How it runs
- Engine: Ollama (`<models>`)
- Path: client → `mini-llm-gateway` `<endpoint>` → Ollama
- Load: `<exact loadgen / bench command>`
- Controls: GPU rate `$0.80/hr`, concurrency cap `<n>`, `<other>`

## How the data is metered (the trail to the screenshot)
`<endpoint>` → gateway records `<metrics>` → Prometheus scrapes `/metrics` every 5 s → Grafana panel **<panel name>** plots `<promql>`. (Plus benchmark chart `<chart.png>` from `bench/<script>.py` where relevant.)

## Screenshots I will capture, and what each decides
| Screenshot | Shows | Supports hypothesis if… | Disproves if… |
|---|---|---|---|
| `NN-...png` | <panel/chart> | <pattern> | <pattern> |
