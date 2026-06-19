# Examples — Benchmark Case Studies

Each subdirectory here is **one self-contained experiment**: a single property is varied (reasoning vs not, traffic shape, model size, …) while everything else is held constant. Every experiment tells the same four-beat story, so it reads like a case study and converts directly into a blog post.

These case studies use **both repos**:
- [`llm-serving-benchmarks`](..) (this repo) — the controlled benchmarks + matplotlib charts.
- [`mini-llm-gateway`](https://github.com/JasperNLiberty/mini-llm-gateway) — the cost-metered serving gateway + the live Grafana dashboard the screenshots come from.

## Anatomy of an experiment

```
NN-short-name/
  hypothesis.md     # what we test, why, what we expect, how it runs, how data is metered
  analysis.md       # the evidence: each screenshot + what the curve means + verdict
  story.md          # a hypothetical company, its problem, and the decision this informs
  screenshots/      # the captured evidence (named, time-pinned, never overwritten)
```

| File | Beat | Blog-post role |
|---|---|---|
| `hypothesis.md` | *What & why* | the setup / intro |
| `screenshots/` | *Evidence* | the figures |
| `analysis.md` | *What it means* | the body |
| `story.md` | *So what* | the takeaway / call to action |

## The workflow (how to run one)

1. **Write `hypothesis.md` first.** State the one variable, the controls, the prediction, the exact run commands, and **how the data flows and is metered** (which endpoint → gateway → Prometheus → which Grafana panel). Name the screenshots you intend to capture and say what each will prove or disprove.
2. **Bring the stack up** (from `mini-llm-gateway/`):
   ```sh
   ollama serve          # engine
   make serve            # gateway  :8000
   make observe          # Prometheus :9090 + Grafana :3000
   ```
3. **Run the experiment and pin the time window.** Record the start epoch, run the load, record the end epoch, and capture Grafana for *exactly that window* so the screenshot can't drift:
   ```sh
   START=$(($(date +%s%3N)))                       # ms
   python observability/loadgen.py --reasoning --duration 120   # the run
   END=$(($(date +%s%3N)))
   # capture only that window, into this experiment's folder, with a prefix:
   python observability/capture.py --no-load \
     --from $((START-5000)) --to $((END+5000)) \
     --out-dir <path>/examples/NN-name/screenshots --prefix "NN-" \
     --panels 1,5,11,12
   ```
   `--from/--to` take epoch-ms (or `now-15m`); `--out-dir`/`--prefix` keep each experiment's PNGs separate and stable. Benchmark matplotlib charts (from `bench/*.py`) are copied in alongside where they tell the story better than a live panel.
4. **Write `analysis.md`.** Embed each screenshot, explain what the curve *is*, and give the verdict against the hypothesis (supported / disproved / surprised).
5. **Write `story.md`.** Wrap it in a concrete business decision.

## Screenshot conventions

- **Named, not numbered-by-accident.** `NN-panel-<id>.png` and `NN-dashboard-full.png`, plus copied benchmark charts keep their descriptive names.
- **Time-pinned.** Always capture an explicit `--from/--to` window matching the run, so re-runs don't overwrite a different experiment's evidence and the curves line up with the narrative.
- **One folder per experiment.** Never write to the shared `mini-llm-gateway/observability/screenshots/` for a case study.

## Index

| # | Experiment | Variable | Decision it informs |
|---|---|---|---|
| 01 | [reasoning-vs-non-reasoning](01-reasoning-vs-non-reasoning/) | reasoning model vs not | route to reasoning, don't default to it |
| 02 | [traffic-and-capacity](02-traffic-and-capacity/) | load vs fixed capacity | autoscale to the demand curve |
| 03 | [model-size](03-model-size/) | 1B vs 7B params | right-size per task |
| 04 | [context-length](04-context-length/) | prompt size (~400 → ~12k tok) | trim retrieval; context is a budget |
| 05 | [quantization](05-quantization/) | precision (Q4 vs Q8) | quantize for memory, not speed |
| 06 | [self-hosted-vs-api](06-self-hosted-vs-api/) | build vs buy | stay on API until ~64M tok/day steady |

Each varies **one** property and holds the rest constant. Together they cover the dimensions a real serving team actually decides on: model **capability** (01), **capacity/utilization** (02), model **size** (03), **context budget** (04), **precision** (05), and **procurement** (06).

See [`_TEMPLATE/`](_TEMPLATE/) to start a new one.
