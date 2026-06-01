# LLM Serving Benchmarks

Ollama model, prompt, and device comparison under concurrent load.

## Setup

```sh
pip install -r requirements.txt
```

## Run Benchmarks

Make sure [mini-llm-gateway](https://github.com/noslack/mini-llm-gateway) is running:

```sh
cd ../mini-llm-gateway
uvicorn main:app --reload
```

Then run the benchmark:

```sh
python bench/load.py
```

This script sweeps:
- models
- prompts
- CPU vs GPU device targets
- concurrency levels
- latency and token throughput metrics

Outputs:
- `results/results.csv` — combined benchmark data
- `results/by_prompt/` — prompt-specific CSV summaries
- `results/by_device/` — device-specific CSV summaries

## Key Finding

Smaller models (1b) maintain lower latency under concurrency.
Larger models (7b) saturate faster due to memory/compute limits.

- [Analysis notebook](notebooks/analysis.ipynb)
- [Raw results](results/results.csv)
