"""Backfill the cost_per_million_tokens column into legacy result CSVs.

Some result CSVs were saved before cost instrumentation existed, so they carry
throughput (tokens_per_sec) but no dollar column. The live benchmark scripts now
append cost on every run; this script brings the already-committed artifacts up to
the same schema so every saved CSV is self-describing.

Cost is recomputed deterministically from the existing tokens_per_sec using the
same cost_calculator and the same GPU_HOURLY_RATE the charts use, so a backfilled
row is identical to one that would have been written at collection time.

Idempotent: a CSV that already has cost_per_million_tokens is left untouched.

Usage:
    python bench/backfill_cost.py                 # backfill everything under results/
    python bench/backfill_cost.py path/to.csv ... # backfill specific files
    GPU_HOURLY_RATE=1.20 python bench/backfill_cost.py
"""

import csv
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent))
from cost_calculator import cost_per_million_tokens, get_gpu_hourly_rate

RESULTS_DIR = Path("results")
THROUGHPUT_COL = "tokens_per_sec"
COST_COL = "cost_per_million_tokens"


def backfill_file(path: Path, rate: float) -> str:
    """Add COST_COL to one CSV. Returns a short status string for logging."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if THROUGHPUT_COL not in fieldnames:
        return f"skip (no {THROUGHPUT_COL})"
    if COST_COL in fieldnames:
        return "skip (already has cost)"

    # Place cost immediately after throughput so the schema matches fresh runs.
    new_fields = list(fieldnames)
    new_fields.insert(new_fields.index(THROUGHPUT_COL) + 1, COST_COL)

    filled = 0
    for row in rows:
        raw = row.get(THROUGHPUT_COL, "")
        try:
            tps = float(raw)
        except (TypeError, ValueError):
            tps = 0.0
        if tps > 0:
            row[COST_COL] = f"{cost_per_million_tokens(rate, tps):.6f}"
            filled += 1
        else:
            row[COST_COL] = ""

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fields)
        writer.writeheader()
        writer.writerows(rows)

    return f"backfilled {filled}/{len(rows)} rows"


def discover(args: List[str]) -> List[Path]:
    if args:
        return [Path(a) for a in args]
    return sorted(RESULTS_DIR.rglob("*.csv"))


def main() -> None:
    rate = get_gpu_hourly_rate()
    paths = discover(sys.argv[1:])
    print(f"Backfilling {COST_COL} at GPU_HOURLY_RATE=${rate:.2f}/hr\n")
    changed = 0
    for path in paths:
        if not path.exists():
            print(f"  {path}: not found")
            continue
        status = backfill_file(path, rate)
        if status.startswith("backfilled"):
            changed += 1
        print(f"  {path}: {status}")
    print(f"\nDone. {changed} file(s) updated.")


if __name__ == "__main__":
    main()
