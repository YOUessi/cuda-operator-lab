#!/usr/bin/env python3
"""Generate a conservative LayerNorm V2/V4 dispatch profile."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> dict[tuple[int, int, str], float]:
    out: dict[tuple[int, int, str], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[(int(row["rows"]), int(row["cols"]), row["variant"])] = float(
                row["median_us"]
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--min-speedup", type=float, default=1.05)
    parser.add_argument("--min-runs", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/layernorm_dispatch_profile.csv"),
    )
    args = parser.parse_args()

    runs = [load(path) for path in args.inputs]
    speedups_by_shape: dict[tuple[int, int], list[float]] = defaultdict(list)

    for run in runs:
        shapes = {
            (rows, cols)
            for rows, cols, variant in run
            if variant == "v2_warp_shuffle"
        }
        for rows, cols in shapes:
            v2 = run.get((rows, cols, "v2_warp_shuffle"))
            v4 = run.get((rows, cols, "v4_float4_io"))
            if v2 is not None and v4 is not None:
                speedups_by_shape[(rows, cols)].append(v2 / v4)

    records: list[dict[str, object]] = []
    for (rows, cols), speedups in sorted(speedups_by_shape.items()):
        enough_runs = len(speedups) >= args.min_runs
        accepted = enough_runs and all(s >= args.min_speedup for s in speedups)
        record = {
            "rows": rows,
            "cols": cols,
            "decision": "v4_float4_io" if accepted else "v2_warp_shuffle",
            "runs": len(speedups),
            "min_speedup": min(speedups),
            "max_speedup": max(speedups),
            "threshold": args.min_speedup,
            "min_runs": args.min_runs,
        }
        records.append(record)
        print(
            f"{rows}x{cols}: {record['decision']} "
            f"speedups={[round(x, 4) for x in speedups]}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
