#!/usr/bin/env python3
"""Generate conservative fused residual LayerNorm V2/V3 dispatch profile."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> dict[tuple[int, int, str], float]:
    out: dict[tuple[int, int, str], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[(int(row["rows"]), int(row["cols"]), row["variant"])] = float(row["median_us"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--min-speedup", type=float, default=1.05)
    parser.add_argument("--min-runs", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/fused_residual_layernorm_dispatch_profile.csv"),
    )
    args = parser.parse_args()

    speedups: dict[tuple[int, int], list[float]] = defaultdict(list)
    for path in args.inputs:
        run = load(path)
        shapes = {(r, c) for r, c, v in run if v == "v2_warp_shuffle"}
        for rows, cols in shapes:
            v2 = run.get((rows, cols, "v2_warp_shuffle"))
            v3 = run.get((rows, cols, "v3_float4_io"))
            if v2 is not None and v3 is not None:
                speedups[(rows, cols)].append(v2 / v3)

    records = []
    for (rows, cols), vals in sorted(speedups.items()):
        accepted = len(vals) >= args.min_runs and all(v >= args.min_speedup for v in vals)
        record = {
            "rows": rows,
            "cols": cols,
            "decision": "v3_float4_io" if accepted else "v2_warp_shuffle",
            "runs": len(vals),
            "min_speedup": min(vals),
            "max_speedup": max(vals),
            "threshold": args.min_speedup,
            "min_runs": args.min_runs,
        }
        records.append(record)
        print(f"{rows}x{cols}: {record['decision']} speedups={[round(v,4) for v in vals]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
