#!/usr/bin/env python3
"""Generate a conservative RMSNorm dispatch profile from repeated benchmark CSVs."""

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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/rmsnorm_dispatch_profile.csv"),
    )
    args = parser.parse_args()

    runs = [load(path) for path in args.inputs]
    shapes = sorted(
        {
            (rows, cols)
            for run in runs
            for rows, cols, variant in run
            if variant == "v2_warp_shuffle"
        }
    )

    records: list[dict[str, object]] = []
    for rows, cols in shapes:
        speedups: list[float] = []
        complete = True
        for run in runs:
            v2 = run.get((rows, cols, "v2_warp_shuffle"))
            v3 = run.get((rows, cols, "v3_float4_io"))
            if v2 is None or v3 is None:
                complete = False
                break
            speedups.append(v2 / v3)

        if not complete:
            continue

        use_v3 = all(speedup >= args.min_speedup for speedup in speedups)
        record = {
            "rows": rows,
            "cols": cols,
            "decision": "v3_float4_io" if use_v3 else "v2_warp_shuffle",
            "min_speedup": min(speedups),
            "max_speedup": max(speedups),
            "runs": len(speedups),
            "threshold": args.min_speedup,
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