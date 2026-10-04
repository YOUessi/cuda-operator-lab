#!/usr/bin/env python3
"""Stable RMSNorm V2/V3/V4 profiling with L2 eviction and alternating order."""

from __future__ import annotations

import argparse
import csv
import random
import statistics
from pathlib import Path

import torch

from cuda_operator_lab.bindings import (
    rmsnorm_v2_into,
    rmsnorm_v3_into,
    rmsnorm_v4_into,
)


DEFAULT_SHAPES = [
    (1024, 512),
    (1536, 512),
    (1024, 1024),
    (1536, 1024),
    (128, 4096),
    (512, 4096),
    (1024, 4096),
    (1536, 4096),
    (2048, 4096),
    (128, 8192),
    (512, 8192),
    (1024, 8192),
    (1280, 8192),
    (1536, 8192),
]


def parse_shape(value: str) -> tuple[int, int]:
    rows_text, sep, cols_text = value.lower().partition("x")
    if not sep:
        raise argparse.ArgumentTypeError("shape must be ROWSxCOLS")
    rows = int(rows_text)
    cols = int(cols_text)
    if rows <= 0 or cols <= 0:
        raise argparse.ArgumentTypeError("rows > 0 and cols > 0 are required")
    return rows, cols


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def timed_one(fn, flush: torch.Tensor) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    flush.add_(1.0)
    torch.cuda.synchronize()
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", nargs="*", type=parse_shape, default=DEFAULT_SHAPES)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=80)
    parser.add_argument("--flush-mib", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/rmsnorm_stable_profile.csv"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    variants = {
        "v2_warp_shuffle": rmsnorm_v2_into,
        "v3_float4_io": rmsnorm_v3_into,
        "v4_shape_dispatch": rmsnorm_v4_into,
    }

    flush_elems = args.flush_mib * 1024 * 1024 // 4
    flush = torch.empty(flush_elems, device="cuda", dtype=torch.float32)
    rng = random.Random(args.seed)

    print(f"device={torch.cuda.get_device_name(0)}")
    print(
        f"rounds={args.rounds} repeats={args.repeats} "
        f"flush_mib={args.flush_mib} seed={args.seed}"
    )

    records: list[dict[str, object]] = []

    for rows, cols in args.shapes:
        gen = torch.Generator(device="cuda")
        gen.manual_seed(args.seed + rows * 10000 + cols)
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float32, generator=gen)
        w = torch.randn(cols, device="cuda", dtype=torch.float32, generator=gen)
        outputs = {name: torch.empty_like(x) for name in variants}

        round_medians: dict[str, list[float]] = {name: [] for name in variants}

        for round_idx in range(args.rounds):
            per_variant: dict[str, list[float]] = {name: [] for name in variants}

            for sample_idx in range(args.repeats):
                order = list(variants)
                if (round_idx + sample_idx) % 2 == 0:
                    rng.shuffle(order)
                else:
                    order.reverse()

                for name in order:
                    impl = variants[name]
                    out = outputs[name]
                    per_variant[name].append(
                        timed_one(
                            lambda impl=impl, out=out: impl(x, w, out),
                            flush,
                        )
                    )

            for name in variants:
                round_medians[name].append(statistics.median(per_variant[name]))

        medians = {
            name: statistics.median(values)
            for name, values in round_medians.items()
        }

        for name, values in round_medians.items():
            median_us = statistics.median(values)
            mean_us = statistics.mean(values)
            stdev_us = statistics.stdev(values) if len(values) > 1 else 0.0
            p10_us = percentile(values, 0.10)
            p90_us = percentile(values, 0.90)
            cv = stdev_us / mean_us if mean_us else 0.0
            record = {
                "rows": rows,
                "cols": cols,
                "variant": name,
                "rounds": args.rounds,
                "repeats_per_round": args.repeats,
                "flush_mib": args.flush_mib,
                "median_us": round(median_us, 6),
                "mean_us": round(mean_us, 6),
                "stdev_us": round(stdev_us, 6),
                "p10_us": round(p10_us, 6),
                "p90_us": round(p90_us, 6),
                "cv": round(cv, 6),
                "speedup_vs_v2": round(medians["v2_warp_shuffle"] / median_us, 6),
            }
            records.append(record)

        print(
            f"{rows}x{cols}: "
            f"v2={medians['v2_warp_shuffle']:.3f} us, "
            f"v3={medians['v3_float4_io']:.3f} us, "
            f"v4={medians['v4_shape_dispatch']:.3f} us, "
            f"v3/v2={medians['v2_warp_shuffle']/medians['v3_float4_io']:.3f}x, "
            f"v4/v2={medians['v2_warp_shuffle']/medians['v4_shape_dispatch']:.3f}x"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
