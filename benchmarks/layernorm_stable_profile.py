#!/usr/bin/env python3
"""Stable LayerNorm V2/V4 profiling with L2 eviction and interleaved samples."""

from __future__ import annotations

import argparse
import csv
import random
import statistics
from pathlib import Path

import torch

from cuda_operator_lab.bindings import layernorm_v2_into, layernorm_v4_into


DEFAULT_SHAPES = [
    (128, 128),
    (128, 512),
    (128, 1024),
    (128, 4096),
    (128, 8192),
    (1024, 128),
    (1024, 512),
    (1024, 1024),
    (1024, 4096),
    (2048, 4096),
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
        default=Path("benchmarks/results/layernorm_stable_profile.csv"),
    )
    args = parser.parse_args()

    variants = {
        "v2_warp_shuffle": layernorm_v2_into,
        "v4_float4_io": layernorm_v4_into,
    }
    flush = torch.empty(
        args.flush_mib * 1024 * 1024 // 4,
        device="cuda",
        dtype=torch.float32,
    )
    rng = random.Random(args.seed)
    records: list[dict[str, object]] = []

    print(f"device={torch.cuda.get_device_name(0)}")
    print(
        f"rounds={args.rounds} repeats={args.repeats} "
        f"flush_mib={args.flush_mib} seed={args.seed}"
    )

    for rows, cols in args.shapes:
        g = torch.Generator(device="cuda")
        g.manual_seed(args.seed + rows * 10000 + cols)
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float32, generator=g)
        weight = torch.randn(cols, device="cuda", dtype=torch.float32, generator=g)
        bias = torch.randn(cols, device="cuda", dtype=torch.float32, generator=g)
        outputs = {name: torch.empty_like(x) for name in variants}
        round_medians = {name: [] for name in variants}

        for round_idx in range(args.rounds):
            samples = {name: [] for name in variants}
            for sample_idx in range(args.repeats):
                order = list(variants)
                if (round_idx + sample_idx) % 2 == 0:
                    rng.shuffle(order)
                else:
                    order.reverse()
                for name in order:
                    fn = variants[name]
                    out = outputs[name]
                    samples[name].append(
                        timed_one(
                            lambda fn=fn, out=out: fn(
                                x, weight, bias, out
                            ),
                            flush,
                        )
                    )
            for name in variants:
                round_medians[name].append(statistics.median(samples[name]))

        v2_median = statistics.median(round_medians["v2_warp_shuffle"])
        v4_median = statistics.median(round_medians["v4_float4_io"])
        print(
            f"{rows}x{cols}: v2={v2_median:.3f} us "
            f"v4={v4_median:.3f} us speedup={v2_median/v4_median:.3f}x"
        )

        for name, values in round_medians.items():
            median_us = statistics.median(values)
            mean_us = statistics.mean(values)
            stdev_us = statistics.stdev(values) if len(values) > 1 else 0.0
            records.append(
                {
                    "rows": rows,
                    "cols": cols,
                    "variant": name,
                    "rounds": args.rounds,
                    "repeats_per_round": args.repeats,
                    "flush_mib": args.flush_mib,
                    "median_us": round(median_us, 6),
                    "mean_us": round(mean_us, 6),
                    "stdev_us": round(stdev_us, 6),
                    "p10_us": round(percentile(values, 0.10), 6),
                    "p90_us": round(percentile(values, 0.90), 6),
                    "cv": round(stdev_us / mean_us if mean_us else 0.0, 6),
                    "speedup_vs_v2": round(v2_median / median_us, 6),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
