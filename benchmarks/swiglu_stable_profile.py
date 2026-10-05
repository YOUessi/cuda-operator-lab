#!/usr/bin/env python3
"""Stable SwiGLU V0/V1 profiling with L2 eviction."""

from __future__ import annotations

import argparse
import csv
import random
import statistics
from pathlib import Path

import torch

from cuda_operator_lab.bindings import swiglu_v0_into, swiglu_v1_into


DEFAULT_SHAPES = [
    (128, 128),
    (128, 512),
    (128, 1024),
    (128, 4096),
    (256, 512),
    (256, 1024),
    (256, 4096),
    (512, 512),
    (512, 1024),
    (512, 4096),
    (1024, 512),
    (1024, 1024),
    (1024, 4096),
    (2048, 512),
    (2048, 1024),
    (2048, 4096),
]


def parse_shape(value: str) -> tuple[int, int]:
    a, sep, b = value.lower().partition("x")
    if not sep:
        raise argparse.ArgumentTypeError("shape must be ROWSxCOLS")
    return int(a), int(b)


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
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", nargs="*", type=parse_shape, default=DEFAULT_SHAPES)
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--repeats", type=int, default=80)
    p.add_argument("--flush-mib", type=int, default=64)
    p.add_argument("--seed", type=int, default=20261014)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    variants = {
        "v0_scalar": swiglu_v0_into,
        "v1_float4": swiglu_v1_into,
    }
    flush = torch.empty(args.flush_mib * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
    rng = random.Random(args.seed)
    records = []

    for rows, cols in args.shapes:
        g = torch.Generator(device="cuda")
        g.manual_seed(args.seed + rows * 10000 + cols)
        gate = torch.randn(rows, cols, device="cuda", dtype=torch.float32, generator=g)
        up = torch.randn(rows, cols, device="cuda", dtype=torch.float32, generator=g)
        outs = {k: torch.empty_like(gate) for k in variants}
        round_medians = {k: [] for k in variants}

        for ridx in range(args.rounds):
            samples = {k: [] for k in variants}
            for sidx in range(args.repeats):
                order = list(variants)
                if (ridx + sidx) % 2 == 0:
                    rng.shuffle(order)
                else:
                    order.reverse()
                for name in order:
                    fn = variants[name]
                    out = outs[name]
                    samples[name].append(
                        timed_one(lambda fn=fn, out=out: fn(gate, up, out), flush)
                    )
            for name in variants:
                round_medians[name].append(statistics.median(samples[name]))

        v0 = statistics.median(round_medians["v0_scalar"])
        v1 = statistics.median(round_medians["v1_float4"])
        print(f"{rows}x{cols}: v0={v0:.3f} us v1={v1:.3f} us speedup={v0/v1:.3f}x")
        for name, values in round_medians.items():
            records.append({
                "rows": rows,
                "cols": cols,
                "variant": name,
                "median_us": statistics.median(values),
                "speedup_vs_v0": v0 / statistics.median(values),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)


if __name__ == "__main__":
    main()
