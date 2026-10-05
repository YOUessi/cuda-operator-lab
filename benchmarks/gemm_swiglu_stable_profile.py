#!/usr/bin/env python3
"""Stable direct V2/V3 GEMM+SwiGLU profiling with L2 eviction."""

from __future__ import annotations

import argparse
import csv
import random
import statistics
from pathlib import Path

import torch

from cuda_operator_lab.bindings import (
    gemm_swiglu_v2_into,
    gemm_swiglu_v3_into,
)


DEFAULT_SHAPES = [
    (32, 128, 256),
    (128, 512, 512),
    (128, 1024, 4096),
    (256, 1024, 4096),
    (512, 1024, 4096),
    (256, 4096, 4096),
    (512, 4096, 4096),
]


def parse_shape(value: str) -> tuple[int, int, int]:
    parts = value.lower().split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be MxKxN")
    m, k, n = (int(x) for x in parts)
    if m <= 0 or k <= 0 or n <= 0:
        raise argparse.ArgumentTypeError("M, K, and N must be > 0")
    return m, k, n


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
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--flush-mib", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/gemm_swiglu_stable_profile.csv"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    variants = {
        "v2_packed_scalar_post": gemm_swiglu_v2_into,
        "v3_packed_float4_post": gemm_swiglu_v3_into,
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

    for m, k, n in args.shapes:
        g = torch.Generator(device="cuda")
        g.manual_seed(args.seed + m * 1000000 + k * 1000 + n)

        x = torch.randn(m, k, device="cuda", dtype=torch.float32, generator=g)
        gate_w = torch.randn(n, k, device="cuda", dtype=torch.float32, generator=g)
        up_w = torch.randn(n, k, device="cuda", dtype=torch.float32, generator=g)
        packed_w = torch.cat((gate_w, up_w), dim=0).contiguous()

        workspaces = {
            name: torch.empty(m, 2 * n, device="cuda", dtype=torch.float32)
            for name in variants
        }
        outputs = {
            name: torch.empty(m, n, device="cuda", dtype=torch.float32)
            for name in variants
        }
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
                    workspace = workspaces[name]
                    out = outputs[name]
                    samples[name].append(
                        timed_one(
                            lambda fn=fn, workspace=workspace, out=out: fn(
                                x, packed_w, workspace, out
                            ),
                            flush,
                        )
                    )

            for name in variants:
                round_medians[name].append(statistics.median(samples[name]))

        v2_median = statistics.median(
            round_medians["v2_packed_scalar_post"]
        )
        v3_median = statistics.median(
            round_medians["v3_packed_float4_post"]
        )

        print(
            f"{m}x{k}x{n}: "
            f"v2={v2_median:.3f} us "
            f"v3={v3_median:.3f} us "
            f"speedup={v2_median / v3_median:.4f}x"
        )

        for name, values in round_medians.items():
            median_us = statistics.median(values)
            mean_us = statistics.mean(values)
            stdev_us = statistics.stdev(values) if len(values) > 1 else 0.0
            records.append(
                {
                    "m": m,
                    "k": k,
                    "n": n,
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
