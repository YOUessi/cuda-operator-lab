#!/usr/bin/env python3
"""Benchmark the serial CUDA reduction baseline against torch.sum."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Callable

import torch

from cuda_operator_lab.bindings import reduction_v0_into
from cuda_operator_lab.references import reduction_sum


DEFAULT_SIZES = [2**10, 2**14, 2**18, 2**22, 2**24]


def measure_us(fn: Callable[[], object], *, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return samples


def percentile(samples: list[float], q: float) -> float:
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(samples: list[float]) -> tuple[float, float, float]:
    return statistics.median(samples), percentile(samples, 0.50), percentile(samples, 0.95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="*", default=DEFAULT_SIZES)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/reduction_v0.csv"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    rows: list[dict[str, object]] = []
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"warmup={args.warmup} repeats={args.repeats}")

    for n in args.sizes:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(20261004 + n)
        x = torch.rand(n, device="cuda", dtype=torch.float32, generator=generator)
        out = torch.empty(1, device="cuda", dtype=torch.float32)

        expected = reduction_sum(x)
        actual = reduction_v0_into(x, out)
        torch.cuda.synchronize()

        abs_error = abs(float(actual.item()) - float(expected.item()))
        rel_error = abs_error / max(abs(float(expected.item())), 1e-12)

        ours_samples = measure_us(
            lambda: reduction_v0_into(x, out),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        torch_samples = measure_us(
            lambda: reduction_sum(x),
            warmup=args.warmup,
            repeats=args.repeats,
        )

        ours_median, ours_p50, ours_p95 = summarize(ours_samples)
        torch_median, _, _ = summarize(torch_samples)
        bytes_read = n * x.element_size()
        effective_gbps = bytes_read / (ours_median * 1e-6) / 1e9 if ours_median else 0.0

        row = {
            "operator": "reduction_sum",
            "variant": "v0_serial",
            "dtype": "float32",
            "n": n,
            "bytes_read": bytes_read,
            "median_us": round(ours_median, 3),
            "p50_us": round(ours_p50, 3),
            "p95_us": round(ours_p95, 3),
            "effective_gbps": round(effective_gbps, 6),
            "torch_median_us": round(torch_median, 3),
            "slowdown_vs_torch": round(ours_median / torch_median, 3),
            "abs_error": abs_error,
            "rel_error": rel_error,
        }
        rows.append(row)
        print(
            f"n={n:>9}  v0={ours_median:>10.3f} us  "
            f"torch={torch_median:>9.3f} us  "
            f"slowdown={ours_median / torch_median:>8.2f}x  "
            f"BW={effective_gbps:>8.3f} GB/s  "
            f"rel_err={rel_error:.3e}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()