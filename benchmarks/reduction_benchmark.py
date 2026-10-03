#!/usr/bin/env python3
"""Compare reduction implementations against torch.sum."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable
from pathlib import Path

import torch

from cuda_operator_lab.benchmarking import measure_us, summarize
from cuda_operator_lab.bindings import reduction_v0_into, reduction_v1_into
from cuda_operator_lab.references import reduction_sum


DEFAULT_SIZES = [2**10, 2**14, 2**18, 2**22, 2**24]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="*", default=DEFAULT_SIZES)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/reduction_compare.csv"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    variants: list[tuple[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]] = [
        ("v0_serial", reduction_v0_into),
        ("v1_parallel_atomic", reduction_v1_into),
    ]

    rows: list[dict[str, object]] = []
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"warmup={args.warmup} repeats={args.repeats}")

    for n in args.sizes:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(20261004 + n)
        x = torch.rand(n, device="cuda", dtype=torch.float32, generator=generator)
        expected = reduction_sum(x)
        torch.cuda.synchronize()
        expected_value = float(expected.item())

        torch_samples = measure_us(
            lambda: reduction_sum(x),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        torch_median, _, torch_p95 = summarize(torch_samples)

        for name, implementation in variants:
            out = torch.empty(1, device="cuda", dtype=torch.float32)
            actual = implementation(x, out)
            torch.cuda.synchronize()

            actual_value = float(actual.item())
            abs_error = abs(actual_value - expected_value)
            rel_error = abs_error / max(abs(expected_value), 1e-12)

            samples = measure_us(
                lambda implementation=implementation, out=out: implementation(x, out),
                warmup=args.warmup,
                repeats=args.repeats,
            )
            median_us, p50_us, p95_us = summarize(samples)
            bytes_read = n * x.element_size()
            effective_gbps = (
                bytes_read / (median_us * 1e-6) / 1e9 if median_us else 0.0
            )

            row = {
                "operator": "reduction_sum",
                "variant": name,
                "dtype": "float32",
                "n": n,
                "bytes_read": bytes_read,
                "median_us": round(median_us, 3),
                "p50_us": round(p50_us, 3),
                "p95_us": round(p95_us, 3),
                "effective_gbps": round(effective_gbps, 6),
                "torch_median_us": round(torch_median, 3),
                "torch_p95_us": round(torch_p95, 3),
                "slowdown_vs_torch": round(median_us / torch_median, 3),
                "abs_error": abs_error,
                "rel_error": rel_error,
            }
            rows.append(row)
            print(
                f"n={n:>9}  {name:<18} "
                f"{median_us:>10.3f} us  "
                f"torch={torch_median:>9.3f} us  "
                f"slowdown={median_us / torch_median:>8.2f}x  "
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
