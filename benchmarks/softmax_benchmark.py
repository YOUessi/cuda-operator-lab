#!/usr/bin/env python3
"""Compare CUDA row-wise Softmax variants against PyTorch."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable
from pathlib import Path

import torch

from cuda_operator_lab.benchmarking import measure_us, summarize
from cuda_operator_lab.bindings import (
    softmax_v0_into,
    softmax_v1_into,
    softmax_v2_into,
    softmax_v3_into,
    softmax_v4_into,
    softmax_v5_into,
)
from cuda_operator_lab.references import row_softmax


DEFAULT_SHAPES = [
    (1, 128),
    (32, 128),
    (128, 128),
    (128, 512),
    (128, 1024),
    (128, 4096),
    (1024, 128),
    (1024, 512),
    (1024, 4096),
]

SoftmaxFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

VARIANTS: dict[str, SoftmaxFn] = {
    "v0_serial_row": softmax_v0_into,
    "v1_block_shared": softmax_v1_into,
    "v2_warp_shuffle": softmax_v2_into,
    "v3_width_aware": softmax_v3_into,
    "v4_warp_rows": softmax_v4_into,
    "v5_shape_dispatch": softmax_v5_into,
}


def parse_shape(value: str) -> tuple[int, int]:
    rows_text, sep, cols_text = value.lower().partition("x")
    if not sep:
        raise argparse.ArgumentTypeError("shape must be ROWSxCOLS")
    rows = int(rows_text)
    cols = int(cols_text)
    if rows < 0 or cols <= 0:
        raise argparse.ArgumentTypeError("rows >= 0 and cols > 0 are required")
    return rows, cols


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        type=parse_shape,
        nargs="*",
        default=DEFAULT_SHAPES,
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        choices=list(VARIANTS),
        default=list(VARIANTS),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/softmax_compare.csv"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    variants = [(name, VARIANTS[name]) for name in args.variants]
    rows_out: list[dict[str, object]] = []

    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"warmup={args.warmup} repeats={args.repeats}")

    for rows, cols in args.shapes:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(20261004 + rows * 10000 + cols)
        x = torch.randn(
            rows,
            cols,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        expected = row_softmax(x)
        torch.cuda.synchronize()

        torch_samples = measure_us(
            lambda: row_softmax(x),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        torch_median, _, torch_p95 = summarize(torch_samples)

        for name, implementation in variants:
            out = torch.empty_like(x)
            actual = implementation(x, out)
            torch.cuda.synchronize()

            max_abs_error = (
                float((actual - expected).abs().max().item())
                if x.numel()
                else 0.0
            )
            max_row_sum_error = (
                float((actual.sum(dim=-1) - 1.0).abs().max().item())
                if rows
                else 0.0
            )

            samples = measure_us(
                lambda implementation=implementation, out=out: implementation(x, out),
                warmup=args.warmup,
                repeats=args.repeats,
            )
            median_us, p50_us, p95_us = summarize(samples)
            logical_io_bytes = rows * cols * x.element_size() * 2
            logical_io_gbps = (
                logical_io_bytes / (median_us * 1e-6) / 1e9
                if median_us
                else 0.0
            )

            v5_uses_packed = (
                name == "v5_shape_dispatch"
                and (
                    (cols <= 64 and rows >= 4096)
                    or (64 < cols <= 128 and rows >= 2048)
                )
            )
            v5_uses_v3 = name == "v5_shape_dispatch" and not v5_uses_packed
            uses_width_aware_threads = (
                name == "v3_width_aware" or v5_uses_v3
            )
            threads_per_block = (
                128
                if name == "v0_serial_row"
                else (
                    32
                    if uses_width_aware_threads and cols <= 32
                    else 64
                    if uses_width_aware_threads and cols <= 64
                    else 128
                    if uses_width_aware_threads and cols <= 128
                    else 256
                )
            )
            v5_uses_packed = (
                name == "v5_shape_dispatch"
                and (
                    (cols <= 64 and rows >= 4096)
                    or (64 < cols <= 128 and rows >= 2048)
                )
            )
            rows_per_block = (
                8
                if (
                    (name == "v4_warp_rows" and cols <= 128)
                    or v5_uses_packed
                )
                else 1
            )

            record = {
                "operator": "row_softmax",
                "variant": name,
                "dtype": "float32",
                "rows": rows,
                "cols": cols,
                "threads_per_block": threads_per_block,
                "rows_per_block": rows_per_block,
                "elements": rows * cols,
                "logical_io_bytes": logical_io_bytes,
                "median_us": round(median_us, 3),
                "p50_us": round(p50_us, 3),
                "p95_us": round(p95_us, 3),
                "logical_io_gbps": round(logical_io_gbps, 6),
                "torch_median_us": round(torch_median, 3),
                "torch_p95_us": round(torch_p95, 3),
                "slowdown_vs_torch": round(median_us / torch_median, 3)
                if torch_median
                else None,
                "max_abs_error": max_abs_error,
                "max_row_sum_error": max_row_sum_error,
            }
            rows_out.append(record)

            print(
                f"shape={rows:>4}x{cols:<5} "
                f"{name:<16} {median_us:>10.3f} us  "
                f"threads={threads_per_block:>3}  "
                f"rows/block={rows_per_block:>2}  "
                f"torch={torch_median:>9.3f} us  "
                f"slowdown={median_us / torch_median:>8.2f}x  "
                f"max_abs={max_abs_error:.3e}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows_out[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
