#!/usr/bin/env python3
"""Benchmark serial-row RMSNorm V0 against a PyTorch reference."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from cuda_operator_lab.benchmarking import measure_us, summarize
from cuda_operator_lab.bindings import (
    rmsnorm_v0_into,
    rmsnorm_v1_into,
    rmsnorm_v2_into,
)
from cuda_operator_lab.references import rmsnorm


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
    (2048, 4096),
    (128, 8192),
]


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
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/rmsnorm_compare.csv"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    variants = [
        ("v0_serial_row", rmsnorm_v0_into),
        ("v1_block_shared", rmsnorm_v1_into),
        ("v2_warp_shuffle", rmsnorm_v2_into),
    ]
    rows_out: list[dict[str, object]] = []
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"warmup={args.warmup} repeats={args.repeats} eps={args.eps}")

    for rows, cols in args.shapes:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(20261005 + rows * 10000 + cols)
        x = torch.randn(
            rows,
            cols,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        weight = torch.randn(
            cols,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        expected = rmsnorm(x, weight, args.eps)
        torch.cuda.synchronize()

        torch_samples = measure_us(
            lambda: rmsnorm(x, weight, args.eps),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        torch_median, _, torch_p95 = summarize(torch_samples)

        for name, implementation in variants:
            out = torch.empty_like(x)
            actual = implementation(x, weight, out, args.eps)
            torch.cuda.synchronize()

            max_abs_error = (
                float((actual - expected).abs().max().item())
                if x.numel()
                else 0.0
            )

            ours_samples = measure_us(
                lambda implementation=implementation, out=out: implementation(
                    x, weight, out, args.eps
                ),
                warmup=args.warmup,
                repeats=args.repeats,
            )
            median_us, p50_us, p95_us = summarize(ours_samples)
            logical_io_bytes = rows * cols * x.element_size() * 3
            logical_io_gbps = (
                logical_io_bytes / (median_us * 1e-6) / 1e9
                if median_us
                else 0.0
            )

            record = {
                "operator": "rmsnorm",
                "variant": name,
                "dtype": "float32",
                "rows": rows,
                "cols": cols,
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
            }
            rows_out.append(record)

            print(
                f"shape={rows:>4}x{cols:<5} "
                f"{name:<16} {median_us:>10.3f} us  "
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
