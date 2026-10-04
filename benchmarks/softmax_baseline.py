#!/usr/bin/env python3
"""Benchmark the serial-per-row CUDA Softmax baseline against PyTorch."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from cuda_operator_lab.benchmarking import measure_us, summarize
from cuda_operator_lab.bindings import softmax_v0_into
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
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/softmax_v0.csv"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

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
        out = torch.empty_like(x)

        expected = row_softmax(x)
        actual = softmax_v0_into(x, out)
        torch.cuda.synchronize()

        max_abs_error = (
            float((actual - expected).abs().max().item()) if x.numel() else 0.0
        )
        max_row_sum_error = (
            float((actual.sum(dim=-1) - 1.0).abs().max().item())
            if rows
            else 0.0
        )

        ours_samples = measure_us(
            lambda: softmax_v0_into(x, out),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        torch_samples = measure_us(
            lambda: row_softmax(x),
            warmup=args.warmup,
            repeats=args.repeats,
        )

        ours_median, ours_p50, ours_p95 = summarize(ours_samples)
        torch_median, _, torch_p95 = summarize(torch_samples)

        logical_io_bytes = rows * cols * x.element_size() * 2
        logical_io_gbps = (
            logical_io_bytes / (ours_median * 1e-6) / 1e9
            if ours_median
            else 0.0
        )

        record = {
            "operator": "row_softmax",
            "variant": "v0_serial_row",
            "dtype": "float32",
            "rows": rows,
            "cols": cols,
            "elements": rows * cols,
            "logical_io_bytes": logical_io_bytes,
            "median_us": round(ours_median, 3),
            "p50_us": round(ours_p50, 3),
            "p95_us": round(ours_p95, 3),
            "logical_io_gbps": round(logical_io_gbps, 6),
            "torch_median_us": round(torch_median, 3),
            "torch_p95_us": round(torch_p95, 3),
            "slowdown_vs_torch": round(ours_median / torch_median, 3)
            if torch_median
            else None,
            "max_abs_error": max_abs_error,
            "max_row_sum_error": max_row_sum_error,
        }
        rows_out.append(record)

        print(
            f"shape={rows:>4}x{cols:<5} "
            f"v0={ours_median:>10.3f} us  "
            f"torch={torch_median:>9.3f} us  "
            f"slowdown={ours_median / torch_median:>8.2f}x  "
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
