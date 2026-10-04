#!/usr/bin/env python3
"""Benchmark serial-row LayerNorm V0 against PyTorch layer_norm."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from cuda_operator_lab.benchmarking import measure_us, summarize
from cuda_operator_lab.bindings import layernorm_v0_into, layernorm_v1_into
from cuda_operator_lab.references import layernorm


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
    parser.add_argument("--shapes", type=parse_shape, nargs="*", default=DEFAULT_SHAPES)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/layernorm_v0.csv"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    records: list[dict[str, object]] = []
    print(f"device={torch.cuda.get_device_name(0)}")

    for rows, cols in args.shapes:
        g = torch.Generator(device="cuda")
        g.manual_seed(20261005 + rows * 10000 + cols)
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float32, generator=g)
        weight = torch.randn(cols, device="cuda", dtype=torch.float32, generator=g)
        bias = torch.randn(cols, device="cuda", dtype=torch.float32, generator=g)
        out = torch.empty_like(x)

        expected = layernorm(x, weight, bias, args.eps)
        actual = layernorm_v0_into(x, weight, bias, out, args.eps)
        torch.cuda.synchronize()

        max_abs_error = (
            float((actual - expected).abs().max().item()) if x.numel() else 0.0
        )

        variants = [
            ("v0_serial_row", layernorm_v0_into),
            ("v1_block_shared", layernorm_v1_into),
        ]
        torch_times = measure_us(
            lambda: layernorm(x, weight, bias, args.eps),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        torch_median, _, torch_p95 = summarize(torch_times)

        for name, implementation in variants:
            out = torch.empty_like(x)
            actual = implementation(x, weight, bias, out, args.eps)
            torch.cuda.synchronize()
            max_abs_error = (
                float((actual - expected).abs().max().item()) if x.numel() else 0.0
            )
            ours = measure_us(
                lambda implementation=implementation, out=out: implementation(
                    x, weight, bias, out, args.eps
                ),
                warmup=args.warmup,
                repeats=args.repeats,
            )
            ours_median, _, ours_p95 = summarize(ours)

            record = {
                "operator": "layernorm",
                "variant": name,
                "dtype": "float32",
                "rows": rows,
                "cols": cols,
                "median_us": round(ours_median, 3),
                "p95_us": round(ours_p95, 3),
                "torch_median_us": round(torch_median, 3),
                "torch_p95_us": round(torch_p95, 3),
                "slowdown_vs_torch": round(ours_median / torch_median, 3)
                if torch_median
                else None,
                "max_abs_error": max_abs_error,
            }
            records.append(record)

            print(
                f"shape={rows:>4}x{cols:<5} "
                f"{name:<16} {ours_median:>10.3f} us  "
                f"torch={torch_median:>9.3f} us  "
                f"slowdown={ours_median/torch_median:>8.2f}x  "
                f"max_abs={max_abs_error:.3e}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
