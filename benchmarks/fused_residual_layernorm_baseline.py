#!/usr/bin/env python3
"""Benchmark fused residual-add + LayerNorm V0."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from cuda_operator_lab.benchmarking import measure_us, summarize
from cuda_operator_lab.bindings import fused_residual_layernorm_v0_into, fused_residual_layernorm_v1_into, fused_residual_layernorm_v2_into, fused_residual_layernorm_v3_into
from cuda_operator_lab.references import fused_residual_layernorm


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
    rows, cols = int(rows_text), int(cols_text)
    if rows < 0 or cols <= 0:
        raise argparse.ArgumentTypeError("rows >= 0 and cols > 0 are required")
    return rows, cols


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", nargs="*", type=parse_shape, default=DEFAULT_SHAPES)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/fused_residual_layernorm_v0.csv"),
    )
    args = parser.parse_args()

    records = []
    print(f"device={torch.cuda.get_device_name(0)}")

    for rows, cols in args.shapes:
        g = torch.Generator(device="cuda")
        g.manual_seed(20261005 + rows * 10000 + cols)
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float32, generator=g)
        residual = torch.randn(rows, cols, device="cuda", dtype=torch.float32, generator=g)
        weight = torch.randn(cols, device="cuda", dtype=torch.float32, generator=g)
        bias = torch.randn(cols, device="cuda", dtype=torch.float32, generator=g)
        out = torch.empty_like(x)

        expected = fused_residual_layernorm(x, residual, weight, bias, args.eps)
        actual = fused_residual_layernorm_v0_into(
            x, residual, weight, bias, out, args.eps
        )
        torch.cuda.synchronize()

        variants = [
            ("v0_serial_fused", fused_residual_layernorm_v0_into),
            ("v1_block_shared", fused_residual_layernorm_v1_into),
            ("v2_warp_shuffle", fused_residual_layernorm_v2_into),
            ("v3_float4_io", fused_residual_layernorm_v3_into),
        ]
        unfused_times = measure_us(
            lambda: fused_residual_layernorm(
                x, residual, weight, bias, args.eps
            ),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        unfused_median, _, unfused_p95 = summarize(unfused_times)

        for name, implementation in variants:
            out = torch.empty_like(x)
            actual = implementation(x, residual, weight, bias, out, args.eps)
            torch.cuda.synchronize()
            max_abs = float((actual - expected).abs().max().item()) if x.numel() else 0.0
            fused_times = measure_us(
                lambda implementation=implementation, out=out: implementation(
                    x, residual, weight, bias, out, args.eps
                ),
                warmup=args.warmup,
                repeats=args.repeats,
            )
            fused_median, _, fused_p95 = summarize(fused_times)

            print(
                f"shape={rows:>4}x{cols:<5} "
                f"{name:<16} {fused_median:>10.3f} us "
                f"torch_unfused={unfused_median:>10.3f} us "
                f"ratio={fused_median/unfused_median:>7.2f}x "
                f"max_abs={max_abs:.3e}"
            )

            records.append(
                {
                    "rows": rows,
                    "cols": cols,
                    "variant": name,
                    "median_us": round(fused_median, 3),
                    "p95_us": round(fused_p95, 3),
                    "torch_unfused_median_us": round(unfused_median, 3),
                    "torch_unfused_p95_us": round(unfused_p95, 3),
                    "ratio_vs_torch_unfused": round(fused_median / unfused_median, 3)
                    if unfused_median
                    else None,
                    "max_abs_error": max_abs,
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
