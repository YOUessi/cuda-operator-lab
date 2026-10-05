#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from cuda_operator_lab.benchmarking import measure_us, summarize
from cuda_operator_lab.bindings import fused_bias_gelu_v0_into, fused_bias_gelu_v1_into
from cuda_operator_lab.references import fused_bias_gelu


DEFAULT_SHAPES = [
    (1, 128),
    (128, 128),
    (128, 512),
    (128, 1024),
    (128, 4096),
    (1024, 512),
    (1024, 4096),
    (2048, 4096),
]


def parse_shape(value: str) -> tuple[int, int]:
    a, sep, b = value.lower().partition("x")
    if not sep:
        raise argparse.ArgumentTypeError("shape must be ROWSxCOLS")
    return int(a), int(b)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", nargs="*", type=parse_shape, default=DEFAULT_SHAPES)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--output", type=Path, default=Path("benchmarks/results/fused_bias_gelu_v0.csv"))
    args = p.parse_args()

    records = []
    print(f"device={torch.cuda.get_device_name(0)}")

    for rows, cols in args.shapes:
        g = torch.Generator(device="cuda")
        g.manual_seed(20261005 + rows * 10000 + cols)
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float32, generator=g)
        bias = torch.randn(cols, device="cuda", dtype=torch.float32, generator=g)
        out = torch.empty_like(x)

        actual = fused_bias_gelu_v0_into(x, bias, out)
        expected = fused_bias_gelu(x, bias)
        torch.cuda.synchronize()
        max_abs = float((actual - expected).abs().max().item()) if x.numel() else 0.0

        unfused = measure_us(lambda: fused_bias_gelu(x, bias), warmup=args.warmup, repeats=args.repeats)
        um, _, up95 = summarize(unfused)

        variants = [
            ("v0_scalar", fused_bias_gelu_v0_into),
            ("v1_float4", fused_bias_gelu_v1_into),
        ]

        for name, fn in variants:
            out = torch.empty_like(x)
            actual = fn(x, bias, out)
            torch.cuda.synchronize()
            max_abs = float((actual - expected).abs().max().item()) if x.numel() else 0.0
            samples = measure_us(lambda fn=fn,out=out: fn(x,bias,out), warmup=args.warmup, repeats=args.repeats)
            fm, _, fp95 = summarize(samples)
            print(f"{rows}x{cols} {name}={fm:.3f} us torch_unfused={um:.3f} us speedup={um/fm:.3f}x max_abs={max_abs:.3e}")
            records.append({
                "rows": rows,
                "cols": cols,
                "variant": name,
                "fused_median_us": fm,
                "fused_p95_us": fp95,
                "torch_unfused_median_us": um,
                "torch_unfused_p95_us": up95,
                "speedup_vs_torch_unfused": um / fm if fm else None,
                "max_abs_error": max_abs,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)


if __name__ == "__main__":
    main()
