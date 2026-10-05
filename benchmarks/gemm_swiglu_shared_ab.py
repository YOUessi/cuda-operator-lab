#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from cuda_operator_lab.benchmarking import measure_us, summarize
from cuda_operator_lab.bindings import (
    gemm_swiglu_v4_into,
    gemm_swiglu_v7_into,
    gemm_swiglu_v8_into,
)
from cuda_operator_lab.references import gemm_swiglu_bf16_dual_fp32_reference


DEFAULT_SHAPES = [
    (32, 128, 256),
    (128, 512, 512),
    (128, 1024, 4096),
    (512, 1024, 4096),
    (512, 4096, 4096),
]


def parse_shape(value: str) -> tuple[int, int, int]:
    parts = value.lower().split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be MxKxN")
    m, k, n = (int(x) for x in parts)
    if m % 32 or k % 16 or n % 64:
        raise argparse.ArgumentTypeError(
            "V8 benchmark requires M multiple of 32, K multiple of 16, N multiple of 64"
        )
    return m, k, n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", nargs="*", type=parse_shape, default=DEFAULT_SHAPES)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    records = []
    print(f"device={torch.cuda.get_device_name(0)}")

    for m, k, n in args.shapes:
        g = torch.Generator(device="cuda")
        g.manual_seed(20261020 + m * 1000000 + k * 1000 + n)

        x = torch.randn(m, k, device="cuda", dtype=torch.float32, generator=g).to(torch.bfloat16)
        gate_w = torch.randn(n, k, device="cuda", dtype=torch.float32, generator=g).to(torch.bfloat16)
        up_w = torch.randn(n, k, device="cuda", dtype=torch.float32, generator=g).to(torch.bfloat16)
        packed_w = torch.cat((gate_w, up_w), dim=0).contiguous()

        ws_v4 = torch.empty(m, 2 * n, device="cuda", dtype=torch.float32)
        out_v4 = torch.empty(m, n, device="cuda", dtype=torch.float32)
        out_v7 = torch.empty_like(out_v4)
        out_v8 = torch.empty_like(out_v4)

        ref = gemm_swiglu_bf16_dual_fp32_reference(x, gate_w, up_w)

        y7 = gemm_swiglu_v7_into(x, gate_w, up_w, out_v7)
        y8 = gemm_swiglu_v8_into(x, gate_w, up_w, out_v8)
        torch.cuda.synchronize()

        d7 = (y7 - ref).abs()
        d8 = (y8 - ref).abs()

        v4 = measure_us(
            lambda: gemm_swiglu_v4_into(x, packed_w, ws_v4, out_v4),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        v7 = measure_us(
            lambda: gemm_swiglu_v7_into(x, gate_w, up_w, out_v7),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        v8 = measure_us(
            lambda: gemm_swiglu_v8_into(x, gate_w, up_w, out_v8),
            warmup=args.warmup,
            repeats=args.repeats,
        )

        v4m, _, v4p95 = summarize(v4)
        v7m, _, v7p95 = summarize(v7)
        v8m, _, v8p95 = summarize(v8)

        print(
            f"{m}x{k}x{n} "
            f"v4={v4m:.3f} us "
            f"v7={v7m:.3f} us "
            f"v8={v8m:.3f} us "
            f"v7_to_v8={v7m/v8m:.3f}x "
            f"v8_over_v4={v8m/v4m:.3f}x "
            f"max_abs={float(d8.max()):.3e}"
        )

        records.append({
            "m": m,
            "k": k,
            "n": n,
            "v4_cublas_median_us": v4m,
            "v4_cublas_p95_us": v4p95,
            "v7_shared_a_median_us": v7m,
            "v7_shared_a_p95_us": v7p95,
            "v8_shared_ab_median_us": v8m,
            "v8_shared_ab_p95_us": v8p95,
            "v7_to_v8_speedup": v7m / v8m if v8m else None,
            "v8_over_v4_ratio": v8m / v4m if v4m else None,
            "v7_max_abs": float(d7.max().item()),
            "v8_max_abs": float(d8.max().item()),
            "v8_mean_abs": float(d8.mean().item()),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)


if __name__ == "__main__":
    main()
