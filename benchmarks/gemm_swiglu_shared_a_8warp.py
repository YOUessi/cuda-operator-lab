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
    gemm_swiglu_v9_into,
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
    if m % 16 or k % 16 or n % 128:
        raise argparse.ArgumentTypeError(
            "V9 benchmark requires M,K multiples of 16 and N multiple of 128"
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
        g.manual_seed(20261021 + m * 1000000 + k * 1000 + n)

        x = torch.randn(m, k, device="cuda", dtype=torch.float32, generator=g).to(torch.bfloat16)
        gate_w = torch.randn(n, k, device="cuda", dtype=torch.float32, generator=g).to(torch.bfloat16)
        up_w = torch.randn(n, k, device="cuda", dtype=torch.float32, generator=g).to(torch.bfloat16)
        packed_w = torch.cat((gate_w, up_w), dim=0).contiguous()

        ws_v4 = torch.empty(m, 2 * n, device="cuda", dtype=torch.float32)
        out_v4 = torch.empty(m, n, device="cuda", dtype=torch.float32)
        out_v7 = torch.empty_like(out_v4)
        out_v9 = torch.empty_like(out_v4)

        ref = gemm_swiglu_bf16_dual_fp32_reference(x, gate_w, up_w)

        y9 = gemm_swiglu_v9_into(x, gate_w, up_w, out_v9)
        torch.cuda.synchronize()
        d9 = (y9 - ref).abs()

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
        v9 = measure_us(
            lambda: gemm_swiglu_v9_into(x, gate_w, up_w, out_v9),
            warmup=args.warmup,
            repeats=args.repeats,
        )

        v4m, _, v4p95 = summarize(v4)
        v7m, _, v7p95 = summarize(v7)
        v9m, _, v9p95 = summarize(v9)

        print(
            f"{m}x{k}x{n} "
            f"v4={v4m:.3f} us "
            f"v7={v7m:.3f} us "
            f"v9={v9m:.3f} us "
            f"v7_to_v9={v7m/v9m:.3f}x "
            f"v9_over_v4={v9m/v4m:.3f}x "
            f"max_abs={float(d9.max()):.3e}"
        )

        records.append({
            "m": m,
            "k": k,
            "n": n,
            "v4_cublas_median_us": v4m,
            "v4_cublas_p95_us": v4p95,
            "v7_4warp_shared_a_median_us": v7m,
            "v7_4warp_shared_a_p95_us": v7p95,
            "v9_8warp_shared_a_median_us": v9m,
            "v9_8warp_shared_a_p95_us": v9p95,
            "v7_to_v9_speedup": v7m / v9m if v9m else None,
            "v9_over_v4_ratio": v9m / v4m if v4m else None,
            "v9_max_abs": float(d9.max().item()),
            "v9_mean_abs": float(d9.mean().item()),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)


if __name__ == "__main__":
    main()
