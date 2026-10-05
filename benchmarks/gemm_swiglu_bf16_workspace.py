#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from cuda_operator_lab.benchmarking import measure_us, summarize
from cuda_operator_lab.bindings import gemm_swiglu_v4_into, gemm_swiglu_v5_into
from cuda_operator_lab.references import (
    gemm_swiglu_bf16_fp32_reference,
    gemm_swiglu_bf16_workspace_reference,
)


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
    return tuple(int(x) for x in parts)


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
        g.manual_seed(20261017 + m * 1000000 + k * 1000 + n)

        x = torch.randn(m, k, device="cuda", dtype=torch.float32, generator=g).to(torch.bfloat16)
        packed_weight = torch.randn(
            2 * n, k, device="cuda", dtype=torch.float32, generator=g
        ).to(torch.bfloat16).contiguous()

        ws_v4 = torch.empty(m, 2 * n, device="cuda", dtype=torch.float32)
        out_v4 = torch.empty(m, n, device="cuda", dtype=torch.float32)

        ws_v5 = torch.empty(m, 2 * n, device="cuda", dtype=torch.bfloat16)
        out_v5 = torch.empty(m, n, device="cuda", dtype=torch.float32)

        ref_v4 = gemm_swiglu_bf16_fp32_reference(x, packed_weight)
        ref_v5 = gemm_swiglu_bf16_workspace_reference(x, packed_weight)

        y4 = gemm_swiglu_v4_into(x, packed_weight, ws_v4, out_v4)
        y5 = gemm_swiglu_v5_into(x, packed_weight, ws_v5, out_v5)
        torch.cuda.synchronize()

        d4 = (y4 - ref_v4).abs()
        d5 = (y5 - ref_v5).abs()
        quality_drop = (y5 - ref_v4).abs()

        v4 = measure_us(
            lambda: gemm_swiglu_v4_into(x, packed_weight, ws_v4, out_v4),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        v5 = measure_us(
            lambda: gemm_swiglu_v5_into(x, packed_weight, ws_v5, out_v5),
            warmup=args.warmup,
            repeats=args.repeats,
        )

        v4m, _, v4p95 = summarize(v4)
        v5m, _, v5p95 = summarize(v5)

        print(
            f"{m}x{k}x{n} "
            f"v4_fp32ws={v4m:.3f} us "
            f"v5_bf16ws={v5m:.3f} us "
            f"speedup={v4m/v5m:.3f}x "
            f"v5_ref_max={float(d5.max()):.3e} "
            f"quality_vs_v4ref_max={float(quality_drop.max()):.3e}"
        )

        records.append({
            "m": m,
            "k": k,
            "n": n,
            "v4_fp32_workspace_median_us": v4m,
            "v4_fp32_workspace_p95_us": v4p95,
            "v5_bf16_workspace_median_us": v5m,
            "v5_bf16_workspace_p95_us": v5p95,
            "v4_to_v5_speedup": v4m / v5m if v5m else None,
            "v4_max_abs_vs_fp32_workspace_ref": float(d4.max().item()),
            "v5_max_abs_vs_bf16_workspace_ref": float(d5.max().item()),
            "v5_mean_abs_vs_bf16_workspace_ref": float(d5.mean().item()),
            "v5_max_abs_vs_v4_reference": float(quality_drop.max().item()),
            "v5_mean_abs_vs_v4_reference": float(quality_drop.mean().item()),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)


if __name__ == "__main__":
    main()
