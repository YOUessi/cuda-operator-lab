#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from cuda_operator_lab.benchmarking import measure_us, summarize
from cuda_operator_lab.bindings import gemm_swiglu_v2_into, gemm_swiglu_v4_into
from cuda_operator_lab.references import gemm_swiglu_bf16_fp32_reference


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
        g.manual_seed(20261016 + m * 1000000 + k * 1000 + n)

        x_fp32 = torch.randn(m, k, device="cuda", dtype=torch.float32, generator=g)
        packed_fp32 = torch.randn(
            2 * n, k, device="cuda", dtype=torch.float32, generator=g
        ).contiguous()

        x_bf16 = x_fp32.to(torch.bfloat16)
        packed_bf16 = packed_fp32.to(torch.bfloat16).contiguous()

        ws_fp32 = torch.empty(m, 2 * n, device="cuda", dtype=torch.float32)
        out_fp32 = torch.empty(m, n, device="cuda", dtype=torch.float32)

        ws_bf16tc = torch.empty_like(ws_fp32)
        out_bf16tc = torch.empty_like(out_fp32)

        torch_bf16_tmp = torch.empty(
            m, 2 * n, device="cuda", dtype=torch.bfloat16
        )
        torch_bf16_out = torch.empty(
            m, n, device="cuda", dtype=torch.bfloat16
        )

        expected = gemm_swiglu_bf16_fp32_reference(x_bf16, packed_bf16)

        actual = gemm_swiglu_v4_into(
            x_bf16, packed_bf16, ws_bf16tc, out_bf16tc
        )
        torch.cuda.synchronize()
        diff = (actual - expected).abs()
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        mean_abs = float(diff.mean().item()) if diff.numel() else 0.0

        v2 = measure_us(
            lambda: gemm_swiglu_v2_into(
                x_fp32, packed_fp32, ws_fp32, out_fp32
            ),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        v4 = measure_us(
            lambda: gemm_swiglu_v4_into(
                x_bf16, packed_bf16, ws_bf16tc, out_bf16tc
            ),
            warmup=args.warmup,
            repeats=args.repeats,
        )

        def torch_bf16_native() -> None:
            torch.mm(
                x_bf16,
                packed_bf16.transpose(0, 1),
                out=torch_bf16_tmp,
            )
            gate = torch_bf16_tmp[:, :n]
            up = torch_bf16_tmp[:, n:]
            torch.ops.aten.silu.out(gate, out=torch_bf16_out)
            torch_bf16_out.mul_(up)

        tbf16 = measure_us(
            torch_bf16_native,
            warmup=args.warmup,
            repeats=args.repeats,
        )

        v2m, _, v2p95 = summarize(v2)
        v4m, _, v4p95 = summarize(v4)
        tbm, _, tbp95 = summarize(tbf16)

        print(
            f"{m}x{k}x{n} "
            f"v2_fp32={v2m:.3f} us "
            f"v4_bf16tc={v4m:.3f} us "
            f"torch_bf16={tbm:.3f} us "
            f"speedup={v2m/v4m:.3f}x "
            f"max_abs={max_abs:.3e} mean_abs={mean_abs:.3e}"
        )

        records.append({
            "m": m,
            "k": k,
            "n": n,
            "v2_fp32_median_us": v2m,
            "v2_fp32_p95_us": v2p95,
            "v4_bf16tc_median_us": v4m,
            "v4_bf16tc_p95_us": v4p95,
            "torch_bf16_native_median_us": tbm,
            "torch_bf16_native_p95_us": tbp95,
            "v2_to_v4_speedup": v2m / v4m if v4m else None,
            "max_abs_vs_bf16_fp32_ref": max_abs,
            "mean_abs_vs_bf16_fp32_ref": mean_abs,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)


if __name__ == "__main__":
    main()
