#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from cuda_operator_lab.benchmarking import measure_us, summarize
from cuda_operator_lab.bindings import gemm_bias_gelu_v0_into, gemm_bias_gelu_v1_into, gemm_bias_gelu_v2_into
from cuda_operator_lab.references import gemm_bias_gelu


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
    p.add_argument("--output", type=Path, default=Path("benchmarks/results/gemm_bias_gelu_v0.csv"))
    args = p.parse_args()

    records = []
    print(f"device={torch.cuda.get_device_name(0)}")

    for m, k, n in args.shapes:
        g = torch.Generator(device="cuda")
        g.manual_seed(20261005 + m * 1000000 + k * 1000 + n)
        x = torch.randn(m, k, device="cuda", dtype=torch.float32, generator=g)
        weight = torch.randn(n, k, device="cuda", dtype=torch.float32, generator=g)
        bias = torch.randn(n, device="cuda", dtype=torch.float32, generator=g)
        out = torch.empty(m, n, device="cuda", dtype=torch.float32)

        gemm_bias_gelu_v0_into(x, weight, bias, out)
        expected = gemm_bias_gelu(x, weight, bias)
        torch.cuda.synchronize()
        max_abs = float((out - expected).abs().max().item()) if out.numel() else 0.0

        tmp = torch.empty(m, n, device="cuda", dtype=torch.float32)
        torch_out = torch.empty_like(tmp)

        def torch_unfused_preallocated() -> None:
            torch.mm(x, weight.transpose(0, 1), out=tmp)
            tmp.add_(bias)
            torch.ops.aten.gelu.out(tmp, approximate="none", out=torch_out)

        variants = [
            ("v0_scalar_epilogue", gemm_bias_gelu_v0_into),
            ("v1_float4_epilogue", gemm_bias_gelu_v1_into),
            ("v2_cublaslt_fused", gemm_bias_gelu_v2_into),
        ]
        torch_alloc = measure_us(
            lambda: gemm_bias_gelu(x, weight, bias),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        torch_prealloc = measure_us(
            torch_unfused_preallocated,
            warmup=args.warmup,
            repeats=args.repeats,
        )

        ta, _, ta95 = summarize(torch_alloc)
        tp, _, tp95 = summarize(torch_prealloc)

        for name, fn in variants:
            out = torch.empty(m, n, device="cuda", dtype=torch.float32)
            fn(x, weight, bias, out)
            torch.cuda.synchronize()
            max_abs = float((out - expected).abs().max().item()) if out.numel() else 0.0
            samples = measure_us(
                lambda fn=fn,out=out: fn(x, weight, bias, out),
                warmup=args.warmup,
                repeats=args.repeats,
            )
            om, _, op95 = summarize(samples)
            print(
                f"{m}x{k}x{n} {name}={om:.3f} us "
                f"torch_prealloc={tp:.3f} us "
                f"ratio_prealloc={om/tp:.3f}x max_abs={max_abs:.3e}"
            )
            records.append({
                "m": m,
                "k": k,
                "n": n,
                "variant": name,
                "ours_median_us": om,
                "ours_p95_us": op95,
                "torch_prealloc_median_us": tp,
                "torch_prealloc_p95_us": tp95,
                "torch_alloc_median_us": ta,
                "torch_alloc_p95_us": ta95,
                "ratio_vs_torch_prealloc": om / tp if tp else None,
                "ratio_vs_torch_alloc": om / ta if ta else None,
                "max_abs_error": max_abs,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)


if __name__ == "__main__":
    main()
