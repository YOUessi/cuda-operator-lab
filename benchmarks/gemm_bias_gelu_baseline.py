#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from cuda_operator_lab.benchmarking import measure_us, summarize
from cuda_operator_lab.bindings import gemm_bias_gelu_v0_into
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

        ours = measure_us(
            lambda: gemm_bias_gelu_v0_into(x, weight, bias, out),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        torch_times = measure_us(
            lambda: gemm_bias_gelu(x, weight, bias),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        om, _, op95 = summarize(ours)
        tm, _, tp95 = summarize(torch_times)

        print(
            f"{m}x{k}x{n} ours={om:.3f} us "
            f"torch={tm:.3f} us ratio={om/tm:.3f}x max_abs={max_abs:.3e}"
        )
        records.append({
            "m": m,
            "k": k,
            "n": n,
            "ours_median_us": om,
            "ours_p95_us": op95,
            "torch_median_us": tm,
            "torch_p95_us": tp95,
            "ratio_vs_torch": om / tm if tm else None,
            "max_abs_error": max_abs,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)


if __name__ == "__main__":
    main()
