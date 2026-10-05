#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from cuda_operator_lab.benchmarking import measure_us, summarize
from cuda_operator_lab.bindings import gemm_swiglu_v4_into, gemm_swiglu_v10_into


DEFAULT_SHAPES = [
    (16, 64, 64),
    (32, 128, 256),
    (64, 128, 1024),
    (128, 64, 2048),
    (32, 256, 64),
    (128, 512, 512),
    (128, 1024, 4096),
    (512, 1024, 4096),
]


def parse_shape(value: str) -> tuple[int, int, int]:
    parts = value.lower().split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be MxKxN")
    return tuple(int(x) for x in parts)


def uses_custom(m: int, k: int, n: int) -> bool:
    return (
        m <= 128
        and k <= 128
        and n <= 2048
        and m % 16 == 0
        and k % 16 == 0
        and n % 64 == 0
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", nargs="*", type=parse_shape, default=DEFAULT_SHAPES)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    records = []
    print(f"device={torch.cuda.get_device_name(0)}")

    for m, k, n in args.shapes:
        g = torch.Generator(device="cuda")
        g.manual_seed(20261024 + m * 1000000 + k * 1000 + n)

        x = torch.randn(m, k, device="cuda", dtype=torch.float32, generator=g).to(torch.bfloat16)
        packed_w = torch.randn(
            2 * n, k, device="cuda", dtype=torch.float32, generator=g
        ).to(torch.bfloat16).contiguous()

        ws_v4 = torch.empty(m, 2 * n, device="cuda", dtype=torch.float32)
        out_v4 = torch.empty(m, n, device="cuda", dtype=torch.float32)
        ws_v10 = torch.empty_like(ws_v4)
        out_v10 = torch.empty_like(out_v4)

        v4 = measure_us(
            lambda: gemm_swiglu_v4_into(x, packed_w, ws_v4, out_v4),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        v10 = measure_us(
            lambda: gemm_swiglu_v10_into(x, packed_w, ws_v10, out_v10),
            warmup=args.warmup,
            repeats=args.repeats,
        )

        v4m, _, v4p95 = summarize(v4)
        v10m, _, v10p95 = summarize(v10)
        path = "custom_v7" if uses_custom(m, k, n) else "cublas_v4"

        print(
            f"{m}x{k}x{n} path={path} "
            f"v4={v4m:.3f} us "
            f"v10={v10m:.3f} us "
            f"speedup={v4m/v10m:.3f}x"
        )

        records.append({
            "m": m,
            "k": k,
            "n": n,
            "v10_path": path,
            "v4_median_us": v4m,
            "v4_p95_us": v4p95,
            "v10_median_us": v10m,
            "v10_p95_us": v10p95,
            "v4_to_v10_speedup": v4m / v10m if v10m else None,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)


if __name__ == "__main__":
    main()
