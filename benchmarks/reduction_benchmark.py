#!/usr/bin/env python3
"""Compare reduction implementations against torch.sum."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable
from pathlib import Path

import torch

from cuda_operator_lab.benchmarking import measure_us, summarize
from cuda_operator_lab.bindings import (
    reduction_v0_into,
    reduction_v1_into,
    reduction_v2_into,
    reduction_v3_into,
    reduction_v4_into,
    reduction_v5_into,
)
from cuda_operator_lab.references import reduction_sum


DEFAULT_SIZES = [2**10, 2**14, 2**18, 2**22, 2**24]

ReductionFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

VARIANTS: dict[str, ReductionFn] = {
    "v0_serial": reduction_v0_into,
    "v1_parallel_atomic": reduction_v1_into,
    "v2_shared_memory": reduction_v2_into,
    "v3_warp_shuffle": reduction_v3_into,
    "v4_float4": reduction_v4_into,
    "v5_shape_aware_float4": reduction_v5_into,
}


def make_cache_preparer(
    mode: str,
) -> tuple[Callable[[], object] | None, int]:
    if mode == "hot":
        return None, 0

    properties = torch.cuda.get_device_properties(0)
    l2_bytes = int(properties.L2_cache_size)
    flush_bytes = max(2 * l2_bytes, 128 * 1024 * 1024)
    flush_elements = (flush_bytes + 3) // 4
    flush = torch.zeros(flush_elements, device="cuda", dtype=torch.float32)

    def prepare() -> object:
        return flush.add_(1.0)

    return prepare, flush.numel() * flush.element_size()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="*", default=DEFAULT_SIZES)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--variants",
        nargs="*",
        choices=list(VARIANTS),
        default=list(VARIANTS),
        help="CUDA variants to benchmark; defaults to all.",
    )
    parser.add_argument(
        "--cache-mode",
        choices=["hot", "cold"],
        default="hot",
        help=(
            "hot reuses the same input normally; cold touches a buffer at least "
            "2x L2 before each timed launch to evict the input working set."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/reduction_compare.csv"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    variants = [(name, VARIANTS[name]) for name in args.variants]
    cache_prepare, flush_bytes = make_cache_preparer(args.cache_mode)

    rows: list[dict[str, object]] = []
    properties = torch.cuda.get_device_properties(0)
    print(f"device={properties.name}")
    print(
        f"warmup={args.warmup} repeats={args.repeats} "
        f"cache_mode={args.cache_mode} l2_bytes={properties.L2_cache_size} "
        f"flush_bytes={flush_bytes}"
    )

    for n in args.sizes:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(20261004 + n)
        x = torch.rand(n, device="cuda", dtype=torch.float32, generator=generator)
        alignment_mod16 = x.data_ptr() % 16
        expected = reduction_sum(x)
        torch.cuda.synchronize()
        expected_value = float(expected.item())

        torch_samples = measure_us(
            lambda: reduction_sum(x),
            warmup=args.warmup,
            repeats=args.repeats,
            before_each=cache_prepare,
        )
        torch_median, _, torch_p95 = summarize(torch_samples)

        for name, implementation in variants:
            out = torch.empty(1, device="cuda", dtype=torch.float32)
            actual = implementation(x, out)
            torch.cuda.synchronize()

            actual_value = float(actual.item())
            abs_error = abs(actual_value - expected_value)
            rel_error = abs_error / max(abs(expected_value), 1e-12)

            samples = measure_us(
                lambda implementation=implementation, out=out: implementation(x, out),
                warmup=args.warmup,
                repeats=args.repeats,
                before_each=cache_prepare,
            )
            median_us, p50_us, p95_us = summarize(samples)
            bytes_read = n * x.element_size()
            logical_input_gbps = (
                bytes_read / (median_us * 1e-6) / 1e9 if median_us else 0.0
            )

            row = {
                "operator": "reduction_sum",
                "variant": name,
                "dtype": "float32",
                "cache_mode": args.cache_mode,
                "l2_bytes": int(properties.L2_cache_size),
                "flush_bytes": flush_bytes,
                "input_ptr_mod16": alignment_mod16,
                "n": n,
                "bytes_read": bytes_read,
                "median_us": round(median_us, 3),
                "p50_us": round(p50_us, 3),
                "p95_us": round(p95_us, 3),
                "logical_input_gbps": round(logical_input_gbps, 6),
                "torch_median_us": round(torch_median, 3),
                "torch_p95_us": round(torch_p95, 3),
                "slowdown_vs_torch": round(median_us / torch_median, 3),
                "abs_error": abs_error,
                "rel_error": rel_error,
            }
            rows.append(row)
            print(
                f"n={n:>9}  {name:<18} "
                f"{median_us:>10.3f} us  "
                f"torch={torch_median:>9.3f} us  "
                f"slowdown={median_us / torch_median:>8.2f}x  "
                f"logical={logical_input_gbps:>8.3f} GB/s  "
                f"align16={alignment_mod16:>2}  "
                f"rel_err={rel_error:.3e}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
