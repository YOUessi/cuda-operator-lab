#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from cuda_operator_lab.benchmarking import measure_us, summarize
from cuda_operator_lab.bindings import gemm_swiglu_v0_into, gemm_swiglu_v1_into, gemm_swiglu_v2_into, gemm_swiglu_v3_into
from cuda_operator_lab.references import gemm_swiglu


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
    p.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/gemm_swiglu_v0.csv"),
    )
    args = p.parse_args()

    records = []
    print(f"device={torch.cuda.get_device_name(0)}")

    for m, k, n in args.shapes:
        g = torch.Generator(device="cuda")
        g.manual_seed(20261006 + m * 1000000 + k * 1000 + n)
        x = torch.randn(m, k, device="cuda", dtype=torch.float32, generator=g)
        gate_w = torch.randn(n, k, device="cuda", dtype=torch.float32, generator=g)
        up_w = torch.randn(n, k, device="cuda", dtype=torch.float32, generator=g)

        workspace = torch.empty(m, n, device="cuda", dtype=torch.float32)
        out = torch.empty_like(workspace)

        gate_tmp = torch.empty_like(workspace)
        up_tmp = torch.empty_like(workspace)
        torch_out = torch.empty_like(workspace)
        packed_weight = torch.cat((gate_w, up_w), dim=0).contiguous()
        packed_tmp = torch.empty(m, 2 * n, device="cuda", dtype=torch.float32)
        packed_out = torch.empty(m, n, device="cuda", dtype=torch.float32)

        expected = gemm_swiglu(x, gate_w, up_w)
        actual = gemm_swiglu_v0_into(x, gate_w, up_w, workspace, out)
        torch.cuda.synchronize()
        max_abs = float((actual - expected).abs().max().item()) if out.numel() else 0.0

        def torch_preallocated() -> None:
            torch.mm(x, gate_w.transpose(0, 1), out=gate_tmp)
            torch.mm(x, up_w.transpose(0, 1), out=up_tmp)
            torch.ops.aten.silu.out(gate_tmp, out=torch_out)
            torch_out.mul_(up_tmp)

        variants = [
            ("v0_scalar_post", gemm_swiglu_v0_into),
            ("v1_float4_post", gemm_swiglu_v1_into),
        ]
        def torch_packed_preallocated() -> None:
            torch.mm(x, packed_weight.transpose(0, 1), out=packed_tmp)
            gate_view = packed_tmp[:, :n]
            up_view = packed_tmp[:, n:]
            torch.ops.aten.silu.out(gate_view, out=packed_out)
            packed_out.mul_(up_view)

        torch_pre = measure_us(
            torch_preallocated,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        torch_packed = measure_us(
            torch_packed_preallocated,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        torch_alloc = measure_us(
            lambda: gemm_swiglu(x, gate_w, up_w),
            warmup=args.warmup,
            repeats=args.repeats,
        )

        tp, _, tp95 = summarize(torch_pre)
        tpack, _, tpack95 = summarize(torch_packed)
        ta, _, ta95 = summarize(torch_alloc)

        for name, fn in variants:
            workspace = torch.empty(m, n, device="cuda", dtype=torch.float32)
            out = torch.empty_like(workspace)
            actual = fn(x, gate_w, up_w, workspace, out)
            torch.cuda.synchronize()
            max_abs = float((actual - expected).abs().max().item()) if out.numel() else 0.0

            samples = measure_us(
                lambda fn=fn,workspace=workspace,out=out: fn(
                    x, gate_w, up_w, workspace, out
                ),
                warmup=args.warmup,
                repeats=args.repeats,
            )
            om, _, op95 = summarize(samples)

            print(
                f"{m}x{k}x{n} {name}={om:.3f} us "
                f"torch_prealloc={tp:.3f} us "
                f"ratio={om/tp:.3f}x max_abs={max_abs:.3e}"
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
                "torch_packed_median_us": tpack,
                "torch_packed_p95_us": tpack95,
                "torch_alloc_median_us": ta,
                "torch_alloc_p95_us": ta95,
                "ratio_vs_torch_prealloc": om / tp if tp else None,
                "ratio_vs_torch_packed": om / tpack if tpack else None,
                "max_abs_error": max_abs,
            })


        packed_workspace = torch.empty(
            (m, 2 * n), device="cuda", dtype=torch.float32
        )
        packed_output = torch.empty(
            (m, n), device="cuda", dtype=torch.float32
        )
        actual_v2 = gemm_swiglu_v2_into(
            x, packed_weight, packed_workspace, packed_output
        )
        torch.cuda.synchronize()
        max_abs_v2 = (
            float((actual_v2 - expected).abs().max().item())
            if packed_output.numel()
            else 0.0
        )
        v2_samples = measure_us(
            lambda: gemm_swiglu_v2_into(
                x, packed_weight, packed_workspace, packed_output
            ),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        v2m, _, v2p95 = summarize(v2_samples)

        print(
            f"{m}x{k}x{n} v2_packed_gemm={v2m:.3f} us "
            f"torch_two_gemm={tp:.3f} us "
            f"torch_packed={tpack:.3f} us "
            f"ratio_two={v2m/tp:.3f}x "
            f"ratio_packed={v2m/tpack:.3f}x "
            f"max_abs={max_abs_v2:.3e}"
        )
        records.append({
            "m": m,
            "k": k,
            "n": n,
            "variant": "v2_packed_gemm",
            "ours_median_us": v2m,
            "ours_p95_us": v2p95,
            "torch_prealloc_median_us": tp,
            "torch_prealloc_p95_us": tp95,
            "torch_packed_median_us": tpack,
            "torch_packed_p95_us": tpack95,
            "torch_alloc_median_us": ta,
            "torch_alloc_p95_us": ta95,
            "ratio_vs_torch_prealloc": v2m / tp if tp else None,
            "ratio_vs_torch_packed": v2m / tpack if tpack else None,
            "max_abs_error": max_abs_v2,
        })


        packed_workspace_v3 = torch.empty(
            (m, 2 * n), device="cuda", dtype=torch.float32
        )
        packed_output_v3 = torch.empty(
            (m, n), device="cuda", dtype=torch.float32
        )
        actual_v3 = gemm_swiglu_v3_into(
            x, packed_weight, packed_workspace_v3, packed_output_v3
        )
        torch.cuda.synchronize()
        max_abs_v3 = (
            float((actual_v3 - expected).abs().max().item())
            if packed_output_v3.numel()
            else 0.0
        )
        v3_samples = measure_us(
            lambda: gemm_swiglu_v3_into(
                x, packed_weight, packed_workspace_v3, packed_output_v3
            ),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        v3m, _, v3p95 = summarize(v3_samples)

        print(
            f"{m}x{k}x{n} v3_packed_float4={v3m:.3f} us "
            f"v2_packed={v2m:.3f} us "
            f"post_speedup={v2m/v3m:.3f}x "
            f"torch_packed={tpack:.3f} us "
            f"max_abs={max_abs_v3:.3e}"
        )
        records.append({
            "m": m,
            "k": k,
            "n": n,
            "variant": "v3_packed_float4_post",
            "ours_median_us": v3m,
            "ours_p95_us": v3p95,
            "torch_prealloc_median_us": tp,
            "torch_prealloc_p95_us": tp95,
            "torch_packed_median_us": tpack,
            "torch_packed_p95_us": tpack95,
            "torch_alloc_median_us": ta,
            "torch_alloc_p95_us": ta95,
            "ratio_vs_torch_prealloc": v3m / tp if tp else None,
            "ratio_vs_torch_packed": v3m / tpack if tpack else None,
            "max_abs_error": max_abs_v3,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)


if __name__ == "__main__":
    main()
