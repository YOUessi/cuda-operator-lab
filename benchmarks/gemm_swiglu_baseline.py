#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from cuda_operator_lab.benchmarking import measure_us, summarize
from cuda_operator_lab.bindings import gemm_swiglu_v0_into, gemm_swiglu_v1_into, gemm_swiglu_v2_into
from cuda_operator_lab.references import gemm_swiglu, gemm_swiglu_packed


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
        packed_w = torch.cat([gate_w, up_w], dim=0).contiguous()

        workspace = torch.empty(m, n, device="cuda", dtype=torch.float32)
        out = torch.empty_like(workspace)

        gate_tmp = torch.empty_like(workspace)
        up_tmp = torch.empty_like(workspace)
        torch_out = torch.empty_like(workspace)
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

        def torch_packed_preallocated() -> None:
            torch.mm(x, packed_w.transpose(0, 1), out=packed_tmp)
            gate_view = packed_tmp[:, :n]
            up_view = packed_tmp[:, n:]
            torch.ops.aten.silu.out(gate_view, out=packed_out)
            packed_out.mul_(up_view)

        variants = [
            ("v0_scalar_post", gemm_swiglu_v0_into),
            ("v1_float4_post", gemm_swiglu_v1_into),
        ]
        torch_pre = measure_us(
            torch_preallocated,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        torch_packed_pre = measure_us(
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
        tpp, _, tpp95 = summarize(torch_packed_pre)
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
                "torch_alloc_median_us": ta,
                "torch_alloc_p95_us": ta95,
                "ratio_vs_torch_prealloc": om / tp if tp else None,
                "max_abs_error": max_abs,
            })


        packed_workspace = torch.empty(m, 2 * n, device="cuda", dtype=torch.float32)
        packed_output = torch.empty(m, n, device="cuda", dtype=torch.float32)
        packed_actual = gemm_swiglu_v2_into(
            x, packed_w, packed_workspace, packed_output
        )
        packed_expected = gemm_swiglu_packed(x, packed_w)
        torch.cuda.synchronize()
        packed_max_abs = (
            float((packed_actual - packed_expected).abs().max().item())
            if packed_output.numel()
            else 0.0
        )
        packed_samples = measure_us(
            lambda: gemm_swiglu_v2_into(
                x, packed_w, packed_workspace, packed_output
            ),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        pm, _, pp95 = summarize(packed_samples)
        print(
            f"{m}x{k}x{n} v2_packed_projection={pm:.3f} us "
            f"torch_two_gemm={tp:.3f} us "
            f"torch_packed={tpp:.3f} us "
            f"ratio_vs_two={pm/tp:.3f}x "
            f"ratio_vs_packed={pm/tpp:.3f}x "
            f"max_abs={packed_max_abs:.3e}"
        )
        records.append({
            "m": m,
            "k": k,
            "n": n,
            "variant": "v2_packed_projection",
            "ours_median_us": pm,
            "ours_p95_us": pp95,
            "torch_prealloc_median_us": tp,
            "torch_prealloc_p95_us": tp95,
            "torch_packed_median_us": tpp,
            "torch_packed_p95_us": tpp95,
            "torch_alloc_median_us": ta,
            "torch_alloc_p95_us": ta95,
            "ratio_vs_torch_prealloc": pm / tp if tp else None,
            "ratio_vs_torch_packed": pm / tpp if tpp else None,
            "max_abs_error": packed_max_abs,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)


if __name__ == "__main__":
    main()
