#!/usr/bin/env python3
"""Stable crossover profiling for cuBLAS V4 vs custom WMMA V7."""

from __future__ import annotations

import argparse
import csv
import random
import statistics
from pathlib import Path

import torch

from cuda_operator_lab.bindings import gemm_swiglu_v4_into, gemm_swiglu_v7_into


DEFAULT_M = [16, 32, 64, 128]
DEFAULT_K = [64, 128, 256, 512, 1024]
DEFAULT_N = [64, 128, 256, 512, 1024, 2048]


def timed_one(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--m-values", nargs="*", type=int, default=DEFAULT_M)
    p.add_argument("--k-values", nargs="*", type=int, default=DEFAULT_K)
    p.add_argument("--n-values", nargs="*", type=int, default=DEFAULT_N)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=40)
    p.add_argument("--seed", type=int, default=20261022)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    rng = random.Random(args.seed)
    records = []

    print(f"device={torch.cuda.get_device_name(0)}")
    print(
        f"rounds={args.rounds} repeats={args.repeats} seed={args.seed} "
        "mode=steady_state_interleaved"
    )

    for m in args.m_values:
        for k in args.k_values:
            for n in args.n_values:
                if m % 16 or k % 16 or n % 64:
                    continue

                g = torch.Generator(device="cuda")
                g.manual_seed(args.seed + m * 1000000 + k * 1000 + n)

                x = torch.randn(
                    m, k, device="cuda", dtype=torch.float32, generator=g
                ).to(torch.bfloat16)
                gate_w = torch.randn(
                    n, k, device="cuda", dtype=torch.float32, generator=g
                ).to(torch.bfloat16)
                up_w = torch.randn(
                    n, k, device="cuda", dtype=torch.float32, generator=g
                ).to(torch.bfloat16)
                packed_w = torch.cat((gate_w, up_w), dim=0).contiguous()

                workspace = torch.empty(
                    m, 2 * n, device="cuda", dtype=torch.float32
                )
                out_v4 = torch.empty(
                    m, n, device="cuda", dtype=torch.float32
                )
                out_v7 = torch.empty_like(out_v4)

                # Warm both implementations before interleaved measurement.
                for _ in range(10):
                    gemm_swiglu_v4_into(
                        x, packed_w, workspace, out_v4
                    )
                    gemm_swiglu_v7_into(
                        x, gate_w, up_w, out_v7
                    )
                torch.cuda.synchronize()

                round_medians = {"v4_cublas": [], "v7_custom": []}

                for ridx in range(args.rounds):
                    samples = {"v4_cublas": [], "v7_custom": []}
                    for sidx in range(args.repeats):
                        order = ["v4_cublas", "v7_custom"]
                        if (ridx + sidx) % 2 == 0:
                            rng.shuffle(order)
                        else:
                            order.reverse()

                        for name in order:
                            if name == "v4_cublas":
                                samples[name].append(
                                    timed_one(
                                        lambda: gemm_swiglu_v4_into(
                                            x, packed_w, workspace, out_v4
                                        )
                                    )
                                )
                            else:
                                samples[name].append(
                                    timed_one(
                                        lambda: gemm_swiglu_v7_into(
                                            x, gate_w, up_w, out_v7
                                        )
                                    )
                                )

                    for name in round_medians:
                        round_medians[name].append(
                            statistics.median(samples[name])
                        )

                v4 = statistics.median(round_medians["v4_cublas"])
                v7 = statistics.median(round_medians["v7_custom"])
                speedup = v4 / v7

                print(
                    f"{m}x{k}x{n}: "
                    f"v4={v4:.3f} us "
                    f"v7={v7:.3f} us "
                    f"custom_speedup={speedup:.3f}x"
                )

                for name, values in round_medians.items():
                    records.append({
                        "m": m,
                        "k": k,
                        "n": n,
                        "variant": name,
                        "rounds": args.rounds,
                        "repeats_per_round": args.repeats,
                        "median_us": statistics.median(values),
                        "mean_round_median_us": statistics.mean(values),
                        "stdev_round_median_us": (
                            statistics.stdev(values)
                            if len(values) > 1
                            else 0.0
                        ),
                        "custom_speedup": speedup,
                    })

                del x, gate_w, up_w, packed_w, workspace, out_v4, out_v7

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
