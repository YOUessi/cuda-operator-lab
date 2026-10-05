#!/usr/bin/env python3
"""Run a representative end-to-end benchmark suite and write one summary.

This is intentionally a quick/reproducible project-level suite. Dedicated
stable/cold/crossover profilers remain the source of truth for dispatch policy.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def one(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise RuntimeError(f"missing row {key}={value}")


def fnum(row: dict[str, str], *keys: str) -> float:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    raise RuntimeError(f"none of {keys!r} found in row")


def speedup(a: float, b: float) -> str:
    return f"{a / b:.2f}x" if b else "-"


def write_summary(results: Path) -> None:
    records: list[dict[str, object]] = []

    rows = read_csv(results / "reduction.csv")
    b = one(rows, "variant", "v0_serial")
    f = one(rows, "variant", "v5_shape_aware_float4")
    records.append({
        "operator": "Reduction",
        "shape": f"N={b['n']}",
        "baseline": fnum(b, "median_us"),
        "recommended": fnum(f, "median_us"),
        "reference": fnum(f, "torch_median_us"),
        "note": "cold-cache",
    })

    rows = read_csv(results / "softmax.csv")
    b = one(rows, "variant", "v0_serial_row")
    f = one(rows, "variant", "v5_shape_dispatch")
    records.append({
        "operator": "Softmax",
        "shape": f"{b['rows']}x{b['cols']}",
        "baseline": fnum(b, "median_us"),
        "recommended": fnum(f, "median_us"),
        "reference": fnum(f, "torch_median_us"),
        "note": "PyTorch softmax",
    })

    rows = read_csv(results / "rmsnorm.csv")
    b = one(rows, "variant", "v0_serial_row")
    f = one(rows, "variant", "v4_shape_dispatch")
    records.append({
        "operator": "RMSNorm",
        "shape": f"{b['rows']}x{b['cols']}",
        "baseline": fnum(b, "median_us"),
        "recommended": fnum(f, "median_us"),
        "reference": fnum(f, "torch_median_us"),
        "note": "PyTorch expression ref",
    })

    rows = read_csv(results / "layernorm.csv")
    b = one(rows, "variant", "v0_serial_row")
    f = one(rows, "variant", "v5_shape_dispatch")
    records.append({
        "operator": "LayerNorm",
        "shape": f"{b['rows']}x{b['cols']}",
        "baseline": fnum(b, "median_us"),
        "recommended": fnum(f, "median_us"),
        "reference": fnum(f, "torch_median_us"),
        "note": "PyTorch layer_norm",
    })

    rows = read_csv(results / "fused_residual_layernorm.csv")
    b = one(rows, "variant", "v0_serial_fused")
    f = one(rows, "variant", "v4_shape_dispatch")
    records.append({
        "operator": "Residual+LayerNorm",
        "shape": f"{b['rows']}x{b['cols']}",
        "baseline": fnum(b, "median_us"),
        "recommended": fnum(f, "median_us"),
        "reference": fnum(f, "torch_unfused_median_us"),
        "note": "PyTorch unfused",
    })

    rows = read_csv(results / "fused_bias_gelu.csv")
    b = one(rows, "variant", "v0_scalar")
    f = one(rows, "variant", "v1_float4")
    records.append({
        "operator": "Bias+GELU",
        "shape": f"{b['rows']}x{b['cols']}",
        "baseline": fnum(b, "fused_median_us"),
        "recommended": fnum(f, "fused_median_us"),
        "reference": fnum(f, "torch_unfused_median_us"),
        "note": "V2 dispatch selects V1 here",
    })

    rows = read_csv(results / "swiglu.csv")
    b = one(rows, "variant", "v0_scalar")
    f = one(rows, "variant", "v1_float4")
    records.append({
        "operator": "SwiGLU",
        "shape": f"{b['rows']}x{b['cols']}",
        "baseline": fnum(b, "fused_median_us"),
        "recommended": fnum(f, "fused_median_us"),
        "reference": fnum(f, "torch_unfused_median_us"),
        "note": "V2 dispatch selects V1 here",
    })

    rows = read_csv(results / "gemm_bias_gelu.csv")
    b = one(rows, "variant", "v0_scalar_epilogue")
    f = one(rows, "variant", "v6_cublaslt_robust_autotune")
    records.append({
        "operator": "GEMM+Bias+GELU",
        "shape": f"{b['m']}x{b['k']}x{b['n']}",
        "baseline": fnum(b, "ours_median_us"),
        "recommended": fnum(f, "ours_median_us"),
        "reference": fnum(f, "torch_prealloc_median_us"),
        "note": "V2+ uses tanh-approx GELU",
    })

    rows = read_csv(results / "gemm_swiglu.csv")
    first = rows[0]
    records.append({
        "operator": "GEMM+SwiGLU",
        "shape": f"{first['m']}x{first['k']}x{first['n']}",
        "baseline": fnum(first, "v4_median_us"),
        "recommended": fnum(first, "v10_median_us"),
        "reference": fnum(first, "v4_median_us"),
        "note": f"hybrid path={first['v10_path']}; V11 equivalent",
    })

    lines = [
        "# Fresh Benchmark Summary",
        "",
        "Generated by benchmarks/run_all.py on the current machine.",
        "",
        "| Operator | Shape | Baseline / comparison | Recommended | Framework/vendor ref | Speedup | Note |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for rec in records:
        lines.append(
            "| {operator} | {shape} | {baseline:.3f} us | {recommended:.3f} us | "
            "{reference:.3f} us | {sp} | {note} |".format(
                sp=speedup(float(rec["baseline"]), float(rec["recommended"])),
                **rec,
            )
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- This suite is a quick representative regression check, not the source of truth for hardware dispatch thresholds.",
        "- Stable dispatch decisions use the dedicated cold-cache / interleaved / multi-seed profilers under benchmarks/ and reports/data/.",
        "- GEMM+Bias+GELU V2+ uses tanh-approximate GELU to match cuBLASLt epilogue semantics.",
        "- GEMM+SwiGLU V11 executes the same generated policy as V10; the V10 hybrid benchmark is used here as the lightweight path-level regression check.",
        "",
    ]
    (results / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "benchmarks" / "results" / "final_suite",
    )
    args = p.parse_args()

    results = args.results_dir.resolve()
    results.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    python_path = str(ROOT / "python")
    env["PYTHONPATH"] = (
        python_path
        if not env.get("PYTHONPATH")
        else python_path + os.pathsep + env["PYTHONPATH"]
    )

    py = sys.executable
    w = str(args.warmup)
    r = str(args.repeats)

    jobs = [
        [
            py, "benchmarks/reduction_benchmark.py",
            "--sizes", "4194304",
            "--variants", "v0_serial", "v5_shape_aware_float4",
            "--cache-mode", "cold",
            "--warmup", w, "--repeats", r,
            "--output", str(results / "reduction.csv"),
        ],
        [
            py, "benchmarks/softmax_benchmark.py",
            "--shapes", "128x4096",
            "--variants", "v0_serial_row", "v5_shape_dispatch",
            "--warmup", w, "--repeats", r,
            "--output", str(results / "softmax.csv"),
        ],
        [
            py, "benchmarks/rmsnorm_baseline.py",
            "--shapes", "128x4096",
            "--warmup", w, "--repeats", r,
            "--output", str(results / "rmsnorm.csv"),
        ],
        [
            py, "benchmarks/layernorm_baseline.py",
            "--shapes", "128x4096",
            "--warmup", w, "--repeats", r,
            "--output", str(results / "layernorm.csv"),
        ],
        [
            py, "benchmarks/fused_residual_layernorm_baseline.py",
            "--shapes", "128x4096",
            "--warmup", w, "--repeats", r,
            "--output", str(results / "fused_residual_layernorm.csv"),
        ],
        [
            py, "benchmarks/fused_bias_gelu_baseline.py",
            "--shapes", "1024x4096",
            "--warmup", w, "--repeats", r,
            "--output", str(results / "fused_bias_gelu.csv"),
        ],
        [
            py, "benchmarks/swiglu_baseline.py",
            "--shapes", "1024x4096",
            "--warmup", w, "--repeats", r,
            "--output", str(results / "swiglu.csv"),
        ],
        [
            py, "benchmarks/gemm_bias_gelu_baseline.py",
            "--shapes", "128x512x512",
            "--warmup", w, "--repeats", r,
            "--output", str(results / "gemm_bias_gelu.csv"),
        ],
        [
            py, "benchmarks/gemm_swiglu_v10_hybrid.py",
            "--shapes", "32x128x256",
            "--warmup", w, "--repeats", r,
            "--output", str(results / "gemm_swiglu.csv"),
        ],
    ]

    for job in jobs:
        run(job, env)

    write_summary(results)
    print(f"\nSummary written to {results / 'summary.md'}")


if __name__ == "__main__":
    main()
