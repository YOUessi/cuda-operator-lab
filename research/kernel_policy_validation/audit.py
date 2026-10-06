#!/usr/bin/env python3
"""Read-only kernel audit. No kernel edits, automatic policy updates, or cache flushes.

Raw CUDA-event intervals are stored as integer nanoseconds (rounded to 1 ns).
This unit does NOT imply 1 ns timer accuracy. Eager intervals may include GPU idle
submission gaps. graph32 is a warm repeated-work diagnostic, NOT request latency.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE = "8507be588349bfb3206e4356d9a86995396a97c0"
SHAPES = [(256, 512), (128, 1024), (64, 2048)]
GEMM_SHAPES = [(16, 64, 64), (32, 128, 256), (64, 128, 1024),
               (128, 128, 2048), (32, 256, 64), (128, 512, 512)]


def balanced_orders(labels, repeats, seed):
    labels = tuple(labels)
    n = len(labels)
    if not n or len(set(labels)) != n or repeats <= 0 or repeats % (2 * n):
        raise ValueError("unique labels and repeats divisible by 2*len(labels) required")
    rng = random.Random(seed)
    orders = []
    for _ in range(repeats // (2 * n)):
        base = list(labels)
        rng.shuffle(base)
        for i in range(n):
            order = base[i:] + base[:i]
            orders.extend([order, list(reversed(order))])
    rng.shuffle(orders)
    return orders


def validate_same_elements(shapes):
    if not shapes or any(len(s) != 2 or min(s) <= 0 for s in shapes):
        raise ValueError("positive nonempty 2-D shapes required")
    sizes = {math.prod(s) for s in shapes}
    if len(sizes) != 1:
        raise ValueError("shape views must have identical element counts")
    return sizes.pop()


def quantile(values, q):
    values = sorted(values)
    pos = q * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def paired_summary(a, b, seed=17):
    if len(a) < 2 or len(a) != len(b):
        raise ValueError("at least two matched rounds required")
    for x, y in zip(a, b):
        if not x or len(x) != len(y):
            raise ValueError("matched samples within every round required")
        if any(not math.isfinite(v) or v <= 0 for v in itertools.chain(x, y)):
            raise ValueError("timings must be finite and positive")
    ma = [statistics.median(x) for x in a]
    mb = [statistics.median(x) for x in b]
    logs = [math.log(x / y) for x, y in zip(ma, mb)]
    rng = random.Random(seed)
    draws = [math.exp(statistics.mean(rng.choices(logs, k=len(logs)))) for _ in range(2000)]
    low, high = quantile(draws, 0.025), quantile(draws, 0.975)
    ratio = math.exp(statistics.mean(logs))
    return {
        "a_median_ns": statistics.median(ma),
        "b_median_ns": statistics.median(mb),
        "a_p95_ns": quantile(list(itertools.chain.from_iterable(a)), 0.95),
        "b_p95_ns": quantile(list(itertools.chain.from_iterable(b)), 0.95),
        "speedup_a_over_b": ratio,
        "ci95_low": low, "ci95_high": high,
        "decision": "b_faster" if low > 1.05 else "a_faster" if high < 1 / 1.05 else "unresolved",
    }


def command(args):
    try:
        result = subprocess.run(args, text=True, capture_output=True, timeout=15)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def gpu_state():
    return command(["nvidia-smi", "--query-gpu=name,uuid,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu", "--format=csv"])


def tensor_info(t):
    return {"shape": list(t.shape), "stride": list(t.stride()), "dtype": str(t.dtype),
            "data_ptr": t.data_ptr(), "storage_offset": t.storage_offset(),
            "ptr_mod_16": t.data_ptr() % 16, "device": str(t.device)}


def make_swiglu(torch, bindings, data_seed):
    total = validate_same_elements(SHAPES)
    generator = torch.Generator(device="cuda").manual_seed(data_seed)
    gate = torch.randn(total, device="cuda", generator=generator)
    up = torch.randn(total, device="cuda", generator=generator)
    out = torch.empty_like(gate)
    expected = torch.nn.functional.silu(gate) * up
    variants, outputs, infos = {}, {}, {}
    keep = [gate, up, out, expected]
    for shape in SHAPES:
        g, u, o = gate.view(shape), up.view(shape), out.view(shape)
        for version in (0, 1):
            label = f"v{version}_{shape[0]}x{shape[1]}"
            fn = getattr(bindings, f"swiglu_v{version}_into")
            variants[label] = lambda fn=fn, g=g, u=u, o=o: fn(g, u, o)
            outputs[label] = o
            infos[label] = {"input": tensor_info(g), "up": tensor_info(u), "out": tensor_info(o)}
        keep.extend([g, u, o])
    g, u, o = gate.view(SHAPES[0]), up.view(SHAPES[0]), out.view(SHAPES[0])
    for label in ("alias_a", "alias_b"):
        variants[label] = lambda g=g, u=u, o=o: bindings.swiglu_v0_into(g, u, o)
        outputs[label] = o
        infos[label] = {"input": tensor_info(g), "up": tensor_info(u), "out": tensor_info(o)}
    assert len({x["input"]["data_ptr"] for x in infos.values()}) == 1
    assert len({x["out"]["data_ptr"] for x in infos.values()}) == 1
    pairs = [(f"v0_{m}x{n}", f"v1_{m}x{n}") for m, n in SHAPES]
    pairs += [(f"v{v}_256x512", f"v{v}_{m}x{n}") for v in (0, 1) for m, n in SHAPES[1:]]
    pairs += [("alias_a", "alias_b")]
    return {"case": "swiglu_same_storage", "variants": variants, "outputs": outputs,
            "reference": expected, "rtol": 5e-6, "atol": 5e-6, "pairs": pairs,
            "pointers": infos, "keep": keep, "require_shape_equality": True}


def make_gemm(torch, bindings, shape, data_seed):
    m, k, n = shape
    generator = torch.Generator(device="cuda").manual_seed(data_seed + m * 100000 + k * 100 + n)
    x = torch.randn(m, k, device="cuda", generator=generator).to(torch.bfloat16)
    # Scale weights before quantization so the probe is not dominated by huge products.
    packed = (torch.randn(2 * n, k, device="cuda", generator=generator) / math.sqrt(k)).to(torch.bfloat16)
    gate, up = packed[:n], packed[n:]
    workspace = torch.empty(m, 2 * n, device="cuda", dtype=torch.float32)
    out = torch.empty(m, n, device="cuda", dtype=torch.float32)
    projection64 = x.double() @ packed.double().T
    expected = (torch.nn.functional.silu(projection64[:, :n]) * projection64[:, n:]).float()
    variants = {"v4_cublas": lambda: bindings.gemm_swiglu_v4_into(x, packed, workspace, out),
                "v7_custom": lambda: bindings.gemm_swiglu_v7_into(x, gate, up, out)}
    return {"case": f"gemm_{m}x{k}x{n}", "variants": variants,
            "outputs": dict.fromkeys(variants, out), "reference": expected,
            "rtol": 1e-3, "atol": 1e-3,
            "pairs": [("v4_cublas", "v7_custom")],
            "pointers": {"x": tensor_info(x), "packed": tensor_info(packed),
                         "gate_view": tensor_info(gate), "up_view": tensor_info(up),
                         "out": tensor_info(out), "workspace": tensor_info(workspace)},
            "keep": [x, packed, gate, up, workspace, out, expected],
            "require_shape_equality": False}


def validate_outputs(torch, case, callables, stream):
    errors, canonical = {}, {}
    for label, fn in callables.items():
        case["outputs"][label].fill_(float("nan"))
        fn()
        stream.synchronize()
        actual = case["outputs"][label].flatten()
        expected = case["reference"].flatten()
        torch.testing.assert_close(actual, expected, rtol=case["rtol"], atol=case["atol"])
        delta = (actual - expected).abs()
        errors[label] = {"max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
                         "relative_l2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(expected).clamp_min(1e-20))}
        if case["require_shape_equality"]:
            key = label.split("_")[0] if label.startswith("v") else "v0"
            if key not in canonical:
                canonical[key] = actual.clone()
            else:
                torch.testing.assert_close(actual, canonical[key], rtol=0, atol=0)
    return errors


def prepare(torch, case, mode, stream, warmup):
    # Initialize libraries and warm up on the same explicit side stream used for capture.
    for fn in case["variants"].values():
        for _ in range(warmup):
            fn()
    stream.synchronize()
    if mode == "eager":
        return case["variants"], [], 1
    count = int(mode.removeprefix("graph"))
    callables, graphs = {}, []
    for label, fn in case["variants"].items():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(count):
                fn()
        graphs.append(graph)
        callables[label] = graph.replay
        for _ in range(warmup):
            graph.replay()
        stream.synchronize()
    return callables, graphs, count


def measure(torch, case, mode, stream, args, group_seed):
    callables, graphs, count = prepare(torch, case, mode, stream, args.warmup)
    # Both eager and graph correctness must pass BEFORE any samples are accepted.
    errors = validate_outputs(torch, case, callables, stream)
    labels = list(callables)
    event_ns = {label: [] for label in labels}
    wall_ns = {label: [] for label in labels}
    orders_recorded = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record(stream); end.record(stream); end.synchronize()  # prime event allocation
    for ridx in range(args.rounds):
        orders = balanced_orders(labels, args.samples, seed=group_seed + ridx)
        orders_recorded.append([[labels.index(label) for label in order] for order in orders])
        samples = {label: [] for label in labels}
        walls = {label: [] for label in labels}
        for order in orders:
            for label in order:
                wall_start = time.perf_counter_ns()
                start.record(stream)
                callables[label]()
                end.record(stream)
                end.synchronize()
                wall = time.perf_counter_ns() - wall_start
                elapsed = int(round(start.elapsed_time(end) * 1_000_000))
                if elapsed <= 0:
                    raise RuntimeError("nonpositive event interval; increase captured work")
                samples[label].append(elapsed)
                walls[label].append(wall)
        for label in labels:
            event_ns[label].append(samples[label])
            wall_ns[label].append(walls[label])
    # All tensor owners and graphs stay alive throughout capture and measurement.
    raw = {"case": case["case"], "mode": mode, "calls_per_sample": count,
           "labels": labels, "orders": orders_recorded, "event_ns": event_ns,
           "sync_wall_ns": wall_ns, "correctness": errors, "pointers": case["pointers"]}
    summary = []
    for a, b in case["pairs"]:
        normalized_a = [[v / count for v in row] for row in event_ns[a]]
        normalized_b = [[v / count for v in row] for row in event_ns[b]]
        row = {"case": case["case"], "mode": mode, "a": a, "b": b,
               "calls_per_sample": count, **paired_summary(normalized_a, normalized_b, group_seed)}
        summary.append(row)
        print(f"{row['case']} {mode} {a}/{b}: {row['a_median_ns']/1000:.3f}/{row['b_median_ns']/1000:.3f} us, ratio={row['speedup_a_over_b']:.3f}, CI=[{row['ci95_low']:.3f},{row['ci95_high']:.3f}] {row['decision']}", flush=True)
    return raw, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--data-seed", type=int, default=1307)
    parser.add_argument("--order-seed", type=int, default=1101)
    parser.add_argument("--modes", nargs="+", choices=["eager", "graph1", "graph32"], default=["eager", "graph1", "graph32"])
    parser.add_argument("--suite", choices=["all", "swiglu", "gemm"], default="all")
    args = parser.parse_args()
    if args.rounds < 2 or args.samples <= 0 or args.samples % 16 or args.warmup < 3:
        parser.error("rounds>=2, samples a positive multiple of 16, warmup>=3 required")
    if len(set(args.modes)) != len(args.modes):
        parser.error("modes must be unique")
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "python"))
    import torch
    from cuda_operator_lab import bindings
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; CPU timings are not a replacement")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=False)
    library = bindings._library()
    library_path = Path(bindings._default_library_path())
    metadata = {"status": "running", "base_commit": BASE,
                "source_commit": command(["git", "-C", str(root), "rev-parse", "HEAD"]),
                "working_tree": command(["git", "-C", str(root), "status", "--porcelain"]),
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "pid": os.getpid(), "python": platform.python_version(),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "library_path": str(library_path),
                "library_sha256": hashlib.sha256(library_path.read_bytes()).hexdigest(),
                "gpu_initial": gpu_state(), "gpu_snapshots": [],
                "clock_policy": "unlocked; no device configuration changes",
                "cache_policy": "natural warm reuse; no explicit flush",
                "units": "integer ns rounded from event elapsed_time; not timer resolution",
                "precision": "SwiGLU FP32; GEMM BF16 inputs, FP32 output, scaled weights, FP64 reference",
                "limitations": ["single device exploratory audit", "CIs resample rounds within one process, not independent GPU sessions", "graph32 warm repeated-work throughput is not single-request latency", "no Nsight causal attribution", "no full-module measurements yet"]}
    raw_groups, summaries = [], []
    def save():
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        data = json.dumps({"schema": 1, "groups": raw_groups}, separators=(",", ":")).encode()
        (args.output / "raw.json.gz").write_bytes(gzip.compress(data, mtime=0))
        (args.output / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        if summaries:
            with (args.output / "summary.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
                writer.writeheader(); writer.writerows(summaries)
    save()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    jobs = []
    if args.suite in ("all", "swiglu"):
        jobs.append(("swiglu", None))
    if args.suite in ("all", "gemm"):
        jobs.extend(("gemm", shape) for shape in GEMM_SHAPES)
    random.Random(args.order_seed).shuffle(jobs)
    try:
        with torch.no_grad(), torch.cuda.stream(stream):
            for idx, (kind, shape) in enumerate(jobs):
                case = make_swiglu(torch, bindings, args.data_seed) if kind == "swiglu" else make_gemm(torch, bindings, shape, args.data_seed)
                for mode in args.modes:
                    before = gpu_state()
                    group_seed = args.order_seed * 10000 + idx * 100
                    raw, result = measure(torch, case, mode, stream, args, group_seed)
                    raw_groups.append(raw); summaries.extend(result)
                    metadata["gpu_snapshots"].append({"case": case["case"], "mode": mode, "before": before, "after": gpu_state()})
                    save()
                del case
            stream.synchronize()
        metadata["status"] = "complete"
    except BaseException as exc:
        metadata["status"] = "failed"
        metadata["error"] = repr(exc)
        raise
    finally:
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["gpu_final"] = gpu_state()
        save()
    print(f"COMPLETE {args.output}: {len(raw_groups)} groups, {len(summaries)} comparisons", flush=True)


if __name__ == "__main__":
    main()
