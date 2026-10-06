#!/usr/bin/env python3
"""G01: observe readiness without importing a GPU runtime or launching a server.

A successful metadata check is not a completed runtime compatibility check.
Memory residuals are unclassified: they must NEVER be labelled graph metadata.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import io
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

EXPECTED = {"sglang": "0.5.21", "torch": "2.13.0", "transformers": "5.12.1"}
UPSTREAM = "e00930c5489053f26d86b179cee0d087f846acbb"


def assess(samples, packages, compute_pids):
    """Finite preflight only. None process inventory fails closed."""
    blockers = []
    if len(samples) < 5:
        blockers.append("insufficient_gpu_samples")
    gpu_ids = {s.get("gpu_uuid") for s in samples}
    if samples and (len(gpu_ids) != 1 or not next(iter(gpu_ids))):
        raise ValueError("samples must name one identical GPU")
    for sample in samples:
        u = sample.get("utilization_pct")
        if isinstance(u, bool) or not isinstance(u, (float, int)) or not math.isfinite(u) or not 0 <= u <= 100:
            raise ValueError("finite GPU utilization in [0,100] required")
    if any(s["utilization_pct"] > 5 for s in samples):
        blockers.append("gpu_busy")
    if compute_pids is None:
        blockers.append("process_inventory_unknown")
    elif compute_pids:
        blockers.append("active_compute_processes")
    for name, version in EXPECTED.items():
        actual = packages.get(name)
        if not isinstance(actual, str) or actual.split("+", 1)[0] != version:
            blockers.append(name + "_version_mismatch")
    return dict(status="blocked" if blockers else "runtime_validation_required",
                blockers=blockers, performance_started=False,
                disclaimer="Low utilization and package metadata do not prove exclusive access, complete dependencies, driver compatibility or a usable model.")


def ledger_delta(before, after):
    """Disjoint views of SAME process/GPU snapshots, not an attribution model.

    allocated is contained in reserved. process_used is independently sampled
    by NVML; its remainder includes more than graph metadata and may be noisy.
    """
    if before.get("gpu_uuid") != after.get("gpu_uuid") or before.get("pid") != after.get("pid"):
        raise ValueError("cross-process/GPU subtraction is invalid")
    if not before.get("gpu_uuid") or type(before.get("pid")) is not int or before["pid"] <= 0:
        raise ValueError("GPU identity and positive PID required")
    for s in (before, after):
        for key in ("allocated_bytes", "reserved_bytes"):
            if type(s.get(key)) is not int or s[key] < 0:
                raise ValueError("nonnegative integer allocator byte counters required")
        if s["allocated_bytes"] > s["reserved_bytes"]:
            raise ValueError("allocated cannot exceed reserved")
        p = s.get("process_used_bytes")
        if p is not None and (type(p) is not int or p < 0):
            raise ValueError("process usage must be nonnegative bytes or unknown")
    reserved = after["reserved_bytes"] - before["reserved_bytes"]
    b, a = before.get("process_used_bytes"), after.get("process_used_bytes")
    process = None if b is None or a is None else a - b
    return dict(torch_allocated_delta_bytes=after["allocated_bytes"]-before["allocated_bytes"],
                torch_reserved_delta_bytes=reserved, process_used_delta_bytes=process,
                unclassified_remainder_delta_bytes=None if process is None else process-reserved,
                graph_metadata_bytes=None,
                warning="Do not add allocated to reserved; unclassified remainder is NOT graph metadata. Validate phase, pool sharing and external activity separately.")


def command(argv):
    try:
        p = subprocess.run(argv, text=True, capture_output=True, timeout=15, check=False)
        return {"argv": argv, "returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "returncode": None, "stdout": "", "stderr": str(exc)}


def package_versions():
    out = {}
    for name in (*EXPECTED, "vllm", "flashinfer-python", "triton", "huggingface-hub", "safetensors"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def collect(gpu_uuid=None, count=5, interval=0.5):
    if type(count) is not int or not 5 <= count <= 20 or not math.isfinite(interval) or not 0.1 <= interval <= 2:
        raise ValueError("5..20 samples, interval 0.1..2 seconds required")
    samples, raw, errors = [], [], []
    chosen = gpu_uuid
    start = datetime.now(timezone.utc).isoformat()
    for index in range(count):
        rec = command(["nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"])
        rec["observed_utc"] = datetime.now(timezone.utc).isoformat()
        raw.append(rec)
        if rec["returncode"] != 0:
            errors.append("GPU telemetry unavailable")
        else:
            try:
                rows = list(csv.reader(io.StringIO(rec["stdout"]), skipinitialspace=True))
                if chosen is None:
                    if len(rows) != 1:
                        raise ValueError("multiple/no GPUs: select an explicit UUID")
                    chosen = rows[0][0].strip()
                selected = [r for r in rows if r and r[0].strip() == chosen]
                if len(selected) != 1 or len(selected[0]) != 7:
                    raise ValueError("selected GPU missing or malformed telemetry")
                r = selected[0]
                samples.append(dict(gpu_uuid=r[0].strip(), name=r[1].strip(), driver_version=r[2].strip(),
                                    total_mib=float(r[3]), used_mib=float(r[4]), utilization_pct=float(r[5]),
                                    temperature_c=float(r[6]), observed_utc=rec["observed_utc"]))
            except (ValueError, IndexError) as exc:
                errors.append(str(exc))
        if index + 1 < count:
            time.sleep(interval)
    proc = command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader,nounits"])
    pids = None
    if proc["returncode"] == 0 and chosen:
        try:
            rows = list(csv.reader(io.StringIO(proc["stdout"]), skipinitialspace=True))
            pids = sorted({int(r[1]) for r in rows if r and r[0].strip() == chosen})
        except (ValueError, IndexError):
            errors.append("invalid process telemetry")
    versions = package_versions()
    try:
        decision = assess(samples, versions, pids)
    except ValueError as exc:
        errors.append(str(exc))
        decision = dict(status="blocked", blockers=["invalid_gpu_samples"], performance_started=False)
    if errors:
        decision["status"] = "blocked"
        decision["blockers"].append("telemetry_errors")
    root = Path(__file__).resolve().parents[2]
    return dict(schema=1, started_utc=start, finished_utc=datetime.now(timezone.utc).isoformat(),
                python=platform.python_version(), executable=sys.executable,
                source=command(["git", "-C", str(root), "rev-parse", "HEAD"]),
                upstream_sglang_commit=UPSTREAM, expected_packages=EXPECTED, packages=versions,
                gpu_samples=samples, compute_pids=pids, raw_gpu_commands=raw,
                raw_process_command=proc, errors=errors, decision=decision,
                gpu_runtime_imported=False, model_weights_loaded=False,
                engine_started=False, no_environment_changes=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu-uuid")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("refuse to overwrite evidence")
    report = collect(args.gpu_uuid)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report["decision"], ensure_ascii=False), flush=True)
    print("Evidence:", args.output, flush=True)
    return 2 if report["decision"]["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
