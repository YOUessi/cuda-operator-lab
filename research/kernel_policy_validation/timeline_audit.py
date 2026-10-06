#!/usr/bin/env python3
"""R04 instrumented timeline diagnosis; never use these times as benchmark results.

Formal latency measurements use unchanged module_audit.py in separate processes.
This file only records diagnostic traces and extracts synchronized call structure.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import audit
import module_audit

TRACE_SHAPES = [(32, 128, 256), (128, 512, 512), (128, 1024, 4096)]


def interval_union_us(intervals):
    """Measure union, not sum, of finite half-open intervals."""
    clean = []
    for left, right in intervals:
        left, right = float(left), float(right)
        if not math.isfinite(left) or not math.isfinite(right) or right < left:
            raise ValueError('invalid finite time interval')
        clean.append((left, right))
    if not clean:
        return 0.0
    clean.sort()
    left, right = clean[0]
    total = 0.0
    for lo, hi in clean[1:]:
        if lo > right:
            total += right - left
            left, right = lo, hi
        else:
            right = max(right, hi)
    return total + right - left


def span(event):
    left, duration = float(event['ts']), float(event['dur'])
    right = left + duration
    if not math.isfinite(left) or not math.isfinite(duration) or duration < 0:
        raise ValueError('invalid trace timestamp/duration')
    return left, right


def trace_metrics(trace, expected_calls):
    """Extract CUDA kernels inside synchronized CPU markers (Chrome times are us).

    Uncovered intervals between kernels are not proof of CPU causality. Other
    GPU activities and instrumentation can also occupy/alter those intervals.
    """
    if type(expected_calls) is not int or expected_calls <= 0:
        raise ValueError('positive expected_calls required')
    events = [e for e in trace['traceEvents'] if e.get('ph') == 'X']
    markers = sorted([e for e in events if e.get('cat') == 'user_annotation'
                      and e.get('name', '').startswith('r04_call:')], key=lambda e: e['ts'])
    if len(markers) != expected_calls or {e['name'] for e in markers} != {
            f'r04_call:{i}' for i in range(expected_calls)}:
        raise ValueError('missing or duplicate synchronized call markers')
    kernels = [e for e in events if e.get('cat') == 'kernel']
    rows = []
    for marker in markers:
        lo, hi = span(marker)
        selected = []
        for kernel in kernels:
            a, b = span(kernel)
            if b <= lo or a >= hi:
                continue
            if a < lo or b > hi:
                raise ValueError('kernel crosses synchronized marker boundary')
            selected.append(kernel)
        if not selected:
            raise ValueError('no CUDA kernel activities in synchronized call')
        selected.sort(key=lambda e: e['ts'])
        intervals = [span(e) for e in selected]
        busy = interval_union_us(intervals)
        first, last = min(a for a, _ in intervals), max(b for _, b in intervals)
        runtimes = [e for e in events if e.get('cat') in ('cuda_runtime', 'cuda_driver')
                    and lo <= e['ts'] < hi]
        launches = [e for e in runtimes if 'launch' in e.get('name', '').lower()]
        rows.append(dict(call=marker['name'], instrumented=True,
            cpu_sync_range_us=hi-lo, kernel_count=len(selected),
            kernel_union_us=busy, device_span_us=last-first,
            uncovered_between_kernels_us=max(0.0, last-first-busy),
            launch_api_count=len(launches),
            launch_api_names=[e['name'] for e in launches],
            kernel_names=[e['name'] for e in selected],
            kernels=[dict(name=e['name'], relative_start_us=e['ts']-lo,
                          duration_us=e['dur'], stream=e.get('args',{}).get('stream'))
                     for e in selected]))
    return rows


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--calls', type=int, default=3)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--data-seed', type=int, default=1702)
    p.add_argument('--order-seed', type=int, default=2501)
    args = p.parse_args()
    if args.calls < 1 or args.warmup < 3:
        p.error('calls>=1, warmup>=3 required')
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root/'python'))
    import torch
    from cuda_operator_lab import bindings
    if not torch.cuda.is_available() or torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        raise RuntimeError('CUDA profiler activities unavailable; no CPU substitute')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    bindings._library()
    lib = Path(bindings._default_library_path())
    meta = dict(status='running', instrumented=True, base_commit=audit.BASE,
        source_commit=audit.command(['git','-C',str(root),'rev-parse','HEAD']),
        started_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
        torch=torch.__version__, cuda=torch.version.cuda, python=platform.python_version(),
        library_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),
        gpu_initial=audit.gpu_state(), gpu_snapshots=[],
        config={k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        limits=['instrumented times are not steady-state benchmark results',
                'uncovered kernel intervals do not uniquely identify CPU causes',
                'single GPU, synthetic mixed-precision FFN, three diagnostic shapes',
                'no stack, shape or memory tracing; profiling overhead still exists'])
    records, files = [], []
    def save():
        write(args.output/'metadata.json', meta)
        write(args.output/'timeline_metrics.json', dict(schema=1, instrumented=True, records=records, files=files))
    save()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    rng = random.Random(args.order_seed)
    shapes = list(TRACE_SHAPES)
    rng.shuffle(shapes)
    try:
        with torch.no_grad(), torch.cuda.stream(stream):
            for shape in shapes:
                cases = module_audit.make_module_cases(torch,bindings,shape,args.data_seed)
                jobs = [(scope, mode) for scope in ('isolated','module') for mode in ('eager','graph1')]
                rng.shuffle(jobs)
                for scope, mode in jobs:
                    case = cases[scope]
                    case['prefix']()
                    stream.synchronize()
                    callables, graphs, count = audit.prepare(torch,case,mode,stream,args.warmup)
                    errors = audit.validate_outputs(torch,case,callables,stream)
                    labels = list(callables)
                    rng.shuffle(labels)
                    for label in labels:
                        fn = callables[label]
                        for _ in range(args.warmup):
                            fn()
                        stream.synchronize()
                        before = audit.gpu_state()
                        with torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                            record_shapes=False, profile_memory=False, with_stack=False) as prof:
                            for i in range(args.calls):
                                with torch.profiler.record_function(f'r04_call:{i}'):
                                    fn()
                                    stream.synchronize()
                        filename = f'{case["case"]}__{scope}__{mode}__{label}.json'
                        trace_path = args.output/filename
                        prof.export_chrome_trace(str(trace_path))
                        blob = trace_path.read_bytes()
                        (args.output/(filename+'.gz')).write_bytes(gzip.compress(blob, mtime=0))
                        values = trace_metrics(json.loads(blob), args.calls)
                        for row in values:
                            row.update(case=case['case'], shape=list(shape), scope=scope, mode=mode, variant=label)
                        records.extend(values)
                        files.append(dict(path=filename, bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest(),
                                          correctness=errors[label], pointers=case['pointers']))
                        meta['gpu_snapshots'].append(dict(case=case['case'],scope=scope,mode=mode,variant=label,
                                                          before=before,after=audit.gpu_state()))
                        save()
                        print(filename,'kernels',[r['kernel_count'] for r in values],
                              'kernel_union_us',[round(r['kernel_union_us'],3) for r in values],flush=True)
                    del graphs, callables
                del cases
            stream.synchronize()
        meta['status'] = 'complete'
    except BaseException as exc:
        meta['status'] = 'failed'
        meta['error'] = repr(exc)
        raise
    finally:
        meta['finished_utc'] = datetime.now(timezone.utc).isoformat()
        meta['gpu_final'] = audit.gpu_state()
        save()
    print('DIAGNOSTIC COMPLETE',len(files),'traces',len(records),'calls',flush=True)


if __name__ == '__main__':
    main()
