#!/usr/bin/env python3
"""P02 controlled output-head comparison, not a full-training benchmark.

The busy-device path must exit before CUDA initialization. The optional
--preflight-only mode never runs kernels even when metadata permits a probe.
Run one shape/mode per process; specialized full-package baselines remain a
separate required gate before any research-speedup claim.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
import traceback

import torch

from compile_compat import load_compatible
from p02_audit import balanced_orders, compare_outputs, load_original, make_callable, make_inputs, observe

ROOT = Path(__file__).resolve().parent


def gate(utilizations, processes):
    return (len(utilizations) >= 3 and processes is not None and len(processes) == 0
            and all(math.isfinite(u) and 0 <= u < 10 for u in utilizations))


def parse_shape(text):
    shape = tuple(int(v) for v in text.lower().split('x'))
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError('shape must be positive TxHxV')
    return shape


def command(args):
    p = subprocess.run(args, capture_output=True, text=True, timeout=15)
    return {'argv': args, 'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}


def process_snapshot():
    r = command(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'])
    try:
        pids = [int(line.split(',')[0]) for line in r['stdout'].splitlines() if line.strip()]
        if r['returncode']:
            pids = None
    except ValueError:
        pids = None
    return r, pids


def preflight():
    samples, utilization = [], []
    for _ in range(3):
        r = command(['nvidia-smi', '--query-gpu=uuid,name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu',
                     '--format=csv,noheader,nounits'])
        r['utc'] = datetime.now(timezone.utc).isoformat()
        samples.append(r)
        lines = [v for v in r['stdout'].splitlines() if v.strip()]
        try:
            utilization.append(float(lines[0].split(',')[-2]) if r['returncode'] == 0 and len(lines) == 1 else float('nan'))
        except (ValueError, IndexError):
            utilization.append(float('nan'))
        time.sleep(0.5)
    proc, pids = process_snapshot()
    return {'samples': samples, 'process_query': proc, 'compute_pids': pids,
            'allowed': gate(utilization, pids),
            'note': 'Low utilization is necessary here, not proof of exclusive access.'}


def run(args):
    if args.output.exists():
        raise FileExistsError('refuse to overwrite evidence')
    result = {'schema': 1, 'task': 'P02_GPU_CONTROL',
              'source_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
              'utc': datetime.now(timezone.utc).isoformat(),
              'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'performance_started': False, 'records': []}
    exitcode = 0
    try:
        result['preflight'] = preflight()
        if not result['preflight']['allowed']:
            result['status'] = 'blocked_gpu_busy_or_unknown'
            exitcode = 2
        elif args.preflight_only:
            result['status'] = 'preflight_only_runtime_not_validated'
        else:
            torch.set_num_threads(1)
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA runtime unavailable')
            original, source = load_original(args.cache, args.download)
            compat, compatibility = load_compatible(args.cache)
            result.update(torch=torch.__version__, sources=source, compatibility_edit=compatibility,
                          device=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()))
            torch.backends.cuda.matmul.allow_tf32 = False
            dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float32
            data = make_inputs(args.shape, args.train_weight, dtype, 'cuda', args.seed)
            eager = make_callable(original, 'original', args.mode)
            compat_fn = make_callable(compat, 'original', args.mode)
            minimal = make_callable(original, 'minimal', args.mode)
            variants = {'original_eager': eager, 'compat_eager': compat_fn, 'minimal_eager': minimal,
                        'compat_compiled': torch.compile(compat_fn, fullgraph=True, dynamic=False, mode=args.compile_mode),
                        'minimal_compiled': torch.compile(minimal, fullgraph=True, dynamic=False, mode=args.compile_mode)}
            reference = observe(eager, *data, args.mode)
            result['checks'] = {}
            for name, fn in variants.items():
                started = time.perf_counter()
                got = observe(fn, *data, args.mode)
                torch.cuda.synchronize()
                result['checks'][name] = {'first_call_s': time.perf_counter() - started,
                                         'errors': compare_outputs(got, reference, dtype)}
                del got
                for _ in range(10):
                    observe(fn, *data, args.mode)
                torch.cuda.synchronize()
            del reference
            result['memory'] = {}
            for name, fn in variants.items():
                torch.cuda.synchronize()
                base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                retained = observe(fn, *data, args.mode)
                torch.cuda.synchronize()
                result['memory'][name] = {'base_allocated': base, 'peak_allocated': torch.cuda.max_memory_allocated(),
                                          'peak_reserved': torch.cuda.max_memory_reserved()}
                del retained
            result['performance_started'] = True
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for round_id in range(3):
                state, pids = process_snapshot()
                result.setdefault('during_process_queries', []).append(state)
                if pids is None or any(pid != os.getpid() for pid in pids):
                    raise RuntimeError('foreign or unknown compute process: timed data invalid')
                for sample, order in enumerate(balanced_orders(list(variants), 20, args.seed + round_id)):
                    for position, name in enumerate(order):
                        wall = time.perf_counter()
                        start.record()
                        value = observe(variants[name], *data, args.mode)
                        end.record()
                        end.synchronize()
                        result['records'].append({'round': round_id, 'sample': sample, 'position': position,
                                                  'variant': name, 'event_us': start.elapsed_time(end) * 1000,
                                                  'wall_us': (time.perf_counter() - wall) * 1e6})
                        del value
            result['summary_us'] = {name: statistics.median(r['event_us'] for r in result['records'] if r['variant'] == name)
                                    for name in variants}
            result['status'] = 'controlled_probe_complete_not_full_training'
    except Exception:
        result['status'] = 'failed_no_performance_claim'
        result['exception'] = traceback.format_exc()
        exitcode = 1
    result['cuda_initialized'] = torch.cuda.is_initialized()
    result['finished_utc'] = datetime.now(timezone.utc).isoformat()
    result['limits'] = ['No complete trainer or specialized-backend comparison',
                        'CUDA event interval includes host submission gaps; not pure kernel time',
                        'Shared-device interference is not fully excluded by process polling',
                        'No warm-speedup claims when blocked or failed']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(result['status'], 'performance_started', result['performance_started'], 'cuda_initialized', result['cuda_initialized'])
    return exitcode


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--download', action='store_true')
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--shape', type=parse_shape, default=(256, 512, 32768))
    p.add_argument('--mode', choices=['logprob_only', 'entropy_logged', 'entropy_loss'], default='logprob_only')
    p.add_argument('--dtype', choices=['fp32', 'bf16'], default='bf16')
    p.add_argument('--train-weight', action='store_true')
    p.add_argument('--seed', type=int, default=7207)
    p.add_argument('--compile-mode', default='max-autotune-no-cudagraphs')
    raise SystemExit(run(p.parse_args()))
