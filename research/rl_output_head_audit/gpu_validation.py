#!/usr/bin/env python3
"""P02 non-timing CUDA validation; see P02_GPU_VALIDATION_PROTOCOL.md.

This does not relax gpu_compare.py's idle-only performance gate. No events,
latency, peak-memory comparisons or speedups are collected here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

import torch

from compile_compat import load_compatible
from gpu_compare import parse_shape, preflight, process_snapshot
from p02_audit import compare_outputs, load_original, make_callable, make_inputs, observe

ROOT = Path(__file__).resolve().parent
CCE_COMMIT = '3de376c106a1916bc5e1b619f9c77c87a461ee1c'
PROVIDERS = ('original_eager', 'compat_eager', 'minimal_eager', 'compat_compiled', 'minimal_compiled')
MODES = ('logprob_only', 'entropy_logged', 'entropy_loss')


def diagnostic_allowed(snapshot):
    if snapshot.get('compute_pids') != [] or len(snapshot.get('samples', [])) < 3:
        return False
    try:
        for sample in snapshot['samples']:
            if sample['returncode'] != 0:
                return False
            lines = sample['stdout'].strip().splitlines()
            if len(lines) != 1:
                return False
            fields = [s.strip() for s in lines[0].split(',')]
            if len(fields) != 7:
                return False
            total, used, util, temp = map(float, fields[3:])
            if not all(math.isfinite(n) for n in (total, used, util, temp)):
                return False
            if not (0 <= used <= total and total - used >= 4096 and 0 <= util <= 100 and 0 <= temp < 80):
                return False
    except (ValueError, KeyError, TypeError):
        return False
    return True


def validate_evidence(result):
    if result.get('performance_measured') is not False:
        raise ValueError('diagnostics cannot claim performance')
    forbidden = {'summary_us', 'speedup', 'event_us', 'wall_us', 'latency_us', 'peak_allocated'}

    def walk(obj):
        if isinstance(obj, dict):
            if forbidden.intersection(obj):
                raise ValueError('timing/memory-performance fields forbidden in validation evidence')
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)
    walk(result)


def dense_reference(x, w, y):
    logits = x.float() @ w.float().t()
    return logits.log_softmax(-1).gather(-1, y[:, None]).squeeze(-1), None


def check_foreign(result):
    query, pids = process_snapshot()
    result.setdefault('process_queries', []).append(query)
    if pids is None or any(pid != os.getpid() for pid in pids):
        raise RuntimeError('foreign/unknown compute process; stop our diagnostic')


def pinned_cce(root):
    root = root.resolve()
    sha = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if sha != CCE_COMMIT or dirty:
        raise RuntimeError('CCE must be the clean pinned checkout')
    sys.path.insert(0, str(root))
    import cut_cross_entropy
    if not Path(cut_cross_entropy.__file__).resolve().is_relative_to(root):
        raise RuntimeError('CCE imported from unexpected path')

    def head(x, w, y):
        nll = cut_cross_entropy.linear_cross_entropy(
            x, w, y, reduction='none', shift=0, filter_eps=None,
            filter_e_grad=False, filter_c_grad=False, accum_e_fp32=True, accum_c_fp32=True)
        return -nll, None
    return head, {'commit': sha, 'module': cut_cross_entropy.__file__,
                  'filter_eps': None, 'filter_e_grad': False, 'filter_c_grad': False,
                  'accum_e_fp32': True, 'accum_c_fp32': True, 'temperature': 1.0,
                  'limit': 'Full CCE package API; logp-only; different intermediate rounding from verl'}


def run(args):
    if args.output.exists():
        raise FileExistsError('refuse to overwrite evidence')
    result = {'schema': 1, 'task': 'P02_GPU_VALIDATION_ONLY',
              'utc': datetime.now(timezone.utc).isoformat(),
              'source_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
              'performance_measured': False, 'records': [],
              'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    exitcode = 0
    try:
        result['preflight'] = preflight()
        if not diagnostic_allowed(result['preflight']):
            result['status'] = 'blocked_diagnostic_resources'
            exitcode = 2
        else:
            torch.set_num_threads(1)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch._dynamo.config.suppress_errors = False
            if hasattr(torch._dynamo.config, 'recompile_limit'):
                torch._dynamo.config.recompile_limit = 64
            original, identities = load_original(args.cache)
            compatible, compatibility = load_compatible(args.cache)
            result.update(torch=torch.__version__, cuda=torch.version.cuda,
                          device=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                          sources=identities, compatibility=compatibility,
                          compiler={'mode': 'default', 'fullgraph': True, 'dynamic': False})
            dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float32
            for train_weight in (False, True):
                for mode in args.modes:
                    check_foreign(result)
                    data = make_inputs(args.shape, train_weight, dtype, 'cuda', args.seed)
                    reference_fn = make_callable(original, 'original', mode)
                    reference = observe(reference_fn, *data, mode)
                    for provider in args.providers:
                        check_foreign(result)
                        row = {'provider': provider, 'train_weight': train_weight, 'mode': mode, 'checks': []}
                        try:
                            if provider.startswith('original'):
                                fn = reference_fn
                            elif provider.startswith('compat'):
                                fn = make_callable(compatible, 'original', mode)
                            else:
                                fn = make_callable(original, 'minimal', mode)
                            if provider.endswith('_compiled'):
                                fn = torch.compile(fn, fullgraph=True, dynamic=False, mode='default')
                            for seed in (args.seed, args.seed + 1):
                                inputs = data if seed == args.seed else make_inputs(args.shape, train_weight, dtype, 'cuda', seed)
                                ref = reference if seed == args.seed else observe(reference_fn, *inputs, mode)
                                got = observe(fn, *inputs, mode)
                                torch.cuda.synchronize()
                                row['checks'].append({'seed': seed, 'errors': compare_outputs(got, ref, dtype)})
                                del got, ref, inputs
                            row['status'] = 'pass'
                            del fn
                        except Exception:
                            row.update(status='failed', exception=traceback.format_exc())
                            exitcode = 1
                        result['records'].append(row)
                        print(provider, train_weight, mode, row['status'], flush=True)
                    del data, reference, reference_fn
                    gc.collect()
            if args.cce_root is not None:
                cce_fn, result['cce'] = pinned_cce(args.cce_root)
                for train_weight in (False, True):
                    check_foreign(result)
                    row = {'provider': 'cce_public_api', 'train_weight': train_weight,
                           'mode': 'logprob_only', 'checks': []}
                    try:
                        for seed in (args.seed, args.seed + 1):
                            data = make_inputs(args.shape, train_weight, dtype, 'cuda', seed)
                            ref_math = observe(dense_reference, *data, 'logprob_only')
                            ref_round = observe(make_callable(original, 'original', 'logprob_only', temperature=1.0), *data, 'logprob_only')
                            got = observe(cce_fn, *data, 'logprob_only')
                            torch.cuda.synchronize()
                            row['checks'].append({'seed': seed,
                                'vs_fp32_projection': compare_outputs(got, ref_math, dtype),
                                'vs_verl_rounded_projection': compare_outputs(got, ref_round, dtype)})
                            del data, got, ref_math, ref_round
                        row['status'] = 'pass'
                    except Exception:
                        row.update(status='failed', exception=traceback.format_exc())
                        exitcode = 1
                    result['records'].append(row)
                    print('cce_public_api', train_weight, row['status'], flush=True)
            check_foreign(result)
            result['status'] = 'gpu_validation_complete_no_timing' if not exitcode else 'gpu_validation_partial_no_timing'
    except Exception:
        result.update(status='failed_no_performance_claim', exception=traceback.format_exc())
        exitcode = 1
    result['cuda_initialized'] = torch.cuda.is_initialized()
    result['finished_utc'] = datetime.now(timezone.utc).isoformat()
    result['limits'] = ['No latency, speedup or peak-memory claims; graphics activity allowed only for correctness',
                        'Default CUDA Inductor, not max-autotune performance validation',
                        'Verl source fragments, not a full trainer',
                        'CCE uses a separate FP32 projection reference at temperature=1',
                        'No Liger SM90/cuTile execution, no training step or convergence validation']
    validate_evidence(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('SAVED', args.output, result['status'], 'records', len(result['records']), flush=True)
    return exitcode


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cce-root', type=Path)
    p.add_argument('--shape', type=parse_shape, default=(129, 256, 32768))
    p.add_argument('--dtype', choices=('bf16', 'fp32'), default='bf16')
    p.add_argument('--seed', type=int, default=7307)
    p.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    p.add_argument('--providers', nargs='+', choices=PROVIDERS, default=list(PROVIDERS))
    raise SystemExit(run(p.parse_args()))
