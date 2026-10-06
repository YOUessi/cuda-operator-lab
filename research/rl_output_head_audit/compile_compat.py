#!/usr/bin/env python3
"""P02 compatibility-only control, separate from demand pruning.

The untouched upstream failed fullgraph compilation because its output buffer
factories use requires_grad=True inside custom Function.forward. Remove ONLY
those two redundant factory keywords in an explicit, labelled control. Keep
all eager arithmetic, including unnecessary dW/entropy. Do not call this
control the unmodified upstream.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import traceback

import torch

from p02_audit import (compare_outputs, make_callable, make_inputs, observe,
                       record_aot, load_original)
from source_probe import Calls, extract_definitions, obtain_sources

ROOT = Path(__file__).resolve().parent


def compatibility_bytes(data):
    needle = b', requires_grad=output_requires_grad'
    if data.count(needle) != 2:
        raise ValueError('expected exactly two pinned output-factory keywords')
    return data.replace(needle, b'')


def load_compatible(cache):
    manifest = json.loads((ROOT / 'sources.json').read_text())
    sources, identities = obtain_sources(manifest, cache, False)
    raw = sources['verl_head']
    patched = compatibility_bytes(raw)
    env = extract_definitions(patched,
        ['_fused_linear_for_ppo_fwd', '_fused_linear_for_ppo_bwd', 'FusedLinearForPPOFunction'],
        {'torch': torch, 'math': math, '_FLASH_ATTN_CROSS_ENTROPY_AVAILABLE': False})
    return env['FusedLinearForPPOFunction'], {
        'original_sha256': hashlib.sha256(raw).hexdigest(),
        'compatibility_sha256': hashlib.sha256(patched).hexdigest(),
        'edit': 'remove exactly two output factory requires_grad=output_requires_grad keywords',
        'no_arithmetic_removed': True, 'sources': identities}


def eager_trace(fn, data, mode):
    x, w, y, a, b = data
    fw, bw = Calls(), Calls()
    with fw:
        lp, ent = fn(x, w, y)
    outputs = (lp, ent) if mode == 'entropy_loss' else (lp,)
    upstream = (a, b.to(ent.dtype)) if mode == 'entropy_loss' else (a,)
    with bw:
        torch.autograd.grad(outputs, (x, w) if w.requires_grad else (x,), upstream)
    return {'forward_ops': dict(fw.ops), 'backward_ops': dict(bw.ops),
            'forward_mm': fw.mm, 'backward_mm': bw.mm}


def saved_tensor_observation(fn, data, mode):
    records = []
    input_storages = {t.untyped_storage().data_ptr() for t in data}
    storage_labels = {}

    def pack(t):
        storage = t.untyped_storage()
        ptr = storage.data_ptr()
        if ptr not in storage_labels:
            storage_labels[ptr] = len(storage_labels)
        records.append({'shape': list(t.shape), 'dtype': str(t.dtype),
                        'logical_bytes': t.numel() * t.element_size(),
                        'storage_bytes': storage.nbytes(), 'storage_id': storage_labels[ptr],
                        'aliases_input': ptr in input_storages})
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = observe(fn, *data, mode)
    return out, records


def run(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError("CPU diagnostic requires CUDA_VISIBLE_DEVICES=''")
    if args.output.exists():
        raise FileExistsError('refuse to overwrite evidence')
    torch.set_num_threads(1)
    original, _ = load_original(args.cache, False)
    compat, source = load_compatible(args.cache)
    from torch._dynamo import config
    config.suppress_errors = False
    if hasattr(config, 'recompile_limit'):
        config.recompile_limit = 64
    records = []
    for train_w in (False, True):
        for mode in ('logprob_only', 'entropy_logged', 'entropy_loss'):
            data = make_inputs((7, 5, 11), train_w, torch.float32, 'cpu', 7207)
            ref_fn = make_callable(original, 'original', mode)
            reference = observe(ref_fn, *data, mode)
            raw_trace = eager_trace(ref_fn, data, mode)
            for provider in ('compat_original', 'minimal'):
                fn = make_callable(compat, 'original' if provider == 'compat_original' else 'minimal', mode)
                row = {'provider': provider, 'train_weight': train_w, 'entropy_mode': mode,
                       'shape_T_H_V': [7, 5, 11], 'dtype': 'float32', 'temperature': 0.7}
                row['eager'] = eager_trace(fn, data, mode)
                if provider == 'compat_original':
                    if raw_trace != row['eager']:
                        raise AssertionError('compatibility edit changed eager operation counts')
                    row['eager_work_matches_unmodified'] = True
                value, row['eager_saved_tensors'] = saved_tensor_observation(fn, data, mode)
                row['eager_errors'] = compare_outputs(value, reference, torch.float32)
                try:
                    aot, row['aot_graphs'] = record_aot(fn, data, mode)
                    row['aot_errors'] = compare_outputs(aot, reference, torch.float32)
                    row['aot_status'] = 'pass'
                except Exception:
                    row['aot_status'] = 'failed'
                    row['aot_exception'] = traceback.format_exc()
                try:
                    print('BEGIN_INDUCTOR', provider, train_w, mode, flush=True)
                    compiled = torch.compile(fn, fullgraph=True, backend='inductor', dynamic=False)
                    row['inductor_errors'] = compare_outputs(observe(compiled, *data, mode), reference, torch.float32)
                    alternate = make_inputs((7, 5, 11), train_w, torch.float32, 'cpu', 7208)
                    alt_ref = observe(ref_fn, *alternate, mode)
                    row['inductor_heldout_errors'] = compare_outputs(observe(compiled, *alternate, mode), alt_ref, torch.float32)
                    saved_result, row['inductor_saved_tensors'] = saved_tensor_observation(compiled, data, mode)
                    row['inductor_hooked_errors'] = compare_outputs(saved_result, reference, torch.float32)
                    row['inductor_status'] = 'pass'
                    print('END_INDUCTOR', provider, train_w, mode, flush=True)
                except Exception:
                    row['inductor_status'] = 'failed'
                    row['inductor_exception'] = traceback.format_exc()
                records.append(row)
                mm = [n.get('tensor_arg_shapes') for g in row.get('aot_graphs', {}).get('backward', [])
                      for n in g['nodes'] if n['target'] == 'aten.mm.default']
                print(provider, train_w, mode, row['aot_status'], row['inductor_status'],
                      'AOT_BW_MM', mm, flush=True)
    if torch.cuda.is_initialized():
        raise RuntimeError('unexpected CUDA initialization')
    failures = sum(r['aot_status'] != 'pass' or r['inductor_status'] != 'pass' for r in records)
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    result = {'schema': 1, 'task': 'P02_COMPATIBILITY_CONTROL',
              'status': 'pass' if not failures else 'partial', 'source_commit': commit,
              'utc': datetime.now(timezone.utc).isoformat(), 'torch': torch.__version__,
              'cpu_threads': 1, 'cuda_initialized': False, 'performance_measured': False,
              'compatibility_edit': source, 'records': records,
              'limits': ['Compatibility-only edit is NOT untouched upstream',
                         'Small CPU/AOT/Inductor result is NOT GPU performance',
                         'Saved logical tensor sizes may alias; not peak memory',
                         'No full trainer, specialized CCE/Liger/cuTile or training convergence validation']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('SAVED', args.output, 'records', len(records), 'failures', failures, flush=True)
    return bool(failures)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    raise SystemExit(run(p.parse_args()))
