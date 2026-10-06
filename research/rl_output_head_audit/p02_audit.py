#!/usr/bin/env python3
"""P02 CPU correctness/compiler audit. GPU performance is a separate gate.

Pinned fallback definitions are NOT installed framework packages. AOT graphs
are structural evidence, not CUDA kernel traces. Real CPU Inductor execution
is also checked when --inductor is set, never timed as GPU evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import traceback

import torch

import demand_head
from source_probe import Calls, extract_definitions, obtain_sources

ROOT = Path(__file__).resolve().parent


def balanced_orders(names, count, seed):
    if not names or len(set(names)) != len(names) or count <= 0 or count % len(names):
        raise ValueError('positive count must be a multiple of unique candidates')
    rng = random.Random(seed)
    result = []
    for _ in range(count // len(names)):
        base = list(names)
        rng.shuffle(base)
        block = [base[i:] + base[:i] for i in range(len(base))]
        rng.shuffle(block)
        result.extend(block)
    return result


def error_metrics(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError('shape mismatch')
    a, b = actual.detach().double(), expected.detach().double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('nonfinite comparison')
    delta = (a - b).abs()
    return {'max_abs': float(delta.max()), 'mean_abs': float(delta.mean()),
            'relative_l2': float(torch.linalg.vector_norm(delta) /
                                 torch.linalg.vector_norm(b).clamp_min(1e-30))}


def graph_record(gm):
    nodes = []
    for n in gm.graph.nodes:
        item = {'name': n.name, 'op': n.op, 'target': str(n.target)}
        value = n.meta.get('val')
        if isinstance(value, torch.Tensor):
            item.update(shape=[int(v) for v in value.shape], dtype=str(value.dtype))
        arg_shapes = []
        for arg in n.args:
            if isinstance(arg, torch.fx.Node):
                value = arg.meta.get('val')
                if isinstance(value, torch.Tensor):
                    arg_shapes.append([int(v) for v in value.shape])
        if arg_shapes:
            item['tensor_arg_shapes'] = arg_shapes
        nodes.append(item)
    return {'code': gm.code, 'nodes': nodes,
            'op_counts': dict(Counter(n['target'] for n in nodes if n['op'] == 'call_function'))}


def make_inputs(shape, train_weight, dtype, device, seed):
    t, h, v = shape
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(t, h, generator=g, device=device).to(dtype).requires_grad_()
    w = (torch.randn(v, h, generator=g, device=device) / math.sqrt(h)).to(dtype).requires_grad_(train_weight)
    y = torch.randint(0, v, (t,), generator=g, device=device)
    a = torch.randn(t, generator=g, device=device)
    b = torch.randn(t, generator=g, device=device)
    return x, w, y, a, b


def observe(fn, x, w, y, a, b, mode):
    lp, entropy = fn(x, w, y)
    need_w = w.requires_grad
    if mode == 'entropy_loss':
        outputs, upstream = (lp, entropy), (a, b.to(entropy.dtype))
    else:
        outputs, upstream = (lp,), (a,)
    gradients = torch.autograd.grad(outputs, (x, w) if need_w else (x,), upstream)
    result = {'logp': lp.detach(), 'dX': gradients[0].detach()}
    if mode != 'logprob_only':
        result['entropy'] = entropy.detach()
    if need_w:
        result['dW'] = gradients[1].detach()
    return result


def load_original(cache, download=False):
    manifest = json.loads((ROOT / 'sources.json').read_text())
    sources, identities = obtain_sources(manifest, cache, download)
    ns = {'torch': torch, 'math': math, '_FLASH_ATTN_CROSS_ENTROPY_AVAILABLE': False}
    env = extract_definitions(sources['verl_head'],
        ['_fused_linear_for_ppo_fwd', '_fused_linear_for_ppo_bwd', 'FusedLinearForPPOFunction'], ns)
    return env['FusedLinearForPPOFunction'], identities


def make_callable(original, provider, mode, temperature=0.7, chunk_size=512):
    if mode not in demand_head.MODES or provider not in ('original', 'minimal'):
        raise ValueError('unknown provider or mode')
    if provider == 'original':
        def fn(x, w, y):
            lp, entropy = original.apply(x, w, y, temperature, chunk_size)
            if mode == 'logprob_only':
                return lp, None
            if mode == 'entropy_logged':
                return lp, entropy.detach()
            return lp, entropy
    else:
        def fn(x, w, y):
            return demand_head.head(x, w, y, temperature, mode, chunk_size)
    return fn


def compare_outputs(actual, reference, dtype):
    if set(actual) != set(reference):
        raise ValueError('output/gradient key mismatch')
    rtol, atol = (0.03, 0.02) if dtype == torch.bfloat16 else (2e-5, 2e-6)
    metrics = {}
    for name in actual:
        torch.testing.assert_close(actual[name], reference[name], rtol=rtol, atol=atol)
        metrics[name] = error_metrics(actual[name], reference[name])
    return metrics


def record_aot(fn, inputs, mode):
    # This executes AOT-generated aten graphs. It is NOT an Inductor timing.
    from torch._dynamo.backends.common import aot_autograd
    from functorch.compile import make_boxed_func
    graphs = {'forward': [], 'backward': []}

    def fw(gm, example_inputs):
        graphs['forward'].append(graph_record(gm))
        return make_boxed_func(gm.forward)

    def bw(gm, example_inputs):
        graphs['backward'].append(graph_record(gm))
        return make_boxed_func(gm.forward)

    compiled = torch.compile(fn, backend=aot_autograd(fw_compiler=fw, bw_compiler=bw),
                             fullgraph=True, dynamic=False)
    result = observe(compiled, *inputs, mode)
    if not graphs['forward'] or not graphs['backward']:
        raise RuntimeError('AOT forward/backward not both observed')
    return result, graphs


def cpu_run(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError("CPU audit requires CUDA_VISIBLE_DEVICES=''")
    if args.output.exists():
        raise FileExistsError('refuse to overwrite evidence')
    torch.set_num_threads(1)
    original, identities = load_original(args.cache, args.download)
    from torch._dynamo import config
    config.suppress_errors = False
    # Separate entropy/grad variants are intentional specializations.
    if hasattr(config, 'recompile_limit'):
        config.recompile_limit = 64
    records = []
    for train_weight in (False, True):
        for mode in demand_head.MODES:
            data = make_inputs((7, 5, 11), train_weight, torch.float32, 'cpu', 7207)
            reference = observe(make_callable(original, 'original', mode), *data, mode)
            for provider in ('original', 'minimal'):
                fn = make_callable(original, provider, mode)
                row = {'provider': provider, 'train_weight': train_weight, 'entropy_mode': mode,
                       'shape_T_H_V': [7, 5, 11], 'dtype': 'float32', 'temperature': 0.7}
                x, w, y, a, b = data
                fw, bw = Calls(), Calls()
                with fw:
                    lp, ent = fn(x, w, y)
                outputs = (lp, ent) if mode == 'entropy_loss' else (lp,)
                upstream = (a, b.to(ent.dtype)) if mode == 'entropy_loss' else (a,)
                with bw:
                    eager_grads = torch.autograd.grad(outputs, (x, w) if train_weight else (x,), upstream)
                row['eager'] = {'forward_ops': dict(fw.ops), 'backward_ops': dict(bw.ops),
                                'forward_mm': fw.mm, 'backward_mm': bw.mm}
                row['eager_errors'] = compare_outputs(observe(fn, *data, mode), reference, torch.float32)
                try:
                    aot_out, row['aot_graphs'] = record_aot(fn, data, mode)
                    row['aot_errors'] = compare_outputs(aot_out, reference, torch.float32)
                    row['aot_status'] = 'pass'
                except Exception:
                    row['aot_status'] = 'failed'
                    row['aot_exception'] = traceback.format_exc()
                if args.inductor:
                    try:
                        compiled = torch.compile(fn, backend='inductor', fullgraph=True, dynamic=False)
                        got = observe(compiled, *data, mode)
                        row['inductor_errors'] = compare_outputs(got, reference, torch.float32)
                        # A second data seed checks execution is not tied to trace values.
                        alternate = make_inputs((7, 5, 11), train_weight, torch.float32, 'cpu', 7208)
                        alternate_ref = observe(make_callable(original, 'original', mode), *alternate, mode)
                        row['inductor_heldout_errors'] = compare_outputs(
                            observe(compiled, *alternate, mode), alternate_ref, torch.float32)
                        row['inductor_status'] = 'pass'
                    except Exception:
                        row['inductor_status'] = 'failed'
                        row['inductor_exception'] = traceback.format_exc()
                else:
                    row['inductor_status'] = 'not_requested'
                records.append(row)
                bw_mm = [n.get('tensor_arg_shapes') for g in row.get('aot_graphs', {}).get('backward', [])
                         for n in g['nodes'] if n['target'] == 'aten.mm.default']
                print(provider, train_weight, mode, 'AOT', row['aot_status'],
                      'Inductor', row['inductor_status'], 'AOT backward mm', bw_mm, flush=True)
    if torch.cuda.is_initialized():
        raise RuntimeError('unexpected CUDA initialization')
    failures = sum(r['aot_status'] == 'failed' or r['inductor_status'] == 'failed' for r in records)
    git = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True)
    result = {'schema': 1, 'task': 'P02_CPU_COMPILER', 'status': 'pass' if not failures else 'partial',
              'source_commit': git.stdout.strip(), 'utc': datetime.now(timezone.utc).isoformat(),
              'torch': torch.__version__, 'cpu_threads': torch.get_num_threads(), 'sources': identities,
              'cuda_initialized': torch.cuda.is_initialized(), 'performance_measured': False,
              'inductor_requested': args.inductor, 'records': records,
              'limits': ['CPU source-fragment/AOT/Inductor checks, NOT GPU performance',
                         'No full verl installation or real training run',
                         'FlashAttention optional branch disabled and not benchmarked',
                         'AOT node counts are NOT CUDA kernel counts',
                         'No specialized CCE, Liger SM90 or cuTile kernel execution']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('SAVED', args.output, 'records', len(records), 'failures', failures, flush=True)
    return 0 if not failures else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--inductor', action='store_true')
    raise SystemExit(cpu_run(parser.parse_args()))
