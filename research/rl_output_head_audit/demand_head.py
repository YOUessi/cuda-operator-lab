# SPDX-License-Identifier: Apache-2.0
# Formula and chunked autograd control adapted from verl-project/verl
# 8718ca30a3f002f93b7c4fd99b9b2506718681bc,
# verl/utils/experimental/torch_functional.py (Copyright 2024 Bytedance Ltd.).
# Research control only: no new CUDA kernel and no claimed algorithm novelty.
"""Prune unused entropy and gradient GEMMs, retaining upstream arithmetic.

`head` assumes validated 2-D inputs and legal labels and is compilation-friendly.
`checked_head` validates metadata/labels and implements masks outside that core.
No autocast, distributional approximations, or custom double backward support.
"""
from __future__ import annotations

import math
import torch

MODES = ('logprob_only', 'entropy_logged', 'entropy_loss')


class DemandHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, y, temperature, mode, chunk_size):
        ctx.set_materialize_grads(False)
        need_entropy = mode != 'logprob_only'
        tokens = x.shape[0]
        lp = torch.zeros(tokens, device=x.device, dtype=torch.float32)
        ent = x.new_zeros(tokens) if need_entropy else None
        for start in range(0, tokens, chunk_size):
            end = min(start + chunk_size, tokens)
            logits = (x[start:end] @ w.t()) / temperature
            dtype = logits.dtype
            logits = logits.to(torch.float32)
            if need_entropy:
                probs = logits.softmax(-1)
                value = torch.logsumexp(logits, -1) - torch.sum(probs * logits, -1)
                ent[start:end] = value.to(dtype)
            log_probs = logits.log_softmax(-1)
            lp[start:end] = log_probs.gather(-1, y[start:end, None]).squeeze(-1)
        ctx.save_for_backward(x, w, y)
        ctx.temperature = temperature
        ctx.chunk_size = chunk_size
        if mode == 'entropy_logged':
            ctx.mark_non_differentiable(ent)
        return lp, ent

    @staticmethod
    def backward(ctx, dlp, dent):
        x, w, y = ctx.saved_tensors
        need_dx, need_dw = ctx.needs_input_grad[:2]
        dx = torch.zeros_like(x) if need_dx else None
        dw = torch.zeros_like(w) if need_dw else None
        for start in range(0, x.shape[0], ctx.chunk_size):
            end = min(start + ctx.chunk_size, x.shape[0])
            hidden = x[start:end]
            logits = (hidden @ w.t()) / ctx.temperature
            dtype = logits.dtype
            logits = logits.to(torch.float32)
            probs = logits.softmax(-1)
            dz = 0
            if dlp is not None:
                one_hot = torch.zeros_like(logits).scatter_(-1, y[start:end, None], 1)
                dz += dlp[start:end].to(torch.float32)[:, None] * (one_hot - probs)
            if dent is not None:
                log_probs = logits.log_softmax(-1)
                entropy = torch.logsumexp(logits, -1) - torch.sum(probs * logits, -1)
                dz += probs * (log_probs + entropy[:, None]) * (-dent[start:end, None])
            # Preserve BF16 cast BEFORE temperature gradient division.
            dz = dz.to(dtype) / ctx.temperature
            if need_dx:
                dx[start:end] += dz @ w
            if need_dw:
                dw += dz.t() @ hidden
        return dx, dw, None, None, None, None


def head(x, w, y, temperature=1.0, mode='entropy_loss', chunk_size=512):
    """Unchecked core: input[X,H], weight[V,H], legal target[X]."""
    return DemandHead.apply(x, w, y, temperature, mode, chunk_size)


def checked_head(x, w, y, temperature=1.0, mode='entropy_loss', chunk_size=512,
                 mask=None, ignore_index=-100):
    """Validation adapter. GPU label checks synchronize; do not time them as core."""
    if mode not in MODES:
        raise ValueError('unknown entropy mode')
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError('temperature must be a real scalar')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be positive and finite')
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError('chunk_size must be a positive integer')
    if x.ndim != 2 or w.ndim != 2 or y.ndim != 1:
        raise ValueError('expected 2-D input/weight and 1-D labels')
    if x.shape[0] != y.shape[0] or x.shape[1] != w.shape[1] or min(*x.shape, *w.shape) < 1:
        raise ValueError('invalid or empty dimensions')
    if x.device != w.device or x.device != y.device:
        raise ValueError('devices must match')
    if x.dtype != w.dtype or x.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise TypeError('input/weight must have matching supported floating dtype')
    if y.dtype != torch.int64:
        raise TypeError('labels must be int64')
    valid = y != ignore_index
    if bool(((y < 0) | (y >= w.shape[0]))[valid].any()):
        raise ValueError('target out of range')
    if mask is not None:
        if mask.shape != y.shape or mask.dtype != torch.bool or mask.device != y.device:
            raise ValueError('mask must be a same-device boolean vector')
        valid = valid & mask
    safe_y = torch.where(y == ignore_index, torch.zeros_like(y), y)
    lp, ent = head(x, w, safe_y, temperature, mode, chunk_size)
    lp = torch.where(valid, lp, torch.zeros_like(lp))
    if ent is not None:
        ent = torch.where(valid, ent, torch.zeros_like(ent))
    return lp, ent
