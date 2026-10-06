"""Small CPU FP64 mathematical oracle, NOT an optimized output-head kernel.

The VJP is an explicit calculus reference independent of autograd. Arbitrary
per-row upstream gradients cover log-probability and entropy consumers without
claiming an implementation of the full PPO/GRPO objective.
"""
from __future__ import annotations

import math
import torch


def _validate(x, w, targets, temperature, mask):
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be a real scalar")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if x.ndim != 2 or w.ndim != 2 or targets.ndim != 1:
        raise ValueError("expected X[T,H], W[V,H], target[T]")
    if x.shape[0] != targets.numel() or x.shape[1] != w.shape[1] or min(*x.shape, *w.shape) <= 0:
        raise ValueError("incompatible or empty dimensions")
    if x.dtype != torch.float64 or w.dtype != torch.float64 or targets.dtype != torch.int64:
        raise TypeError("this oracle requires FP64 inputs/weights and int64 labels")
    if any(t.device.type != "cpu" for t in (x, w, targets)):
        raise ValueError("P01 oracle is deliberately CPU-only")
    if not torch.isfinite(x).all() or not torch.isfinite(w).all():
        raise ValueError("finite inputs required")
    valid = targets != -100
    if ((targets[valid] < 0) | (targets[valid] >= w.shape[0])).any():
        raise ValueError("target outside vocabulary")
    if mask is not None:
        if mask.shape != targets.shape or mask.dtype != torch.bool or mask.device.type != "cpu":
            raise ValueError("mask must be CPU bool[T]")
        valid = valid & mask
    return valid, targets.masked_fill(targets == -100, 0)


def outputs(x, w, targets, *, temperature=1.0, mask=None):
    """Return masked log p(target) and full-vocabulary Shannon entropy."""
    valid, safe = _validate(x, w, targets, temperature, mask)
    logall = torch.log_softmax((x @ w.T) / temperature, dim=-1)
    logp = logall.gather(1, safe[:, None]).squeeze(1)
    entropy = -(logall.exp() * logall).sum(-1)
    zero = torch.zeros_like(logp)
    return torch.where(valid, logp, zero), torch.where(valid, entropy, zero)


def _upstream(value, x):
    if value is None:
        return torch.zeros(x.shape[0], dtype=torch.float64)
    if value.shape != (x.shape[0],) or value.dtype != torch.float64 or value.device.type != "cpu":
        raise ValueError("upstream gradient must be CPU FP64[T]")
    if not torch.isfinite(value).all():
        raise ValueError("upstream gradient must be finite")
    return value


def vjp(x, w, targets, grad_logp=None, grad_entropy=None, *, temperature=1.0,
        mask=None, need_dx=True, need_dw=True):
    """Explicit VJP for sum(a*logp + b*entropy), with optional dX/dW.

    For z = XW^T/tau:
      dJ/dz = a*(onehot-p) - b*p*(log(p)+entropy)
      dJ/dX = (dJ/dz)/tau @ W
      dJ/dW = ((dJ/dz)/tau).T @ X

    Entropy logging is represented by grad_entropy=None, not by discarding
    required entropy values in the forward contract.
    """
    valid, safe = _validate(x, w, targets, temperature, mask)
    if grad_logp is None and grad_entropy is None:
        raise ValueError("at least one upstream gradient is required")
    if type(need_dx) is not bool or type(need_dw) is not bool:
        raise TypeError("gradient requirement flags must be booleans")
    a = _upstream(grad_logp, x) * valid
    b = _upstream(grad_entropy, x) * valid
    logall = torch.log_softmax(x @ w.T / temperature, dim=-1)
    p = logall.exp()
    entropy = -(p * logall).sum(-1)
    one_hot = torch.zeros_like(p).scatter(1, safe[:, None], 1.0)
    dz = a[:, None] * (one_hot - p) - b[:, None] * p * (logall + entropy[:, None])
    dmatmul = dz / temperature
    return (dmatmul @ w if need_dx else None,
            dmatmul.T @ x if need_dw else None)


def chunked_outputs(x, w, targets, *, chunk_size, temperature=1.0, mask=None):
    """Reference composition only; not a memory-efficient autograd operator."""
    _validate(x, w, targets, temperature, mask)
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    logp, entropy = [], []
    for start in range(0, x.shape[0], chunk_size):
        stop = min(start + chunk_size, x.shape[0])
        lp, ent = outputs(x[start:stop], w, targets[start:stop], temperature=temperature,
                          mask=None if mask is None else mask[start:stop])
        logp.append(lp); entropy.append(ent)
    return torch.cat(logp), torch.cat(entropy)
