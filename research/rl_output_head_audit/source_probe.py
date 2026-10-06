#!/usr/bin/env python3
"""CPU source-fragment audit. No framework install, CUDA work or timing claims.

Read P01_PROTOCOL.md. Pinned upstream definitions are downloaded verbatim into
an external cache, hash-checked, then explicitly selected via AST. This does
NOT exercise full-package dispatch or the SM90/CuTe/CCE device kernels.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
from urllib.request import urlopen

import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parent


def verify_blob(data, expected):
    actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if actual != expected:
        raise ValueError(f"source identity mismatch: {actual} != {expected}")
    return actual


def extract_definitions(data, names, namespace):
    """Execute only named top-level definitions from already trusted sources.

    This is not a security sandbox. Only the pinned, reviewed upstream files
    are used by run(); imports and unrelated top-level setup are not executed.
    """
    tree = ast.parse(data.decode("utf-8"))
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    counts = Counter(n.name for n in selected)
    if set(counts) != set(names) or any(value != 1 for value in counts.values()):
        raise ValueError("missing or duplicate selected source definition")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future] + selected, type_ignores=[]))
    env = dict(namespace)
    exec(compile(module, "<verified-upstream-fragment>", "exec"), env)
    return env


def mm_role(left, right, tokens, hidden, vocab):
    if len({tokens, hidden, vocab}) != 3:
        raise ValueError("probe axes must differ to attribute matrix multiplies")
    signature = (tuple(left), tuple(right))
    lookup = {
        ((tokens, hidden), (hidden, vocab)): "projection",
        ((tokens, vocab), (vocab, hidden)): "dX",
        ((vocab, tokens), (tokens, hidden)): "dW",
    }
    return lookup.get(signature, "other")


class Calls(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = Counter()
        self.mm = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        self.ops[str(func)] += 1
        if func == torch.ops.aten.mm.default:
            self.mm.append({"left": list(args[0].shape), "right": list(args[1].shape),
                            "role": mm_role(args[0].shape, args[1].shape, 7, 5, 11)})
        return func(*args, **kwargs)


def obtain_sources(manifest, cache, download):
    data, identities = {}, []
    cache.mkdir(parents=True, exist_ok=True)
    for item in manifest["files"]:
        path = cache / (item["id"] + "-" + item["commit"] + ".py")
        url = f'https://raw.githubusercontent.com/{item["repo"]}/{item["commit"]}/{item["path"]}'
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"missing {path}; explicit --download required")
            raw = urlopen(url, timeout=30).read()
            verify_blob(raw, item["blob_sha"])
            path.write_bytes(raw)
        raw = path.read_bytes()
        verify_blob(raw, item["blob_sha"])
        data[item["id"]] = raw
        identities.append(dict(item, raw_url=url, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    return data, identities


def probe(env, provider, train_weight, entropy_mode):
    # Deliberately tiny, distinct dimensions: no wall-time/performance inference.
    generator = torch.Generator().manual_seed(7107)
    x = torch.randn(7, 5, generator=generator, dtype=torch.float32).requires_grad_()
    w = (torch.randn(11, 5, generator=generator, dtype=torch.float32) / math.sqrt(5)).requires_grad_(train_weight)
    target = torch.tensor([1, 0, 5, 10, 3, 4, 9], dtype=torch.int64)
    a = torch.randn(7, generator=generator, dtype=torch.float32)
    b = torch.randn(7, generator=generator, dtype=torch.float32)
    temperature = 0.7
    saved = []

    def pack(t):
        saved.append({"shape": list(t.shape), "dtype": str(t.dtype), "requires_grad": t.requires_grad})
        return t

    forward, backward = Calls(), Calls()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t), forward:
        if provider == "verl_fragment":
            lp, entropy = env["FusedLinearForPPOFunction"].apply(x, w, target, temperature, 512)
        else:
            result = env["_FusedLinearPPOFallbackFunction"].apply(
                x, w, target, temperature, -100, entropy_mode != "logprob_only")
            if entropy_mode == "logprob_only":
                lp, entropy = -result, None
            else:
                nll, entropy = result
                lp = -nll
    loss = (a * lp).sum()
    if entropy_mode == "entropy_logged":
        loss = loss + (b * entropy.detach()).sum()
    elif entropy_mode == "entropy_loss":
        loss = loss + (b * entropy).sum()
    with backward:
        grads = torch.autograd.grad(loss, (x, w) if train_weight else (x,))

    # Compare to separately derived double-precision VJP on exactly these FP32 values.
    import contract
    xx, ww = x.detach().double(), w.detach().double()
    ref_lp, ref_entropy = contract.outputs(xx, ww, target, temperature=temperature)
    dx, dw = contract.vjp(xx, ww, target, a.double(),
                           b.double() if entropy_mode == "entropy_loss" else None,
                           temperature=temperature, need_dw=train_weight)
    pairs = [("logp", lp.double(), ref_lp), ("dX", grads[0].double(), dx)]
    if entropy is not None and entropy_mode != "logprob_only":
        pairs.append(("entropy", entropy.double(), ref_entropy))
    if train_weight:
        pairs.append(("dW", grads[1].double(), dw))
    errors = {}
    for name, actual, reference in pairs:
        torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-6)
        errors[name] = float((actual - reference).abs().max())
    return {
        "provider": provider, "train_weight": train_weight, "entropy_mode": entropy_mode,
        "shape_T_H_V": [7, 5, 11], "dtype": "float32", "temperature": temperature,
        "forward_ops": dict(forward.ops), "backward_ops": dict(backward.ops),
        "forward_mm": forward.mm, "backward_mm": backward.mm,
        "saved_tensors": saved, "numerical_check": "pass", "max_abs_errors": errors,
        "returned_gradient_shapes": [list(t.shape) for t in grads],
        "no_gpu_performance_inference": True,
    }


def run(args):
    if args.output.exists():
        raise FileExistsError("refuse to overwrite evidence")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("run P01 with CUDA_VISIBLE_DEVICES='' to prohibit GPU use")
    torch.set_num_threads(1)
    manifest = json.loads((ROOT / "sources.json").read_text())
    sources, identities = obtain_sources(manifest, args.cache, args.download)
    namespace = {"torch": torch, "math": math, "_FLASH_ATTN_CROSS_ENTROPY_AVAILABLE": False,
                 "_FALLBACK_CHUNK_SIZE": 512}
    verl = extract_definitions(sources["verl_head"], ["_fused_linear_for_ppo_fwd",
        "_fused_linear_for_ppo_bwd", "FusedLinearForPPOFunction"], namespace)
    liger = extract_definitions(sources["liger_scaled"], ["_validate_temperature",
        "_validate_fallback_inputs", "_fallback_forward_chunk", "_fallback_backward_chunk",
        "_FusedLinearPPOFallbackFunction"], namespace)
    records = []
    for name, env in (("verl_fragment", verl), ("liger_fallback_fragment", liger)):
        for train_weight in (True, False):
            for entropy_mode in ("logprob_only", "entropy_logged", "entropy_loss"):
                row = probe(env, name, train_weight, entropy_mode)
                records.append(row)
                print(name, train_weight, entropy_mode,
                      "backward", [m["role"] for m in row["backward_mm"]], flush=True)
    if torch.cuda.is_initialized():
        raise RuntimeError("unexpected CUDA initialization")
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    result = {"schema": 1, "status": "complete", "utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "torch": torch.__version__, "source_commit": git.stdout.strip(),
        "manifest_sha256": hashlib.sha256((ROOT / "sources.json").read_bytes()).hexdigest(),
        "cuda_initialized": torch.cuda.is_initialized(), "cpu_threads": torch.get_num_threads(),
        "sources": identities, "records": records,
        "limits": ["tiny CPU inputs, not large-vocabulary performance", "explicit fallback source fragments, not full packages",
                   "FlashAttention path disabled and not measured", "no CCE or SM90 kernel execution",
                   "an unnecessary eager op might be optimized by compilation; no compiler claim",
                   "mathematical FP64 check does not guarantee BF16 behavior or training convergence"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("COMPLETE", len(records), "CPU cases", args.output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    run(parser.parse_args())
