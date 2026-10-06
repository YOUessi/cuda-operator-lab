#!/usr/bin/env python3
"""R03 full-FFN audit; read R03_PROTOCOL.md before interpreting results.

No device code or production policy edits. Device intervals include all work in
scope; sync wall intervals also include host submission and synchronization.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import audit

STAGES = ("layernorm_fp32", "cast_bf16", "gemm_swiglu", "down_projection_fp32", "residual_add")
SHAPES = [(16,64,64), (32,128,256), (64,128,1024), (128,128,2048),
          (128,512,512), (128,1024,4096)]
ARMS = ("v4_cublas", "v7_custom")


def validate_shape(shape):
    if len(shape) != 3 or any(type(x) is not int or x <= 0 for x in shape):
        raise ValueError("positive integer M,K,N required")
    m, k, n = shape
    if m % 16 or k % 16 or n % 64:
        raise ValueError("both candidates require M,K divisible by16 and N by64")
    return tuple(shape)


def static_choice(shape):
    m, k, n = validate_shape(shape)
    return ARMS[1] if m <= 128 and k <= 128 and n <= 2048 else ARMS[0]


def learn_policy(runs, scope):
    if len(runs) < 2 or scope not in ("isolated", "module"):
        raise ValueError("two independent calibration runs and known scope required")
    tables = []
    for rows in runs:
        table = {}
        for row in rows:
            if row["scope"] != scope:
                continue
            key = row["case"] + "|" + row["mode"]
            if key in table or (row["a"], row["b"]) != ARMS:
                raise ValueError("duplicate calibration key or wrong candidates")
            if row["decision"] not in ("a_faster", "b_faster", "unresolved"):
                raise ValueError("unknown calibration decision")
            table[key] = row["decision"]
        if not table:
            raise ValueError("missing calibration scope")
        tables.append(table)
    if any(set(table) != set(tables[0]) for table in tables):
        raise ValueError("incomplete calibration matrix")
    return {key: ARMS[1] if all(t[key] == "b_faster" for t in tables) else ARMS[0]
            for key in sorted(tables[0])}


def observed_regret(row, choice):
    ratio = float(row["speedup_a_over_b"])
    if not math.isfinite(ratio) or ratio <= 0 or choice not in ARMS:
        raise ValueError("finite positive ratio and known choice required")
    return max(1.0, ratio if choice == ARMS[0] else 1.0 / ratio) - 1.0


def make_module_cases(torch, bindings, shape, seed):
    m, k, n = validate_shape(shape)
    gen = torch.Generator(device="cuda").manual_seed(seed + m * 100000 + k * 100 + n)
    x = torch.randn(m, k, device="cuda", generator=gen)
    norm_w = torch.ones(k, device="cuda")
    norm_b = torch.zeros(k, device="cuda")
    norm = torch.empty_like(x)
    x_bf = torch.empty(m, k, device="cuda", dtype=torch.bfloat16)
    packed = (torch.randn(2*n, k, device="cuda", generator=gen) / math.sqrt(k)).to(torch.bfloat16)
    gate, up = packed[:n], packed[n:]
    wd = torch.randn(k, n, device="cuda", generator=gen) / math.sqrt(n)
    wd_t = wd.T
    workspace = torch.empty(m, 2*n, device="cuda")
    h = torch.empty(m, n, device="cuda")
    down = torch.empty_like(x)
    out = torch.empty_like(x)

    def prefix():
        bindings.layernorm_v2_into(x, norm_w, norm_b, norm)
        x_bf.copy_(norm)

    def suffix():
        torch.mm(h, wd_t, out=down)
        torch.add(down, x, out=out)

    prefix()
    torch.testing.assert_close(norm, torch.nn.functional.layer_norm(x, (k,), norm_w, norm_b),
                               rtol=5e-5, atol=5e-6)
    # The normalized BF16 values define the common projection contract.
    p64 = x_bf.double() @ packed.double().T
    h64 = torch.nn.functional.silu(p64[:, :n]) * p64[:, n:]
    h_ref = h64.float()
    out_ref = (h64 @ wd.double().T + x.double()).float()
    core = {ARMS[0]: lambda: bindings.gemm_swiglu_v4_into(x_bf, packed, workspace, h),
            ARMS[1]: lambda: bindings.gemm_swiglu_v7_into(x_bf, gate, up, h)}
    full = {}
    for label, fn in core.items():
        def forward(fn=fn):
            prefix()
            fn()
            suffix()
        full[label] = forward
    immutable = dict(x=x, norm_w=norm_w, norm_b=norm_b, packed=packed, down_weight=wd)
    scratch = [norm, x_bf, workspace, h, down, out]
    pointers = {name: audit.tensor_info(t) for name, t in dict(
        **immutable, normalized=norm, normalized_bf16=x_bf, gate_view=gate, up_view=up,
        workspace=workspace, h=h, down=down, out=out).items()}
    common = dict(case=f"gemm_{m}x{k}x{n}", rtol=1e-3, atol=1e-3,
                  pairs=[ARMS], pointers=pointers, require_shape_equality=False,
                  immutable=immutable, scratch=scratch, prefix=prefix,
                  keep=[x,norm_w,norm_b,norm,x_bf,packed,gate,up,wd,wd_t,workspace,h,down,out,h_ref,out_ref])
    return {"isolated": dict(common, variants=core, outputs=dict.fromkeys(ARMS,h), reference=h_ref),
            "module": dict(common, variants=full, outputs=dict.fromkeys(ARMS,out), reference=out_ref)}


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    if args.rounds < 2 or args.samples <= 0 or args.samples % 4 or args.warmup < 3:
        raise ValueError("rounds>=2, samples positive multiple of4, warmup>=3 required")
    if len(set(args.modes)) != len(args.modes) or len(set(args.scopes)) != len(args.scopes):
        raise ValueError("duplicate scopes or modes")
    if args.role == "test" and not args.policy:
        raise ValueError("held-out test requires a frozen policy file")
    if args.policy and args.role != "test":
        raise ValueError("calibration must not consume a policy")
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "python"))
    import torch
    from cuda_operator_lab import bindings
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=False)
    bindings._library()
    library = Path(bindings._default_library_path())
    meta = dict(status="running", base_commit=audit.BASE,
        source_commit=audit.command(["git","-C",str(root),"rev-parse","HEAD"]),
        working_tree=audit.command(["git","-C",str(root),"status","--porcelain"]),
        started_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
        python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
        config={k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        library_path=str(library), library_sha256=sha(library),
        gpu_initial=audit.gpu_state(), gpu_snapshots=[], stages=STAGES,
        policy_sha256=sha(args.policy) if args.policy else None,
        clock_policy="unlocked; no power changes", cache_policy="natural warm reuse; no flush",
        precision="FP32 norm -> BF16 input/weights -> FP32 SwiGLU/down projection/residual",
        limits=["synthetic FFN, not full Transformer", "FP32 down projection may dilute candidate effect",
                "fixed weights and warm buffers", "descriptive within-process round bootstrap; no multiplicity correction",
                "same-shape held-out processes, not unseen-shape generalization", "no profiler causal evidence"])
    raw, summaries = [], []
    wall0 = time.perf_counter_ns()

    def save():
        meta["elapsed_wall_ns"] = time.perf_counter_ns() - wall0
        write_json(args.output/"metadata.json",meta)
        write_json(args.output/"summary.json",summaries)
        payload = json.dumps(dict(schema=2,groups=raw), separators=(",",":"),allow_nan=False).encode()
        (args.output/"raw.json.gz").write_bytes(gzip.compress(payload,mtime=0))
        if summaries:
            with (args.output/"summary.csv").open("w",newline="",encoding="utf-8") as f:
                writer=csv.DictWriter(f,fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)

    save()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    shapes = list(SHAPES)
    random.Random(args.order_seed).shuffle(shapes)
    try:
        with torch.no_grad(), torch.cuda.stream(stream):
            for idx, shape in enumerate(shapes):
                cases = make_module_cases(torch, bindings, shape, args.data_seed)
                jobs = [(scope,mode) for scope in args.scopes for mode in args.modes]
                random.Random(args.order_seed+idx).shuffle(jobs)
                for j, (scope,mode) in enumerate(jobs):
                    case = cases[scope]
                    case["prefix"]()
                    stream.synchronize()
                    before = audit.gpu_state()
                    record, results = audit.measure(torch,case,mode,stream,args,args.order_seed*10000+idx*100+j)
                    record.update(scope=scope,shape=list(shape))
                    raw.append(record)
                    for result in results:
                        result.update(scope=scope,m=shape[0],k=shape[1],n=shape[2])
                    summaries.extend(results)
                    meta["gpu_snapshots"].append(dict(case=case["case"],scope=scope,mode=mode,
                                                       before=before,after=audit.gpu_state()))
                    print(f"scope={scope}",flush=True)
                    save()
                del cases
            stream.synchronize()
        meta["status"] = "complete"
    except BaseException as exc:
        meta["status"]="failed"; meta["error"]=repr(exc)
        raise
    finally:
        meta["finished_utc"]=datetime.now(timezone.utc).isoformat()
        meta["gpu_final"]=audit.gpu_state()
        save()
    print(f"COMPLETE {args.output}: {len(raw)} groups",flush=True)


def fit(args):
    if args.output.exists():
        raise FileExistsError("refuse to overwrite a frozen policy")
    start=time.perf_counter_ns()
    if len({p.resolve() for p in args.inputs}) != len(args.inputs):
        raise ValueError("duplicate calibration directory")
    rows, sources = [], []
    for path in args.inputs:
        meta=json.loads((path/"metadata.json").read_text())
        if meta["status"] != "complete" or meta["config"]["role"] != "calibration":
            raise ValueError("only complete calibration runs may fit policy")
        rows.append(json.loads((path/"summary.json").read_text()))
        sources.append(dict(path=str(path),summary_sha256=sha(path/"summary.json"),
                            source_commit=meta["source_commit"],wall_ns=meta["elapsed_wall_ns"],
                            data_seed=meta["config"]["data_seed"],order_seed=meta["config"]["order_seed"]))
    policy=dict(schema=1,sources=sources,rule="V7 only when every calibration process supports >5% via round CI",
                isolated=learn_policy(rows,"isolated"), module=learn_policy(rows,"module"),
                fit_wall_ns=time.perf_counter_ns()-start,
                limits="same-shape offline lookup; no runtime implementation or unseen-shape claim")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,policy)
    print(f"FROZEN {args.output} sha256={sha(args.output)}")


def evaluate(args):
    if args.output.exists():
        raise FileExistsError("refuse to overwrite evaluation")
    policy=json.loads(args.policy.read_text())
    policy_sha=sha(args.policy)
    for source in policy["sources"]:
        if sha(Path(source["path"])/"summary.json") != source["summary_sha256"]:
            raise ValueError("calibration evidence changed after freezing")
    result=[]
    for path in args.inputs:
        meta=json.loads((path/"metadata.json").read_text())
        if meta["status"] != "complete" or meta["config"]["role"] != "test" or meta["policy_sha256"] != policy_sha:
            raise ValueError("evaluation requires complete held-out runs tied to this frozen policy")
        for row in json.loads((path/"summary.json").read_text()):
            if row["scope"] != "module":
                continue
            key=row["case"]+"|"+row["mode"]
            choices=dict(always_v4=ARMS[0], static_v11=static_choice((row["m"],row["k"],row["n"])),
                isolated_eager_lookup=policy["isolated"][row["case"]+"|eager"],
                isolated_mode_lookup=policy["isolated"][key],module_mode_lookup=policy["module"][key])
            for name, choice in choices.items():
                result.append(dict(run=path.name,case=row["case"],mode=row["mode"],policy=name,
                    selected=choice,observed_regret=observed_regret(row,choice),
                    paired_a_over_b=row["speedup_a_over_b"],decision=row["decision"],
                    selected_median_ns=row["a_median_ns"] if choice==ARMS[0] else row["b_median_ns"]))
    write_json(args.output,dict(policy_sha256=policy_sha,records=result,
        metric="relative to measured best of two in held-out full-module groups; no selection-cost adjustment"))
    print(f"EVALUATED {len(result)} selections -> {args.output}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="action",required=True)
    p=commands.add_parser("run")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--role",choices=["calibration","test"],required=True)
    p.add_argument("--policy",type=Path)
    p.add_argument("--rounds",type=int,default=6)
    p.add_argument("--samples",type=int,default=8)
    p.add_argument("--warmup",type=int,default=20)
    p.add_argument("--data-seed",type=int,default=1701)
    p.add_argument("--order-seed",type=int,default=2101)
    p.add_argument("--modes",nargs="+",choices=["eager","graph1"],default=["eager","graph1"])
    p.add_argument("--scopes",nargs="+",choices=["isolated","module"],default=["isolated","module"])
    for name in ("fit","evaluate"):
        p=commands.add_parser(name)
        p.add_argument("--inputs",nargs="+",type=Path,required=True)
        p.add_argument("--output",type=Path,required=True)
        if name=="evaluate":
            p.add_argument("--policy",type=Path,required=True)
    args=parser.parse_args()
    {"run":run,"fit":fit,"evaluate":evaluate}[args.action](args)


if __name__ == "__main__":
    main()
