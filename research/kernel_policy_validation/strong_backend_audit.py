#!/usr/bin/env python3
"""R06 research only: matched BF16-boundary FFN, strong compiler, frozen policy.
See R06_PROTOCOL.md. No kernel edits; every candidate includes the entire module.
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
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import audit

SHAPES = [(16,64,64),(32,128,256),(128,512,512),
          (1,1536,8960),(16,1536,8960),(128,1536,8960)]
MODES = ('eager','graph1')


def validate_shape(shape):
    if len(shape)!=3 or any(type(x) is not int or x<=0 for x in shape):
        raise ValueError('positive integer M,K,N required')
    return tuple(shape)


def custom_supported(shape):
    m,k,n=validate_shape(shape)
    return not (m%16 or k%16 or n%64)


def static_choice(shape,mode):
    m,k,n=validate_shape(shape)
    return 'custom_v7' if custom_supported(shape) and m<=128 and k<=128 and n<=2048 else 'cublas_v4'


def precision_contract():
    return dict(input='bfloat16',weights='bfloat16',norm_statistics='float32',
        normalized_boundary='bfloat16',projection_output='float32',
        activation_boundary='bfloat16',down_projection_output='float32',output='float32',
        reference='float64_with_bfloat16_boundaries',eps=1e-6,rtol=0.02,atol=0.01,relative_l2=0.003)


def fit_table(runs):
    if len(runs)<2:raise ValueError('at least two calibration processes required')
    tables=[]
    for rows in runs:
        table={}
        for row in rows:
            key=row['case']+'|'+row['mode']
            arm=row['arm'];lat=float(row['median_ns'])
            if not math.isfinite(lat) or lat<=0:raise ValueError('invalid latency')
            if arm in table.setdefault(key,{}):raise ValueError('duplicate measurement')
            table[key][arm]=lat
        tables.append(table)
    if not tables[0] or any(set(t)!=set(tables[0]) for t in tables):
        raise ValueError('incomplete case/mode matrix')
    result={}
    for key in sorted(tables[0]):
        arms=set(tables[0][key])
        if 'inductor' not in arms or any(set(t[key])!=arms for t in tables):
            raise ValueError('missing compiler or different candidate coverage')
        accepted=[a for a in arms if a!='inductor' and
                  all(t[key]['inductor']/t[key][a]>=1.05 for t in tables)]
        result[key]=min(accepted,key=lambda a:statistics.mean(math.log(t[key][a]) for t in tables)) if accepted else 'inductor'
    return result


def regret(latencies,choice):
    if choice not in latencies or any(not math.isfinite(v) or v<=0 for v in latencies.values()):
        raise ValueError('valid candidate/latencies required')
    return latencies[choice]/min(latencies.values())-1


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')


def make_case(torch,bindings,shape,seed):
    m,k,n=validate_shape(shape)
    gen=torch.Generator(device='cuda').manual_seed(seed+m*100000+k*100+n)
    setup=time.perf_counter_ns()
    x=torch.randn(m,k,device='cuda',generator=gen).to(torch.bfloat16)
    scale=torch.ones(k,device='cuda',dtype=torch.bfloat16)
    pw=(torch.randn(2*n,k,device='cuda',generator=gen)/math.sqrt(k)).to(torch.bfloat16)
    dw=(torch.randn(k,n,device='cuda',generator=gen)/math.sqrt(n)).to(torch.bfloat16)
    ws=torch.empty(m,2*n,device='cuda',dtype=torch.float32)
    h=torch.empty(m,n,device='cuda',dtype=torch.float32)
    gate,up=pw[:n],pw[n:]
    torch.cuda.synchronize()
    setup_ns=time.perf_counter_ns()-setup

    def prefix(x,scale):
        xf=x.float()
        return (xf*torch.rsqrt(xf.square().mean(-1,keepdim=True)+1e-6)*scale.float()).to(torch.bfloat16)

    def suffix(hidden,x,dw):
        return torch.mm(hidden.to(torch.bfloat16),dw.T,out_dtype=torch.float32)+x.float()

    def torch_forward(x,scale,pw,dw):
        q=prefix(x,scale)
        p=torch.mm(q,pw.T,out_dtype=torch.float32)
        width=pw.shape[0]//2
        hidden=torch.nn.functional.silu(p[:,:width])*p[:,width:]
        return suffix(hidden,x,dw)

    def custom_forward(version):
        def forward():
            q=prefix(x,scale)
            if version==4:bindings.gemm_swiglu_v4_into(q,pw,ws,h)
            else:bindings.gemm_swiglu_v7_into(q,gate,up,h)
            return suffix(h,x,dw)
        return forward

    # Independent high-precision oracle preserves every specified BF16 boundary.
    xd=x.double()
    q64=(xd*torch.rsqrt(xd.square().mean(-1,keepdim=True)+1e-6)*scale.double()).to(torch.bfloat16)
    p64=q64.double()@pw.double().T
    h64=(torch.nn.functional.silu(p64[:,:n])*p64[:,n:]).to(torch.bfloat16)
    reference=(h64.double()@dw.double().T+xd).float()
    variants={'torch_packed':lambda:torch_forward(x,scale,pw,dw),'cublas_v4':custom_forward(4)}
    if custom_supported(shape):variants['custom_v7']=custom_forward(7)
    immutable={'x':x,'scale':scale,'packed_weight':pw,'down_weight':dw}
    return dict(case=f'ffn_{m}x{k}x{n}',shape=shape,variants=variants,
        torch_forward=torch_forward,args=(x,scale,pw,dw),reference=reference,
        immutable=immutable,snapshots={k:v.clone() for k,v in immutable.items()},
        setup_ns=setup_ns,keep=[ws,h,gate,up])


def check(torch,y,ref):
    c=precision_contract()
    if y.dtype!=torch.float32 or y.shape!=ref.shape or not torch.isfinite(y).all():
        raise AssertionError('invalid output dtype/shape/finite contract')
    torch.testing.assert_close(y,ref,rtol=c['rtol'],atol=c['atol'])
    d=(y-ref).abs()
    rel=float(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(ref).clamp_min(1e-20))
    if rel>c['relative_l2']:raise AssertionError(f'relative L2 {rel} exceeds contract')
    return dict(max_abs=float(d.max()),mean_abs=float(d.mean()),relative_l2=rel)


def counter_snapshot(torch):
    return {str(k):{str(a):int(b) for a,b in v.items()} for k,v in torch._dynamo.utils.counters.items()}


def run(args):
    if args.rounds<2 or args.samples<=0 or args.samples%24 or args.warmup<3:
        raise ValueError('rounds>=2, samples positive multiple of24, warmup>=3 required')
    if (args.role=='test')!=bool(args.policy):raise ValueError('only test requires frozen policy')
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'python'))
    import torch
    from cuda_operator_lab import bindings
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch._dynamo.config.suppress_errors=False
    args.output.mkdir(parents=True,exist_ok=False)
    bindings._library()
    meta=dict(status='running',role=args.role,started_utc=datetime.now(timezone.utc).isoformat(),
        pid=os.getpid(),source_commit=audit.command(['git','-C',str(root),'rev-parse','HEAD']),
        base_commit=audit.BASE,torch=torch.__version__,cuda=torch.version.cuda,
        python=platform.python_version(),library_sha256=sha(bindings._default_library_path()),
        config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        contract=precision_contract(),compile_mode='max-autotune-no-cudagraphs',
        fullgraph=True,dynamic=False,mode_options=torch._inductor.list_mode_options(),
        compile_cache=os.environ.get('TORCHINDUCTOR_CACHE_DIR','default; may contain prior compilations'),
        gpu_initial=audit.gpu_state(),cases=[],policy_sha256=sha(args.policy) if args.policy else None,
        limits=['single unlocked GPU','synthetic weights, not model inference','natural warm reuse',
                'no cold-cache claim for compiler setup','observed finite-candidate minimum is noisy',
                'same shape held-out processes, not unseen-shape generalization'])
    policy=json.loads(args.policy.read_text())['choices'] if args.policy else None
    raw=[];summary=[];policies=[]
    wall0=time.perf_counter_ns()
    def save():
        meta['elapsed_wall_ns']=time.perf_counter_ns()-wall0
        dump(args.output/'metadata.json',meta)
        dump(args.output/'summary.json',summary)
        dump(args.output/'policy_results.json',policies)
        data=json.dumps(dict(schema=6,groups=raw),separators=(',',':'),allow_nan=False).encode()
        (args.output/'raw.json.gz').write_bytes(gzip.compress(data,mtime=0))
        if summary:
            with (args.output/'summary.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    save()
    shapes=list(args.shapes or (SHAPES[:1] if args.smoke else SHAPES))
    random.Random(args.order_seed).shuffle(shapes)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.no_grad(),torch.cuda.stream(stream):
            for idx,shape in enumerate(shapes):
                case=make_case(torch,bindings,shape,args.data_seed)
                info=dict(case=case['case'],shape=shape,setup_ns=case['setup_ns'],
                    pointers={k:audit.tensor_info(v) for k,v in case['immutable'].items()},
                    unsupported=[] if custom_supported(shape) else ['custom_v7'],gpu_before=audit.gpu_state())
                meta['cases'].append(info);save()
                torch._dynamo.reset()
                before=counter_snapshot(torch)
                t=time.perf_counter_ns()
                compiled=torch.compile(case['torch_forward'],fullgraph=True,dynamic=False,mode='max-autotune-no-cudagraphs')
                initial=compiled(*case['args']);stream.synchronize()
                info['compile_first_call_ns']=time.perf_counter_ns()-t
                info['compiler_check']=check(torch,initial,case['reference']);del initial
                info['compile_counters_before']=before;info['compile_counters_after']=counter_snapshot(torch)
                case['variants']['inductor']=lambda compiled=compiled,case=case:compiled(*case['args'])
                modes=list(MODES);random.Random(args.order_seed+idx).shuffle(modes)
                for mode in modes:
                    fns=case['variants'];graphs=[]
                    for fn in fns.values():
                        for _ in range(args.warmup):tmp=fn()
                    del tmp
                    stream.synchronize()
                    graph_setup=time.perf_counter_ns()
                    if mode=='graph1':
                        captured={}
                        for label,fn in fns.items():
                            graph=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph,stream=stream):output=fn()
                            def replay(graph=graph,output=output):
                                graph.replay();return output
                            graphs.append((graph,output));captured[label]=replay
                        fns=captured
                    graph_setup_ns=time.perf_counter_ns()-graph_setup
                    errors={}
                    for label,fn in fns.items():
                        for _ in range(args.warmup):tmp=fn()
                        stream.synchronize();errors[label]=check(torch,tmp,case['reference']);del tmp
                    labels=list(fns)
                    events={a:[] for a in labels};walls={a:[] for a in labels};orders_saved=[]
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record(stream);end.record(stream);end.synchronize()
                    for ridx in range(args.rounds):
                        orders=audit.balanced_orders(labels,args.samples,args.order_seed*10000+idx*100+ridx)
                        orders_saved.append(orders)
                        er={a:[] for a in labels};wr={a:[] for a in labels}
                        for order in orders:
                            for label in order:
                                t0=time.perf_counter_ns();start.record(stream)
                                output=fns[label]()
                                end.record(stream);end.synchronize()
                                wr[label].append(time.perf_counter_ns()-t0)
                                er[label].append(int(round(start.elapsed_time(end)*1e6)))
                                del output
                        for a in labels:events[a].append(er[a]);walls[a].append(wr[a])
                    rowlat={}
                    for a in labels:
                        vals=[statistics.median(v) for v in events[a]]
                        rowlat[a]=statistics.median(vals)
                        summary.append(dict(case=case['case'],m=shape[0],k=shape[1],n=shape[2],mode=mode,arm=a,
                            median_ns=rowlat[a],p95_ns=audit.quantile([v for r in events[a] for v in r],0.95),
                            wall_median_ns=statistics.median([statistics.median(v) for v in walls[a]])))
                    pairs={a:audit.paired_summary(events['inductor'],events[a]) for a in labels if a!='inductor'}
                    raw.append(dict(case=case['case'],shape=shape,mode=mode,labels=labels,event_ns=events,
                        wall_ns=walls,orders=orders_saved,errors=errors,pairs_vs_inductor=pairs,
                        graph_setup_ns=graph_setup_ns,gpu_after=audit.gpu_state()))
                    choices={'fixed_cublas':'cublas_v4','fixed_compiler':'inductor','legacy_static':static_choice(shape,mode)}
                    if policy:
                        key=case['case']+'|'+mode
                        if key not in policy:raise ValueError('held-out shape absent in frozen table')
                        choices['frozen_table']=policy[key]
                    for name,choice in choices.items():
                        policies.append(dict(case=case['case'],mode=mode,policy=name,choice=choice,
                            regret=regret(rowlat,choice),chosen_ns=rowlat[choice],best_ns=min(rowlat.values())))
                    print(case['case'],mode,{a:round(v/1000,3) for a,v in rowlat.items()},flush=True)
                    save();del graphs
                for name,tensor in case['immutable'].items():
                    torch.testing.assert_close(tensor,case['snapshots'][name],rtol=0,atol=0)
                info['immutable_check']='pass';info['gpu_after']=audit.gpu_state();save()
                case['variants'].clear();del case,compiled
            stream.synchronize()
        meta['status']='complete'
    except BaseException as exc:
        meta['status']='failed';meta['error']=repr(exc)
        raise
    finally:
        meta['finished_utc']=datetime.now(timezone.utc).isoformat();meta['gpu_final']=audit.gpu_state();save()
    print('COMPLETE',len(raw),'groups',len(summary),'candidate rows',args.output,flush=True)


def fit(args):
    if args.output.exists():raise FileExistsError('do not overwrite frozen policy')
    if len({p.resolve() for p in args.inputs})!=len(args.inputs):raise ValueError('duplicate calibration dir')
    runs=[];sources=[]
    for p in args.inputs:
        meta=json.loads((p/'metadata.json').read_text())
        if meta['status']!='complete' or meta['role']!='calibration':raise ValueError('complete calibration only')
        runs.append(json.loads((p/'summary.json').read_text()))
        sources.append(dict(path=str(p),sha256=sha(p/'summary.json'),pid=meta['pid'],
                            elapsed_wall_ns=meta['elapsed_wall_ns'],source_commit=meta['source_commit']))
    if len({s['pid'] for s in sources})!=len(sources):raise ValueError('distinct calibration processes required')
    dump(args.output,dict(schema=6,choices=fit_table(runs),sources=sources,
        rule='prefer inductor; switch only when >=1.05x in all calibrations; no test input'))
    print('FROZEN',args.output,sha(args.output))


def parse_shape(value):return validate_shape(tuple(int(x) for x in value.lower().split('x')))

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('run');r.add_argument('--output',type=Path,required=True)
    r.add_argument('--role',choices=('calibration','test'),required=True);r.add_argument('--policy',type=Path)
    r.add_argument('--shapes',nargs='+',type=parse_shape);r.add_argument('--smoke',action='store_true')
    r.add_argument('--rounds',type=int,default=6);r.add_argument('--samples',type=int,default=24)
    r.add_argument('--warmup',type=int,default=10);r.add_argument('--data-seed',type=int,default=6061)
    r.add_argument('--order-seed',type=int,default=6061)
    f=sub.add_parser('fit');f.add_argument('inputs',nargs='+',type=Path);f.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args) if args.command=='run' else fit(args)

if __name__=='__main__':main()
