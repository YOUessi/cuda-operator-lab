#!/usr/bin/env python3
"""R06 supplement: max-autotune managed graphs versus explicit graph capture.
Not used to refit the frozen policy. Profile traces are diagnostics, not latency evidence.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import gzip
import json
import os
from pathlib import Path
import statistics
import sys
import time
import audit
import strong_backend_audit as core


def graph_launch_count(names):
    return sum(name in ('cudaGraphLaunch','cudaGraphLaunch_ptsz','cuGraphLaunch','cuGraphLaunch_ptsz') for name in names)


def compile_options(name):
    if name not in ('max-autotune','max-autotune-no-cudagraphs'):raise ValueError('explicit maximum-tuning mode required')
    return dict(fullgraph=True,dynamic=False,mode=name)


def run(args):
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'python'))
    import torch
    from cuda_operator_lab import bindings
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch._dynamo.config.suppress_errors=False
    args.output.mkdir(parents=True,exist_ok=False)
    bindings._library()
    meta=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
        source_commit=audit.command(['git','-C',str(root),'rev-parse','HEAD']),
        torch=torch.__version__,cuda=torch.version.cuda,contract=core.precision_contract(),
        gpu_before=audit.gpu_state(),cases=[],cache=os.environ.get('TORCHINDUCTOR_CACHE_DIR'),
        diagnostic_only=True,not_used_for_policy_fit=True)
    groups=[]
    def save():
        core.dump(args.output/'metadata.json',meta)
        core.dump(args.output/'summary.json',[{k:v for k,v in g.items() if k not in ('event_ns','wall_ns','orders')} for g in groups])
        (args.output/'raw.json.gz').write_bytes(gzip.compress(json.dumps(groups,separators=(',',':')).encode(),mtime=0))
    save()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.no_grad(),torch.cuda.stream(stream):
            for idx,shape in enumerate(core.SHAPES):
                case=core.make_case(torch,bindings,shape,6261)
                torch._dynamo.reset()
                t=time.perf_counter_ns()
                nc=torch.compile(case['torch_forward'],**compile_options('max-autotune-no-cudagraphs'))
                y=nc(*case['args']);stream.synchronize();core.check(torch,y,case['reference']);del y
                no_graph_setup_ns=time.perf_counter_ns()-t
                for _ in range(10):y=nc(*case['args']);del y
                stream.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):graph_output=nc(*case['args'])
                def explicit():graph.replay();return graph_output
                t=time.perf_counter_ns()
                mc=torch.compile(case['torch_forward'],**compile_options('max-autotune'))
                for _ in range(10):
                    torch.compiler.cudagraph_mark_step_begin()
                    y=mc(*case['args']);stream.synchronize();del y
                managed_setup_ns=time.perf_counter_ns()-t
                def managed():return mc(*case['args'])
                fns={'explicit_graph':explicit,'managed_max_autotune':managed}
                errors={}
                for label,fn in fns.items():
                    torch.compiler.cudagraph_mark_step_begin()
                    y=fn();stream.synchronize();errors[label]=core.check(torch,y,case['reference']);del y
                events={a:[] for a in fns};walls={a:[] for a in fns};all_orders=[]
                s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                s.record(stream);e.record(stream);e.synchronize()
                for ridx in range(6):
                    orders=audit.balanced_orders(list(fns),24,6271+idx*100+ridx);all_orders.append(orders)
                    er={a:[] for a in fns};wr={a:[] for a in fns}
                    for order in orders:
                        for label in order:
                            # Boundary marker outside CUDA event interval, same for both arms.
                            torch.compiler.cudagraph_mark_step_begin()
                            t=time.perf_counter_ns();s.record(stream);y=fns[label]()
                            e.record(stream);e.synchronize()
                            wr[label].append(time.perf_counter_ns()-t)
                            er[label].append(int(round(s.elapsed_time(e)*1e6)));del y
                    for a in fns:events[a].append(er[a]);walls[a].append(wr[a])
                diag={}
                for label,fn in fns.items():
                    path=args.output/(case['case']+'_'+label+'.trace.json')
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                        for _ in range(3):
                            torch.compiler.cudagraph_mark_step_begin()
                            y=fn();stream.synchronize();del y
                    names=[ev.name for ev in prof.events()]
                    prof.export_chrome_trace(str(path))
                    diag[label]=dict(graph_launch_apis=graph_launch_count(names),
                        launch_api_counts={k:v for k,v in Counter(names).items() if 'Launch' in k},
                        trace_path=str(path),trace_sha256=core.sha(path))
                record=dict(case=case['case'],shape=shape,event_ns=events,wall_ns=walls,orders=all_orders,
                    median_ns={a:statistics.median([statistics.median(r) for r in v]) for a,v in events.items()},
                    paired=audit.paired_summary(events['explicit_graph'],events['managed_max_autotune']),
                    errors=errors,diagnostic=diag,no_graph_first_call_ns=no_graph_setup_ns,
                    managed_compile_warmup_ns=managed_setup_ns,counters=core.counter_snapshot(torch))
                groups.append(record);meta['cases'].append(case['case']);save()
                print(case['case'],{k:round(v/1000,3) for k,v in record['median_ns'].items()},
                      {k:v['graph_launch_apis'] for k,v in diag.items()},flush=True)
                case['variants'].clear();del case,mc,nc,graph,graph_output
        meta['status']='complete'
    except BaseException as exc:
        meta['status']='failed';meta['error']=repr(exc);raise
    finally:
        meta['finished_utc']=datetime.now(timezone.utc).isoformat();meta['gpu_after']=audit.gpu_state();save()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
