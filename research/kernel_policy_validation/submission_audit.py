#!/usr/bin/env python3
"""R05 submission-layer audit. See R05_PROTOCOL.md; no production kernel changes."""
from __future__ import annotations
import argparse
import ctypes as C
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from datetime import datetime, timezone
import audit

MODES=('python_binding','python_raw','native_eager','native_graph')
ARMS=('v4_cublas','v7_custom')
SHAPES=((16,64,64),(32,128,256),(64,128,1024),(128,128,2048),
        (128,512,512),(128,1024,4096),(16,1536,8960),(128,1536,8960))


def validate_shape(shape):
    if len(shape)!=3 or any(type(x) is not int or x<=0 for x in shape):
        raise ValueError('positive integer M,K,N required')
    m,k,n=shape
    if m%16 or k%16 or n%64 or max(m,k)>2**31-1 or n>(2**31-1)//2:
        raise ValueError('V7 alignment and int32 dimensions required')
    return tuple(shape)


def timing_record(event_ms,wall_ns):
    if any(not math.isfinite(x) or x<=0 for x in (event_ms,wall_ns)):
        raise ValueError('finite positive timings required')
    return round(event_ms*1_000_000),round(wall_ns)


def validate_orders(orders,count):
    if len(orders)!=count or count%2 or any(sorted(x)!=[0,1] for x in orders):
        raise ValueError('invalid paired schedule')
    if sum(x[0]==0 for x in orders)!=count//2:
        raise ValueError('unbalanced first position')
    return True


def new_output(path):
    path.mkdir(parents=True,exist_ok=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(code):
    if code:
        raise RuntimeError(f'R05 CUDA/native error {code}; stop, do not accept partial timing')


class Native:
    def __init__(self,path,library,tensors,shape,stream,warmup):
        self.api=C.CDLL(str(Path(path).resolve()))
        vp=C.c_void_p; u=C.c_uint64; pd=C.POINTER(C.c_double)
        self.api.r05_abi.restype=C.c_int
        if self.api.r05_abi()!=1:
            raise ValueError('native ABI mismatch')
        self.api.r05_create.argtypes=[vp]*6+[u]*3+[vp,C.POINTER(vp)]
        self.api.r05_create.restype=C.c_int
        self.api.r05_launch.argtypes=[vp,C.c_int]
        self.api.r05_capture.argtypes=[vp,C.c_int]
        self.api.r05_measure.argtypes=[vp,C.c_int,C.c_int,pd,pd]
        self.api.r05_destroy.argtypes=[vp]
        for name in ('r05_launch','r05_capture','r05_measure','r05_destroy'):
            getattr(self.api,name).restype=C.c_int
        # Invalid arguments must fail before touching device memory.
        ms=C.c_double(); wall=C.c_double()
        if self.api.r05_measure(None,0,0,C.byref(ms),C.byref(wall))==0:
            raise AssertionError('native null handle incorrectly accepted')
        self.handle=vp()
        x,packed,gate,up,workspace,out=tensors
        f4=library.handle.cuda_operator_gemm_swiglu_v4
        f7=library.handle.cuda_operator_gemm_swiglu_v7
        check(self.api.r05_create(C.cast(f4,vp),C.cast(f7,vp),vp(x.data_ptr()),
            vp(packed.data_ptr()),vp(workspace.data_ptr()),vp(out.data_ptr()),
            *map(u,shape),vp(stream.cuda_stream),C.byref(self.handle)))
        try:
            for arm in (0,1):
                for _ in range(warmup): check(self.api.r05_launch(self.handle,arm))
            stream.synchronize()
            for arm in (0,1): check(self.api.r05_capture(self.handle,arm))
            # No graph or event initialization may be included in accepted samples.
            for arm in (0,1):
                for graph in (False,True):
                    for _ in range(warmup): self.measure(arm,graph)
        except BaseException:
            self.close()
            raise

    def measure(self,arm,graph=False):
        ms=C.c_double(); wall=C.c_double()
        check(self.api.r05_measure(self.handle,arm,int(graph),C.byref(ms),C.byref(wall)))
        return timing_record(ms.value,wall.value)

    def close(self):
        if self.handle.value:
            handle=self.handle; self.handle=C.c_void_p()
            check(self.api.r05_destroy(handle))


def run(args):
    if args.rounds<2 or args.samples<=0 or args.samples%4 or args.warmup<3:
        raise ValueError('rounds>=2, samples positive multiple of4, warmup>=3 required')
    shapes=list(SHAPES if not args.smoke else SHAPES[:2])
    for shape in shapes: validate_shape(shape)
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'python'))
    import torch
    from cuda_operator_lab import bindings
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    new_output(args.output)
    library=bindings._library()
    lib_path=Path(bindings._default_library_path())
    meta=dict(schema=1,status='running',started_utc=datetime.now(timezone.utc).isoformat(),
        pid=os.getpid(),source_commit=audit.command(['git','-C',str(root),'rev-parse','HEAD']),
        working_tree=audit.command(['git','-C',str(root),'status','--porcelain']),
        base_commit=audit.BASE,library_path=str(lib_path),library_sha256=sha(lib_path),
        helper_path=str(args.native),helper_sha256=sha(args.native),
        torch=torch.__version__,cuda=torch.version.cuda,python=sys.version,
        config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        gpu_initial=audit.gpu_state(),gpu_snapshots=[],
        precision='BF16 inputs and weights, FP32 workspace/output; FP64 reference from same BF16 values',
        cache='natural warm reuse; no flush',clock_policy='unlocked; unchanged',
        model_dimensions='Qwen2.5-1.5B K1536 N8960; chosen M16/128; synthetic data, not model inference',
        limits=['native_eager retains CUDA/cuBLAS host launch work',
                'event interval is not pure sum of kernel durations',
                'wall scope differs between Python and native: do not subtract as exact host cost',
                'single core, not full module or torch.compile comparison',
                'single GPU, descriptive round bootstrap, no multiplicity adjustment'])
    records=[]; groups=[]; start_wall=time.perf_counter_ns()

    def save():
        meta['elapsed_wall_ns']=time.perf_counter_ns()-start_wall
        (args.output/'metadata.json').write_text(json.dumps(meta,indent=2,allow_nan=False))
        payload=dict(schema=1,metadata=meta,groups=groups)
        (args.output/'raw.json.gz').write_bytes(gzip.compress(
            json.dumps(payload,separators=(',',':'),allow_nan=False).encode(),mtime=0))
        if records:
            with (args.output/'summary.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n')
                writer.writeheader();writer.writerows(records)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    random.Random(args.order_seed).shuffle(shapes)
    save()
    try:
        with torch.inference_mode(),torch.cuda.stream(stream):
            for idx,shape in enumerate(shapes):
                case=audit.make_gemm(torch,bindings,shape,args.data_seed)
                x,packed,gate,up,workspace,out,expected=case['keep']
                originals=(x.clone(),packed.clone())
                stream.synchronize()
                native=Native(args.native,library,(x,packed,gate,up,workspace,out),shape,stream,args.warmup)
                try:
                    vp=C.c_void_p; u=C.c_uint64
                    tail=tuple(map(u,shape))+(vp(stream.cuda_stream),)
                    raw_args=[(vp(x.data_ptr()),vp(packed.data_ptr()),vp(workspace.data_ptr()),vp(out.data_ptr()))+tail,
                              (vp(x.data_ptr()),vp(gate.data_ptr()),vp(up.data_ptr()),vp(out.data_ptr()))+tail]
                    fns=[library.handle.cuda_operator_gemm_swiglu_v4,library.handle.cuda_operator_gemm_swiglu_v7]
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record(stream);end.record(stream);end.synchronize()

                    def once(mode,arm):
                        if mode.startswith('native_'):
                            return native.measure(arm,mode=='native_graph')
                        wall=time.perf_counter_ns()
                        start.record(stream)
                        if mode=='python_binding': case['variants'][ARMS[arm]]()
                        else: check(fns[arm](*raw_args[arm]))
                        end.record(stream);end.synchronize()
                        elapsed_wall=time.perf_counter_ns()-wall
                        return timing_record(start.elapsed_time(end),elapsed_wall)

                    modes=list(MODES);random.Random(args.order_seed+idx).shuffle(modes)
                    canonical={}
                    for midx,mode in enumerate(modes):
                        for arm in (0,1):
                            for _ in range(args.warmup): once(mode,arm)
                        errors={}
                        for arm in (0,1):
                            out.fill_(float('nan'));once(mode,arm);stream.synchronize()
                            torch.testing.assert_close(out,expected,rtol=1e-3,atol=1e-3)
                            if arm in canonical:
                                torch.testing.assert_close(out,canonical[arm],rtol=0,atol=0)
                            else: canonical[arm]=out.clone()
                            delta=(out-expected).abs()
                            errors[ARMS[arm]]=dict(max_abs=float(delta.max()),mean_abs=float(delta.mean()))
                        stream.synchronize()
                        before=audit.gpu_state()
                        event=[[],[]];walls=[[],[]];orders_saved=[]
                        for ridx in range(args.rounds):
                            orders=audit.balanced_orders([0,1],args.samples,
                                args.order_seed*10000+idx*100+midx*10+ridx)
                            validate_orders(orders,args.samples);orders_saved.append(orders)
                            e=[[],[]];w=[[],[]]
                            for order in orders:
                                for arm in order:
                                    ns,wall=once(mode,arm);e[arm].append(ns);w[arm].append(wall)
                            for arm in (0,1):event[arm].append(e[arm]);walls[arm].append(w[arm])
                        result=dict(case=case['case'],m=shape[0],k=shape[1],n=shape[2],mode=mode,
                            a=ARMS[0],b=ARMS[1],**audit.paired_summary(*event))
                        records.append(result)
                        groups.append(dict(case=case['case'],shape=shape,mode=mode,
                            event_ns=event,wall_ns=walls,orders=orders_saved,errors=errors,
                            pointers=case['pointers'],summary=result))
                        meta['gpu_snapshots'].append(dict(case=case['case'],mode=mode,
                            before=before,after=audit.gpu_state()))
                        print(f"{case['case']} {mode} V4={result['a_median_ns']/1000:.3f} V7={result['b_median_ns']/1000:.3f} ratio={result['speedup_a_over_b']:.3f} {result['decision']}",flush=True)
                        save()
                    torch.testing.assert_close(x,originals[0],rtol=0,atol=0)
                    torch.testing.assert_close(packed,originals[1],rtol=0,atol=0)
                finally:
                    native.close()
                del case,native,canonical,originals,x,packed,gate,up,workspace,out,expected
            stream.synchronize()
        meta['status']='complete'
    except BaseException as exc:
        meta['status']='failed';meta['error']=repr(exc);raise
    finally:
        meta['finished_utc']=datetime.now(timezone.utc).isoformat()
        meta['gpu_final']=audit.gpu_state();save()
    print(f'COMPLETE {len(groups)} groups at {args.output}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=8)
    p.add_argument('--samples',type=int,default=8)
    p.add_argument('--warmup',type=int,default=30)
    p.add_argument('--data-seed',type=int,default=1702)
    p.add_argument('--order-seed',type=int,default=2501)
    p.add_argument('--smoke',action='store_true')
    run(p.parse_args())

if __name__=='__main__':main()
