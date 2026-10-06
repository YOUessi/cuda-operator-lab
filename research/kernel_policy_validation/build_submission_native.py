#!/usr/bin/env python3
"""Build only the R05 host shim; never change global toolkit paths or production libraries."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess


def needed_names(text):
    return re.findall(r'Shared library: \[([^\]]+)\]',text)


def validate_needed(names,major):
    runtimes=[x for x in names if x.startswith('libcudart.so.')]
    if runtimes!=[f'libcudart.so.{major}']:
        raise ValueError(f'expected only CUDA {major} runtime, found {runtimes}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nvcc',type=Path,required=True)
    p.add_argument('--cudart',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    import torch
    major=int(torch.version.cuda.split('.')[0])
    cudart=args.cudart.resolve()
    if cudart.name!=f'libcudart.so.{major}' or not cudart.is_file():
        raise ValueError('explicit cudart must match current PyTorch major')
    if args.output.exists():raise FileExistsError('use a new artifact path, preserve old evidence')
    nvcc=args.nvcc.resolve()
    version=subprocess.check_output([str(nvcc),'--version'],text=True)
    match=re.search(r'release (\d+)\.',version)
    if not match or int(match[1])!=major:raise ValueError('nvcc/PyTorch CUDA major mismatch')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).with_name('submission_native.cu').resolve()
    command=[str(nvcc),'-std=c++17','-O3','-shared','-Xcompiler=-fPIC',
        '--cudart=none',str(source),'-L',str(cudart.parent),'-l',':'+cudart.name,
        '-Xlinker','-rpath','-Xlinker',str(cudart.parent),'-o',str(args.output.resolve())]
    # Original nvcc path may define a project toolkit layout; preserve that path.
    command[0]=str(args.nvcc.absolute())
    result=subprocess.run(command,text=True,capture_output=True)
    record=dict(command=command,nvcc_version=version,torch=torch.__version__,
        torch_cuda=torch.version.cuda,cudart=str(cudart),returncode=result.returncode,
        stdout=result.stdout,stderr=result.stderr,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    if result.returncode==0:
        dynamic=subprocess.check_output(['readelf','-d',str(args.output)],text=True)
        record['needed']=needed_names(dynamic)
        validate_needed(record['needed'],major)
        record['artifact_sha256']=hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix('.build.json').write_text(json.dumps(record,indent=2))
    if result.returncode:raise RuntimeError(result.stderr)
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
