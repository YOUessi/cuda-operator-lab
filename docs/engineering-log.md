# Engineering Log

## Project bootstrap

- Target machine: RTX 4090 Laptop GPU, compute capability 8.9.
- Driver: 580.178.04.
- System PyTorch: 2.10.0+cu128.
- System `/usr/bin/nvcc`: CUDA 11.5, too old to be the primary compiler for Ada SM 8.9.
- Selected local CUDA compiler: CUDA 12.8 nvcc from the existing Anaconda package cache.
- Development strategy: local-first execution on Tang, Git-backed from the first commit, and push small verified milestones to GitHub rather than waiting until the project is finished.
