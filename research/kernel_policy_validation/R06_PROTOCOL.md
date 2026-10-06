# R06：同精度前馈模块的强后端与策略空间审查

Status: preregistered before implementation/timing. Research branch only; production baseline 8507be588349bfb3206e4356d9a86995396a97c0, R05 parent 1c3dd6850f13f64b6c4d3acad57555473ed7da85. Never change kernels, main, driver, power settings or old evidence.

## Question and decision
Does a nontrivial selection opportunity remain after comparing an entire synthetic pre-norm FFN against a fullgraph, maximum-autotuning PyTorch compiler? A winner switch is not itself a new method. No predictor/autotuner will be developed in this batch. Compare fixed library, compiled backend, legacy static rule and a calibration-only table against the observed valid-candidate minimum. The minimum is a noisy finite-candidate reference, not a theoretical oracle.

## Matched numerical contract (new workload; do not subtract R03/R04 times)
1. Input x[M,K], RMSNorm scale[K], packed gate/up weight[2N,K], down weight[K,N] are BF16.
2. RMSNorm mean-square/rsqrt/scale use FP32, eps=1e-6, normalized input explicitly rounds to BF16.
3. Packed projection is BF16 x BF16 with FP32 output/accumulation. Existing cuBLAS V4 or WMMA V7 computes FP32 SiLU(gate)*up. Pure PyTorch uses torch.mm(...,out_dtype=torch.float32) and FP32 activation.
4. Activation explicitly rounds to BF16 before the down projection; BF16 down matmul produces FP32 output; FP32 residual addition produces FP32 final output.
5. Compare all variants to a FP64 calculation with the same required BF16 boundary roundings. Predeclared elementwise rtol=0.02, atol=0.01 AND relative L2<=0.003; finite outputs mandatory. These are numerical-contract tolerances, not a model-quality guarantee. Failed candidates are excluded and failures retained, never silently replaced or tolerances widened.
6. One packed static weight layout for all arms, packing/conversion outside steady-state timings and reported separately. Eager paths return allocated outputs; graph outputs use capture-owned stable storage. No candidate-only input copies in the timed interval.

## Candidates and modes
- torch_packed: functional PyTorch packed matmul FFN.
- cublas_v4: identical PyTorch prefix/suffix, existing V4 core.
- custom_v7: identical prefix/suffix, existing V7 core; requires M%16=K%16=0,N%64=0; record unsupported shapes, do not silently drop workloads or masquerade fallback as V7.
- inductor: pure PyTorch function compiled with fullgraph=True, dynamic=False, mode=max-autotune-no-cudagraphs. No suppressed compiler errors, graph breaks or fallback accepted as a compiled success.
Primary deployment modes: eager submission and explicit SINGLE whole-module CUDA Graph replay (also for the no-graph compiled function, ensuring comparable graph scope). Separately probe max-autotune managed graph mode, recording whether graph launch is actually observed; not nest it in an external CUDA Graph.

## Workload matrix fixed before timing
Controls M,K,N: 16,64,64; 32,128,256; 128,512,512.
Model-dimension probes: 1,1536,8960; 16,1536,8960; 128,1536,8960.
K1536,N8960 from Qwen/Qwen2.5-1.5B official config; row counts are chosen probes, no checkpoint or full-model claim. Input random, weights scaled by inverse sqrt(fan-in); RMS scale ones. Ineligible V7 for M1 remains in coverage table as unsupported; simple strategies fall back to V4.
Source: https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/main/config.json

## Measurement, costs, and held-out policy
GPU tests only on Tang isolated checkout. Every source/test/protocol committed to GitHub before execution. CPU-only contract tests must fail without implementation, then pass. Compile and graph build before timed samples. Save first-call compilation/setup wall time, cache provenance, mode options, graph-break/compile counters and diagnostics. Compiler cache reused if needed: explicitly label warm-cache setup, never equate it with cold tuning cost.
Matched candidate orders using audit.balanced_orders; complete modes/shape ordering randomized. 6 rounds, 24 samples per candidate per round (divisible by 2*3 and 2*4); warmup>=10. Store every event interval, synchronized wall interval, order, correctness and device snapshots. Unlocked device, natural warm buffers. CUDA Event is not a pure device-time oracle. No cache-flush-as-deployment assumption. Within-round bootstrap intervals are descriptive, not independent-session guarantees.
Two calibration processes, then frozen best-per-mode table. Prefer compiled backend unless another candidate is >=5% faster in both calibration processes. Two test processes with a new data seed, reordered groups. No tuning choices changed after held-out results. Report policy regret, max loss, candidate coverage, wall-time effects and actual module microseconds. Uniform averages are exploratory, not production-weighted traffic.

## Gate
If fixed/compiled/simple strategies recover essentially all observed performance (small losses within noise), stop promoting the current complex selector as a method-paper direction. If substantial repeatable module losses survive strong baselines, identify the concrete mechanism and profile-cost economics before designing a method. Single GPU/synthetic weights cannot demonstrate cross-GPU or model-quality claims.

## Execution ledger
- Plan approved in conversation: reuse R05 infrastructure, add only research module/contract tests, execute inline.
- Isolated Tang checkout verified at /home/you/projects/cuda-operator-lab-research; original checkout stays on docs/finalize-project.
- Capability probe: Tang torch 2.10.0+cu128 supports aten.mm.dtype with BF16 inputs and FP32 output. No environment install/change needed.
- Pending: RED/GREEN contract tests, smoke/fullgraph validation, calibration/freeze/test, diagnostics, archive, conclusion.

References checked for installed-version semantics: https://docs.pytorch.org/docs/2.10/generated/torch.compile.html ; https://docs.pytorch.org/docs/2.10/generated/torch.mm.html ; https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html
