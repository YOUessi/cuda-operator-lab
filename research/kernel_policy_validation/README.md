# 科研预实验：内核选择的测量审计

## 隔离与目标

唯一科研分支：`research/kernel-policy-validation`。冻结工程基线：`8507be588349bfb3206e4356d9a86995396a97c0`。
本阶段不改 CUDA 内核、不重写工程结果、不合并 main、不学习或发布新的调度策略。

已批准的总体问题：独立算子测试选出的实现，在图执行和完整前馈模块中是否仍然最好，错误选择是否造成值得研究的实际损失？
本批仅执行前两项审计；完整前馈模块、LayerNorm 对照和第二种 GPU 尚未执行。

## R01：同工作量与同内核对照

SwiGLU V0/V1 使用完全相同的输入、输出存储，只改变二维视图：256×512、128×1024、64×2048。
这些视图均含 131072 个元素，记录地址、步幅、存储偏移和16字节对齐信息。
不通过原有 V2 白名单；直接调用 V0/V1。增加两个相同 V0 包装函数 alias_a/alias_b，作为零效应对照。
每个模式计时前检查数值正确性；相同版本不同视图及别名必须逐元素相等。

## R02：GEMM＋SwiGLU 提交模式对照

候选固定为现有 V4 cuBLAS 与 V7 自定义 WMMA。权重只预打包一次，V7 使用打包存储的上下半区视图，不额外复制权重。
两者共享输入和输出地址；V4 工作空间预分配，不计分配与打包时间。
输入为 BF16，累加与输出为 FP32；权重先除以 sqrt(K) 再量化。以同一 BF16 数值提升到 FP64 计算为参考，事先固定 rtol=1e-3、atol=1e-3。
测试形状：16×64×64、32×128×256、64×128×1024、128×128×2048、32×256×64、128×512×512（M×K×N）。
这些是已知区域与边界附近的审计点，不代表完整模型形状分布，不能据此发表泛化或端到端结论。

## 三种计时语义

- `eager`：CUDA Event 包围一次普通绑定调用。该区间可能包含 GPU 等待主机提交的空隙，不能称为纯内核时间。
- `graph1`：一个图包含一次相同算子调用。捕获、首次初始化和预热不计时。记录图重放设备区间。
- `graph32`：一个图包含32次调用，原始设备区间完整保留，汇总时除以32。这是热状态连续执行诊断，不是单请求延迟，也不是完整模块延迟。

所有图在侧流预热后捕获，张量与图对象持续存活。捕获与预热成本本批不作为稳态指标，但不宣称部署总成本为零。
不人工清理缓存；自然热复用仅是一个受控条件，不宣称代表全部生产缓存状态。
不调整系统功耗或时钟。不安装分析器。记录前后 GPU 状态，但快照无法证明测量期间没有频率变化或其他活动。

## 顺序与统计

每轮使用随机基序列的全部循环移位及其反序，使每个候选在每个位置出现相同次数，同时每一对候选的先后方向均衡。
默认每个模式6轮，每候选每轮16次测量。复用已经初始化的事件对象。
保存每次的完整事件区间、同步式主机墙钟区间和真实执行顺序。事件换算成整数纳秒保存，仅是存储精度，不是硬件计时精度。
两次新进程使用相同数据种子、不同顺序种子，第二次反转模式顺序；不得把更换数据种子当作独立实验重复。

汇总为匹配轮次中位数比值的几何均值，并按轮次有放回抽样2000次给出描述性95%区间。
这个区间不覆盖跨进程、跨时段或跨设备变异，也没有多重比较校正。`a_faster/b_faster` 仅表示该进程内区间越过5%实用差异门槛，不自动生成策略，不构成确认性显著性结论。

## 运行

```bash
python3 -m pytest research/kernel_policy_validation/tests/test_audit.py -q
PYTHONPATH=$PWD/python python3 -m pytest tests -q
python3 research/kernel_policy_validation/audit.py \
  --output reports/research/kernel-policy-validation/20261006-audit/run01 \
  --data-seed 1307 --order-seed 1101
python3 research/kernel_policy_validation/audit.py \
  --output reports/research/kernel-policy-validation/20261006-audit/run02 \
  --data-seed 1307 --order-seed 1102 --modes graph32 graph1 eager
```

结果目录必须不存在，避免覆盖旧结果。每个目录保存 metadata.json、summary.json、summary.csv、raw.json.gz；失败时保留已完成分组并明确标记 failed。

## 决策条件

1. 同工作量结果不稳定：先诊断测量，不训练预测模型。
2. 图模式改变排名：先量化相同模式中的策略损失；不能直接宣称新论文问题成立。
3. 完整模块中无明显损失，或者分别缓存简单策略即可解决：收缩为工程功能。
4. 稳定、显著、简单方法无法低成本解决的模块级损失，才支持研究新方法。

## 实施记录

- 从 GitHub 读取分支与 main：两者起点均为 8507be5。
- Tang 原目录仍在 docs/finalize-project，科研使用独立 worktree：/home/you/projects/cuda-operator-lab-research。
- 首先提交20项 CPU 契约测试；无实现时均被缺失实现检查拦截（pytest fixture setup errors，原始日志保留，非 GPU 数值失败）。
- 基线在科研目录独立构建，通过原有816项测试。
- 本批是对既有基准流程的独立审计扩展；不会修改原内核以追求预期结果。

## 官方方法依据

- PyTorch CUDA Graph：侧流预热、静态地址生命周期与捕获限制。https://docs.pytorch.org/docs/stable/notes/cuda
- NVIDIA Nsight Compute：缓存清理、时钟控制、串行化与性能分析器影响。https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
