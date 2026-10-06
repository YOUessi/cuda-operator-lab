# P02 GPU后续状态（2026-10-07）

读取本文件及 reports/research/rl-output-head-audit/P02-GV/REPORT.md 后再继续，不重复P01/P02 CPU审计。

## 已完成

- gpu_validation.py固定受测提交5e504fec20097df3907b1c7f06d5c50ce554fe99。
- 两个输入形状129x256x32768和513x256x32768，BF16，51项CPU测试通过。
- GPU控制路径五种、W训练/冻结、三种熵需求、两个数据种子；两形状共60个控制组合。
- 固定CCE完整包3de376c的公开API在Ada上实际运行；logp-only、W训练/冻结，两形状共4组；关闭梯度过滤、FP32梯度累加，温度1.0。
- 总64组合、420项张量数值比较通过原预设容差。没有逐位一致或训练收敛保证。
- 三个证据分段已从GitHub重新拉取，包内11文件与Tang原文件逐字节一致。报告及manifest均已发布。

## 仍未完成

- gpu_attempt02和gpu_attempt03分别在25/25/28及18/18/31%利用率下被原正式门槛拦截；都没有初始化CUDA、没有时延样本。
- 不放宽<10%正式计时门槛，不停止别的GPU任务，不将非计时诊断当benchmark。
- 尚无峰值显存、正式速度、真实trainer调用或训练步骤结果。
- gpu_compare.py尚未加入CCE计时通道；加入前必须明确温度、中间舍入和误差契约。CCE不代替entropy任务。
- 尚未运行Liger Hopper/cuTile或verl其他专用融合后端。

## 下一步与边界

先确认Tang适合正式计时，再用已有最小修复/兼容性对照及已就绪的CCE进行同需求性能测量。不要继续新增纯CPU审计版本。编译模式、准备成本、前反向时延及独立内存测量须分开；无性能证据就不提出新算法。

未修改生产CUDA、demand_head.py、compile_compat.py或旧计时门槛。仅在research/rl-output-head-audit写入，不合并main。
