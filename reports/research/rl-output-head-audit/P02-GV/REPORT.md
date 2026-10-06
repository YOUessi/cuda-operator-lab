# P02-GV：CUDA正确性与CCE完整包检查

日期：2026-10-07。状态：GPU数值验证完成，P02正式性能阶段仍未完成。

## 1. 本轮新增证据

不再停留在CPU图分析：在Tang RTX 4090 Laptop上执行了BF16输出层的前向与反向，并执行CUDA Inductor编译后的两个控制路径。还从固定提交的完整CCE包公开接口运行了logp-only计算与梯度。

| 输入 T×H×V | 控制路径组合 | CCE组合 | 每个组合输入种子 |
|---|---:|---:|---:|
| 129×256×32768 | 30 | 2 | 2 |
| 513×256×32768 | 30 | 2 | 2 |

控制路径：original_eager、compat_eager、minimal_eager、compat_compiled、minimal_compiled；各自覆盖W训练/冻结和logprob_only/entropy_logged/entropy_loss。513-token形状跨越默认512-token分块并包含1-token尾块。

合计64个组合、128次带不同种子的候选执行。420个输出/梯度张量比较在既定BF16容差rtol=0.03、atol=0.02下通过。这个计数包含原路径自对照及CCE的两类参考，不是420个独立训练任务。没有事后放宽容差。

两次独立进程的最终状态均为gpu_validation_complete_no_timing。它们不是跨时段性能复现，也没有执行完整模型或PPO/GRPO更新。

## 2. 冻结来源与执行环境

- 科研分支：research/rl-output-head-audit；本轮起点148b84e7bfab3ebab17f5528d81a6f029001d342。
- 受测GPU诊断代码：5e504fec20097df3907b1c7f06d5c50ce554fe99。
- 原gpu_compare.py、demand_head.py、compile_compat.py未在本轮改变；最终diff已核验。
- 设备：RTX4090 Laptop，计算能力8.9；PyTorch2.10.0+cu128。
- CUDA Inductor：mode=default、fullgraph=True、dynamic=False，suppress_errors=False。没有把默认编译的数值通过当作max-autotune性能验证。
- 既有verl源片段继续使用sources.json固定的8718ca30a3f002f93b7c4fd99b9b2506718681bc。
- CCE：apple-aiml-research/ml-cross-entropy，3de376c106a1916bc5e1b619f9c77c87a461ee1c，独立checkout /tmp/output-head-cce-3de376c；校验HEAD及tracked clean，从该路径导入公开包入口。
- CCE所需torch/triton已存在，通过PYTHONPATH加载源码包，无pip安装、无依赖升级、无驱动或功耗变更。
- 没有运行Liger Hopper/cuTile后端，也没有安装或运行完整verl trainer。

## 3. 精度结果与限制

所有主控制继续使用原投影/缩放/升精度顺序，温度0.7，对原样未编译BF16路径比较。编译后不保证逐位相同。

两形状及种子上，compat_compiled与minimal_compiled相对原路径的最坏指标如下。每列分别取最大值，不暗示发生在同一个case或元素上。

| 输出 | 最大绝对误差 | 最大相对L2误差 |
|---|---:|---:|
| logp | 0.0134806633 | 0.0002294752 |
| dX | 0.0078125 | 0.0033734574 |
| entropy | 0.0625 | 0.0005863355 |
| dW | 0.125 | 0.0031916911 |

这是相对原始BF16实现，不是本轮全部对FP64真值的证明，也不是训练收敛保证。两种编译控制具有相同的最坏误差统计，不等于已直接证明它们逐元素相同。这里没有收集GPU AOT图或kernel trace，因此也不新增“GPU编译器确实删除某个GEMM”的结构性结论；此前CPU图分析和本轮GPU数值执行是两层证据。

第二次编译日志包含Constructing input/output tensor meta failed for Extern Choice警告，保留全部原文。对应候选仍完成fullgraph编译调用与数值比较；没有隐藏异常或开启suppress_errors。但不能据此声称所有调优候选均已正常覆盖。

## 4. CCE强后端边界

实际调用cut_cross_entropy.linear_cross_entropy，reduction=none、shift=0、filter_eps=None、filter_e_grad=False、filter_c_grad=False、accum_e_fp32=True、accum_c_fp32=True。

只测logp-only；没有省略目标任务需要的熵来制造等价对照。CCE对照温度为1.0，并分别与以下参考比较：

1. 同BF16输入提升FP32后执行投影、log_softmax及自动求导；
2. 原verl BF16舍入路径，温度也为1.0。

每种权重状态使用两个种子、两个形状，两类参考均通过预设容差。相对FP32投影参考，最大logp绝对误差0.0077486038、dX绝对误差0.00390625、dW绝对误差0.0625。原始文件逐项保存max_abs、mean_abs、relative_l2。

结论仅为：这份固定CCE源码包可以在当前Ada环境实际运行，具备加入logp-only强基线的前提。中间舍入与温度契约需要在正式基准中继续显式区分；不能从容差通过直接推导逐位相同、精确数学等价或速度优势。现有正式gpu_compare.py尚未加入CCE计时通道。

## 5. 为什么仍然没有正式时延和峰值显存结果

没有放宽P02原有<10%空闲门槛。

- gpu_attempt02：利用率25/25/28%，compute PID查询为空；status=blocked_gpu_busy_or_unknown，零样本，未初始化CUDA。
- 两轮GPU数值检查完成后的一次最终正式尝试gpu_attempt03：利用率18/18/31%，compute PID查询为空；同样在CUDA初始化之前退出，零时延样本。

非计时诊断的不同准入规则在执行前单独写入协议：无外来compute PID、至少4GiB空闲、温度低于80°C；允许图形负载，但绝不产出计时或峰值性能字段。每个候选前和结束时检查外来compute PID，记录原始查询。该规则不能复用为正式benchmark放行条件。

两个诊断的performance_measured均为False，脚本没有CUDA Event计时、无峰值显存采集、无加速比。后续一次只读pmon还观察到了图形进程与另一个Python计算进程，未停止任何其他任务；不能将本轮持续占用唯一归因于浏览器或图形桌面。

## 6. 实现与测试

新增gpu_validation.py以及8项CPU保护测试，总51项测试通过。顺序：先提交测试并观察8项因缺实现失败；实现初稿将nvidia-smi七字段误写为八字段，正向准入测试失败；修正为七字段后51项通过，随后才运行CUDA诊断。

保护覆盖缺失/非法metadata、外来compute进程、空闲内存与温度、非计时证据禁止性能字段、独立FP32参考。原正式门槛继续被测试为拒绝28%利用率。

最后重新执行51项测试，复核数据、代码差异与归档。无本轮Sanitizer、全仓GPU回归或独立审阅者；自审不冒充独立审查。

## 7. 完整原始归档

本目录evidence/包含3个Base64分段与manifest.json。拼接、Base64解码、XZ解压后的JSON包167102字节，SHA256：

3a2efad15f276926d375c4ace151df9b05d23e4050220a7845bf454ab55e9320

包内11个完整文件包括两次CUDA数值结果、两份运行日志、两次本轮正式计时阻塞记录及日志、测试失败与通过日志。重新从GitHub fetch后，三段哈希与包内11个原文件全部逐字节比对通过。没有只挑通过的局部输出，也没有将阻塞计入时延样本。

还原示例（仅在新目录写文件）：

```python
import base64, hashlib, json, lzma
from pathlib import Path
root = Path('reports/research/rl-output-head-audit/P02-GV/evidence')
m = json.loads((root / 'manifest.json').read_bytes())
parts = []
for p in m['parts']:
    b = (root / p['name']).read_bytes()
    assert len(b) == p['bytes'] and hashlib.sha256(b).hexdigest() == p['sha256']
    parts.append(b)
raw = lzma.decompress(base64.b64decode(b''.join(b''.join(parts).split()), validate=True))
assert len(raw) == m['bundle_raw_bytes']
assert hashlib.sha256(raw).hexdigest() == m['bundle_raw_sha256']
bundle = json.loads(raw)
assert set(bundle['files']) == {f['name'] for f in m['files']}
out = Path('/tmp/p02-gv-restored')
out.mkdir(exist_ok=False)
for f in m['files']:
    assert Path(f['name']).name == f['name']
    b = bundle['files'][f['name']].encode()
    assert len(b) == f['bytes'] and hashlib.sha256(b).hexdigest() == f['sha256']
    with (out / f['name']).open('xb') as stream:
        stream.write(b)
```

## 8. 决策

GPU正确性门槛和一个完整强后端的可运行性已获得正面证据；没有新增速度、省显存或论文新颖性证据。不会因此立即设计新内核。

P02仍待：真正空闲GPU上的正式前向+反向与峰值显存；CCE的同需求性能对照与其他可用强后端；真实训练调用与完整步骤。下一次不再重复扩展CPU诊断，应优先解决正式测量条件，随后直接比较最小工程修复和强基线。若编译器/强后端已覆盖收益，按工程修复收尾。
