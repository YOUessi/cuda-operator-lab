# P02阶段报告：最小需求裁剪与CPU编译验证；GPU性能阶段未完成

日期：2026-10-07。状态：**部分完成，不是GPU性能结论或完整训练加速结果。**

## 结论

P01发现的未编译源码冗余可由最小条件判断删除。本轮更重要的发现是：在显式修复编译兼容性后，AOT图也自动删除冻结权重的dW和logp-only时未使用的熵。因此，不能仅凭eager操作冗余就推导新优化方法的必要性。

原样verl源码的整图编译实际失败，失败原文已保存。之后单独增加只移除两个输出张量工厂requires_grad关键字的兼容性对照，不删除其计算；其eager操作计数与原样源码逐项一致。不得把这个对照称作未修改上游。

最终12组CPU AOT和真实CPU Inductor执行均通过；190个输出/梯度张量比较通过，最大绝对误差9.5367431640625e-07。43项CPU单元测试通过。GPU入口因利用率门槛未通过而在CUDA初始化之前退出，零正式时延样本。专用后端、显存峰值和完整训练步骤尚未验证。

## 1. 来源与隔离

- 分支：research/rl-output-head-audit；起点27a9824a3579f5be8d273b4abe2f05027e7c82b4。
- Tang工作树：/home/you/projects/cuda-operator-lab-output-head。
- 最终受测计算代码：33f125359b94d0342cc52af3eba20232e0c75221。
- 原样编译失败探针：3efee10ba7498480c72113681543d9d3bf730b7d。
- GPU阻塞记录对应代码：8034758a3c84314de8b27a83431fa6ef4d0c3576。
- 上游文件继续由P01 sources.json固定并验证Git blob；缓存/tmp/p02-upstream。
- 没有升级torch、安装完整verl/Liger/CCE、改变驱动或功耗，也没有修改生产CUDA文件或合并main。
- 协议P02_PROTOCOL.md先提交，所有实现先GitHub提交，再由隔离工作树拉取。

调用关系审查：verl固定commit 8718ca30a3f002f93b7c4fd99b9b2506718681bc的verl/models/transformers/dense_common.py中，forward_with_torch_backend将self.lm_head.weight传入FusedLinearForPPO；同文件还有forward_with_triton_backend，后者使用verl/utils/kernel/linear_cross_entropy.py。对应blob分别31d034f64b74afcb427b901d614adbccafa09dbb和84191d748f7e95368e7b29b684e73e11c865ee62。

这是源码调用路径核验，不是完整trainer实际调用轨迹。当前解释器未安装完整训练框架，本轮没有声称真实PPO/GRPO训练已运行，也没有把其他融合后端当作不存在。

## 2. 最小控制变量

新增demand_head.py，保持上游投影、默认chunk=512、温度计算位置和统计公式。仅按需要计算dX/dW和熵，熵作为日志时仍返回数值但标记为不参与梯度。

三种需求：logprob_only、entropy_logged、entropy_loss。任意逐token正负上游梯度均测试；冻结W时仍保留dX。checked_head额外提供掩码/ignore/元数据验证，未宣称已完成向完整模型接口传播新mode参数。原样verl未支持的输入不能直接硬塞入对照。

BF16约定保留投影/温度缩放后再升FP32统计，以及反向先转原dtype再除温度的舍入次序；这与某些FP32投影/缩放专用实现并非逐位相同。小尺寸BF16数值检查通过，不等于大GPU输入或训练收敛已验证。

自审发现初稿使用dz=dz+而非上游dz+=，可能引入额外临时分配。添加实际操作计数回归测试，先观察到2!=3失败，再恢复原地累加。最终43项测试及全部兼容性编译检查重新执行通过；报告采用cpu_compat02.json而不是修复前的cpu_compat01.json。未放宽数值容差。

## 3. 原样源码编译失败及单独兼容性对照

原始源码在FusedLinearForPPOFunction.forward中使用：

```python
torch.zeros(..., requires_grad=output_requires_grad)
hidden_states.new_zeros(..., requires_grad=output_requires_grad)
```

本机PyTorch 2.10.0的torch.compile(fullgraph=True)报：Attempted to use tensor creation function with requires_grad=True。原样路径6种模式组合的AOT和Inductor均失败，保留在cpu_compiler01.json；没有开启suppress_errors或静默回退。

compile_compat.py严格验证并仅移除两个", requires_grad=output_requires_grad"文本；custom Function本身依然提供autograd关系。此改动不删dW/entropy算术，6组eager操作跟踪与原始实现相同。该对照名称为compat_original。

比较compat_original与minimal各6组：W冻结/可训练乘三种熵需求，共12组。CPU输入T=7,H=5,V=11、FP32、tau=0.7，seed7207，并以seed7208测试编译后执行。不是大词表性能尺寸。

## 4. 实际编译结果

### 权重冻结时

| 实现 | eager反向矩阵乘次数 | AOT反向矩阵乘次数 | AOT中dW矩阵乘 |
|---|---:|---:|---:|
| 兼容性修复后的原计算 | 3 | 2 | 0 |
| 最小需求裁剪 | 2 | 2 | 0 |

三种熵需求均得到此结果。W可训练时两方的AOT反向均保留3次矩阵乘，包括需要的dW。

### 不需要熵时

logprob_only下，两方AOT前向都只保留所需log_softmax，没有用于熵的softmax与logsumexp。entropy_logged和entropy_loss仍保留熵前向，这符合需求。

完整12行数据在compiler_summary.csv；原始JSON保留全部前后向图代码、节点形状、操作计数、异常、保存张量以及误差。AOT节点数不是CUDA kernel数。

### 数值与执行层次

- 12组AOT前向/反向图捕获及执行通过。
- 同12组真实CPU Inductor执行通过，不把AOT aten执行冒充Inductor。
- 每组包含eager、AOT、Inductor、另一输入seed、保存张量hook观测的数值比较，总190个张量比较。
- 相对未修改FP32上游源码输出/梯度，最大绝对误差9.5367431640625e-07；rtol=2e-5、atol=2e-6预先固定。
- FP64数学参考及BF16舍入检查属于43项单元测试的一部分，不把上述190项全部误称为对FP64检查。
- 所有CPU执行CUDA_VISIBLE_DEVICES为空，torch.cuda.is_initialized()为False。

### 编译后的保存张量值得后续核验，但不是已经发现显存退化

在最终12组CPU Inductor saved_tensors_hooks记录中，都有一个[7,11]的非输入别名保存张量（逻辑大小308字节），另有部分[7,1]统计量。未编译custom Function主要保存X/W/y。

这提示编译阶段可能改变保存与重算安排。仅凭小CPU张量形状不能推导大GPU峰值显存、总存储，不能把逻辑字节重复相加，也不能把它宣称为已确认的新瓶颈。GPU对照应同时测完整前后向时间与峰值显存。

## 5. GPU阶段为何未运行

早期复查：设备利用率97/97/99%，显存6498MiB，活动计算PID183902约5952MiB。没有停止这个任务。

正式gpu_attempt01.json采样UTC 2026-10-06 19:50:51至19:50:52（东京2026-10-07 04:50），此时compute PID列表为空，但三次GPU利用率41/41/37%，显存527/527/528MiB，仍未满足预注册<10%门槛。

返回状态blocked_gpu_busy_or_unknown，退出码2；performance_started=False、records=[]、cuda_initialized=False。不要把正式阻塞归因于当时仍存在计算PID，也不要将阻塞记录当零吞吐或性能失败。

gpu_compare.py已提供五种控制（original_eager、compat_eager、minimal_eager、compat_compiled、minimal_compiled）、先正确性、预热、独立显存采样、均衡顺序和外来进程检测；本轮只验证了CPU helper与阻塞分支。GPU正向执行分支尚未运行，不能称该GPU基准已经验证成功。没有后台重试或排队任务。

## 6. 证据与复现

43项CPU单元测试：原19，加需求9、审计工具6、兼容性3、GPU门槛5、原地累加1；均看到相关测试失败后再实现/修复。最终绿色日志包含在归档中。

主要证据10文件通过evidence/part01.b64至part04.b64无损发布。拼接Base64->XZ解压->JSON，原始包770955字节，SHA256 7ae759053d47e50874b85344c26475b072c2feba25ec10bdbd9efa7262385237。

从GitHub重新fetch后四段均逐字节哈希匹配；9个JSON/测试日志与Tang原文件逐字节一致。compiler_summary.csv在read_text时仅CRLF转换LF，所有CSV字段完全一致，原与归档哈希分别存manifest。第一次严格字节比较正确发现此区别，随后明确记录，没有改数值。

三个verbose编译器输出代码日志仍仅在Tang，manifest有字节与SHA256；主要JSON包含全量AOT图和执行结果，但不是所有Inductor生成C++日志已发布。正常HTTPS/SSH push dry-run均因现有凭证不可用而失败，没有读取或修改凭证；主要证据用GitHub连接器发布。

复现CPU最终检查：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m unittest discover -s research/rl_output_head_audit/tests -v
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 python3 research/rl_output_head_audit/compile_compat.py --cache /tmp/p02-upstream --output /tmp/p02-new-compat.json
```

第二条需先按sources.json准备并校验公开源码缓存，不覆盖已有证据。

还原主要归档，从仓库根目录执行（新目录，不覆盖）：

```python
import base64, hashlib, json, lzma
from pathlib import Path
root = Path('reports/research/rl-output-head-audit/P02/evidence')
m = json.loads((root / 'manifest.json').read_text())
parts = []
for part in m['parts']:
    data = (root / part['name']).read_bytes()
    assert len(data) == part['bytes']
    assert hashlib.sha256(data).hexdigest() == part['sha256']
    parts.append(data)
raw = lzma.decompress(base64.b64decode(b''.join(b''.join(parts).split()), validate=True))
assert len(raw) == m['bundle_raw_bytes']
assert hashlib.sha256(raw).hexdigest() == m['bundle_raw_sha256']
bundle = json.loads(raw)
assert set(bundle['files']) == {f['name'] for f in m['files']}
out = Path('/tmp/p02-restored-evidence')
out.mkdir(exist_ok=False)
for f in m['files']:
    data = bundle['files'][f['name']].encode()
    assert len(data) == f['bytes'] and hashlib.sha256(data).hexdigest() == f['sha256']
    assert Path(f['name']).name == f['name']
    with (out / f['name']).open('xb') as stream:
        stream.write(data)
print(out)
```

## 7. 当前研究决定

“冻结权重但无条件计算dW”及“未使用熵仍计算”的未编译工程问题被证实，最小修复可消除；在可编译的控制上编译器已消除同类死计算。现在没有证据支持把需求裁剪包装成新方法。

P02仍缺：同GPU正式前后向时延、峰值显存、已确认适用的CCE/Liger/cuTile/verl融合强后端，以及完整后训练步骤。没有实际训练调用验证，不能据源码路径声称真实训练收益。GPU可用后先运行这些门槛；若强基线已解决或局部收益不能改善完整训练，就按工程修复结题。

本轮无新增CUDA kernel、全仓GPU回归、Sanitizer、独立审阅者、训练收敛证明或新颖性结论。自审不冒充独立审查。
