# P01：后训练大词表输出层——源码审查与CPU工作量验证

## 本轮结论

完成了三个上游实现的固定版本审查、FP64数学契约测试，以及两个默认回退路径的CPU操作跟踪。没有运行GPU性能实验，没有实现新CUDA内核，没有进行完整训练或证明论文新颖性。

发现一个明确但很可能属于简单工程修复的问题：在测试的 verl 与 Liger 单设备回退源码片段中，即使输出权重W被冻结，反向仍执行 dW 矩阵乘法，然后外层不返回该权重梯度。只需要对数概率时，两个被测前向也仍执行熵相关的计算。该结论限于这里实际执行的未编译源码片段；不能推出GPU加速比例，不能推广到Liger所有后端，也不能忽略编译器可能消除死计算。

更重要的强基线发现：Liger Hopper实现已经支持不物化完整logits的逐token NLL和可求导熵，反向也已有融合与固定大小临时缓冲区；“对数概率+熵融合”“分块重算”不能直接列为新贡献。

## 1. 隔离与固定版本

分支：research/rl-output-head-audit。
工程起点：8507be588349bfb3206e4356d9a86995396a97c0。
Tang工作树：/home/you/projects/cuda-operator-lab-output-head。
CPU探针正式源提交：7ebac09c084070b90eb56ad683f6e7fba61636f1。

| 上游 | 固定commit |
|---|---|
| verl-project/verl | 8718ca30a3f002f93b7c4fd99b9b2506718681bc |
| linkedin/Liger-Kernel | 29eb8ca2239661f6469f9cf935bb7a455d8b0171 |
| apple-aiml-research/ml-cross-entropy | 3de376c106a1916bc5e1b619f9c77c87a461ee1c |

这些是审查时main快照，不是笼统声称某稳定发行版具有所有特性。CCE原apple/ml-cross-entropy返回迁移，通过GitHub repository_id=887718293确认规范仓库所有者，没有使用第三方fork替代。

六份主要源码的Git blob SHA在 sources.json 和原始证据中保存；下载时计算包含Git blob头的SHA1逐一核验。上游源码保存在Tang外部缓存，不在本仓库复制发布。

main及两个既有科研分支不写入，不创建合并PR。没有安装软件包，没有改变驱动、CUDA或功耗设置。本轮没有检查GPU是否空闲，因为所有测试明确隐藏CUDA设备并仅用CPU单线程。

## 2. 首先固定计算契约

X[T,H]为隐藏状态，W[V,H]为词表投影权重，y[T]为已选token，tau为有限正温度。

z = XW^T / tau；p = softmax(z)

l_t = log p[t,y_t]

E_t = -sum_v p[t,v] log p[t,v]

本轮检查J = sum_t mask_t (a_t*l_t + b_t*E_t)的向量-雅可比积。a、b允许逐token非均匀正负值，不用一个标量loss梯度代替任意上游梯度。它检验输出层契约，不是完整实现PPO/GRPO的外层算法。

显式公式：

dJ/dz = a*(onehot(y)-p) - b*p*(log(p)+E)

dX = ((dJ/dz)/tau) W

dW = ((dJ/dz)/tau)^T X

分开三种需求：只需要logp；熵只作日志并detach；熵参与损失。冻结W时仍可需要dX，不把冻结输出头等同于整个LoRA模型训练。

contract.py是小尺寸CPU FP64数学参考，支持掩码、ignore=-100、温度和可选梯度；它不是节省显存的autograd实现。分块参考仅用于检验数学结果及尾块，不拿它作为GPU竞争实现。

## 3. 上游实现对照

| 实现入口 | 已有功能与中间状态 | 本轮验证范围 |
|---|---|---|
| verl FusedLinearForPPOFunction | 默认512-token分块；已联合返回logp与熵；保存X/W/labels；反向重算投影 | 未编译CPU源码片段；FlashAttention可选分支明确关闭 |
| Liger scaled CE单设备fallback | 与verl公式相近；可不返回熵，但辅助函数仍计算熵；反向helper返回dX+dW | 未编译CPU源码片段，不是完整Liger包 |
| Liger SM90 CuTe DSL scaled CE | BF16 Hopper；逐token NLL+可求导熵；前向不将完整logits写HBM；保存LSE；反向dZ+dX+dW融合，1024-token波次复用BF16 dZ工作区 | 只审阅源码；没有在Ada/CPU上冒充执行Hopper内核 |
| CCE | NLL reduction=none、可返回LSE、梯度过滤/累加选项及词表并行；保存LSE等并重算必要中间量 | 只审阅源码；直接API没有return_entropy，不把LSE等同于熵 |
| Liger普通fused linear CE | 与scaled CE/entropy不同的入口；当前固定版本chunk_mem_const默认1且可配置，已有梯度需求相关选项 | 只审阅源码；历史C=16补丁不能当作当前默认状态 |

重要限制：默认单设备scaled CE入口在SM90选专用实现，在Ada等环境选回退。但后续搜索还发现独立的cuTile scaled CE实现，不能因此说“Liger只有Hopper优化”。该cuTile文件前90行包含256MiB工作区和部分Blackwell形状512MiB策略；此附加文件仅做初步阅读，未完成依赖、设备适用性、dispatch接入和性能核验。它不属于已执行的六文件清单。

TP fallback是另一条路径，源码本身包含按梯度需求控制矩阵乘的分支。单设备fallback冗余不推广为所有TP路径的缺陷。

CCE适合加入logp-only的强基线，但不能在需要可求导熵时省略熵后声称等价。若扩展/组合CCE，新增计算和中间状态必须计入。无过滤模式、精度与累加设置应显式记录，不能把近似配置与精确任务混为一谈。

## 4. 实际执行结果：CPU操作而非GPU性能

环境：Tang上的PyTorch 2.10.0+cu128；CUDA_VISIBLE_DEVICES为空，CPU线程数1，结束时torch.cuda.is_initialized()为False。没有模型权重加载、GPU分配或GPU时延测量。

只从固定且hash校验过的文件选出白名单AST定义，保持其函数体不变；加入future annotations以避免未导入的注解类型，显式提供torch/math等依赖。没有执行上游模块顶层初始化，也没有声称完整框架运行。此提取不是安全沙箱。

探针尺寸T=7,H=5,V=11，FP32，tau=0.7，固定CPU种子7107。尺寸互异用于辨认投影/dX/dW操作，不是大词表性能代表。每个provider执行W训练/冻结乘三种熵需求，共12组。

| 权重状态与需求 | 前向mm | 反向mm | 对外需要/返回的梯度 |
|---|---|---|---|
| W可训练，logp-only | 投影 | 投影重算、dX、dW | dX、dW |
| W可训练，熵只日志 | 投影 | 投影重算、dX、dW | dX、dW |
| W可训练，熵参与损失 | 投影 | 投影重算、dX、dW | dX、dW |
| W冻结，logp-only | 投影 | 投影重算、dX、dW | 仅dX |
| W冻结，熵只日志 | 投影 | 投影重算、dX、dW | 仅dX |
| W冻结，熵参与损失 | 投影 | 投影重算、dX、dW | 仅dX |

上述表格分别在verl_fragment与liger_fallback_fragment上出现，原始文件保留全部12组，而非合并后丢弃记录。

两个provider的每种需求，前向均观察到softmax、logsumexp和log_softmax各一次，包括不需要熵输出的case。反向logp-only与熵日志case只观察到softmax；熵参与loss时又观察到log_softmax与logsumexp。这是调用计数，不是对应独立GPU kernel数，也不意味着这些算子具有相同成本。

数值验证用同一组实际FP32输入值提升到FP64后，分别计算数学值与显式VJP。38个值/梯度张量比较全部通过，最大绝对误差5.739047912456385e-07；预设rtol=2e-5、atol=2e-6，没有根据结果放宽。

另有19项CPU unittest全部通过：12项数学契约、7项源码与工作量归属保护。数学测试还覆盖掩码、非整分块、不同温度与entropy-only VJP；实际上游片段的12组探针没有覆盖全部这些组合，两者不要混淆。

## 5. 对研究的意义

已经排除的简单创新叙述：

- 新提出logp+entropy融合：现有verl接口和Liger Hopper路径已经具备。
- 新提出避免完整词表logits：CCE及Liger Hopper等已有实现。
- 修改分块常数就视为新方法：当前标准CE和cuTile已提供相关配置，且不同入口不能混用。

保留一个小且可证伪的工程假设：真实后训练任务若冻结输出头、关闭熵或只将熵用于日志，某些执行路径是否因未按需求裁剪而支付不必要成本？目前已验证eager CPU片段存在冗余；其GPU重要性、编译后是否仍存在、实际训练步占比都未知。

下一道关应先比较最小条件分支修复、编译后的同语义PyTorch，以及相同设备上真正可用的强实现。需要同时报告前向+反向、峰值显存、梯度误差与完整训练步骤；全参训练本来就需要dW，不能用冻结权重结果代替全参结果。

如果几行梯度需求判断就解决，按工程修复结束；如果只能击败Ada回退而无法比较已存在的专用强内核，也不宣称通用新算法。只有剩余重要成本在强基线与真实任务中成立，才研究新的计算组织。

本轮不宣称方向已经值得发文，也不因部分问题已有实现就否定所有输出层研究。结论是把问题从宽泛的融合，缩小到明确的需求与执行路径。

## 6. 复现、日志与发布完整性

测试先失败后通过：12项契约测试在缺contract.py时失败，随后通过；7项探针测试在缺source_probe.py时失败，随后通过；正式探针前再次19项全部通过。

运行示例，从仓库根目录：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m unittest discover -s research/rl_output_head_audit/tests -v
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 research/rl_output_head_audit/source_probe.py --download --cache /tmp/p01-upstream --output /tmp/p01-new-evidence.json
```

第二条会下载公开固定源码；已有输出拒绝覆盖。源码及Git blob identities由sources.json控制。

完整原始JSON和五份日志均通过evidence/part01.b64至part03.b64发布，拼接解压后72255字节，SHA256为77b419d62e53f136ee8e84b4c96f493fe5d8f5e7f15c7101f5720607481a4aa5。重新从GitHub获取并与Tang六个原文件逐字节比较通过。还原说明在evidence/README.md，不再出现“只上传汇总，原始数据缺失”的状态。

本轮有两项执行错误，均保留历史并明确修正：sources.json最初误录一个Liger blob，在执行前通过固定路径元数据纠正；第一次整段Base64发布损坏，解码与哈希校验发现后改分段上传并移除错误单文件。上游源码和原始CPU结果未被修改来迎合检查。

原日志还包含最大误差转Python scalar的requires_grad提示，不影响已核验结果；本轮未测时延，不从警告推断性能。

无本轮GPU回归、Compute Sanitizer、完整软件包集成或独立审阅者。此轮为源码审查、CPU测试和可还原证据的会话内自检，不复用旧项目测试数量冒充本轮验证。

## 7. 主要源码定位

- verl固定文件：https://github.com/verl-project/verl/blob/8718ca30a3f002f93b7c4fd99b9b2506718681bc/verl/utils/experimental/torch_functional.py；forward 36–58，backward helper 61–93，autograd 96–217。
- Liger单设备入口：https://github.com/linkedin/Liger-Kernel/blob/29eb8ca2239661f6469f9cf935bb7a455d8b0171/src/liger_kernel/ops/fused_linear_scaled_cross_entropy.py；fallback 219–317，dispatch 320–368。
- Liger SM90：https://github.com/linkedin/Liger-Kernel/blob/29eb8ca2239661f6469f9cf935bb7a455d8b0171/src/liger_kernel/ops/cutedsl/ops/fused_scaled_cross_entropy_sm90.py；实现说明1–64，backward与autograd 211–322。
- CCE API：https://github.com/apple-aiml-research/ml-cross-entropy/blob/3de376c106a1916bc5e1b619f9c77c87a461ee1c/cut_cross_entropy/linear_cross_entropy.py；实现117–223。
- CCE autograd：https://github.com/apple-aiml-research/ml-cross-entropy/blob/3de376c106a1916bc5e1b619f9c77c87a461ee1c/cut_cross_entropy/cce.py；47–214。
- 标准CE：https://github.com/linkedin/Liger-Kernel/blob/29eb8ca2239661f6469f9cf935bb7a455d8b0171/src/liger_kernel/ops/fused_linear_cross_entropy.py；参数51–100，不能混作scaled CE。
- 补充cuTile：https://github.com/linkedin/Liger-Kernel/blob/29eb8ca2239661f6469f9cf935bb7a455d8b0171/src/liger_kernel/ops/cutile/ops/fused_linear_scaled_cross_entropy.py；仅初读1–90，blob86f62abf324e640f3f6b32350b4bff6ad9b56a5b，未运行。
