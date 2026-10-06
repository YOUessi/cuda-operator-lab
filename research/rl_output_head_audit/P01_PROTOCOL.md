# P01：后训练大词表输出层的源码与计算契约审查

## 目的与边界

用户批准首先比较现有 verl、Liger、CCE 的实际计算与反向路径。此轮是可行性审查，不开发新 CUDA 内核，不预设有论文创新，不把 CPU 操作计数当作 GPU 性能。

分支 research/rl-output-head-audit 从工程基线 8507be588349bfb3206e4356d9a86995396a97c0 创建。main、已结题 kernel-policy-validation、graph-kv-capacity-audit 均不修改。所有代码、协议和结果先在本分支记录；需要真机性能时再使用 Tang，不影响其他 GPU 任务。

## 固定上游版本

- verl-project/verl：8718ca30a3f002f93b7c4fd99b9b2506718681bc。
- linkedin/Liger-Kernel：29eb8ca2239661f6469f9cf935bb7a455d8b0171。
- apple-aiml-research/ml-cross-entropy：3de376c106a1916bc5e1b619f9c77c87a461ee1c。原 apple/ml-cross-entropy 已返回仓库迁移；使用 GitHub 仓库 ID 887718293 解析出的规范名称，不替换为第三方 fork。

上游 main 是审查时的快照，不声称为稳定发布版；运行支持范围须以实际 dispatch 和依赖为准。

## 任务定义

给定 X[T,H]、W[V,H]、标签 y[T]、温度 tau>0：

z = X W^T / tau；p = softmax(z)；l = log p_y；E = -sum_v p_v log p_v。

不做词表近似截断。主任务是任意逐 token 上游权重 a,b 的向量-雅可比积：J = sum_t mask_t (a_t l_t + b_t E_t)。最终 PPO clipping 或 GRPO 外层逻辑不属于本轮算子实现；任意 a,b 用于检查它们对输出层施加的上游梯度契约。

必须分开：
1. 只要 log-prob，无熵；
2. 熵仅日志，detach 后不产生熵梯度；
3. 熵参与损失；
4. 输出权重训练/冻结（冻结 W 仍可能需要 dX；不等于完整 LoRA 训练）。

数学参考使用 CPU float64；验证显式公式与 autograd，测试温度、掩码、分块尾部和有符号逐行梯度。FP64参考不冒充任何上游低精度契约。

## 源码审查

分别记录：前向是否物化整张或分块词表分数、保存的张量、反向重算、温度与 dtype 边界、是否计算未请求的 dX/dW、熵可选性、梯度过滤、设备 dispatch。

Liger 普通 fused linear CE 的 chunk 常数与 scaled CE/entropy 是不同入口，不能混作同一 baseline。CCE 的 NLL/LSE 不等于直接支持熵；只有同任务候选可放同一性能表。

## 本轮 CPU 探针

允许下载固定提交的公开源码到隔离缓存，核验 Git blob SHA；不修改上游文件，不安装整个框架。只加载白名单内、不带依赖装饰器的 verl/Liger fallback 定义，显式关闭可选 FlashAttention 路径。此为固定代码片段执行，不是完整 verl/Liger 软件包运行，更不是 GPU 编译结果。

用 PyTorch 调用跟踪记录前向、反向矩阵乘形状；分别运行 W 可训练/冻结、熵日志/参与损失。记录返回梯度、数值误差和被 autograd 丢弃前实际已发生的工作。操作计数不是耗时，不推出GPU节省比例。不会通过导入或伪装SM90来运行Hopper内核。

## 去留规则

发现冗余调用只标为工程优化候选，必须与最小条件分支修复及现有强实现对照；不视为论文必要性。若 joint NLL/entropy、分块反向已被强实现覆盖，明确缩小选题，不追加同类框架。后续性能实验至少包含更新的 Liger、适用的 CCE exact/no-filter、编译 PyTorch、现有 verl，检查实际 dispatch，并分别测前向+反向、峰值内存、梯度误差和真实训练步占比。当前未授权在用户繁忙 GPU 上叠加任务。

## 本轮交付

固定版本与 blob 证据、计算契约测试、CPU工作量探针及原始JSON、源码比较报告。无 GPU 时延、完整模型收敛或新方法加速声明。
