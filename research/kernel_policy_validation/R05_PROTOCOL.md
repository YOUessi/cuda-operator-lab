# R05：提交路径消融（预先固定）

## 问题与边界

接续 R04：检查自定义 V7 相对 cuBLAS V4 的收益是否依赖 Python 提交路径。不是新内核、完整模型或跨 GPU 方法论文。所有新增文件只在 research/kernel-policy-validation；不修改 csrc/、python/cuda_operator_lab/ 或 main。
工程基线 8507be588349bfb3206e4356d9a86995396a97c0；本批起点 fdcca7a36c8212840d5ba95e0123c664d4eed1d3。

## 四种条件

1. python_binding：现有参数校验和 ctypes 绑定，Python 记录 CUDA Event。
2. python_raw：复用同一库的函数指针与预先准备的参数，跳过每次张量校验；仍由 Python 记录 Event。
3. native_eager：C++ 宿主辅助库内部记录 Event、调用同一个导出函数、同步并读时间。计时区间内部没有 Python；仍有 CUDA/cuBLAS 提交开销。
4. native_graph：C++ 捕获单次同样工作，并在内部对单次图重放计时。捕获、初始化、预热不计入正式区间。不是把连续32次取均值伪称单请求时延。

同一形状使用完全相同输入、打包权重、输出地址与流；V7 的门控/上投影权重是同一打包权重的视图。只有提交/计时层改变。Python 原始接口与宿主辅助库仅用于可信实验，不作为放宽校验的用户 API。

## 固定矩阵

旧对照：16x64x64、32x128x256、64x128x1024、128x128x2048、128x512x512、128x1024x4096。
模型维度探针：16x1536x8960、128x1536x8960。
K=1536,N=8960 来自 Qwen/Qwen2.5-1.5B 官方 config.json；M=16/128 是本实验人为选择的 token 数。使用合成随机输入/缩放权重，不下载模型，不代表 Qwen 推理或模型质量结果。
来源：https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/main/config.json （2026-10-06读取，hidden_size=1536, intermediate_size=8960）。

## 验证与统计

- BF16 输入/权重，FP32 workspace/output。权重在量化前按 1/sqrt(K) 缩放。
- 同一 BF16 量化值提升 FP64 做 matmul + SwiGLU 参考；rtol=atol=1e-3，预先固定。
- 每种条件计时前将输出污染为 NaN、检查两候选正确性；同候选四条件输出逐元素一致；输入/权重不可变。
- 2 个新进程，相同数据种子1702；独立顺序种子2501/2502。8轮，每轮每候选8样本，AB/BA均衡。模式/形状顺序随机，第二进程改变顺序。保存原始纳秒数（不代表纳秒精度）、顺序、墙钟、库哈希、提交及设备快照。
- 使用既有配对轮次中位数比值与bootstrap区间，只有区间完全跨过1.05或1/1.05才判定>5%赢家。探索性、无多重比较校正；两个进程不是两张卡，也不提供跨时间保证。
- 自然热复用，不清缓存、不提权、不修改GPU时钟。分析器关闭；桌面GPU外部负载可能存在。

## 结论约束

native_eager 仍不是纯设备计算真值；不同模式的墙钟边界不同，不能直接相减当作准确Python开销。图重放改变执行方式，不能说同一部署免费加速。
若小形状收益在移除Python提交后消失，缩小旧“自定义更快”结论的适用范围；若库路径在模型维度与图模式足够好，暂停复杂选择方法，不为维持选题制造新的指标。
本轮只隔离 core；真实BF16完整模块、torch.compile强后端、跨日/跨GPU及低预算重校准不在本轮完成范围。

## 执行账本

先提交本协议和契约测试；观察测试因缺实现失败，再实现宿主辅助库和驱动；复核所有旧回归；先smoke，再两进程正式采样；发布完整核心表与原始证据校验清单。
Ruling：先完成宿主提交因果消融，不同时重写完整模块——避免精度/计算图/候选集和主机层一起改变。
参考：CUDA 12.8 Event API https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/group__CUDART__EVENT.html ；PyTorch CUDA Graph语义 https://docs.pytorch.org/docs/stable/notes/cuda 。
