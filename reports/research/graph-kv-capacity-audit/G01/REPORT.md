# G01：上游基线审查与实验前提检查

## 本轮结果与决策

完成源码资格审查、无GPU负载的环境采样及18项CPU契约测试。**没有运行图捕获、模型推理、吞吐测试或KV容量扫描，不能报告任何新加速比或性能退化。**

新信息：SGLang已包含后捕获KV定容与图池借用等可选功能。新实验必须把这些功能及其启用条件纳入强基线，不能把“捕获后再确定KV大小”或“共享图池”当作待发明的方法。这些功能本身不证明图规格/容量/吞吐取舍已经解决。

服务实测被环境前提拦住：有限采样期间GPU95–96%利用率并存在活动计算进程，默认Python也未安装目标引擎。没有停止别的任务，没有安装/升级系统依赖，没有修改驱动、功耗、CUDA或既有工程内核。

## 1. 隔离与来源

- 新分支：research/graph-kv-capacity-audit。
- 起点：107dd7a3dc7ec6bc586ad949994402033c292a87（已结题R06成果）。
- 原kernel-policy-validation分支和main均不写入；不创建合并PR。
- Tang独立工作树：/home/you/projects/cuda-operator-lab-graph-kv。
- G01正式采样代码：a32995b08d2432ee623c01b1885481e05d16654f。
- 审阅的SGLang：v0.5.21，release published_at=2026-10-02T01:09:04Z；tag对象7337b27853377544a322fd4390df332d8ca7d8cb，对应commit e00930c5489053f26d86b179cee0d087f846acbb。

上述为源码/元数据固定版本，不代表该引擎已在Tang安装或运行。

## 2. 深读源码后必须补入的强基线

### 已共享的图池和捕获流

`python/sglang/srt/model_executor/runner_utils/pool.py`包含进程级共享图池与capture stream；注释限定prefill和decode不会并发重放。不能把同一个共享池按图数重复计入，也不能无条件推广到并发图重放。
Blob：fb259065bbf3df6e15320f77eb52e59e26da9a13。

### 后捕获KV定容不是新想法

`model_runner_components/kv_pool_runtime.py`中的`compute_post_capture_kv_resize`在同步后读取可用显存，扣除headroom、特定额外reservation/workspace，加回已有pool backing，再通过配置器确定token容量，调用`finalize_backing`和allocator resize；必要时同步降低最大运行请求数。
Blob：321e686a67cffa26c481b129bcaeda648550e374。

这说明上游已有“图捕获后再据实际资源确定KV容量”的实现。它不等于自动优化捕获规格，更不等于已达到服务吞吐最优。

### 开关及适用性不能省略

`environ.py`声明`SGLANG_ENABLE_POST_CAPTURE_KV_SIZING=False`、`SGLANG_ENABLE_GRAPH_POOL_BORROW=False`，以及`SGLANG_ENABLE_CUDA_GRAPH_DEDUP=False`。这里是源代码默认声明，不代替实际部署的resolved配置。

`arg_groups/overrides.py`的`post_capture_kv_sizing_planned`还排除MLA、非CUDA、DCP!=1、unified memory、memory saver、某些传输/模型等。非decode-only模式要求prefill图覆盖最大prefill buffer工作量。关掉decode图也影响资格。
Blob：a2de0bc2937bcf30f6f477d7b81925719721d93a。

因此不能说“最新SGLang默认自动解决全部容量问题”，也不能在比较中关闭可适用的新功能来制造弱基线。TensorRT-LLM官方DeepSeek案例与单卡Qwen架构不可直接等同。

### 图池借用的边界

`runner_utils/pool.py`会识别空闲存储范围供短生命周期分配借用，并检查live user与图重放的冲突。它不是允许常驻KV无条件驻留于图重放会覆盖的空间。
`CUDA_GRAPH_DEDUP`本轮只确认开关存在，未完成其实现、覆盖范围或效果审计，不报告相关收益。

### 引擎依赖与既有环境不能混用

该固定commit的`python/pyproject.toml`要求`torch==2.13.0`、`transformers==5.12.1`，且含`flashinfer_python[cu13]`等CUDA13依赖。不能直接向原torch2.10环境安装并修改它，也不能把原环境当作此源码基线已验证。
Blob：50394749d90be93edd94533310d2a4aafa909fdd。
独立引擎环境、完整依赖解析、驱动/运行库兼容性和模型smoke尚未完成。

## 3. Tang实测环境证据（不是性能实验）

采样UTC：2026-10-06 18:26:54.781950至18:26:56.903270，对应东京时间2026-10-07 03:26。Python3.10.12，/usr/bin/python3。

| 样本 | GPU利用率 | 设备已用显存MiB | 温度°C |
|---|---:|---:|---:|
| 1 | 96% | 6449 | 69 |
| 2 | 96% | 6443 | 68 |
| 3 | 96% | 6445 | 68 |
| 4 | 95% | 6431 | 69 |
| 5 | 96% | 6430 | 69 |

设备为RTX4090 Laptop，总显存16376MiB，driver580.178.04；发现活动计算PID2815986，占5830MiB。没有检查其命令行内容、停止它或将其占用误归因于本实验。

默认Python元数据：torch2.10.0、triton3.6.0；sglang、vllm、transformers、flashinfer-python未安装于这个解释器。没有断言全机器所有隔离环境均未安装。

环境检查状态blocked：gpu_busy、active_compute_processes、sglang/torch/transformers目标版本不满足。返回码2是预注册的阻塞状态，不是运行基准后失败，更不是零吞吐结果。

无torch/sglang/vllm运行库导入，无模型加载，无GPU分配，无服务监听端口。

## 4. 新增的测量保护

仅新增CPU-only `preflight.py`与18项unittest。环境元数据符合也只报告runtime_validation_required，不承诺可立即服务。
`ledger_delta`是未来资源账本的纯函数契约：allocated包含在reserved中，不相加；只能相减同进程/同GPU数据；缺失进程指标不当作0；差额不截断为正值，也永远不直接标成图元数据。
未来真正的资源分配测量仍未实现；该函数通过测试不是图开销已量化。

测试过程：协议52de6a5先提交；测试cfd6c7b在缺实现时18项失败（含子测试共23错误）；实现a32995b后18项全部通过；采样后再次运行18项通过。
本轮没有全仓GPU回归、Compute Sanitizer或独立审阅者，不复用历史933项来冒充本轮结果。

## 5. 证据归档与核验

`preflight01.json`已完整上传，包括5个GPU原始命令结果、计算进程查询、版本信息、UTC时间和源commit。
发布版改变了JSON排版，但与Tang原文件逐字段完全一致；不是摘要替代原始观测。

- Tang原JSON SHA256：0bf5e7a043388024413f20759a2fd48f548eb69625f7fc08cd7a7d23b6f56117。
- GitHub发布JSON SHA256：660690a6569f4dcdd29a86033b2371b67ce46bb29ff74b41b1db3c4260259355。
- 单元测试原始stdout日志在Tang的G01目录保留。正文仅报告看到的18项CPU检查结果。

## 6. 下一步的边界

当前状态不是“方向成功”或“方向失败”，而是来源审查发现强基线更强，GPU实测尚未获准进入。
下一次先确认GPU空闲并准备不破坏旧工程的独立引擎环境，完整核验模型；然后运行单一模型的资源账本，确认可选基线功能是否真正启用，再做同请求轨迹的服务对照。没有后台排队、定时任务或自动重试服务启动。

不以故意占满显存、禁用上游可用优化或退回未经说明的旧版本来制造缺口。若资源账本显示没有重要容量损失，或者简单联合配置已足够，则不立项复杂系统。

## 上游来源

https://github.com/sgl-project/sglang/releases/tag/v0.5.21
https://github.com/sgl-project/sglang/tree/e00930c5489053f26d86b179cee0d087f846acbb
固定源码相对路径在第2节逐项列出。
TensorRT-LLM官方问题来源：https://nvidia.github.io/TensorRT-LLM/1.3.0rc14/blogs/tech_blog/blog20_Tuning_CUDA_Graph_Batch_Sizes_for_Higher_Output_Throughput.html
其GB200多卡数据仅是第三方来源，不是本轮Tang实测。
