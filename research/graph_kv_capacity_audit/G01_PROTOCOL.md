# G01：图资源、KV容量与请求接纳的可行性核验

状态：已批准的有限核验；不是新调度算法，不恢复R01–R06复杂selector选题。
工作分支research/graph-kv-capacity-audit，从107dd7a3dc7ec6bc586ad949994402033c292a87创建；main和原科研分支均不得修改。

## 第一关：环境与强基线资格

1. 读取GPU占用和活动计算进程；不停止其他任务、不改功耗/时钟/驱动。
2. 无其他计算任务、至少5个有限连续采样GPU利用率<=5%才允许考虑smoke；这个门槛是实验管理规则，不是设备隔离证明，正式测量仍需检查前中后状态。
3. 不向现有Python环境安装或升级引擎。SGLang v0.5.21源代码快照固定为e00930c5489053f26d86b179cee0d087f846acbb，tag对象7337b27853377544a322fd4390df332d8ca7d8cb。
4. 该快照pyproject要求torch==2.13.0、transformers==5.12.1并含CUDA13依赖。Tang原有torch2.10不能视为该基线已就绪；独立环境的依赖、驱动和smoke须另行验证。
5. 模型目录或config.json存在不代表完整权重可用。正式服务实验必须先核验所有分片、tokenizer和模型commit，默认不加载不可信远程代码。

## 已发现的上游实现，不可重新包装为创新

- runner_utils/pool.py：prefill/decode在互不并发的前提下共享图池与capture stream。
- 环境变量SGLANG_ENABLE_POST_CAPTURE_KV_SIZING默认False；启用后仍须通过post_capture_kv_sizing_planned条件。
- 后捕获KV定容排除MLA、非CUDA、DCP!=1、unified memory、memory saver等；非decode-only部署要求prefill图覆盖最大预填充缓冲工作量。
- kv_pool_runtime.py：捕获后按真实free memory、headroom、已有backing及额外workspace求budget，finalize_backing，resize allocator，必要时降低max_running_requests。
- SGLANG_ENABLE_GRAPH_POOL_BORROW默认False。只可借用符合生命周期限制的空闲图存储，不是把常驻KV任意放入图池；重放会覆盖别名，源码有live-user检查。
- SGLANG_ENABLE_CUDA_GRAPH_DEDUP默认False；这里只确认该开关存在，不声称本轮已审完实现或兼容性。

## 第二关：资源账本（未运行，不可提前报告数字）

同一引擎/模型commit、精度、attention backend、请求上限、prefix缓存设置和GPU总预算。逐阶段记录：进程初始化、权重加载、KV初始池、图捕获前后、可选KV最终定容、warmup和稳态。
保留device total/free、NVML进程占用、PyTorch allocated/reserved、graph pool标识和唯一segment集合、引擎最终KV token/block数、最大请求数和启动耗时。
allocated已包含在reserved内，不得相加。图池segment按唯一地址范围去重，不得把共享池按图数重复相加。
NVML process used减torch reserved仅为未分类差额，不能直接称为图元数据；VMM、上下文、库workspace和指标口径都可能参与。设备总使用量受其他进程影响，不用于单进程归因。
KV字节/token仅对已确认架构和布局计算，最终容量以引擎token/block报告为准。

## 第三关：服务性能空间（未运行）

先比较同最大batch的稀疏/合理中间/密集图规格，另设图关闭控制；再扫描合法KV预算。不跨配置偷偷改变模型、精度、并发上限或请求。
至少包含：当前默认、官方建议调优、基于校准轨迹的简单离线配置、有限联合扫描。适用时必须纳入后捕获KV定容及图池借用；不适用时保存明确理由，不能强开或忽略。
固定到达时间轨迹并记录实际发送偏差。纳入等待、拒绝、错误、超时请求；主要指标为预注册TTFT/生成延迟约束下的有效吞吐，另报原始TPS、排队和启动成本。
校准轨迹、测试轨迹隔离。合成负载不得称为生产流量；故障注入只可作为诊断，不用占位数组制造显存不足。
图规格改变可能影响最终KV大小，但固定静态KV分配模式下也可能只是减少剩余headroom或导致OOM；两种机制必须分开。

## 停止条件

环境繁忙或基线不兼容：只做来源审查/CPU契约测试并报告阻塞，不继续GPU负载。
图资源在真实合法部署中不构成重要限制：不扩大假问题。
充分调优的简单配置已接近候选最好结果：不立项复杂控制器。
只有强基线下存在跨进程/跨时段稳定且重要的服务损失，并排除旧版本缺陷后才设计新方法。

## 来源

固定SGLang源码根：https://github.com/sgl-project/sglang/tree/e00930c5489053f26d86b179cee0d087f846acbb
相关文件：python/sglang/srt/model_executor/runner_utils/pool.py；model_runner_components/kv_pool_runtime.py；python/sglang/srt/arg_groups/overrides.py；python/sglang/srt/environ.py；python/pyproject.toml。
官方问题报告：https://nvidia.github.io/TensorRT-LLM/1.3.0rc14/blogs/tech_blog/blog20_Tuning_CUDA_Graph_Batch_Sizes_for_Higher_Output_Throughput.html
其v1.3.0rc8/GB200多卡/DeepSeek结果仅作问题来源，不能作为Tang实测或预计收益。
