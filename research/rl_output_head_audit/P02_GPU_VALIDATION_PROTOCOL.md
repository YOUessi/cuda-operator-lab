# P02-GV：GPU正确性与后端就绪检查

日期：2026-10-07。继续用户已批准的P02，不改生产CUDA或main。

## 原因与边界

148b84e的gpu_compare.py在gpu_attempt02中仍被正式计时门槛拦住：三次利用率25/25/28%，compute PID为空。禁止为了跑出速度结果放宽原<10%门槛。

本子阶段只回答CPU结果能否在CUDA上执行、编译后输出与反向是否正确、CCE完整包是否可用。它不计时、不发布加速比、不把共享桌面上的显存观测作为峰值显存比较。不是重试正式benchmark，也不是重新定义P02成功标准。

Ruling: 允许独立的非计时CUDA正确性检查在无外来compute PID、至少4GiB空闲且温度低于80°C时执行；全部metadata必须可解析，单GPU。检查期间出现外来compute PID则停止本实验，不终止他人任务。这不满足正式性能实验的空闲要求，故记录performance_measured=false，禁止写summary_us/speedup/latency字段。没有后台任务。

## 不改变的计算

沿用sources.json固定上游及compile_compat.py显式兼容性修改。比较original_eager、compat_eager、minimal_eager、compat_compiled、minimal_compiled。

覆盖W训练/冻结与logprob_only/entropy_logged/entropy_loss。CPU数学参考与GPU编译验证分开；CUDA主对照使用既定BF16或FP32容差并保留max_abs/mean_abs/relative_l2，不放宽容差使之通过。两组输入种子用于数值核验，不是性能重复次数。

正式基准函数与计算内核均不修改。新工具复用现有make_inputs、make_callable、observe、compare_outputs。GPU诊断采用default模式、fullgraph=True、dynamic=False，不代表最大调优模式已验证。

## 强后端

CCE使用sources.json固定的apple-aiml-research/ml-cross-entropy提交3de376c106a1916bc5e1b619f9c77c87a461ee1c，在独立的只读源码checkout中通过公开包入口导入，不运行安装脚本、不升级已有环境。校验checkout SHA和tracked clean。选reduction=none、filter_eps=None、禁用梯度过滤、FP32梯度累加，且temperature=1.0。仅对logp-only使用CCE；不假装其API返回可求导熵。

CCE与原路径的中间舍入契约不同：增加一个显式FP32数学运算参考；报告对原路径与FP32参考的误差，不要求逐位一致，不由正确性容差推导性能公平或训练收敛。CCE不能运行时保存异常，不省略这个强基线，也不声称全部强后端比较完成。

## 交付与去留

GPU数值结果、每个候选异常、源版本、设备状态完整保存并通过GitHub发布。代码先提交GitHub再在Tang拉取测试。CPU helper先失败后实现。

此阶段通过仅表示GPU正确性/后端就绪，P02正式时间/显存与完整训练仍未完成。若发现编译后的误差，保留失败并在放行前查明。若后续正式计时无空闲窗口，如实报告，不无限循环轮询。
