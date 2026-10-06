# P02执行状态

日期2026-10-07。用户批准P02；只用research/rl-output-head-audit，不合并main。

## 已完成

- 预注册P02_PROTOCOL.md，新增需求裁剪、编译兼容性单独控制、AOT/CPU Inductor审计及带GPU空闲门槛的基准入口。
- 最终计算代码33f125359b94d0342cc52af3eba20232e0c75221。43项CPU测试通过，保留各次RED和最终GREEN。
- 未修改的verl片段6种组合都因输出factory requires_grad=True导致fullgraph编译失败，原始错误保留。
- 只去除两个factory关键字的compat_original与原实现eager操作完全相同；它和minimal各6组，AOT和CPU Inductor执行全部通过。
- 最终190个输出/梯度张量比较通过，最大绝对误差9.5367431640625e-07，相对于同输入未修改FP32源输出，不误称这190项都是FP64真值比较。
- AOT自动删除冻结W的dW及logprob_only未使用熵。最小裁剪没有在此证据层面显示编译器之外的新收益。
- CPU Inductor保存[7,11]中间张量，提示保存/重算安排需进一步检查，不是大词表GPU显存结论。
- 自审通过RED回归纠正了初稿out-of-place dz累加，恢复上游+=后全部重新验证。

## GPU与训练尚未完成

gpu_attempt01.json：采样利用率41/41/37%，无compute PID，未通过<10%门槛，退出码2，零计时样本，CUDA未初始化。更早独立检查曾有活动计算进程；不能把它混成正式采样时状态。
没有停止用户任务，没有安装全框架，没有升级torch/驱动。GPU runner只执行了阻塞分支，正向性能分支尚未验证。不要连续重试或后台等待。

尚缺GPU前后向/显存、同设备强专用实现、真正trainer路径执行及完整训练步骤。保持三种熵需求和两种W梯度状态的相同语义，不用冻结W结果代表全参训练；不把源码调用链当实际训练运行。

## 证据

报告reports/research/rl-output-head-audit/P02/REPORT.md；12行compiler_summary.csv；主要原始结果与测试日志10文件在evidence/part01..04.b64及manifest.json。主要包770955字节、SHA256 7ae759053d47e50874b85344c26475b072c2feba25ec10bdbd9efa7262385237。
四段从GitHub fetch核验通过。9文件逐字节等于Tang原件；CSV仅CRLF->LF且字段完全一致，两种hash均记录。verbose Inductor日志仍在Tang，未全部上传，不声称所有生成代码日志已发布。

## 下一步

先确认合法GPU空闲条件，再运行预注册控制；强基线后如无显著剩余成本或无法传递至完整训练，按工程修复结束，不训练预测器。不要重做P01，也不重新创建研究分支。保留全部失败、原始数据与其他研究分支。
