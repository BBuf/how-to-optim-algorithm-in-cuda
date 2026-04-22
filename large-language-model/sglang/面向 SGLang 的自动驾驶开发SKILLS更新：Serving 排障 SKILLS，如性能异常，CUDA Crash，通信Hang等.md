> 欢迎关注：https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS 用Codex/Claude Code一起自动驾驶SGLang开发。

# 0x0. 前言

AI-Infra-Auto-Driven-SKILLS 更新一个新的 SKILLS ，这次更新的的是一个更偏真实 serving 排障的 SKILL：

https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/sglang-prod-incident-triage

如果后续合并进 SGLang 主仓，对应目录就是：

https://github.com/sgl-project/sglang/tree/main/.claude/skills/sglang-prod-incident-triage

这个 SKILL 不是替代 `torch profiler`、`cuda crash debug`、`distributed hang debug` 这些专项技能，而是先把真实服务里的问题收成一条规范路径。这个SKILL之前已经有很多铺垫了，例如debug cuda crash（https://github.com/sgl-project/sglang/tree/main/.claude/skills/debug-cuda-crash） ，debug distributed hang(https://github.com/sgl-project/sglang/tree/main/.claude/skills/debug-distributed-hang)，sglang torch profiler analysis（https://github.com/sgl-project/sglang/tree/main/.claude/skills/sglang-torch-profiler-analysis）等等。这个 SKILL 做真实serving排障功能的时候高强度的调用了上面一系列已有SKILL，只是让整个生产环境的排障流程更加清楚，对Agent debug提供更加精准的上下文信息。

整体思路就是：

**先保留现场，再 replay 事故，再把问题送进正确的专项技能。**

使用的方法就是带上这个SKILL让他帮你去debug你会发生异常的服务，只需要把原始的启动方式和请求方式告诉Agent，接下来抓现场，replay，debug都可以被Agent连贯的接起来。

# 0x1. Motivation

它的思路很直接：

- 先抓只读 bundle，不一上来就开 profiler。
- 先把触发问题的 request 或 crash dump 留下来，不靠手搓 prompt 猜现场。
- 先 replay，同一个现象能在干净目标上重放出来，后面的 trace、torch profile、cuda coredump、`git bisect run` 才有意义。
- 不重复造轮子，后面还是接到现有的 `debug-cuda-crash`、`debug-distributed-hang`、`sglang-torch-profiler-analysis` 等等这些SGLang已经有的专用SKILL里。

这个 SKILL 在Ref里面提供了三个例子 ：

- TTFT 异常，但 queue time 很低
- 请求链路里的 CUDA Crash（一个真实的生产场景发生的问题）
- 多卡 TP 通信 Hang

# 0x2. Examples

主要展示一下会正确触发这个SKILL的排障流程，更细节的信息就没贴了，用Codex和GPT5.4 xHigh在H200复现确实可以找到每个bug发生的具体代码位置。

## 0x2.1 TTFT 异常，但 queue 不是主因

![TTFT](https://img.shields.io/badge/Case-TTFT%20slowdown-f59e0b)

先看现场信号，不要先入为主：

```text
Health: /health=ok /health_generate=ok
Point-in-time load: running=1 waiting=0 total=1 token_usage=0.410 throughput=29.800 cache_hit_rate=0.970
Metrics: requests=2 prompt_tokens=1540 generation_tokens=128 avg_ttft_s=3.210 avg_e2e_s=4.150 avg_queue_s=0.030
Stage Averages (max across TP ranks): prefill_forward=2.900s, request_process=0.090s
```

<mark>waiting=0</mark>
<mark>avg_queue_s=0.030</mark>
<mark>prefill_forward=2.900s</mark>

这一步已经足够说明：

- 不是 queue pressure 在顶着
- 更像 prefill-side compute 或 request path slowdown

规范路径就是：

```text
baseline bundle
  -> save the slow request
  -> replay the same request
  -> trace / torch profile
```

## 0x2.2 CUDA Crash：挂掉的 kernel，不一定是根因 kernel

![Crash](https://img.shields.io/badge/Case-CUDA%20crash-ef4444)

这个例子不是写一个孤立 kernel 小 demo，而是故意把脏数据写在上游 routing kernel 里，让下游 `moe_align_block_size_kernel` 真正挂掉。来自真正的生产例子：https://zhuanlan.zhihu.com/p/1984750078074839122 

你最后看到的现场输出会更像这样：

```text
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
Dumped 1 finished and 1 unfinished requests before crash to /tmp/.../crash_dump_2026-04-20_14-23-15.pkl
CUDA Exception: Warp Out-of-range Address
#0 0x7f7fe1dfac00 _Z27moe_align_block_size_kernelIiEvPKT_PiS3_S3_iimS3_bii
```

<mark>faulting kernel = moe_align_block_size_kernel</mark>
<mark>root cause kernel = topkGatingSoftmax</mark>

整体流程是：

- crash dump 先把真实 request mix 留了下来
- replay 能把 crash 重新打出来
- cuda-gdb 指向的是下游 faulting kernel
- 但真正写脏数据的是上一个 routing kernel

规范路径就是：

```text
crash dump
  -> summarize dump
  -> replay
  -> CUDA coredump
  -> cuda-gdb
  -> walk one kernel upstream
```

## 0x2.3 通信 Hang：先 replay 事故，再做 distributed hang 深挖

![Hang](https://img.shields.io/badge/Case-TP%20hang-2563eb)

这个例子是故意让真实 request 链路里的 TP collective mismatch 把服务挂住。触发 request 很简单：

```text
"hello " * 768
prompt_tokens = 769
```

服务日志里先会看到：

```text
Prefill batch, #new-seq: 1, #new-token: 769, #cached-token: 0
```

重放期间再抓现场，bundle 会开始变坏：

```text
health.txt.error.json:
  TimeoutError: timed out

health_generate.txt.error.json:
  TimeoutError: timed out

loads_all.json:
  ConnectionResetError: [Errno 104] Connection reset by peer

loads_core_queues_disagg.json:
  URLError: <urlopen error [Errno 111] Connection refused>
```

再往后，watchdog / py-spy 会落到这种路径：

```text
cuEventSynchronize
cudaEventSynchronize
synchronize (torch/cuda/streams.py:231)
process_batch_result_prefill
process_batch_result
event_loop_overlap
```

<mark>服务 baseline 时是健康的</mark>
<mark>同一个 trigger request 可以 replay</mark>
<mark>control plane 是在 replay 挂住之后才开始退化的</mark>

规范路径就是：

```text
baseline bundle
  -> save the trigger request
  -> replay on a clean target
  -> collect replay-time bundle
  -> watchdog / py-spy
  -> debug-distributed-hang
```

# 0x3. 总结

这个 SKILL 真正有用的地方，不是“帮你查一个 kernel”，而是把真实 serving 里的慢、crash、hang 统一成：

**先抓现场，先 replay，再决定后面该进 trace、性能 profile、cuda coredump，还是直接 bisect。**

