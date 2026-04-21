# 0x0. 前言

最近把 SGLang 的自动驾驶开发技能又补了一轮，这次补的是一个更偏真实 serving 排障的 SKILL：

https://github.com/BBuf/SGLang-Auto-Driven-SKILLS/tree/main/skills/sglang-prod-incident-triage

如果后续合并进 SGLang 主仓，对应目录就是：

https://github.com/sgl-project/sglang/tree/main/.claude/skills/sglang-prod-incident-triage

这个 SKILL 不是替代 `torch profiler`、`cuda crash debug`、`distributed hang debug` 这些专项技能，而是先把真实服务里的问题收成一条规范路径：

**先保留现场，再 replay 事故，再把问题送进正确的专项技能。**

# 0x1. 这个 SKILL 做什么

它的思路很直接：

- 先抓只读 bundle，不一上来就开 profiler。
- 先把触发问题的 request 或 crash dump 留下来，不靠手搓 prompt 猜现场。
- 先 replay，同一个现象能在干净目标上重放出来，后面的 trace、torch profile、cuda coredump、`git bisect run` 才有意义。
- 不重复造轮子，后面还是接到现有的 `debug-cuda-crash`、`debug-distributed-hang`、`sglang-torch-profiler-analysis` 这些专项技能里。

这个 SKILL 现在自带 3 条最常见的分支：

- TTFT 异常，但 queue time 很低
- 请求链路里的 CUDA Crash
- 多卡 TP 通信 Hang

# 0x2. 演示

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

这个例子不是写一个孤立 kernel 小 demo，而是故意把脏数据写在上游 routing kernel 里，让下游 `moe_align_block_size_kernel` 真正挂掉。

你最后看到的现场输出会更像这样：

```text
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
Dumped 1 finished and 1 unfinished requests before crash to /tmp/.../crash_dump_2026-04-20_14-23-15.pkl
CUDA Exception: Warp Out-of-range Address
#0 0x7f7fe1dfac00 _Z27moe_align_block_size_kernelIiEvPKT_PiS3_S3_iimS3_bii
```

<mark>faulting kernel = moe_align_block_size_kernel</mark>
<mark>root cause kernel = topkGatingSoftmax</mark>

这条链路最值得看的不是解释，而是这几个事实同时成立：

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

这个 SKILL 真正有用的地方，不是“帮你查一个 kernel”，而是把真实 serving 里的慢、crash、hang 收成一条统一路径：

**先抓现场，先 replay，再决定后面该进 trace、torch profile、cuda coredump，还是直接 bisect。**

这样做的好处是，后面的专项技能不再是盲查，而是接在一个已经被保留、被重放、被收敛过的事故现场后面。
