# 0x0. 前言

这篇文章整理了笔者近一段时间高频使用的几个 SGLang 相关 SKILL，内容覆盖 debug、benchmark、远程开发以及性能分析等场景。可以将其理解为把一部分日常开发经验、验证流程和排查方法抽取出来，交给 Codex 以 skill 的形式复用。希望这些内容对 Auto Driven AI Infra 的实践能提供一些参考。以下提到的 skill 都已经在多个模型和多个场景中做过验证，当然也仍然可能存在需要继续修正的地方。

SGLang CUDA Debug Crash SKILL 和 SGLang Auto-Driven Benchmark SKILL 因为要侵入一点SGLang系统才可以正常工作，所以分别放在：https://github.com/sgl-project/sglang/tree/main/.claude/skills/debug-cuda-crash & https://github.com/sgl-project/sglang/pull/21736

其它的SKILLS都放在：

**https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS.git** **欢迎关注**

目标是让更多 SGLang 相关工作逐步进入 Agent 可自动完成、自动分析和自动优化的流程。

下面按功能分类介绍这些 SKILL。

# 0x1. 远程连接 SKILL

- https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/b200
- https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/h100
- H200 diffusion 远端 skill（同仓库内，本文不展开具体主机标识）

这个 skill 的作用比较基础，但实际开发里非常重要。原因在于，很多 SGLang 的验证工作本来就依赖远程 GPU 服务器，包括模型加载、kernel smoke test、端到端服务验证、benchmark、profiler 采集等。如果继续沿用“每台机器单独配置 agent、单独记环境细节”的方式，切换成本会很高，也容易出错。

这个 skill 的核心目标，就是让本地运行的 Codex 或 Claude Code 直接接入远程 GPU 机器，把远程主机变成统一的执行后端，而不是在每台服务器上重复部署一套 agent。

主要功能如下：

- 约定远程主机、默认 container、默认 repo 路径，减少手工定位环境的成本。
- 提供统一的主机检查流程，包括 `hostname`、`docker ps`、`nvidia-smi`、GPU 空闲情况确认。
- 支持直接进入默认开发容器，并检查 `HF_TOKEN`、Hugging Face cache、FlashInfer 相关环境是否齐全。
- 提供安全的远程工作流，默认先检查远端 repo 状态，再决定直接使用、创建 detached worktree，还是同步本地工作树到远端临时目录。
- 支持将本地当前 working tree 流式同步到远端验证目录，适合“验证当前本地改动”这类场景。
- 提供统一的远端验证流程，包括 `py_compile`、`compileall`、`pytest`、GPU smoke test、server-level validation 等。
- 针对不同机器保留专门的 skill 版本，例如 `b200`、`h100`、`h200`，将 GPU 架构、容器名称、repo 路径、清理方式、适合的任务类型等前提写入 skill，降低选错环境的概率。
- 对 diffusion、`torch.compile`、多机 serving 等特殊验证场景，允许在不同机器上复用一致的操作方式。

下面是一组当前可直接复用的远端探测输出。以下示例仅保留必要字段，主机名和容器名均做了脱敏处理：

```bash
$ ssh h100_sglang 'hostname && whoami'
[redacted-h100-host]
sglang

$ ssh b200 'hostname && whoami'
[redacted-b200-host]
lmsys

$ ssh h200_diffusion 'hostname && whoami'
[redacted-h200-host]
sglang-rl

$ ssh h100_sglang 'docker ps --format "{{.Names}}|{{.Status}}" | sed -n "1,3p"'
spec-dev-container|Up 2 days
sglang_bbuf|Up 4 days
aux-dev-container|Up 3 days

$ ssh b200 'nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | sed -n "1,3p"'
0, NVIDIA B200, 160491, 183359
1, NVIDIA B200, 4, 183359
2, NVIDIA B200, 4, 183359

$ ssh h200_diffusion 'docker ps --format "{{.Names}}|{{.Status}}" | sed -n "1,3p"'
diffusion-dev-a|Up 21 hours
diffusion-dev-b|Up 34 hours
omni-dev-container|Up 33 hours
```

这一类 skill 没有直接的性能提升数据可展示，但它解决的是更上游的工程问题：如何在本地保持一个统一的 agent 会话，同时稳定访问多台不同类型的 GPU 服务器，并把重复性的环境准备工作压缩到最低。

# 0x2. SGLang CUDA Debug Crash SKILL

- https://github.com/sgl-project/sglang/tree/main/.claude/skills/debug-cuda-crash

这个 skill 的目标，是把 CUDA crash 的定位过程从“只剩异常栈和猜测”变成“可以在 kernel/custom op 边界拿到充分上下文”的问题。对于 SGLang 这种同时包含 JIT kernel、custom op、Triton kernel、模型封装逻辑的系统来说，这一点尤其重要。

它的实现思路来自 FlashInfer 的 API logging。既然 crash 最终一定发生在某个调用边界附近，那么就应该在执行之前记录输入，避免程序异常退出之后只剩一条模糊的 CUDA 错误。

主要功能如下：

- 覆盖 SGLang 中最关键的 kernel 边界，包括 `register_custom_op(...)`、`register_custom_op_from_extern(...)`、LLM attention/linear/quantization wrapper、diffusion attention/linear/rotary wrapper，以及部分 `torch.ops.sglang.*` 热点路径。
- 支持分级日志输出，便于按需控制信息量。
- `SGLANG_KERNEL_API_LOGLEVEL=1` 输出 API 调用边界和异常边界。
- `SGLANG_KERNEL_API_LOGLEVEL=3` 输出输入 tensor 的 shape、dtype、device、contiguous 等元数据。
- `SGLANG_KERNEL_API_LOGLEVEL=5` 进一步输出 min/max/mean、NaN/Inf 统计，便于排查数值问题。
- `SGLANG_KERNEL_API_LOGLEVEL=10` 在执行前自动保存 `inputs.pt`、`metadata.json`，在异常情况下保留完整调用快照。
- 支持将 dump 落盘到指定目录，方便离线复现。
- 在开启 CUDA Graph 时会自动跳过不安全的 tensor dump，但仍保留调用边界日志，避免调试逻辑反过来影响图捕获。
- 同时适用于 LLM 和 diffusion 侧的问题定位，而不是只面向一种模型类型。

下面是一段笔者在远端 H100 上实际运行这个 skill 得到的日志摘录。测试脚本故意构造了一个越界的 embedding index，使 custom op 在 `torch.cuda.synchronize()` 处触发 `device-side assert`：

```bash
===STDERR===
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1478: indexSelectSmallIndex: block: [0,0,0], thread: [0,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
...
torch.AcceleratorError: CUDA error: device-side assert triggered

===LEVEL3===
[2026-04-01 15:42:32] SGLang Kernel API Call: sglang_llm_crash.mock_llm_cuda_crash
Positional input arguments:
  arg[0]=Tensor(shape=(2,), dtype=torch.int64, device=cuda:0)
  arg[1]=Tensor(shape=(4, 8), dtype=torch.float16, device=cuda:0)
[2026-04-01 15:42:32] SGLang Kernel API Exception: sglang_llm_crash.mock_llm_cuda_crash

===LEVEL10_DUMPS===
/tmp/sglang_kernel_api_demo/.../inputs.pt
/tmp/sglang_kernel_api_demo/.../metadata.json

===LEVEL10_META===
{
  "function_name": "sglang_llm_crash.mock_llm_cuda_crash",
  "execution_status": "exception",
  "input_tensor_keys": ["arg_0", "arg_1"],
  "exception": {"type": "AcceleratorError", "message": "CUDA error: device-side assert triggered"}
}
```

这段输出可以说明三件事。第一，异常发生时最后一个 kernel API 边界已经被记录下来。第二，`level 3` 已经能把输入 tensor 的 shape、dtype 和 device 打印出来。第三，`level 10` 会把本次调用的 `inputs.pt` 与 `metadata.json` 落盘，从而把一次原本瞬时消失的 CUDA crash 转成可以继续离线分析的问题样本。更完整的验证过程可以参考 [PR #20910](https://github.com/sgl-project/sglang/pull/20910#issuecomment-4088048145)。

因此，这个 skill 的主要价值并不是多输出一些日志，而是把一次原本不稳定、难以追踪的 CUDA crash，转成可以离线复现和继续分析的问题样本。

# 0x3. SGLang Auto-Driven Benchmark SKILL

- https://github.com/sgl-project/sglang/pull/21736

这个 skill 的目标，是把 SGLang 里最耗时的一类经验工作，也就是 server flag 搜索和 workload benchmark，变成可以自动执行、自动记录、可以中断恢复的流程。它并不只是一个 benchmark 脚本，而是一套围绕“配置生成、服务启动、压测、SLA 判断、结果汇总”建立起来的自动化闭环。

主要功能如下：

- 提供 `run`、`convert`、`validate` 三个入口。
- `convert` 支持把 `sharegpt`、`custom`、`random`、`generated-shared-prefix` 等输入格式统一转成 canonical autobench JSONL。
- `validate` 支持对 canonical autobench 数据做结构校验，避免 benchmark 运行过程中才发现数据格式问题。
- `run` 支持通过 YAML 配置驱动完整的自动 benchmark 流程。
- 支持根据 prompt 或 YAML 自动生成候选 server flags，而不是手工逐条拼启动命令。
- 支持根据数据格式自动选择 benchmark backend，例如 `sglang`、`sglang-oai`、`sglang-oai-chat`。
- 支持三档搜索等级：
  - `tier 1`：最小、最快的 sanity sweep。
  - `tier 2`：默认的平衡搜索。
  - `tier 3`：最大、最慢的全量搜索。
- 支持 `max_candidates`，可以在搜索空间较大时手工限制候选数量。
- 支持固定 QPS 列表模式，也支持带 `lower / upper / tolerance` 的 QPS binary search。
- 支持 `max_concurrency` 维度联合搜索，而不是只看单一请求速率。
- 支持并行相关配置的搜索与推导，例如 `tp`、`dp`、`pp`、`ep` 相关组合。
- 支持多场景 dataset 展开，例如对同一份配置同时生成 chat、summarization 等不同 workload 场景。
- 支持 base 阶段和 speculative 阶段的两段式搜索，后者可继续搜索 speculative/EAGLE 相关参数。
- 支持实时回传进度，运行过程中会持续写入 `live_results.jsonl`。
- 支持在中断时保留已完成 trial 的结果，避免长时间实验因为意外退出而完全丢失。
- 支持 `resume`，中断之后可以读取已有 trial 结果继续执行，而不是从头重跑。
- 支持按场景输出 `prepared_dataset.jsonl`、`results.jsonl`、`results.csv`、`summary.md`，多场景时还会额外输出 `scenario_summary.jsonl/csv`。
- 支持自动记录每个 trial 的启动参数、请求速率、SLA 是否通过、吞吐和延迟指标，便于后续横向比较。

下面是一段笔者在远端 H100 上实际运行这个 skill 得到的日志摘录。为了控制运行时间，这里使用了一个单卡、小模型、`tier 1` 的最小搜索配置：

```bash
=== Auto Benchmark Plan ===
search.tier=1 (tier 1: smallest and fastest sanity sweep)
qps_plan=fixed qps values=[1.0]
Planned base candidates:
  [1/3] {"model_path": "Qwen/Qwen2.5-0.5B-Instruct", "tp_size": 1, ...}
  [2/3] {"model_path": "Qwen/Qwen2.5-0.5B-Instruct", "tp_size": 1, "chunked_prefill_size": 512, ...}
  [3/3] {"model_path": "Qwen/Qwen2.5-0.5B-Instruct", "tp_size": 1, "chunked_prefill_size": 1024, ...}
scenario=demo
prepared_dataset=/tmp/auto_bench_qwen05_demo/prepared_dataset.jsonl
selected_backend=sglang-oai

=== SUMMARY ===
| Candidate | Stage | QPS | Output tok/s | TTFT ms | TPOT ms | SLA |
| 0 | base | 1.0 | 51.78 | 6.26 | 1.20 | pass |
| 1 | base | 1.0 | 51.78 | 6.05 | 1.21 | pass |
| 2 | base | 1.0 | 51.76 | 6.50 | 1.21 | pass |

=== RESUME RUN ===
resume=true loaded_records=3 scenario=demo
results_jsonl=/tmp/auto_bench_qwen05_demo/results.jsonl
```

这段结果表明，skill 已经能够完整跑通候选生成、数据准备、服务启动、压测、结果汇总和 `resume` 复用。在这组最小演示里，3 个 candidate 都满足 SLA，第二次运行会直接复用已有 trial。更大规模的搜索结果可以参考 [PR #21736](https://github.com/sgl-project/sglang/pull/21736#issuecomment-4159966660)。

此外，这个 skill 的意义不仅是找到更优配置，还包括系统性排除那些看起来合理、但在真实 workload 下并不成立的参数选择。

# 0x4. SGLang Torch Profiler Analysis SKILL

- https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/sglang-torch-profiler-analysis

这个 skill 是对 torch profiler 分析流程的一次统一整理。过去做 SGLang profiler 分析时，往往需要自己分开处理 trace 采集、stage 拆分、Perfetto 渲染修复、kernel 分类、源码映射、overlap 判断以及 fuse 机会识别。这个 skill 的作用，就是把这些动作整合到一个统一入口里。

主要功能如下：

- 提供统一入口脚本 `analyze_sglang_torch_profile.py`，而不是分散使用多套脚本。
- 提供四个子命令：
  - `triage`：默认工作流，直接输出三张主表。
  - `breakdown`：单 trace 的 kernel/category 占比分析。
  - `overlap`：两阶段 overlap 分析。
  - `perfetto-fix`：修复部分 overlapped event 在 Perfetto 中渲染缺失的问题。
- 支持直接分析已有 `trace.json(.gz)` 或 profile 目录。
- 支持对运行中的 SGLang server 直接触发 `sglang.profiler`，并自动发送 probe request 采集真实 workload。
- 支持 `profile_by_stage`，将 `extend/prefill` 与 `decode` 分开分析。对于普通服务也推荐使用这一模式；如果开启 PD 分离，则需要分别对 prefill worker 和 decode worker 采集。
- `triage` 默认输出三张表：
  - `Kernel Table`
  - `Overlap Opportunity Table`
  - `Fuse Opportunity Table`
- `breakdown` 支持按 attention、communication、MoE、norm、quantization、memory 等类别统计 GPU 时间占比。
- `overlap` 采用两阶段方法：
  - 第一阶段采集 graph-off 的 `mapping trace`，用于恢复 `kernel -> cpu op -> python scope` 的对应关系。
  - 第二阶段采集 graph-on 的 `formal trace`，用于判断真实 serving 形态下的 overlap 空间。
- 能够把 kernel 回指到 Python 代码位置，而不是只停留在 kernel 名称层面。
- 能够区分“仍有 overlap headroom 的 kernel”和“已经被其它 compute 覆盖、优化收益较低的 kernel”。
- 能输出 dependency risk、actionable overlap rows 以及 ASCII timeline，方便在不打开 Perfetto 的情况下先完成第一轮判断。
- 能对照 `fuse-overlap-catalog`，将分析结果与 SGLang 中已存在的 fuse 或 multi-stream overlap pattern 进行匹配，避免把现有优化缺失误判为全新机会。
- `perfetto-fix` 可以对 trace 做后处理，修复 overlapped kernel 在 Perfetto 中显示不完整的问题。

下面这次展示来自笔者在远端 B200 上实际运行 `triage` 的结果。输入是一组真实的 `mapping trace + formal trace`，模型为 `Qwen/Qwen2.5-0.5B-Instruct`，输出头部如下：

```bash
Triage View
Mapping traces: ...TP-0-DECODE.trace.json.gz
Formal traces: ...TP-0-EXTEND.trace.json.gz, ...TP-0-DECODE.trace.json.gz
Model: Qwen/Qwen2.5-0.5B-Instruct
```

这次运行对应的目标，是检查一个故意拆掉 `TP all-reduce + residual/RMSNorm` 的路径能否被重新识别出来。从结果来看，skill 的判断是成立的：`Kernel Table` 在 `decode` 阶段重新暴露出 `cross_device_reduce_1stage<__nv_bfloat16, 2>`，`Overlap Opportunity Table` 将其标记为高优先级的 headroom，`Fuse Opportunity Table` 则重新将问题指回 `layernorm.py::_forward_with_allreduce_fusion`。换句话说，它不仅能看出“哪里慢”，还能够把一个被人为移除的优化重新找出来。

考虑到文章里的 markdown 表格宽度有限，下面贴一版节选后的主表，字段做了压缩，但信息来源仍然是 skill 默认 `triage` 的真实输出。

`Kernel Table（节选）`

| Stage | Kernel | Share | Python location |
| --- | --- | ---: | --- |
| extend/prefill | `cross_device_reduce_1stage` | 89.2% | `custom_all_reduce_ops.py:45 all_reduce` |
| decode | `cross_device_reduce_1stage` | 31.0% | `custom_all_reduce_ops.py:45 all_reduce` |
| decode | `nvjet_tst_32x64_64x16_4x1_v_bz_TNN` | 9.9% | `unquant.py:134 apply` |
| decode | `FusedAddRMSNormKernel` | 9.2% | `unresolved` |
| decode | `nvjet_tst_64x8_64x16_4x1_v_bz_TNT` | 7.6% | `unquant.py:134 apply` |
| decode | `act_and_mul_kernel` | 3.9% | `activation.py:73 forward_cuda` |

`Overlap Opportunity Table（节选）`

| Stage | Priority | Kernel | Python scope | Recommendation |
| --- | --- | --- | --- | --- |
| extend/prefill | P2 | `cross_device_reduce_1stage` | `device_communicators/custom_all_reduce_ops.py:45 all_reduce` | `check deps` |
| decode | P2 | `cross_device_reduce_1stage` | `device_communicators/custom_all_reduce_ops.py:45 all_reduce` | `check deps` |
| decode | P2 | `FusedAddRMSNormKernel` | `layers/quantization/unquant.py:134 apply` | `check deps` |
| decode | P1 | `nvjet_tst_32x64_64x16_4x1_v_bz_TNN` | `layers/quantization/unquant.py:134 apply` | `try fusion` |
| decode | P5 | `vectorized_elementwise_kernel` | `managers/scheduler.py:2583 run_batch` | `skip overlap` |

`Fuse Opportunity Table（节选）`

| Stage | Pattern | Share | Current location | Candidate fused path |
| --- | --- | ---: | --- | --- |
| extend/prefill | `TP all-reduce + residual/RMSNorm` | 89.2% | `custom_all_reduce_ops.py:45 all_reduce` | `layernorm.py:89 _forward_with_allreduce_fusion` |
| decode | `TP all-reduce + residual/RMSNorm` | 31.0% | `custom_all_reduce_ops.py:45 all_reduce` | `layernorm.py:89 _forward_with_allreduce_fusion` |



因此，这个 skill 更适合用来回答三个问题：当前 trace 的主要耗时来自哪里，哪些位置仍然存在 overlap 空间，以及这些问题是否已经在 SGLang 中有现成的 fuse/overlap 路径可复用。


# 0x5. 总结

最近这段时间持续将 SGLang 开发、benchmark、debug、profile 的流程整理为 SKILL，比较明显的感受是：不少过去高度依赖个人经验的工作，已经可以逐步交给 Agent 去完成。当然，这并不意味着把 Agent 接入之后问题就自然消失，更重要的是先把经验、上下文和验证流程整理清楚。

这篇里聊的四类 skill，分别解决了四种不同层面的问题：

- 远程连接 skill 解决的是多机环境切换的问题。
- CUDA crash skill 解决的是 kernel 级 debug 的问题。
- auto benchmark skill 解决的是 server flag 调参的问题。
- torch profiler analysis skill 解决的是从 trace 到优化结论这条链路的问题。

后面应该还会继续补这个系列。一方面是继续完善已有 skill，另一方面也是把更多真正有价值的 SGLang 开发流程继续整理出来，解放兄弟们。
