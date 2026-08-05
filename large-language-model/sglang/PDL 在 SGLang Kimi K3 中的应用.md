# PDL 在 SGLang Kimi K3 中的应用

Kimi K3 是一个 2.8T 参数的混合注意力模型。decode 一个 token 需要经过 93 个注意力层（69 层 KDA + 24 层 MLA）和 92 层 latent-MoE。bs=1 时，单层的计算量很小，瓶颈主要来自 kernel 数量和 launch 延迟，而不是 FLOPs。模型刚接入时，一个 decode step 会启动数百个小 kernel。

SGLang 后续通过 kernel 融合、替换 NVIDIA kernel 和通信融合等方式优化了这条路径。在推测解码介入前，day-0 版本的 decode 吞吐约为 113 tok/s（PR #32541 中的最新结果约为 118 tokens/s）。下文只引用已经公开的绝对性能数据，未公开的测试结果只保留定性结论。这些优化新增或改写了不少 kernel，其中多数接入了 PDL（Programmatic Dependent Launch），用于重叠相邻 kernel 的 launch 和 prolog。

本文按调用链梳理 K3 中接入 PDL 的 kernel，记录相邻依赖关系、wait/trigger 的位置、能够重叠的工作和现有性能数据，最后整理实现中遇到的问题。

**TL;DR**

- **PDL 的作用不是让单个 kernel 算得更快，而是缩短相邻 kernel 之间的空隙。** Consumer 可以提前完成 launch 和不依赖 producer 输出的 prolog，在第一次读取依赖数据前再通过 `griddepcontrol.wait` 等待。
- **K3 的 bs=1 decode 是 PDL 的典型使用场景。** 一次 step 要穿过 93 个注意力层和 92 层 latent-MoE，数百个小 kernel 的 launch gap 会直接进入关键路径。
- **K3 同时在计算链和通信链中使用 PDL。** 计算 kernel 会在 wait 前预取权重或 bias；通信 kernel 则会在等待其他 rank 时放行后继 kernel 的 prolog。
- **正确性取决于 wait、trigger 和数据依赖的位置。** Producer 必须在相关输出写回后 trigger；consumer 只能在 wait 前读取与紧邻 producer 无关的数据。错误的 trigger、`__ldg()` 重排或资源无法共驻，都可能让优化失效，甚至产生竞态。
- **公开结果证明“融合 + PDL”有效，但还不能把收益全部归因于 PDL。** K3 Day-0 博客中包含 PDL 的“重叠与 prolog 融合”阶段合计增加 10.4 tok/s；目前没有只切换 PDL 开关的完整 e2e A/B。

本文主要基于 K3 Day-0 支持总 PR [#32541](https://github.com/sgl-project/sglang/pull/32541) 和独立 kernel 导出 PR [#32890](https://github.com/sgl-project/sglang/pull/32890)。后者增加 27,808 行，`PDL`/`griddepcontrol` 在 diff 中命中约 300 处。PDL 原理部分参考 Yifan Yang 的[《使用 Programmatic Dependent Launch（PDL）降低端到端延迟》](https://yang-yifan.github.io/blogs/pdl/pdl_cn.html)，正确性问题部分参考是小肖啊的[《PDL 遇上 __ldg()：Bug 还是 Feature？》](https://zhuanlan.zhihu.com/p/2067263583239533156)。下文分析以代码 diff 和公开性能结果为准。

## PDL 是什么

同一条 stream 上的两个 kernel 默认串行执行：前一个 grid 的所有 block 退出并完成全局内存可见性之后，后一个 kernel 才开始 launch。但 consumer 的 launch 和部分 prolog 通常不依赖 producer 的输出。Yifan Yang 在博客中写道：“FC2 的 launch 开销和 prolog 并不*依赖*于 FC1 的结果，只有 FC2 的 mainloop 的执行才依赖于 FC1 的结果”。

Hopper 及更新架构提供的 PDL 通过两条 PTX 指令表达这种依赖：

- `griddepcontrol.launch_dependents`：producer 在输出可供 consumer 使用后放行后继 kernel。若 trigger 之后仍有 consumer 会读取的写操作，提前放置这条指令也会影响正确性；
- `griddepcontrol.wait`：consumer 在第一次读取 producer 输出前等待。wait 放得越后，可重叠的 prolog 越多，但越过依赖数据的读取会造成竞态。

launch 侧还要通过 extensible launch API（`cudaLaunchKernelEx`）设置 `cudaLaunchAttributeProgrammaticStreamSerialization` 属性。设置 launch attribute，并在 producer/consumer 中使用对应指令后，consumer 的 launch 和部分 prolog（初始化、加载无关数据）可以与 producer 的 mainloop 及收尾阶段（grid-ending membar）重叠。

<p align="center">
  <img src="https://files.mdnice.com/user/59/5285ca55-33a0-49e3-94bd-c32766ac997c.png" width="98%" alt="默认串行执行与 PDL 重叠执行的时间线对比">
</p>

<p align="center">
  <em><b>默认串行执行与 PDL。</b>PDL 允许 FC2 的 launch 和 prolog 与 FC1 的执行重叠。来源：Yifan Yang，《使用 Programmatic Dependent Launch（PDL）降低端到端延迟》。</em>
</p>

在同步关系正确的前提下，trigger 的位置决定了实际重叠量。放得太晚，FC2 的 prolog 无法充分重叠；放得太早，FC2 可能提前走完 prolog，然后停在 wait 上，并与 FC1 争用执行资源。

<p align="center">
  <img src="https://files.mdnice.com/user/59/498ed057-36ee-482e-9600-191d4c171553.png" width="98%" alt="launch_dependents 放置过晚和过早的对比">
</p>

<p align="center">
  <em><b>Trigger 的位置决定重叠量。</b><code>griddepcontrol.launch_dependents</code> 放得过晚时无法充分重叠，放得过早时 consumer 可能停在 wait 上并与 producer 争用资源。来源：Yifan Yang，《使用 Programmatic Dependent Launch（PDL）降低端到端延迟》。</em>
</p>

Yifan Yang 将 kernel 间同步分为三类：默认串行的硬件同步、PDL 这种软件辅助的硬件同步，以及由 L2 atomic 管理依赖的纯软件同步（megakernel）。SGLang K3 主要采用第二种方式。各个融合单元仍是独立 kernel，便于单独 A/B 或替换实现；相邻 kernel 之间则通过 PDL 建立依赖并重叠执行。

K3 对 PDL 的封装只有两个模板函数和一个 launch 参数：

```c++
// 来源：sglang/python/sglang/kernels/jit/include/sgl_kernel/utils.cuh
template <bool kUsePDL>
__device__ __forceinline__ void PDLWaitPrimary() {
  if constexpr (kUsePDL) {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;");
#endif
  }
}

template <bool kUsePDL>
__device__ __forceinline__ void PDLTriggerSecondary() {
  if constexpr (kUsePDL) {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;");
#endif
  }
}
```

host 侧统一使用 `LaunchKernel(grid, block, ...).enable_pdl(kUsePDL)`，由内部代码设置 launch attribute。Python wrapper 通过 `is_arch_support_pdl()` 判断架构。Triton kernel 使用 `tl.extra.cuda.gdc_wait()`、`gdc_launch_dependents()` 和 `launch_pdl=True`；CuTe-DSL kernel 使用 `cute.arch.griddepcontrol_wait()` 和 `use_pdl=True`。K3 中的 CUDA C++、Triton 和 CuTe-DSL kernel 都有对应实现。

## PDL 在 K3 bs=1 decode 中解决什么问题

bs=1 decode 中没有其他请求的计算可用于填充 launch gap。大 batch 下能够被并行工作掩盖的空隙，在 bs=1 的串行路径上通常会直接计入 step 时间。中文文章[**SGLang 和 Miles 为 Kimi K3 提供 Day-0 支持**](https://mp.weixin.qq.com/s/H6fstE6NmGnG7LhgQz_lVA)对此有一段总结：“All-reduce 是一个同步点，因此在那里节省一微秒，就会一比一地转化为 step 时间的缩短；而位于另一个 stream 的重叠空隙中的 kernel，转化比例大约只有十分之一。”

下面两张图来自 [SGLang Kimi K3 Day-0 博客](https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support)，分别给出 bs=1 优化的分类收益和吞吐变化。对应的中文文章是[《SGLang 和 Miles 为 Kimi K3 提供 Day-0 支持》](https://mp.weixin.qq.com/s/H6fstE6NmGnG7LhgQz_lVA)。

<p align="center">
  <img src="https://files.mdnice.com/user/59/4eb4b81d-a2f1-46d0-b4fe-6e3cde44fc94.png" width="98%" alt="batch size 1 时按优化类别划分的瀑布图">
</p>

<p align="center">
  <em><b>bs=1 优化的分类收益。</b>按优化类别划分 K3 decode 吞吐的提升。来源：SGLang Kimi K3 Day-0 博客。</em>
</p>

<p align="center">
  <img src="https://files.mdnice.com/user/59/acdc19a8-c5f9-4117-a918-bc85b2961262.png" width="98%" alt="batch size 1 时的 decode 吞吐优化阶梯">
</p>

<p align="center">
  <em><b>bs=1 decode 吞吐优化阶梯。</b>各项优化按合入顺序带来的吞吐变化。来源：SGLang Kimi K3 Day-0 博客。</em>
</p>

瀑布图中的“重叠与 prolog 融合”（P10、P11、P14、P15）合计增加 10.4 tok/s，其中 P15 明确写的是“将 MLA decode 的 prolog 融合为一个 kernel，并用 PDL 启动其后的注意力 kernel”。“消除 launch”“NVIDIA kernel”和“通信融合”中的多处新 kernel 也启用了 PDL。优化后的 trace 中可以看到负 launch gap，即后继 kernel 的启动时间戳早于前驱 kernel 的结束时间戳。

## 计算链上的 PDL

下表列出 #32890 导出的 K3 计算 kernel。“整体 A/B”表示该项收益包含在所在 PR 的整体测试中，无法单独归因。

| kernel | 链上位置（producer → consumer） | PDL 用法要点 | 性能结论 |
|---|---|---|---|
| `tiny_gemm`（n/k 两变体） | qkvg GEMM/norm → KDA 窄投影、router gate | 权重在 wait 前预取 | 微基准：开启预取后延迟下降 |
| `route_radix_v2` | fused-front GEMM → MoE 路由 | bias 在 wait 前预取；选完 topk 即 trigger | 微基准：延迟下降 |
| `route_quant_fused` | fused-front GEMM → {路由 + FP8 量化} | 量化 CTA 自带独立 wait/trigger | 整体 A/B：吞吐提升 |
| `align_single_token` | route → grouped GEMM | 3 launch+memset 折成 1×32 线程 launch 并入链 | 与 `topk_sum` 合测：ITL 下降 |
| `topk_sum` | marlin MoE 输出 → top-k 求和 | 入口 wait / 尾部 trigger | 微基准：快于通用 `torch.sum` |
| `add3` / `moe_tail_add` | up_proj + shared + prefix_sum 三路加 | **开头就 trigger**；可选 b/c 预取 | 整体 A/B：ITL 下降 |
| `situ_and_mul` 系 | gate_up GEMM → SiTU 激活 | 入口 wait / 尾部 trigger | 纳入整体 A/B |
| `mla_output_gate` | MLA attn → x·sigmoid(g) | 入口 wait / 尾部 trigger | 纳入整体 A/B |
| `attn_res_*`（score/combine/fused_tma） | o_proj/AR → 注意力残差链 | fused_tma 只对 prefix_sum 精确 wait | 纳入整体 A/B |
| `set_mla_kv_concat_q` | q/kv 投影 → trtllm-gen fmha | 融合省 1 launch/层且不断链 | 整体 A/B：单 step 时间下降 |
| `kda_packed_decode` / `kda_fused_decode` | KDA 投影 → 状态更新 | 入口 wait，**所有出口分支都 trigger** | 微基准：开启 PDL 后延迟下降 |
| TGV bf16 GEMM（CuTe-DSL） | bs=1 skinny GEMM | `pdl=True` | 整体 A/B：吞吐提升 |

下面看几个具体实现。

### 在 wait 前预取不依赖前驱的数据

`tiny_gemm` 的注释写道：“Weight prefetch: address is input-independent, load before the PDL wait”。权重地址不依赖 producer 的输出，因此 HBM load 可以在 `griddepcontrol.wait` 前发出；激活 x 必须在 wait 后读取。对 bs=1 的 skinny GEMM，权重加载占据了主要时间。微基准结果显示，bfa 投影、forget gate 和 router gate 三种形状的延迟都明显下降。

`route_radix_v2` 在 wait 前预取 bias，微基准也能看到延迟下降。CuTe-DSL 的 `kda_decode_mtp` 会提前发起状态 tile 的 TMA load，TGV GEMM 的 `pdl=True` 也用于提前加载权重。前提是被预取数据的地址和内容都不能依赖紧邻的 producer。

### 提前放置 trigger

`add3` 在 kernel 入口执行 trigger，代码注释是：“Trigger early, so that the next kernel gets a chance to prefetch”。后继 kernel 可以尽早开始不依赖 `add3` 输出的预取工作。

`attn_res` 的 TMA 版本只等待 `prefix_sum`。这个输入由紧邻的前驱写入，其他 bank 行来自更早的 kernel，因此代码只在第一笔 prefix load 前执行一次 wait。原注释为：“Only prefix_sum is written by the immediately-preceding kernel; one wait before the first token's prefix load covers the rest”。

### 融合时保留 PDL 链

PR #32541 中 `_set_kv_and_concat_q_fused` 的 docstring 写明了两个目的：“saves one launch per MLA layer and keeps the PDL chain intact”。KV scatter 和 q 拼接原本由两个小 kernel 完成。融合后少一次 launch，后面的 trtllm-gen fmha 也可以通过 `enable_pdl=is_arch_support_pdl()` 与 query preparation 的尾部重叠。对应注释是：“Arm PDL on the trtllm-gen decode launch so its prolog overlaps the tail of the query-prep kernels”。

不支持 PDL 的小 kernel 会中断这条链。例如，每层执行一次的 `seq_lens.to(int32)` dtype copy 在 Python 代码中并不显眼，但可以从 trace 中重新出现的 launch gap 定位出来。

### `prefetch_bc` 的安全条件

PR #32541 在 `_add3` 的调用点给出了 `prefetch_bc` 的使用条件：只有确认 `b/c` 在 `_add3` 提前读取时已经写完，才能将它设为 `True`。`PDLWaitPrimary` 只处理当前 kernel 与紧邻前驱 grid 之间的依赖，不会替提前读取的 `b/c` 补上同步。对应的 CUDA 代码在 `elementwise/add3.cuh` 中：

```c++
if constexpr (kPrefetchBC) {
  b.load(params.b, vid);
  c.load(params.c, vid);
  device::PDLWaitPrimary<kUsePDL>();
  a.load(params.a, vid);
} else {
  device::PDLWaitPrimary<kUsePDL>();
  a.load(params.a, vid);
  b.load(params.b, vid);
  c.load(params.c, vid);
}
```

这段代码把三个输入分成了两组。`a` 是紧邻 `_add3` 的 `routed_expert_up_proj` 产生的。由于 `up_proj` 和 `_add3` 之间使用 PDL，`_add3` 的 block 开始执行时，前一个 GEMM 可能还没有写完 `a`，所以 `a.load` 必须放在 wait 之后。`b` 是更早的 all-reduce 写出的 `shared_output`，`c` 则是更早生成的 `prefix_sum`。

这条调用链的顺序可以简化成：

```text
all-reduce 写 b → norm → up_proj 写 a ──PDL──> add3
c 更早完成                              先 load b/c → wait → load a
```

这里的 all-reduce 使用 plain launch，没有跨过它做 PDL 重叠，相当于同一条 stream 上的一道完整边界。后面的 norm 和 `up_proj` 能够开始时，`b` 已经写完，`c` 也早已就绪。因此 `_add3` 可以在等待 `a` 的同时先发起 `b/c` 的读取，把这部分显存访问藏在 `up_proj` 的尾部；wait 返回后只需再读取 `a` 并完成三路相加。如果关闭 `prefetch_bc`，三次读取都要等到 wait 之后才开始。

判断 `prefetch_bc` 是否安全时，不需要机械地数中间隔了几个 kernel，只要确认两件事：

1. `b/c` 的 producer 在提前 load 发生前已经执行完；
2. 从提前 load 到 `_add3` 完成计算期间，不会再有其他 kernel 修改 `b/c`。

当前 MoE 尾部满足这两个条件。如果以后调整调用链，让 `b` 或 `c` 变成由仍可能与 `_add3` 重叠的前驱写入，就必须关闭 `prefetch_bc`，或者增加一个能够保证写入已经完成的同步边界。

## 通信 kernel 如何使用 PDL

计算 kernel 通常用 PDL 重叠 launch 和 prolog。通信 kernel 还可以利用等待其他 rank 的时间运行后继 kernel 的 prolog。相关实现位于 #32890 的 `kimi_k3/comm/` 目录，几个主要文件都在头部注释中说明了同步条件。

### `ar_fusion.cuh`

`ar_fusion.cuh` 实现了融合 residual 和 RMSNorm 的 all-reduce。push 系算法在 stage-1 的 multimem push 之后、stage-2 的 poll+reduce 之前执行 trigger，注释为“launch pdl early for low latency case”。本 rank 发出数据后，后继 kernel 可以在当前 kernel 等待其他 rank 的同时执行 prolog。

pull 系算法在 wait 前执行用于进入屏障的 `atomicAdd`，以免 RMW 延迟进入 wait 后的关键路径。代码注释为“keeps the RMW latency off the post-wait critical path”。multimem 归约仍放在 wait 之后，因为这一步要求 producer grid 已完成 flush，原注释是“it asserts the producer grid has flushed”。

### `gemm_ar.cuh`

`gemm_ar.cuh` 融合了 o_proj GEMM 和 all-reduce。文件头部对跨 kernel 重叠和 wait 的语义有如下说明：

> the NEXT launch on the stream may start its feed while THIS grid sits in the boundary spin + reduce; its epilogue then griddepcontrol.wait's until this grid fully completes… an unfused cublas composite cannot cooperate across the boundary
>
> PDLWaitPrimary — prior grid reached ITS trigger (k-loop end) — NOT done

GEMM 完成 k-loop 后执行 trigger。随后进行的 boundary spin 和归约可以与下一个 kernel 的权重加载重叠。需要注意的是，consumer 的 wait 对应前驱的 trigger，并不表示前驱已经完成；如果 producer 在写回完成前 trigger，就可能产生竞态。

该实现还要求两个 kernel 能够同时驻留在 SM 上。为保证 2 CTA/SM，kernel 使用 100% smem carveout，并将 smem 控制在 113KB 以内。文件注释指出，默认 carveout 会“blocks dual residency and with it the whole tail-hiding scheme”。

### `gemm_ag.cuh`

`gemm_ag.cuh` 连接 fused-norm all-reduce 和 up_proj GEMV。消费侧的 `spin_add3` 在入口 trigger，但不执行 PDL wait。注释写的是：“Deliberately NO PDLWaitPrimary: the dependency is carried through data”。这里的依赖由 phase counter 自旋维护，PDL 只负责放行后继 kernel。

GEMV 仍在 wait 前预取权重。PR #32541 对这条链的描述是：“the gemm_ag tail wants the normed latent straight out of the fused-norm AR (its GEMV chains on it via PDL)”。

下面列出 MLA、MoE 和 KDA 层中与 PDL 相关的主要调用链：

```
MLA 层：norm → q/kv 投影 GEMM → set_mla_kv_concat_q(PDL)
        → trtllm-gen fmha(enable_pdl) → mla_output_gate(PDL)
        → gemm_ar(GEMM+AR 融合，AR 尾巴藏进下一 launch)
        → attn_res_score_fused_add(PDL) → attn_res_combine(PDL) → 下一层

MoE 层：fused-front GEMM → route_radix_v2(PDL，bias 预取)
        → align_single_token(PDL) → grouped GEMM(trtllm-gen)
        → situ_and_mul(PDL) → GEMM2 → topk_sum(PDL)
        → fused-norm AR(push_norm，push 后即 trigger)
        → gemm_ag GEMV(权重预取) → spin_add3(不 wait，数据依赖)

KDA 层：qkvg GEMM → tiny_gemm(权重预取) → kda_fused_decode(全出口 trigger) → 输出投影
```

## 现有性能数据

目前没有一组完整的 e2e A/B 只切换 PDL 开关。多数改动同时包含 kernel 融合和 PDL，因此不能把全部收益单独归因于 PDL。本节只保留公开数据和可公开的定性结论，内部测试的绝对吞吐、ITL、单 kernel 延迟和 launch gap 均不展开。

kernel 微基准可以直接观察 wait 前预取的收益。`tiny_gemm` 和 `route_radix_v2` 在开启预取后都能看到明显的延迟下降。bs=1 的一个 step 中有数百个类似站点，但它们之间已经存在部分重叠，因此不能把每个站点节省的时间简单相加作为 e2e 收益。

组合 A/B 中，tail fusions 与 align/topk_sum 降低了 ITL；包含 `tiny_gemm` 的一组改动和 TGV GEMM 都提高了吞吐。这些都是“融合 + PDL”的整体结果。公开的 day-0 博客中，包含 P15 的“重叠与 prolog 融合”一组增加了 10.4 tok/s。

MLA decode 路径还出现过一次由 dtype copy 中断 PDL 链的问题。`seq_lens.to(int32)` 原本在每层都会触发一次小 copy。将它移出循环，并为 fmha 设置 `enable_pdl` 后，trace 中的 PDL 链恢复连续。该修复随 #32541 合入主线。

PDL 只能重叠 launch 边界附近的工作，不能消除 launch 本身携带的固定成本。在 all-reduce 路径上，PDL 只能遮住每次通信后的一小部分 gap，其余开销来自通信实现的固定 launch 和参数传递成本，最终通过更换为 custom AR v2 解决。

SGLang 主线的 custom allreduce v2 也在三个算法中使用 PDL，入口调用 `PDLWaitPrimary`，出口调用 `PDLTriggerSecondary`。eager 模式的输入输出拷贝则使用支持 PDL 的 memcpy kernel。具体实现见[**SGLang Custom AllReduce v1 与 v2 实现原理详解**](https://zhuanlan.zhihu.com/p/2065205306540531895) 的 0x7.5 节。

## 实现中遇到的问题

### trigger-before-store 竞态

旧版 `concat_mla_absorb_q` 在读完输入、写回输出之前执行 trigger。后继 fmha 尚未启用 PDL 时，提前 trigger 没有产生可见影响；fmha 设置 `enable_pdl` 后，consumer 可能在 concat 写回完成前继续执行，因为它的 wait 对应 concat 的 trigger，而不是 concat kernel 的完成。

`set_mla_kv_concat_q` 融合时将 trigger 移到了全部写回之后。`route_radix` 曾在选出 top-k 后、renorm 写回前 trigger，`situ_and_mul_masked_post_quant` 也出现过类似问题。`gemm_ar.cuh` 的注释概括了这条语义：“PDL's wait pairs with the prior grid's TRIGGER (not completion)”。因此，提前 trigger 前需要确认之后不再有 consumer 会读取的写操作。

### ptxas 将 load 调度到 wait 前

是小肖啊在[《PDL 遇上 __ldg()：Bug 还是 Feature？》](https://zhuanlan.zhihu.com/p/2067263583239533156)中记录了一个 B300/CUDA 13.2 环境中的问题。

PDL consumer 在源码和 PTX 中都先执行 `griddepcontrol.wait`，再通过 `ld.global.nc`（由 `__ldg()` 生成的非一致只读 load）读取 producer 输出；ptxas 生成 SASS 时，却可能把该 load 调度到 wait 之前。

手工预取只适用于不依赖 producer 的数据，而这里被重排的是依赖数据。因此 PDL consumer 应谨慎使用 `__ldg()` 读取 producer 输出。对于影响正确性的 kernel，还需要检查 SASS，确认相关 load 没有越过 `griddepcontrol.wait`。[**SGLang 贴近硬件：用编译产物（PTX/SASS）评审 PR #26588、DeepGEMM 与架构迁移**](https://mp.weixin.qq.com/s/NDtOu-tWzmfZf-sHbINnVw) 一文介绍了这种检查方法。

### profiler 中的 kernel duration

启用 PDL 后，torch profiler 和 Nsight Systems 中的 per-kernel duration 不再等同于 kernel 独占执行时间。后继 kernel 从被放行时开始计时，其中包含在 wait 上停留的时间；前驱的收尾又可能与后继的 prolog 重叠。

<p align="center">
  <img src="https://files.mdnice.com/user/59/9f87449a-7df1-4017-af14-4ccae6413c94.png" width="98%" alt="Nsight Systems 中启用和未启用 PDL 的执行对比">
</p>

<p align="center">
  <em><b>Profiler 中的 PDL。</b>Nsight Systems 中启用和未启用 PDL 时的 kernel 时间线。来源：Yifan Yang，《使用 Programmatic Dependent Launch（PDL）降低端到端延迟》。</em>
</p>

K3 的调试记录也验证了这一点：启用 PDL 后，per-kernel duration 会混入 wait 时间，不能直接用来衡量单个 kernel 的收益。性能判断应以 e2e ITL（或 tok/s）和 NCU cycle 计数为准。trace 中的 kernel duration 更适合用来检查调用链，例如是否出现负 gap 或意外断链。

### 资源共驻限制

PDL 只有在两个 kernel 能够同时驻留在 SM 上时才能产生执行重叠。两个 kernel 的 smem、寄存器和线程需求之和不能超过 SM 容量。K3 的 `gemm_ar` 要求 100% smem carveout、smem 不超过 113KB，并保持 2 CTA/SM。设计此类 kernel 时，occupancy 需要按相邻 kernel 共驻计算，而不是只计算单个 kernel 的占用。

## 小结

K3 在 bs=1 decode 路径中主要用 PDL 做两件事：计算 kernel 在 wait 前预取不依赖前驱的数据；通信 kernel 在等待远端数据时允许后继执行 prolog。producer 应在相关输出写回后 trigger，consumer 则在第一次读取依赖数据前 wait。`spin_add3` 不使用 PDL wait，它的正确性由 phase counter 维护。

PDL 不能消除通信实现本身的固定 launch 成本，重叠还受到 smem、寄存器和线程数的限制。实现时需要检查 trigger 后是否仍有写回，并留意 ptxas 对 `.nc` load 的重排。性能测试应以 e2e 指标和 NCU cycle 为主，trace 主要用于检查 PDL 链是否连续。

## 参考资料

- [SGLang PR #32541：Kimi K3 Day-0 支持](https://github.com/sgl-project/sglang/pull/32541)
- [SGLang PR #32890：独立 Kimi K3 kernel 导出](https://github.com/sgl-project/sglang/pull/32890)
- [Yifan Yang：《使用 Programmatic Dependent Launch（PDL）降低端到端延迟》](https://yang-yifan.github.io/blogs/pdl/pdl_cn.html)
- [是小肖啊：《PDL 遇上 __ldg()：Bug 还是 Feature？》](https://zhuanlan.zhihu.com/p/2067263583239533156)
- [SGLang Kimi K3 Day-0 博客](https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support)
- [《SGLang 和 Miles 为 Kimi K3 提供 Day-0 支持》中文翻译](https://mp.weixin.qq.com/s/H6fstE6NmGnG7LhgQz_lVA)
- [《SGLang Custom AllReduce v1 与 v2 实现原理详解》](https://zhuanlan.zhihu.com/p/2065205306540531895)，参见 0x7.5 节中主线 Custom AllReduce v2 的 PDL 实现。

