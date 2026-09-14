# SGLang 某些模型的 CUDA Graph 为什么会有上百条 decode cuda stream？

## 0x0 先看这张超难受的profiler图

最近优化 DeepSeek-V4.1 Flash，打开 torch profiler 现在是这样的：**往下一层走，主计算就换到新的 stream，整个 trace 看起来像楼梯。** 想把相邻几层的执行过程连起来看，需要一直往下翻。层数多了之后根本无法正常阅读。

下面是实际 trace，在 Firefox 的 Perfetto 里只展开了其中 12 条轨道。整份 TP0 trace 有 **143 条执行 kernel 的 stream，计入 copy 则是 144 条**。

![原始 trace：每层主计算不断切换 stream](https://files.mdnice.com/user/59/930cc0ee-f7a2-4c12-86ea-a3e241a470dd.png)

这份数据来自 PR #39370 的 `f0f2d12b5e`：4×GB300、TP4 / EP1、DSpark、BS=1、随机 4K 输入 / 1K 输出、模拟 acceptance length 5.5。后面的整理前后对比都用它，没有重新跑性能测试。

这种现象与模型的多 stream 分叉、汇合以及 CUDA Graph replay 有关，其他 LLM 走到类似路径也可能遇到。不过，**不是所有模型都必现，也不是严格“每层只新增一条”**。这里用 V4.1 和 PR #39420(https://github.com/sgl-project/sglang/pull/39420) 作为案例。

## 0x1 为什么 capture 时没那么多 stream，replay 却爆炸了？

CUDA Graph 保存的是节点和依赖关系，replay 时不保证照搬 capture 时的 stream 分配。V4.1 这里有两组比较典型的分支：mHC 的 mix-stats 与 attention / FFN 并行，MoE 的 routed MXFP8 pre-quant 与 router top-k 并行。

PR 在当前环境观察到：**汇合节点倾向沿着“先录制的父分支”所在的执行轨道继续。** 原来 side-stream 的计算录得早，汇合之后，主计算也跟着被带到了新的 stream；一层层重复，就铺开了一大堆cuda stream轨道。

这是本次实验观察到的调度行为，不是 CUDA 对所有版本的接口保证。NVIDIA 也介绍过 CUDA Graph 节点创建顺序会影响调度(https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements)。因此，不能仅凭截图中的 stream 行数，反推出 Python 每层创建了多少个 `torch.cuda.Stream()`。

这个 PR 的改法很巧：把 `_hc_mix_and_combine` 拆开，**先录 attention / FFN，再录 side-stream 的 stats，紧接着 join**。MoE 的 pre-quant 也移到 router top-k 后面录制。

![PR 的关键 diff：改变录制顺序，保留 fork 位置](https://files.mdnice.com/user/59/bea6931e-47c9-420f-9130-8b553405bcc3.png)

重点看右边：fork 的 `wait_stream` 仍在原来的位置。以 tiny 分支为例，依赖还是 `combine → {attention, stats} → hc_post`。虽然 stats 的 Python 调用写到后面了，Graph 中并没有因此新增 `attention → stats` 这条依赖，原来的 overlap 仍然有机会保留。

![PR 报告的 stream 数与 overlap 结果](https://files.mdnice.com/user/59/963143e3-8d36-4dfc-81a1-933cd9340e05.png)

上图是 **PR 的 TP4 / EP4、64K context 实验**：144 → 19 条，mix-stats 的重叠比例仍是 62%。其实可以理解为这个修改对overlap以及速度的影响几乎可以忽略，但是可以较大的缓解stream爆炸的问题。

但是SGLang支持的模型很多，不可避免的经常发生这种问题，我们能不做一个skill来直接把已经stream爆炸的torch profiler结果换成可读性非常好的profile图呢？同时我们再加一个layer层数id的滑轨那就更好看了。。


## 0x2 不改模型代码，也能把已经stream爆炸的 trace 看清楚

我做了一个 [torch-profiler-layer-track skill](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/torch-profiler-layer-track)，给 GPU 时间线加层号导航。现在它还可以把零散的 GPU 活动整理到 10 条以内的显示轨道。

同一份 trace 处理之后是这样：**6 条 GPU lane，加一条 L0–L39 导航，就能看完一次 target verify。**

![同一份 trace：六条显示轨道加层号导航](https://files.mdnice.com/user/59/22bafcf6-104b-47f6-a96a-03cd21a84067.png)

先说层号怎么来的。脚本找一个每个 target layer 恰好出现一次的 anchor kernel，按时间排序，再结合模型配置和源码确定 L0。这次使用 `_q_rope_store`，40 层、20 次 target verify，共 800 个 anchor；draft 走不同的 RoPE 路径，要单独排除。最后一层再用确认过的结束 kernel 收尾，避免把 draft 或下一轮等待也算进去。

**能被 40 整除，只能帮助检查，不能证明第一个 anchor 就是 L0。** 换模型、换 kernel 实现后，这一步需要重做。

再说压缩轨道。它按 GPU PID / device 分组，把 kernel、memcpy 等时间区间排序，用最小堆保存各条 lane 的结束时间：最早结束的 lane 已经空闲，就复用；否则新开一条。需要超过 10 条同时重叠的轨道时，直接报错，不会移动 kernel 来硬凑数量。

![轨道整理的实际代码与本次校验结果](https://files.mdnice.com/user/59/fe8724d4-fcc9-4c05-ba75-893fe56ed766.png)

原始 `name / ts / dur / args.stream` 都保留，展示用的 `pid / tid` 会重映射，原身份放进 `_compact_gpu_track`；关联的 GPU flow 端点也一起处理，CPU scope 保持原样。这次验证了 **31,416 个 kernel 全部保留，原始事件数组可以完整还原**，并实际导入 Perfetto 检查每条 lane 没有堆叠出额外行。

所以，**6 条 synthetic lane 是显示结果，不代表运行时只用了 6 个 CUDA stream，更不代表模型变快了。** PR #39420 和这个 skill 解决的是不同层面的问题。

还有一个容易踩的坑，放大看下面这张图。

![选中的 C2 kernel：显示 lane 与原始 stream 身份](https://files.mdnice.com/user/59/6c8b86c9-9ab8-4137-840e-52a452030610.png)

选中的 C2 kernel 显示在 lane 2，下面 `args.stream` 仍是 **1736**。它落在 L13 导航条下面，但结合源码和 C2 source layer 顺序核对，属于 **L14 的 compressor**：它在 L14 的 Q-store anchor 之前就执行了。

这也是为什么我把它叫“层号导航”：**anchor 到下一个 anchor 的区间，不等于完整 layer 边界，更不等于这段时间里所有 kernel 的归属。** 看 overlap、统计某层耗时时，仍然要回到真实依赖和源码。

