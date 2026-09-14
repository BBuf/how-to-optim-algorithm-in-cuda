# SGLang 某些模型的 CUDA Graph 为什么会有上百条 decode cuda stream？

## 0x0 先看这张超难受的profiler图

最近优化 DeepSeek-V4.1 Flash，打开 torch profiler 就是下面这样。一个 decode layer 跑完，下一层就换到另一条 stream，想看相邻两层的 kernel 都得往下翻。40 层排下来，基本没法看了。

我数了一下，这份 TP0 trace 有 143 条执行 kernel 的 stream，算上 copy 是 144 条。下面在 Firefox 的 Perfetto 里只展开了其中 12 条。

![原始 trace：每层主计算不断切换 stream](https://files.mdnice.com/user/59/930cc0ee-f7a2-4c12-86ea-a3e241a470dd.png)

这份 trace 来自 PR #39370 的 `f0f2d12b5e`：4×GB300、TP4 / EP1、DSpark、BS=1、随机 4K 输入 / 1K 输出、模拟 acceptance length 5.5。后面处理 profiler 的几张图也用这份文件。

模型用了多 stream，并且在 CUDA Graph 中反复 fork、join，就可能碰到这种情况，具体还得看代码和运行环境。有的 layer 会多出好几条 stream。PR #39420(https://github.com/sgl-project/sglang/pull/39420) 正好在修 V4.1 的这个问题。

## 0x1 为什么 capture 时没那么多 stream，replay 却爆炸了？

CUDA Graph 保存的是 node 和 dependency，replay 时可能重新分配 stream。V4.1 的 mHC 会把 mix-stats 放到 side stream，和 attention / FFN overlap；MoE 的 routed MXFP8 pre-quant 也在 side stream 上，和 router top-k overlap。

PR 作者测试后发现，在他的环境里，两个分支 join 之后，后续 node 会沿着先 capture 的那个 parent node 所在的 stream 继续执行。原来的代码先 capture 了 side stream 上的 kernel，join 后主计算也换到了那条 stream。每层都来这么几次，最后就有了一百多条。

这个规律是作者在当前环境测出来的，CUDA 没有保证所有版本都这样调度。NVIDIA 也写过 node 创建顺序对调度的影响：https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements 。图里的一百多条 stream，也就不能直接理解成 Python 创建了一百多个 `torch.cuda.Stream()`。

PR 把 `_hc_mix_and_combine` 拆成了两个函数，先 capture attention / FFN，再 capture side stream 上的 stats，然后马上 join。MoE 那边也一样，把 pre-quant 移到 router top-k 后面 capture。

![PR 中调整 capture 顺序的代码](https://files.mdnice.com/user/59/bea6931e-47c9-420f-9130-8b553405bcc3.png)

图右边的 `wait_stream` 没挪位置。tiny 分支仍然是 `combine → {attention, stats} → hc_post`，stats 只需要等 combine。把它的 Python 调用写到 attention 后面，不会让 Graph 中的 stats 也等 attention 跑完，所以 replay 时两者仍然可以 overlap。

![PR 报告的 stream 数与 overlap 结果](https://files.mdnice.com/user/59/963143e3-8d36-4dfc-81a1-933cd9340e05.png)

PR 作者用 TP4 / EP4、64K context 测到 stream 从 144 条降到了 19 条，mix-stats 的 overlap 比例还是 62%。这组测试里 overlap 保住了，单次 cycle latency 是 6.36 → 6.41 ms，作者认为是测量波动。至少从这份结果看，stream 少了很多，速度没什么变化。

SGLang 支持的模型多，其他模型也可能碰到类似问题。我就想先把已有的 profiler 文件处理一下，少显示几行，再加上 layer id，找 kernel 的时候方便一点。

## 0x2 用 skill 处理一下 profiler 文件

之前做的 [torch-profiler-layer-track skill](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/torch-profiler-layer-track) 已经能给 timeline 加 layer id，这次又加了个功能，把 GPU kernel 和 copy 排到最多 10 行里显示。

处理完是下面这样。这份 trace 只需要 6 行，上面从 L0 标到 L39，一屏就能看完一次 target verify。

![处理后的 trace，上面一行是 layer id](https://files.mdnice.com/user/59/22bafcf6-104b-47f6-a96a-03cd21a84067.png)

layer id 是靠 anchor kernel 标出来的。找一个每个 target layer 恰好执行一次的 kernel，按时间排好，再对照模型配置和源码确认哪一个是 L0。这次用的是 `_q_rope_store`，40 层跑了 20 次 target verify，共 800 个。draft 用的是另一套 RoPE，统计时要排除。L39 的结束位置也单独找了对应的 kernel，免得这条 layer 标记一直画到 draft 里面。

L0 要查源码确认，光看 kernel 数量能被 40 整除还不够。换了模型或者 kernel 实现，anchor 也要重新选。

减少显示行数用的是 min-heap。脚本先按 GPU PID / device 分组，把 kernel、memcpy 按开始时间排序，再用 heap 记录每一行最后一个 event 的结束时间。如果最早空出来的那一行能放下当前 event，就放进去，否则另开一行。超过 10 行就报错，同时发生的 event 必须分开放，不能为了少几行把 overlap 藏掉。

![min-heap 的实现和这次处理后的检查结果](https://files.mdnice.com/user/59/fe8724d4-fcc9-4c05-ba75-893fe56ed766.png)

文件里改的是显示用的 `pid / tid`，原来的值记在 `_compact_gpu_track` 里。`name / ts / dur / args.stream` 都没改，GPU flow 连线跟着换位置，CPU scope 不动。我也把处理后的 event 按记录恢复回去，和原文件逐个比较，结果一致，31,416 个 kernel 一个没少。重新导入 Perfetto 后，每一行也没有因为 event overlap 再撑出额外的行。

图里的 GPU lane 是脚本重新排的显示行。模型运行时用了多少 stream、每个 kernel 跑了多久，都和处理前一样。

下面这张放大图还能看到原来的 stream id。

![C2 kernel 的原始 stream id 仍然保存在 args 中](https://files.mdnice.com/user/59/6c8b86c9-9ab8-4137-840e-52a452030610.png)

选中的 C2 kernel 被排到了 lane 2，`args.stream` 还是 1736。它虽然在 L13 的标记下面，查代码和 C2 source layer 的顺序后，对应的却是 L14 的 compressor，因为它在 L14 的 Q-store 之前就跑了。

脚本从 Q-store 开始画 L14，所以前面的 compressor 落到了 L13 那段里。layer id 用来找位置很方便，真要统计完整的 layer latency，还得把 anchor 前后的 kernel 对照代码查清楚。
