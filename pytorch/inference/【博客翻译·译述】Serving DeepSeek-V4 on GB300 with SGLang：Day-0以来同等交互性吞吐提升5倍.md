![Serving DeepSeek-V4 on GB300 with SGLang](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/57da5a7f-a794-4919-abb9-78949d12067c.png)

> 原文：Serving DeepSeek-V4 on GB300 with SGLang: 5x Higher Throughput at the Same Interactivity Since Day-0
> 原文地址：PyTorch Blog（https://pytorch.org/blog/serving-deepseek-v4-on-gb300-with-sglang-5x-higher-throughput-at-the-same-interactivity-since-day-0/）
> 作者：SGLang Team and NVIDIA Team
> 发布时间：2026 年 6 月 23 日
>
> 说明：PyTorch 官网页脚标注 All rights reserved。本文是面向公众号阅读的中文翻译/译述稿，保留原文技术主线、图和关键信息，但不是逐字全译。文中的图片已转存到 mdnice 图床，外部链接统一写成“文本（链接）”形式，方便公众号排版。

# 【博客翻译·译述】Serving DeepSeek-V4 on GB300 with SGLang：Day-0 以来同等交互性下吞吐提升 5 倍

这篇 PyTorch Blog 讲的是 DeepSeek-V4 在 SGLang 上从 Day-0 可用，到发布后数月内持续优化的过程。核心结论很直接：DeepSeek-V4 在 Day-0 就已经可以用 SGLang 跑起来，但 Day-0 只是起点。到 2026 年 6 月，SGLang 和 NVIDIA 团队围绕 kernel、runtime、PD 分离式部署、CUDA Graph、SWA 内存管理、FP4 MoE 以及 speculative decoding 做了一系列优化和修复，让公开 benchmark 曲线明显上移。

最醒目的数字来自 SemiAnalysis InferenceX 的 GB300 disaggregated lane。配置是 DeepSeek-V4 Pro、FP4、ISL=8192、OSL=1024、dynamo-sglang。2026 年 6 月的 MTP 曲线在约 50 tok/s/user 的交互性下达到约 11200 tok/s/GPU，而 Day-0，也就是 2026 年 4 月的 no-MTP 曲线约为 2200 tok/s/GPU。同样用户可见交互性下，吞吐大约提升了 5 倍。

这里的重点不是某一个 isolated optimization，而是一整条 serving path 的成熟：MHC fusion、token-bucket prewarm、KV Compression V2、W4A4 MegaMoE、SWA budgeting 和 eviction 行为、disaggregated decode admission、DeepSeek-V4 prefill path 上的 breakable CUDA graph，以及 SGLang 和 Dynamo 侧的一批稳定性修复。

## 1. 性能结果

原文的两张图都来自公开的 SemiAnalysis InferenceX dashboard。每张图固定模型、GPU 家族、precision、workload、serving framework 和 serving mode，只比较 SGLang 自身在不同时期的演进。

### 1.1 NVIDIA GB300 Disaggregated 8K/1K

![DeepSeek V4 Pro on NVIDIA GB300](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/f214afdf-f309-4573-9424-beac490e8b77.png)

图 1：DeepSeek-V4 Pro 在 NVIDIA GB300 disaggregated lane 上的性能曲线。横轴是每用户 token 速度，纵轴是每 GPU 吞吐。

这张图里有两个明显信号。

第一，不只是单点提升。no-MTP 和 MTP 两条曲线在整个 interactivity range 内都抬高了。换句话说，优化不是只对某个并发点、某个 batch size 或某个特殊 recipe 有效。

第二，对真实线上部署更重要的是，高交互区间的曲线维持得更稳。Day-0 曲线在约 40 tok/s/user 之后掉得很快；6 月的新曲线在部署常见的高交互区域还能维持更高吞吐。原文给出的量化结果是：no-MTP 在 40 tok/s/user 处吞吐提升约 2.1 倍，MTP 在 80 tok/s/user 处吞吐提升约 2.6 倍。

no-MTP 和 MTP 曲线共用同一套 DeepSeek-V4 serving stack。只要 prefill 侧 kernel overhead 降下来、runtime compile 行为更可预测、decode 侧 SWA accounting 更准确、FP4 MoE 路径和部署 recipe 更匹配，speculative path 的 correctness 问题被修掉，两条曲线都会被一起推高。

### 1.2 NVIDIA Blackwell Ultra Aggregated 8K/1K

![DeepSeek-V4 Pro on NVIDIA Blackwell Ultra](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/83ea5813-4c16-474a-bc39-340266de4bfc.png)

图 2：DeepSeek-V4 Pro 在 NVIDIA Blackwell Ultra aggregated lane 上的性能曲线。

Blackwell Ultra aggregated lane 也吃到了同一批优化的收益。原文给出的两个代表点是：no-MTP 在 30 tok/s/user 处吞吐提升 2.91 倍，MTP 在 90 tok/s/user 处吞吐提升 2.85 倍。

另外，no-MTP 峰值吞吐相比 Day-0 曲线提升超过 6 倍。这里有一个背景：早期 no-MTP 曲线来自更保守的 fallback recipe，主要是 TP-only 执行，没有 DP attention，没有 speculative decoding，搜索空间也更窄。到后续公开提交时，Blackwell Ultra aggregated lane 已经切到更强的 recipe family：不同 concurrency 使用不同 recipe，decode worker 可持续 batch size 更高，FP4/MoE 路径也更成熟。

## 2. Day-0 时已经有什么

Day-0 support 不是“刚刚能跑”的状态。到 DeepSeek-V4 发布窗口，SGLang 已经有一条可工作的 serving path，也有 DeepSeek-V4 recipe family 提供部署指导，并且在多种平台上验证过 launch 配置。

这条 Day-0 stack 已经覆盖了 DeepSeek-V4 serving 里最关键的几类组件：

- FP4 inference。
- MoE execution，包括高吞吐场景里的 DeepGEMM MegaMoE kernel，以及低延迟场景里的 FlashInfer TRT-LLM MoE backend。
- FlashMLA 相关 attention/kernel 基础设施。
- DP、EP、TP 等 recipe variants。
- Disaggregated deployment。
- Speculative decoding support。
- Decode 侧 full CUDA graph support。

这层基础很重要。后续优化不是从“模型能不能启动”开始，而是从一个已经能在真实硬件上 serving DeepSeek-V4 的系统开始。Day-0 之后的工作，是把它变得更快、更稳定、更接近生产部署形态。

## 3. Day-0 之后改了什么

原文把后续工作分成三类：kernel 相关优化、runtime 相关优化，以及 bug fix / hardening。更工程化地看，这三类并不是彼此独立的。DeepSeek-V4 这种模型的服务边界，经常由 prefill kernel、MoE 路径、KV compression、内存预算、graph capture、speculative path correctness 共同决定。

### 3.1 Kernel 相关优化

最明显的 kernel-side 工作集中在 MHC pipeline。

PR #24775（https://github.com/sgl-project/sglang/pull/24775）重做了 DeepSeek-V4 的 MHC 路径，把 large `mhc_pre` 移到更强的 DeepGEMM-backed flow，并把 RMSNorm 融合进 MHC 路径，不再把它留成独立 stage boundary。同时，这个改动也加入了专门的 fused `hc_head` kernel。效果是减少中间 tensor traffic，也减少 scheduler 能看到的零散边界。

接着，PR #25976（https://github.com/sgl-project/sglang/pull/25976）继续加入 fused `mhc_fused_post_pre` kernel，进一步减少 MHC 路径中昂贵的 stage 边界。这个方向的本质是：不要把 MHC 当成一堆独立 point operation，而要尽量把数据搬运和 launch overhead 压下去。

KV Compression V2 是另一个关键跳跃。PR #24890（https://github.com/sgl-project/sglang/pull/24890）加入 DeepSeek-V4 的 V2 compression kernel，包括 c4、c128、online c128 compression kernel，同时更新 compressor plumbing 和 fused norm/rope V2 组件。DeepSeek-V4 不只需要快的 matmul，也需要 compression 和 indexer 这类 kernel 在高并发下保持效率。

FP4 MoE 路径也有实际提升。PR #25052（https://github.com/sgl-project/sglang/pull/25052）之前，DeepSeek-V4 的 DeepGEMM MegaMoE path 使用 W4A8 kernel：expert weight 量化到 MXFP4，但 activation path 仍量化到 MXFP8。这个 PR 加入 W4A4 MegaMoE 选项，让 activation path 也走 MXFP4。在原文描述里，这带来的精度损失可以忽略，收益主要体现在高吞吐工作区间的 MoE efficiency。

所以 kernel 侧可以概括成四句话：减少 unfused boundary，加入 DeepSeek-V4-specific compression kernel，补强 FP4 MoE kernel，并把 prefill/compression path 里那些看起来“不像 matmul”的 overhead 也认真处理掉。

### 3.2 Runtime 相关优化

Runtime 侧同样关键。对 DeepSeek-V4 来说，serving frontier 往往取决于 runtime 能否在真实请求混合下正确预算、分配、graph-capture 和回收状态。

首先是 disaggregated decode 里的 SWA 内存预算。PR #24036（https://github.com/sgl-project/sglang/pull/24036）修复 disaggregated decode SWA preallocation sizing，把 full-length accounting 和真正需要常驻 sliding-window pool 的 SWA tail 区分开来。PR #24857（https://github.com/sgl-project/sglang/pull/24857）继续改进 full-token / SWA-token budgeting、waiting/running/transfer 状态下的 reservation logic，以及 SWA tail 的 preallocation 行为。

这些改动对吞吐很直接：decode worker 在触及内存上限前，可以承载更大的 effective batch size 或更多 concurrent requests。decode worker 能持续更大 batch，GPU 就更容易被喂饱。

其次是 runtime recipe。Blackwell Ultra aggregated lane 不再使用一套 recipe 打天下，而是根据 concurrency 选择不同 recipe。GB300 disaggregated lane 也加入了一组调过的 recipe。调参对象包括 prefill-to-decode ratio、并行执行计划，例如 wide-EP 配置、full attention 和 sliding window attention 的 KV cache 内存分配、token/request size limit、tokenizer 配置等。

第三是 CUDA Graph。DeepSeek-V4 runtime path 里有不少 irregular behavior，会让 graph capture 比较难。PR #25195（https://github.com/sgl-project/sglang/pull/25195）给 DeepSeek-V4 DP attention 引入 breakable CUDA graph，后续 PR #25795（https://github.com/sgl-project/sglang/commit/c4a7d1209231e662c4447fe3d3326d8c3d1087b7）把它推进到 speculative path。这样更多 prefill path 能回到 graph-friendly 执行模式，减少 host-bound 影响，提高 prefill worker 的 GPU 利用率。decode 侧 Day-0 已经有 full CUDA graph support，所以这里新增价值主要在更容易被 irregular path 打断的 prefill 侧。

还有一些没有覆盖在 SemiAnalysis InferenceX benchmark 场景里的优化，但对真实部署仍然重要。比如 Day-0 已经支持 TP、DP、EP 等常见 parallelism 下的 PD deployment，PR #24704（https://github.com/sgl-project/sglang/pull/24704）进一步给 DeepSeek-V4 PD deployment 加入 pipeline parallelism support。DeepEP 路径也有 PR #25391（https://github.com/sgl-project/sglang/pull/25391）这类 DeepEP Waterfill 相关改动。

### 3.3 Bug fix 和 hardening 也是性能工作

这篇文章里一个很好的观点是：很多性能提升看起来像 bug fix。它们没有引入新算法，却移除了 runtime 维持更好性能曲线时遇到的正确性和稳定性问题。

在 speculative 和 disaggregated path 上，PR #23919（https://github.com/sgl-project/sglang/pull/23919）修复 PD-MTP metadata buffer hidden-size bug，PR #25805（https://github.com/sgl-project/sglang/pull/25805）修复 disaggregated decode + MTP speculation 场景下 SWA memory handling 的 double-free。类似问题可能让 speculative serving 在某个窄 recipe 下看起来没问题，但一旦做 concurrency sweep 就会崩掉。

Runtime compilation 行为也需要 hardening。PR #25810（https://github.com/sgl-project/sglang/pull/25810）加入代表性 MHC token-count bucket prewarm，让第一批真实请求不用再为 DeepSeek-V4 常见 token-count bucket 支付 lazy compile 成本。这既是性能优化，也是可靠性优化：热 shape 应该在服务关键路径之前被预热，而不是让线上请求负责探索。

还有数值正确性修复。PR #25733（https://github.com/sgl-project/sglang/pull/25733）修复 DeepSeek-V4-Pro 在 NVIDIA Blackwell 上的 NaN 问题，做法是把 `fp8_einsum` input scale 转成 `ue8m0`。原文特别提到，这个一行修复主要是 correctness fix，但它对 speculative path 有实际性能影响：Blackwell FP8 einsum scaling 不再破坏 DeepSeek-V4 MTP path 后，acceptance length 也恢复了。在一次观测中，acceptance rate 从 0.57 提升到 0.70。

SGLang 之外，Dynamo（https://github.com/ai-dynamo/dynamo）也有一个重要修复。ai-dynamo/dynamo#9080（https://github.com/ai-dynamo/dynamo/pull/9080）让 bootstrap-room generation 和选中的 prefill DP rank 对齐，从而减少 DP ranks 之间的 workload imbalance。这个修复虽然不在 SGLang 主仓库，但它属于 GB300 disaggregated lane 的公开 serving path，所以也会反映到 live frontier 上。

因此，bug-fix bucket 不是性能故事之外的附录。DeepSeek-V4 runtime 不再错算 metadata、不再 double-free SWA 状态、不再为已知 shape 付 lazy compile 成本、不再在 Blackwell 上踩数值坑、不再让 DP ranks 工作倾斜之后，serving frontier 才会既更快，也更可信。

## 4. 如何复现

原文给出了 SemiAnalysis InferenceX 结果对应的公开脚本和 recipe 入口。

GB300 disaggregated 运行使用 srt-slurm（https://github.com/NVIDIA/srt-slurm）在 Slurm-managed cluster 中启动 Dynamo frontend 和 SGLang prefill/decode servers。GB300 disaggregated performance sweep 使用的配置见 InferenceX GB300 recipe 目录（https://github.com/SemiAnalysisAI/InferenceX/tree/801d1261235f4892d4831de9de70c34f5bea7d98/benchmarks/multi_node/srt-slurm-recipes/sglang/deepseek-v4/8k1k）。

srt-slurm 的复现流程大致如下：

```bash
git clone https://github.com/NVIDIA/srt-slurm
cd srt-slurm
pip install -e .
srtctl apply -f benchmarks/multi_node/srt-slurm-recipes/sglang/deepseek-v4/8k1k/disagg-gb300-10p1d-dep4-dep32-18-c2500.yaml
```

NVIDIA Blackwell Ultra aggregated 的两个脚本如下：

- no-MTP 脚本（https://github.com/SemiAnalysisAI/InferenceX/blob/801d1261235f4892d4831de9de70c34f5bea7d98/benchmarks/single_node/fixed_seq_len/dsv4_fp4_b300_sglang.sh）
- MTP 脚本（https://github.com/SemiAnalysisAI/InferenceX/blob/801d1261235f4892d4831de9de70c34f5bea7d98/benchmarks/single_node/fixed_seq_len/dsv4_fp4_b300_sglang_mtp.sh）

如果要查看图中 NVIDIA GB300 和 NVIDIA Blackwell Ultra 曲线的原始性能数据和最新曲线，可以去 SemiAnalysis InferenceX website（https://inferencex.semianalysis.com/inference）。

## 5. Roadmap

原文最后列了后续方向。它们已经不是“证明 DeepSeek-V4 能不能 serve”，而是继续收紧 production path：

- 继续优化 DeepSeek-V4 kernel performance，包括 DeepEP v2、ragged indexer kernel，以及更多 small kernel fusion。
- 扩展 DeepSeek-V4 在 SM120/SM121 硬件以及 NVFP4 checkpoint 上的支持。
- 继续改进 cache-sensitive 和 routing-sensitive 的 DeepSeek-V4 serving path。
- 优化 DeepSeek-V4 在 agentic benchmark 上的性能。

这些待做项在 roadmap issue #23602（https://github.com/sgl-project/sglang/issues/23602）里跟踪。

## 6. 小结

我觉得这篇文章最值得注意的，不只是“5x throughput”这个数字，而是它把一个大模型 serving 系统从 Day-0 到 production-shaped 的演进拆得很清楚。

Day-0 的意义是：模型已经能在真实硬件和真实 serving framework 上跑起来，关键路径基本打通。Day-0 之后的性能增长，来自很多层面的共同收敛：

- kernel 层减少 MHC、KV compression、MoE 路径里的额外边界和非 matmul overhead。
- runtime 层把 SWA memory budgeting、decode admission、recipe dispatch、CUDA Graph 支持做得更准。
- deployment 层把 GB300 disaggregated 和 Blackwell Ultra aggregated 这两类公开 lane 都调到更强 recipe。
- hardening 层修掉 metadata、SWA memory、lazy compile、数值 NaN、DP imbalance 等问题。

这也是为什么线上 serving 的性能优化经常不像单 kernel benchmark 那么干净。一个 PR 可能只是修了一个 metadata hidden-size bug，或者把某个 input scale 换成更正确的格式，但它可能让 speculative path 的 acceptance 重新稳定，最终把整条吞吐曲线往上推。对 DeepSeek-V4 这种高复杂度模型来说，性能 frontier 本质上就是 kernel、runtime、recipe、正确性和可复现 benchmark 一起决定的。

## 7. 致谢

原文感谢了推动 DeepSeek-V4 kernel、runtime 和 serving path 工作的 SGLang 与 NVIDIA 贡献者，也感谢 SemiAnalysis InferenceX 团队持续公开 benchmark workflow 和 changelog。

SGLang Team and Community Contributors：Yuhao Yang, Cheng Wan, Baizhou Zhang, Pranjal Shankhdhar, Tom Chen, Ziyi Xu, Qiaolin Yu, Ke Bao, Liangsheng Yin, Yuwei An, Chunan Zeng, Shangming Cai, Yanbo Yang, Lianmin Zheng, Banghua Zhu, Ying Sheng。

NVIDIA Team：Yangmin Li, Weiliang Liu, Ishan Dhanani, Hao Lu, Ian Wang, Trevor Morris, Elvis Chen, Shu-Hao Yeh, Julien Lin, Akhil Goel, Nicolas Castet, Kedar Potdar, Ankur Singh, Harshika Shrivastava, Kaixi Matteo Chen, Xuting Zhou, Po-Han Huang, Triston Cao 等。
