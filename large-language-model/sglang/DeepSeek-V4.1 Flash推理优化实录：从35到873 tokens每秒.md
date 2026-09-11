# SGLang 的 DeepSeek-V4.1 Flash Day 0 优化实录：4×GB300 上 BS=1 达到 873 tokens/s

![SGLang DeepSeek-V4.1 Day 0 封面](https://files.mdnice.com/user/59/e48eedf2-117a-4a36-b6ae-466edd0c6330.png)

## 0x0. 前言

SGLang 团队共同完成了 DeepSeek-V4.1 的 Day 0 适配和 kernel 优化。在 **4×GB300** 上，普通 decode 的 BS=1 从 35 tokens/s 提升到 203 tokens/s；接入 DSpark 后，我们继续优化 verify、MoE 和小 batch 投影，再将 MoE 切换到 TP4 + padding。最终使用 **attention TP4、MoE TP4（EP1），4096 tokens 随机输入、1024 tokens 输出，模拟 accept length 固定设为 5.5**，输出速度达到 **BS=1 873.63 tokens/s**。这里介绍模型结构变化，以及这些 kernel 优化是怎么做的。

![普通 decode 与 DSpark 的完整 kernel 优化历程](https://files.mdnice.com/user/59/b27224ec-a296-4d62-ab3a-9a7ccf42c9fb.png)

*01–10 为普通 decode 的累计优化结果；11–16 使用同一份随机 4k/1k 输入，模拟 accept length 固定设为 5.5，吞吐取多轮中位数。11–15 使用 MoE EP4，第 16 点保留全部优化并切换到 MoE TP4 + padding；attention 均为 TP4。图下方列出各节点的优化手段。*

## 0x1. 架构变化与 KV cache 压缩

V4.1 Flash 的 backbone 比 V4 Flash 更大，但处理输入时激活的参数反而更少。

| 项目 | V4 Flash | V4.1 Flash |
|---|---|---|
| Backbone 参数 | 284B | 552B，另有 196B Engram 记忆参数 |
| 每 token 激活参数 | 约 13B | 输入约 8B，输出约 16B |
| 主干结构 | 43 层 decoder | 20 层 causal encoder + 20 层 decoder |
| 全局注意力 | CSA / HCA | CSA2，跨层共享 KV 和索引 |
| 全局 KV / token | 3514 bytes | 890 bytes |

数据来自官方模型配置与技术报告（https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash）。

**CED 减少了 prefill 计算量。** 后 20 层需要的全局 KV 从 encoder 最终输出生成，所以长 prompt 主要经过前 20 层。Decoder 的局部窗口通过最近 128 个 token 的 bounded replay 补齐，主干的 prefill 工作量接近减半。生成新 token 时，仍然经过完整的 40 层。

![CED 与共享 KV cache](https://files.mdnice.com/user/59/be0a830a-e594-42af-9f56-ae6a0af08d24.png)

**KV cache 压缩来自共享、池化和 FP4 存储。** 40 层中只有第 2、8、14、20 层产生全局 KV，其余层复用缓存，部分层重新计算索引。前三份缓存每两个 token 池化成一项，最后一份按 token 保存。主 KV 每项 288 bytes，indexer K 每项 68 bytes，折合每个原始 token：

$$
(288+68)\times(3/2+1)=890\ \text{bytes}
$$

这约为 V4 Flash 的 1/4。局部窗口还能用 bounded replay 重建，减少需要持久保存的状态，官方给出的 SSD 缓存需求约为原来的 1/8。890 bytes 是全局 KV 的逻辑大小；SGLang 部分路径仍使用兼容 FlashMLA 的缓存布局，实际显存占用与此不同。

CSA2 还通过分层候选筛选缩小后续 indexer 的搜索范围，最终只取 Top-512 全局位置参与 attention。Engram 则通过 n-gram 查表补充条件记忆。它们在模型里承担不同工作，也带来了新的索引、归一化和融合需求。

## 0x2. 普通 decode 的 kernel 优化

先回顾普通 decode 从 35 到 203 tokens/s 的优化。下面保留这组测量结果，DSpark 部分使用文末的随机 4k/1k 负载。

### FP8 GEMM：35 → 118 tokens/s

模型的一些 dense 权重已经是 FP8，但量化块和 scale 布局没有适配对应后端，计算进入了较慢的 fallback。我们在加载阶段整理 scale 布局，让它直接进入 Blackwell 的 MXFP8 GEMM 实现。

这一项就把 BS=1 从 **35.2 提升到 117.8 tokens/s**。新模型适配时，确认 GEMM 实际调用了哪个 kernel，往往比先调 tile 更有用。

### 小算子融合与 GEMV

逐 token decode 会反复执行大量短 kernel。RoPE、FP4 量化、压缩器池化、RMSNorm 和缓存写入之间有直接的数据依赖，把相邻操作合在一起，可以减少启动次数和中间结果读写。

| 优化 | 实现方法 | BS=1 的变化（tokens/s） |
|---|---|---:|
| RoPE + FP4 | 融合旋转、量化与反量化 | 117.8 → 133.5 |
| C2 压缩器 | 融合相邻 token 的归一化、池化和状态写入 | 148.4 → 152.1 |
| WO-A、norm、Engram gate | 单行投影使用 GEMV，小尺寸归一化和门控使用融合 kernel | 186.6 → 203.3 |

每步的 request 索引和 scratch 也改为跨层共享，减少重复的转换和初始化，BS=1 从 **146.5 提升到 148.4 tokens/s**。随后将验证过的快速路径设为默认。

### mHC：归约融合与计算重叠

mHC 保留四条残差流，需要计算混合系数，并通过 Sinkhorn 归一化。每层 attention 和 MoE 都会执行一次，小 batch 下开销很明显。按输入行数调整 tile，再融合统计量归约和 Sinkhorn，BS=1 从 **133.5 → 141.1 → 146.5 tokens/s**。

![Single-Pass mHC 的并行计算](https://files.mdnice.com/user/59/61bba574-cff4-4824-8cd7-629be1584cc9.png)

V4.1 的 Single-Pass mHC 使用前一个 sublayer 产生的输入混合系数。当前 attention / MoE 因此可以和当前残差的统计量计算并行，等 post 混合时再汇合。我们用多 stream 实现这一重叠，并结合 compressor / indexer 的融合与重叠，把 BS=1 从约 **152 提升到 186 tokens/s**。

## 0x3. DSpark 适配与优化

DSpark 的权重就在官方 checkpoint 中，包括三个轻量 draft block。它利用主模型后几层的 hidden state，一次为多个位置计算 logits，再用 Markov head 处理 draft token 之间的依赖，最后交给目标模型批量验证。

![DSpark 与 verify 的实际输入行数](https://files.mdnice.com/user/59/876a5cb4-012b-4e4f-ab73-37dc8f68c5e1.png)

本次固定 block size 5，模拟接受长度目标为 5.5。加上锚点，一个请求的 target verify 最多处理 6 行。普通 decode 每个请求只处理一行，原先针对 M=1 的快速路径需要适配这些小 batch。

**第一组优化集中在 verify 和 MoE。** mHC 的计算重叠接入 verify / draft，WO-A 投影直接写入后续需要的布局；candidate mask 融合有效长度判断和候选处理，减少大 buffer 的扫描。MoE router 直接输出所需布局，输入量化与 routing 重叠，最后把专家带权归约、shared expert 加法和 all-reduce 合在一起，减少中间结果写回。

**接着处理小 batch 的投影和归一化。** WO-A 使用 split-K，让更多线程块同时计算；mHC 把四条残差流的混合与 RMSNorm 融合。Draft 的多组 KV 投影复用 MXFP8 权重和 scale，替换原来的 FP8 路径。

**索引后处理与投影继续融合。** Top-K 之后的分数检查、无效位置过滤、KV 页地址转换合在一起；候选块选完后，直接展开成 token mask。Q 的 RoPE 与 attention buffer 写入合并，WO-A 的 split-K 归约直接完成后续 MXFP8 量化，省去中间张量读写。

**再处理 L2、L8、L14 的 verify 压缩（层号从 0 开始）。** 这三层原先要用一串小算子寻找前一个 token、处理 mask，再做池化和缓存写入。Verify 的位置连续，同一请求内直接读取前一行，只有首行需要读取 ring buffer。我们将 pair pooling、RMSNorm、RoPE、量化和主 KV 写入合成一个 kernel，随后复用 index-K 的融合写入。

**最后将 MoE 从 EP4 切到 TP4 + padding。** 专家中间维度按 TP4 切分后，每卡为 576，加载时补齐到 640，以适配 kernel。各卡计算同一批专家的不同分片，减少专家负载不均带来的等待。保留上述全部优化的同轮对照中，EP4 为 **854.64 tokens/s**，TP4 为 **873.63 tokens/s**，提升 **2.22%**；trace 中各 rank 到达 finalize 的时间差中位数从 **11.14 降到 3.40 µs**。

| 配置 | BS=1 输出速度（tokens/s） | 实测 accept length |
|---|---:|---:|
| DSpark，优化前 | 558.24 | 5.505 |
| 加入 verify / MoE 融合与重叠 | 718.75 | 5.505 |
| 再加入小 batch 投影 / mHC 融合 | 761.71 | 5.505 |
| 再融合索引后处理 / Q RoPE / WO-A 量化 | 802.38 | 5.505 |
| 再融合 C2 verify 压缩 | 853.49 | 5.520 |
| 保留全部优化，MoE 切换为 TP4 + padding | **873.63** | **5.505** |

随机 4k/1k、模拟接受长度目标为 5.5 时，输出速度从 **558.24 提升到 873.63 tokens/s**，提升约 **56.5%**。表中保留 C2 verify 融合时的 853.49 历史测量点；上面的 2.22% 使用本轮 EP4 / TP4 对照计算。

**本文这些结果还没有接入 DeepSeek 随 V4.1 发布的那批新 kernel。**

## 0x4. 如何复现：Random 4k/1k，固定模拟 accept length=5.5

本节复现图中 11–16 的 DSpark 数据：**随机输入 4096 tokens，固定输出 1024 tokens，模拟 accept length 目标为 5.5**。接受长度由服务端配置控制，各版本使用相同输入。

使用 SGLang 代码（https://github.com/BBuf/sglang/tree/835c39094ad017c2f54f8ea598002e669e6fa30d）和官方 checkpoint（https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277），环境为 4×GB300。依赖版本：PyTorch 2.13.0+cu130、FlashInfer 0.6.18、Triton 3.7.1、sglang-kernel 0.4.6.post1、sgl-deep-gemm 0.1.7、CUTLASS DSL 4.6.2。

下面是 **873.63 tokens/s 对应的 TP4 serving 命令**，在该版本的 SGLang 源码根目录执行。`--tp 4 --ep-size 1` 表示 attention 和 MoE 都使用 TP4；padding 由该版本在权重加载时完成。

```bash
export MODEL_PATH=/path/to/DeepSeek-V4.1-Flash
export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_SIMULATE_ACC_LEN=5.5
export SGLANG_SIMULATE_ACC_METHOD=match-expected
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH="$PWD/python" MAX_JOBS=16 \
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \
  --tp 4 --ep-size 1 --trust-remote-code \
  --moe-a2a-backend none --moe-runner-backend flashinfer_mxfp4 \
  --mem-fraction-static 0.80 --max-total-tokens 33554432 \
  --chunked-prefill-size 4096 \
  --cuda-graph-bs-decode 1 2 4 8 16 32 64 \
  --max-running-requests 128 \
  --speculative-algorithm DSPARK --speculative-dspark-block-size 5 \
  --skip-server-warmup --reasoning-parser deepseek-v41 \
  --random-seed 42 --decode-log-interval 10 \
  --host 127.0.0.1 --port 30021
```

复现 EP4 对照时，将 `--ep-size 1` 改为 `--ep-size 4`，其余参数不变。完整 TP4 启动脚本（https://raw.githubusercontent.com/BBuf/how-to-optim-algorithm-in-cuda/master/large-language-model/sglang/assets/deepseek-v41-kernel-journey/random-dspark/launch-tp4.sh）也可直接下载使用。

输入由固定随机种子 `42` 生成：从模型词表中排除特殊 token，再均匀抽取 4096 个 token ID。直接提交这组 ID，不套聊天模板。所有配置复用同一份输入。

- 随机输入下载（https://raw.githubusercontent.com/BBuf/how-to-optim-algorithm-in-cuda/master/large-language-model/sglang/assets/deepseek-v41-kernel-journey/random-dspark/prompt.json）。
- 输入构造脚本、测试脚本和逐轮结果（https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/large-language-model/sglang/assets/deepseek-v41-kernel-journey/random-dspark）。

另开终端运行：

```bash
ASSET_URL=https://raw.githubusercontent.com/BBuf/how-to-optim-algorithm-in-cuda/master/large-language-model/sglang/assets/deepseek-v41-kernel-journey/random-dspark
curl -fL "$ASSET_URL/prompt.json" -o prompt.json
curl -fL "$ASSET_URL/benchmark.py" -o benchmark.py
python -m pip install requests
python benchmark.py bench --prompt prompt.json --max-tokens 1024 --out result --repeat 6
```

脚本使用 `temperature=0`、`ignore_eos=True`，每轮检查实际输入为 **4096 tokens**、输出为 **1024 tokens**。输入通过 `/generate` 提交，启动后调用 `/freeze_gc`，每轮前清空缓存，并排除一次预热。每次启动测 6 轮；小 batch 投影、索引后处理 / 投影融合、C2 verify 和 TP4 配置均独立启动两次，各取 12 轮的中位数。本轮 EP4 / TP4 按 TP→EP→TP→EP 顺序交替启动，保留全部测量，吞吐测量时不开 profiler。

`match-expected` 在每轮接受 5 或 6 个 token，使期望接受长度为 **5.5**。有限轮次和最后一步截断会让统计值略有波动，表中保留实际值。这里的接受结果是模拟的，生成文本不用于评价模型回答质量；模拟模式也会关闭图内接受判断路径。

吞吐按“首个流式事件之后新增的 token 数 / 从首个事件到最后事件的耗时”计算，不包含完整 prefill。Accept length 包含目标模型补出的 token，block size 5 时上限为 6。

下面汇总 DSpark 开关及 MoE TP4 + padding 的结果。各项使用同一份代码、相同随机输入、输出长度和计时方式，**attention 均为 TP4**；表头的 EP4 / TP4 指 MoE 配置。

| 负载与指标 | DSpark 启用前 · EP4 | DSpark 启用后 · EP4 | DSpark 启用后 · TP4 + padding |
|---|---:|---:|---:|
| BS=1，输出速度（tokens/s） | 223.50 | 853.49 | **873.63** |
| BS=1，实测 accept length（模拟目标 5.5） | | 5.520 | **5.505** |

## 0x5. 相关链接

- LMSYS：SGLang and Miles Add Day-0 Support for DeepSeek-V4.1（https://www.lmsys.org/blog/2026-09-10-deepseek-v41）：团队的完整 Day 0 支持介绍，涵盖推理和 RL 训练。
- SGLang DeepSeek-V4.1 部署指南（https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1）：启动配置、硬件支持和调优说明。
- SGLang DeepSeek-V4.1 代码（https://github.com/sgl-project/sglang/tree/dsv4.1）。
- Miles DeepSeek-V4.1 Flash 训练指南（https://miles.radixark.com/docs/models/deepseek/deepseek-v4-1-flash）：训练环境、checkpoint 准备和 RL 启动配置。
- Miles GitHub 仓库（https://github.com/radixark/miles）。
- DeepSeek-V4.1 技术报告（https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf）。

Kernel 实现可以直接看 C2 verify 压缩（https://github.com/BBuf/sglang/blob/835c39094ad017c2f54f8ea598002e669e6fa30d/python/sglang/kernels/ops/attention/dsv4/c2.py）、索引后处理（https://github.com/BBuf/sglang/blob/835c39094ad017c2f54f8ea598002e669e6fa30d/python/sglang/kernels/ops/attention/dsv4/indexer_postprocess.py）、Q RoPE / store（https://github.com/BBuf/sglang/blob/835c39094ad017c2f54f8ea598002e669e6fa30d/python/sglang/kernels/ops/attention/dsv4/q_rope_store.py）和 WO-A / MXFP8（https://github.com/BBuf/sglang/blob/835c39094ad017c2f54f8ea598002e669e6fa30d/python/sglang/kernels/ops/attention/dsv4/wo_a_bf16_small_batch.py）。

## 0x6. 致谢

感谢 DeepSeek 团队开源 DeepSeek-V4.1，也感谢 SGLang 和 Miles 团队及社区参与模型适配、kernel 优化、测试和 review 的所有人。

部分 kernel 的开发也使用了 KDA 0.5 框架，感谢 Humanize（https://github.com/humanfia/humanize2）和 Kernel Design Agents（https://github.com/NVlabs/kda）提供的工具与工作流。
