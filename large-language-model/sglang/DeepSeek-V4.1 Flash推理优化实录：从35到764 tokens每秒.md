# SGLang 的 DeepSeek-V4.1 Flash Day 0 优化实录：4×GB300 上 BS=1 达到 787 tokens/s

![SGLang DeepSeek-V4.1 Day 0 封面](https://files.mdnice.com/user/59/02752be9-a583-44f7-a73a-3bb76be64832.png)

## 0x0. 前言

SGLang 团队共同完成了 DeepSeek-V4.1 的 Day 0 适配和 kernel 优化。普通 decode 的 BS=1 从 35 tokens/s 提升到 203 tokens/s；接入 DSpark 后，我们继续优化 verify、MoE 和小 batch 投影。在 **4×GB300、TP4 / EP4** 上，本文选用的 GSM8K 单题达到 **BS=1 787 tokens/s**，真实 accept length 为 **5.545**。这里介绍模型结构变化，以及这些 kernel 优化是怎么做的。

![GSM8K 单题上的 DSpark kernel 优化](https://files.mdnice.com/user/59/b1833bf5-2dd0-4146-a47e-cf31a681f96d.png)

*同一条 GSM8K prompt，76 tokens 输入、固定 122 tokens 输出；各点为实测中位数。它是按较高接受长度挑选的单题，不是 GSM8K 全集的平均吞吐。*

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

先回顾普通 decode 从 35 到 203 tokens/s 的优化。下面保留这组测量结果，DSpark 部分使用文末的 GSM8K 单题负载。

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

每步的 request 索引和 scratch 也改为跨层共享，减少重复的转换和初始化。

### mHC：归约融合与计算重叠

mHC 保留四条残差流，需要计算混合系数，并通过 Sinkhorn 归一化。每层 attention 和 MoE 都会执行一次，小 batch 下开销很明显。按输入行数调整 tile，再融合统计量归约和 Sinkhorn，BS=1 从 **133.5 → 141.1 → 146.5 tokens/s**。

![Single-Pass mHC 的并行计算](https://files.mdnice.com/user/59/61bba574-cff4-4824-8cd7-629be1584cc9.png)

V4.1 的 Single-Pass mHC 使用前一个 sublayer 产生的输入混合系数。当前 attention / MoE 因此可以和当前残差的统计量计算并行，等 post 混合时再汇合。我们用多 stream 实现这一重叠，并结合 compressor / indexer 的融合与重叠，把 BS=1 从约 **152 提升到 186 tokens/s**。

## 0x3. DSpark 适配与优化

DSpark 的权重就在官方 checkpoint 中，包括三个轻量 draft block。它利用主模型后几层的 hidden state，一次为多个位置计算 logits，再用 Markov head 处理 draft token 之间的依赖，最后交给目标模型批量验证。

![DSpark 与 verify 的实际输入行数](https://files.mdnice.com/user/59/e9496fa6-9bad-490c-b3a9-3f156d65c55c.png)

本次固定 block size 5，统计真实接受结果。加上锚点，一个请求的 target verify 最多处理 6 行。普通 decode 每个请求只处理一行，原先针对 M=1 的快速路径需要适配这些小 batch。

**第一组优化集中在 verify 和 MoE。** mHC 的计算重叠接入 verify / draft，WO-A 投影直接写入后续需要的布局；candidate mask 融合有效长度判断和候选处理，减少大 buffer 的扫描。MoE router 直接输出所需布局，输入量化与 routing 重叠，最后把专家带权归约、shared expert 加法和 all-reduce 合在一起，减少中间结果写回。

**接着处理小 batch 的投影和归一化。** WO-A 使用 split-K，让更多线程块同时计算；mHC 把四条残差流的混合与 RMSNorm 融合。Draft 的多组 KV 投影复用 MXFP8 权重和 scale，替换原来的 FP8 路径。我们也把接受判断、结果整理和部分 KV 更新放进 CUDA Graph，减少图外的小算子和 CPU 发射间隔。

| 配置 | BS=1 输出速度（tokens/s） | Accept length |
|---|---:|---:|
| DSpark，优化前 | 524.68 | 5.083 |
| 加入 verify / MoE 融合与重叠 | 669.01 | 5.083 |
| 再加入小 batch 投影 / mHC 融合 | **787.04** | **5.545** |

这条输入上的速度从 **524.68 提升到 787.04 tokens/s**，提升约 **50.0%**。表里同时列出各版本实际的接受长度。

**本文这些结果还没有接入 DeepSeek 随 V4.1 发布的那批新 kernel。**

## 0x4. 如何复现

使用 SGLang 代码（https://github.com/sgl-project/sglang/tree/3b709e55c0f7599f90bdd400e1fe758c5a942cb6）和官方 checkpoint（https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277），环境为 4×GB300、TP4 / EP4。依赖版本：PyTorch 2.13.0+cu130、FlashInfer 0.6.18、Triton 3.7.1、sglang-kernel 0.4.6.post1、sgl-deep-gemm 0.1.7、CUTLASS DSL 4.6.2。

```bash
export MODEL_PATH=/path/to/DeepSeek-V4.1-Flash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH="$PWD/python" MAX_JOBS=16 \
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \
  --tp 4 --ep-size 4 --trust-remote-code \
  --mem-fraction-static 0.80 --max-total-tokens 33554432 \
  --chunked-prefill-size 4096 \
  --cuda-graph-bs-decode 1 2 4 8 16 32 64 \
  --max-running-requests 128 \
  --speculative-algorithm DSPARK --speculative-dspark-block-size 5 \
  --skip-server-warmup --reasoning-parser deepseek-v41 \
  --random-seed 42 --decode-log-interval 10 \
  --host 127.0.0.1 --port 30021
```

这次使用 GSM8K test.jsonl 的第 273 行（从 0 开始的索引为 272），是一道购物找零问题。原始问题不改写，不加 few-shot 示例。经过官方聊天编码后输入为 76 tokens，prompt 文件已保存完整的 `input_ids`，可以直接复用。

- Prompt 下载（https://raw.githubusercontent.com/BBuf/how-to-optim-algorithm-in-cuda/master/large-language-model/sglang/assets/deepseek-v41-kernel-journey/gsm8k-dspark/prompt.json）。
- 测试脚本、逐轮结果和优化版本构造方式（https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/large-language-model/sglang/assets/deepseek-v41-kernel-journey/gsm8k-dspark）。

另开终端运行：

```bash
ASSET_URL=https://raw.githubusercontent.com/BBuf/how-to-optim-algorithm-in-cuda/master/large-language-model/sglang/assets/deepseek-v41-kernel-journey/gsm8k-dspark
curl -fL "$ASSET_URL/prompt.json" -o prompt.json
curl -fL "$ASSET_URL/benchmark.py" -o benchmark.py
python -m pip install requests
python benchmark.py bench --prompt prompt.json --out result --repeat 6
```

脚本使用 `temperature=0`、`ignore_eos=True`，固定生成 **122 tokens**，与这条问题筛选时的自然输出长度一致。输入通过 `/generate` 提交，使用真实接受判断。脚本会先调用 `/freeze_gc`，每轮前清空缓存，并排除一次预热。每次启动测 6 轮；中间版本和最终配置各独立启动两次，合并全部 12 轮取中位数。

吞吐按“首个流式事件之后新增的 token 数 / 从首个事件到最后事件的耗时”计算，不包含完整 prefill。Accept length 是每轮 verify 平均提交的 token 数，包含目标模型补出的 token，block size 5 时上限为 6。输出长度会影响这个值，因此复现时需要同时保持 prompt 和输出长度一致。

同一份最终代码关闭 DSpark，使用相同输入、输出长度和计时方式，结果如下：

| 负载与指标 | DSpark 启用前 | DSpark 启用后 |
|---|---:|---:|
| BS=1，输出速度（tokens/s） | 249.47 | **787.04** |
| BS=1，accept length | | **5.545** |

## 0x5. 相关链接

- LMSYS：SGLang and Miles Add Day-0 Support for DeepSeek-V4.1（https://www.lmsys.org/blog/2026-09-10-deepseek-v41）：团队的完整 Day 0 支持介绍，涵盖推理和 RL 训练。
- SGLang DeepSeek-V4.1 部署指南（https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1）：启动配置、硬件支持和调优说明。
- SGLang DeepSeek-V4.1 代码（https://github.com/sgl-project/sglang/tree/dsv4.1）。
- Miles DeepSeek-V4.1 Flash 训练指南（https://miles.radixark.com/docs/models/deepseek/deepseek-v4-1-flash）：训练环境、checkpoint 准备和 RL 启动配置。
- Miles GitHub 仓库（https://github.com/radixark/miles）。
- DeepSeek-V4.1 技术报告（https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf）。

Kernel 实现细节可以直接看 mHC（https://github.com/sgl-project/sglang/blob/1b742acd2a49ebd7acd017032875552099d12391/python/sglang/kernels/ops/layernorm/mhc.py）、candidate mask（https://github.com/sgl-project/sglang/blob/1b742acd2a49ebd7acd017032875552099d12391/python/sglang/kernels/ops/attention/dsv4/candidate_blocks.py） 和 MoE / all-reduce（https://github.com/sgl-project/sglang/blob/1b742acd2a49ebd7acd017032875552099d12391/python/sglang/kernels/jit/csrc/distributed/all_reduce_fusion.cuh）。

## 0x6. 致谢

感谢 DeepSeek 团队开源 DeepSeek-V4.1，也感谢 SGLang 和 Miles 团队及社区参与模型适配、kernel 优化、测试和 review 的所有人。

部分 kernel 的开发也使用了 KDA 0.5 框架，感谢 Humanize（https://github.com/humanfia/humanize2）和 Kernel Design Agents（https://github.com/NVlabs/kda）提供的工具与工作流。
