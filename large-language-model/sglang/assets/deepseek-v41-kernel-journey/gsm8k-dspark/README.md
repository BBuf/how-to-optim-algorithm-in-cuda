> 历史 GSM8K 单题测量（122 tokens 输出）。本文当前数据已改为 Random 4k/1k、模拟接受长度 5.5，复现请使用 random-dspark（https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/large-language-model/sglang/assets/deepseek-v41-kernel-journey/random-dspark）。本目录保留原始记录。

# GSM8K 单题 DSpark benchmark

这里保存博客使用的原始问题、精确输入 token、测试脚本和逐轮数据。测试为 BS=1，使用真实 DSpark 接受判断，不使用 simulated acceptance；没有运行准确率评测。

## Prompt

原始数据来自 GSM8K（https://github.com/openai/grade-school-math/blob/master/grade_school_math/data/test.jsonl），共 1319 行，文件 SHA256：

```text
3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14
```

本次选择第 273 行，零基索引 272。在已筛选的前 400 条问题中，按较高的真实接受长度挑选这一条。没有修改问题，没有添加 few-shot 示例，也没有把参考答案放进输入。

`question.txt` 是原始问题。`prompt.json` 同时保存问题、聊天编码请求和精确的 76 个输入 token。编码使用官方 `/v1/tokenize`，开启 thinking，`reasoning_effort=high`。复现直接提交保存的 token，避免聊天模板或默认元数据变化。

输出固定为 122 tokens，`temperature=0`、`ignore_eos=True`，与筛选时这条问题的自然输出长度一致。最终配置两次独立启动、共 12 轮的 accept length 都为 5.5454545。它是接近 5.6 的单题结果，不代表 GSM8K 全集的平均接受长度或吞吐。固定到 128 / 1024 tokens 的试测接受长度更低，这些试测保存在 `selection.json`，未并入正式结果。

## 环境和版本

- 4×NVIDIA GB300，TP4 / EP4，ARM64；驱动 595.71.05，GPU power limit 1400 W。
- 模型：DeepSeek-V4.1-Flash（https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277）。
- SGLang：3b709e55c0f7599f90bdd400e1fe758c5a942cb6（https://github.com/sgl-project/sglang/tree/3b709e55c0f7599f90bdd400e1fe758c5a942cb6）。
- PyTorch 2.13.0+cu130、Triton 3.7.1、FlashInfer 0.6.18、sglang-kernel 0.4.6.post1、sgl-deep-gemm 0.1.7、CUTLASS DSL 4.6.2、Transformers 5.12.1、Tokenizers 0.22.2。

三个优化版本保留相同的模型适配与后续正确性修复，仅撤回待对比的 kernel 改动：

| 目录 | 在上述代码上的改动 |
|---|---|
| `base` | 撤回小 batch 优化，再撤回 verify / MoE 优化 |
| `middle` | 仅撤回小 batch 优化 |
| `final` | 无改动 |

两个改动的 commit 分别为 `759baff47f2193008771f0886bb7947a1d91ac64`（小 batch）和 `c36636b7da601639d4a8b19a5d008761252146a9`（verify / MoE）。`prepare_variants.sh` 会在新目录中创建三个独立 worktree，保留现有 checkout。需要先确保本地 SGLang 仓库已有这些 commit：

```bash
bash prepare_variants.sh /path/to/sglang /path/to/new-worktrees
```

普通 decode 对照使用 `final` 的相同代码，仅关闭 DSpark。

## 启动和测试

进入要测量的 SGLang worktree，并设置官方 checkpoint 路径：

```bash
cd /path/to/new-worktrees/final
export MODEL_PATH=/path/to/DeepSeek-V4.1-Flash
bash /path/to/gsm8k-dspark/launch.sh
```

`launch.sh` 保存完整参数。另开终端运行：

```bash
cd /path/to/gsm8k-dspark
python -m pip install requests
python benchmark.py bench --prompt prompt.json --out result --repeat 6
```

脚本验证输入 token 的 SHA256，等待服务就绪，调用 `/freeze_gc`，并在每轮之前清空请求缓存。每个测量序列先排除一次预热，再记录 6 轮。`measurements.json` 包含每轮计时、接受长度和完整响应；`summary.json` 给出中位数及范围。

比较其他版本时，先停止当前服务，再进入对应 worktree 启动。测普通 decode 时使用 `DSPARK=0 bash /path/to/gsm8k-dspark/launch.sh`，其余输入和参数保持一致。

## 统计方式

吞吐为首个非空流式事件之后的新增 token 数，除以从该事件到最后一个响应事件的耗时；不包含完整 prefill，也不等同于整个请求的端到端吞吐。接受长度取服务实际返回的 `spec_accept_length`，包含目标模型补出的 token，block size 5 时上限为 6。

`results.json` 保留各轮数值、预热标记、版本差异摘要、输入及输出哈希。图表汇总时排除各序列的预热：base 和普通 decode 各 6 轮；middle 因首次波动较大，增加一次独立启动，合并全部 12 轮；final 合并两次独立启动的全部 12 轮。没有删除中间版本的慢样本，也没有把各版本接受长度改成同一个值。
