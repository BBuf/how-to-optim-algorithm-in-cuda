# Random 4k/1k · DSpark 模拟接受长度 5.5

这里保存博客使用的随机输入、测试脚本和逐轮数据。BS=1，输入4096 tokens、输出1024 tokens，DSpark block size=5，模拟接受长度目标固定为5.5。

## 随机输入

`prompt.json` 保存精确的4096个输入token ID及SHA256。输入由固定种子42生成：从官方tokenizer词表中排除声明的特殊token，按ID排序后，用Python `random.Random(42).randrange` 独立均匀抽样。没有聊天模板或重复文本填充，各配置复用同一份输入。

可以直接使用已保存的文件，也可以重新构造并核对哈希：

```bash
python -m pip install tokenizers
python prepare_prompt.py --model-path "$MODEL_PATH" --seed 42 --out prompt.json
```

`temperature=0`、`ignore_eos=True`，每轮检查输入4096、实际输出1024。输出文本是这组随机输入和模拟接受条件下生成的内容，不用于评价模型质量。

## 环境和版本

- 4×NVIDIA GB300，TP4 / EP4，ARM64；驱动 595.71.05，GPU power limit 1400 W。
- 模型：DeepSeek-V4.1-Flash（https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277）。
- SGLang：835c39094ad017c2f54f8ea598002e669e6fa30d（https://github.com/BBuf/sglang/tree/835c39094ad017c2f54f8ea598002e669e6fa30d）。
- PyTorch 2.13.0+cu130、Triton 3.7.1、FlashInfer 0.6.18、sglang-kernel 0.4.6.post1、sgl-deep-gemm 0.1.7、CUTLASS DSL 4.6.2、Transformers 5.12.1、Tokenizers 0.22.2。

五个版本对应图中的11–15，完整SHA及各轮结果保存在 `results.json`：

| 目录 | 代码版本与优化 |
|---|---|
| `base` | `3b709e55`，撤回小 batch 及 verify / MoE 优化 |
| `middle` | `3b709e55`，仅撤回小 batch 优化 |
| `small_batch` | `3b709e55`，包含小 batch 投影 / mHC 融合 |
| `index_projection` | `2e4dff1c`，增加索引后处理、Q RoPE/store、WO-A 归约与 MXFP8 量化融合 |
| `final` | `835c3909`，再增加 L2/L8/L14 的 C2 target-verify 压缩融合 |

前两项撤回的 commit 分别为 `759baff47f2193008771f0886bb7947a1d91ac64`（小 batch）和 `c36636b7da601639d4a8b19a5d008761252146a9`（verify / MoE）。`prepare_variants.sh` 在新目录中创建五个独立 worktree，保留现有 checkout。先取得固定版本：

```bash
git -C /path/to/sglang fetch https://github.com/BBuf/sglang.git 835c39094ad017c2f54f8ea598002e669e6fa30d
```

```bash
bash prepare_variants.sh /path/to/sglang /path/to/new-worktrees
```

普通 decode 对照使用 `final` 的相同代码，关闭 DSpark 和模拟接受环境变量。

## 启动和测试

进入要测量的 SGLang worktree，并设置官方 checkpoint 路径：

```bash
cd /path/to/new-worktrees/final
export MODEL_PATH=/path/to/DeepSeek-V4.1-Flash
bash /path/to/random-dspark/launch.sh
```

`launch.sh` 保存完整参数。另开终端运行：

```bash
cd /path/to/random-dspark
python -m pip install requests
python benchmark.py bench --prompt prompt.json --max-tokens 1024 --out result --repeat 6
```

命令行 `--max-tokens` 优先于JSON内的长度。脚本验证输入ID的SHA256、逐轮实际输入/输出计数，等待服务就绪，调用 `/freeze_gc`，并在每轮之前清空请求缓存。每个序列先排除一次预热，再记录6轮。`measurements.json` 包含每轮计时、接受长度和完整响应；`summary.json` 给出中位数及范围。

比较其他版本时，先停止当前服务，再进入对应 worktree 启动。测普通 decode 时使用 `DSPARK=0 bash /path/to/random-dspark/launch.sh`，其余输入和参数保持一致。

## 统计方式

吞吐为首个非空流式事件之后的新增 token 数，除以从该事件到最后一个响应事件的耗时；不包含完整 prefill，也不等同于整个请求的端到端吞吐。接受长度取服务实际返回的 `spec_accept_length`，包含目标模型补出的token，block size5时上限为6。

`results.json` 保留各轮数值、预热标记、版本差异摘要、输入及输出哈希。图表汇总时排除各序列的预热：base、middle 和普通 decode 各 6 轮；small_batch、index_projection、final 各合并两次独立启动的全部 12 轮。不删除慢样本；各配置统一使用模拟目标5.5，保留每轮实际统计值。

## 模拟接受长度

所有DSpark配置使用：

```bash
SGLANG_RAGGED_VERIFY_MODE=static
SGLANG_SIMULATE_ACC_LEN=5.5
SGLANG_SIMULATE_ACC_METHOD=match-expected
```

`match-expected` 在5和6之间抽样，期望为5.5。实际每条请求的均值会受有限步数和最后一步截断影响，因此结果文件保留实际统计值，未强行改写为5.5。模拟模式下使用图外接受/提交路径，图内接受判断融合不会启用。这里比较相同模拟条件下的运行时性能，不是自然接受率或真实任务吞吐。

`launch.sh` 自动设置上述变量；使用 `DSPARK=0` 时自动清除。`results.json` 保存启动参数、环境变量、软件版本和每轮原始数值。普通decode的accept length留空。各配置使用表中固定代码和同一checkpoint。最终配置为853.49 tokens/s，实测accept length中位数5.520；同一代码关闭DSpark为223.50 tokens/s。
