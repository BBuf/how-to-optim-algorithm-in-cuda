> 原博客链接：https://lmsys.org/blog/2025-08-27-gpt-oss/ , 这里仅做翻译和知识传播。


# SGLang支持GPT-OSS：从Day 0支持到性能优化

> 作者：Liangsheng Yin, Ke Bao，2025年8月27日

我们很高兴地宣布SGLang的重大更新，重点关注最近发布的openai/gpt-oss-120b模型的深度性能优化和新功能。**虽然我们从Day 0就开始提供支持，但我们花了最近几周时间来增强我们的引擎，确保您获得最佳性能。**

本文重点介绍了我们最新的成就：GPT-OSS的性能显著提升，预填充阶段吞吐量提升高达**2.1倍**，解码阶段吞吐量提升高达**2.25倍**，开箱即用支持NVIDIA Blackwell & Hopper和AMD MI350 GPU，支持推测解码，以及增强的API来支持复杂的智能体应用程序——所有这些都保持了模型的高精度。

所有更改现在都在我们的主分支中可用。

### 开始使用SGLang

```bash
pip install "sglang[all]>=0.5.1.post3"
python3 -m sglang.launch_server --model-path openai/gpt-oss-120b --tp 4
```

关于环境设置和如何获得最佳性能的详细说明，请参阅我们在awesome-sglang(https://github.com/sgl-project/awesome-sglang/tree/main/gpt-oss)中的指南。

## 数据说话：全面的基准测试结果 📊

为了展示我们优化的影响，我们在各种硬件配置上对SGLang进行了基准测试。对于所有结果，重现命令可以在这里找到(https://github.com/sgl-project/sglang/tree/main/benchmark/gpt_oss)。

##### 低延迟性能（批次大小 = 1）

对于延迟敏感型应用程序，我们测量了B200和H100 GPU上的单批次解码吞吐量，展示了出色的性能。

| 硬件 / 精度 | NVIDIA B200  | NVIDIA H100  |
| ------------ | ------------ | ------------ |
| MXFP4        | 416.02 tok/s | 318.53 tok/s |
| BF16         | 315.63 tok/s | 293.12 tok/s |

<span style="color: grey; font-size: 12px;">
B200使用TP=4测试，H100使用TP=8和triton attention测试。
</span>

##### 高吞吐量性能（批次大小 = 32）

对于高吞吐量应用程序，SGLang相比我们最初的Day 0支持提供了显著的性能提升，在不同硬件上的预填充和解码都表现出色。

<!-- 灰色文字 -->

<span style="color: grey; font-size: 12px;">
AMD MI350的结果使用triton后端测试，该后端尚未完全优化，更多使用AMD AITER的优化将很快发布。
</span>

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/1480db89-7e39-42a6-bc0e-4b5ed63d3eb2.png)

## 性能深度解析 🚀

我们的性能提升来自 kernel 级别的几个关键优化：

- **Blackwell的FlashInfer kernel **：为了在Blackwell GPU上释放GPT-OSS的峰值性能，我们集成了来自FlashInfer的高度优化 kernel 。这加速了新硬件上的核心组件，包括多头注意力和专家混合（MoE）层。
- **Hopper的FlashAttention-3**：我们修改了FlashAttention-3 kernel 以支持attention sinks，为Hopper GPU上的推理提供了显著的加速。
- **kernel 融合和减少** ：我们执行了几个low-level融合以减少开销。这包括将Resdiual_RMS_Norm与all-reduce融合，将set KV Buffer操作合并到RoPE中，以及将隐藏状态 Padding融合到量化中。我们还移除了不必要的 kernel ，为某些 kernel 启用了PDL(https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization)，并减少了CPU开销以提高效率。

## 与官方报告的精度对齐 🎯

我们针对GPQA基准测试验证了我们优化的GPT-OSS实现，并确认我们的结果与官方模型卡片密切对齐，确保这些加速不会损害模型的推理能力。

| 推理难度 | SGLang | vLLM | 官方 |
| -------- | ------ | ---- | ---- |
| 低       | 65.6   | 65.3 | 67.1 |
| 中       | 72.1   | 72.4 | 73.1 |
| 高       | 79.8   | 79.4 | 80.1 |

- vLLM: https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html#accuracy-evaluation-panels
- 官方: https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf

## 推测解码支持 🦅

**推测解码**是提高LLM推理性能的关键技术。[**EAGLE3**](https://arxiv.org/abs/2503.01840)是当前最先进的推测解码方法，SGLang是第一个支持它的框架，这要归功于与EAGLE团队的密切合作。

在SGLang中，您可以轻松启动带有EAGLE3推测解码的GPT-OSS模型：

```bash
# 在Hopper上：
# - 树解码（topk > 1）和链解码（topk = 1）在FA3和Triton后端上都受支持。
python3 -m sglang.launch_server --model openai/gpt-oss-120b --speculative-algorithm EAGLE3 --speculative-draft-model-path lmsys/EAGLE3-gpt-oss-120b-bf16 --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --tp 4
python3 -m sglang.launch_server --model openai/gpt-oss-120b --speculative-algorithm EAGLE3 --speculative-draft-model-path lmsys/EAGLE3-gpt-oss-120b-bf16 --speculative-num-steps 5 --speculative-eagle-topk 4 --speculative-num-draft-tokens 8 --tp 4

# 在Blackwell上：
# - 链解码（topk = 1）在TRTLLM-MHA后端上受支持。树解码（topk > 1）正在进行中，敬请期待！
# - 树解码（topk > 1）和链解码（topk = 1）在Triton后端上都受支持。
python3 -m sglang.launch_server --model openai/gpt-oss-120b --speculative-algorithm EAGLE3 --speculative-draft lmsys/EAGLE3-gpt-oss-120b-bf16 --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --tp 4
python3 -m sglang.launch_server --model openai/gpt-oss-120b --speculative-algorithm EAGLE3 --speculative-draft lmsys/EAGLE3-gpt-oss-120b-bf16 --speculative-num-steps 5 --speculative-eagle-topk 4 --speculative-num-draft-tokens 8 --attention-backend triton --tp 4
```

对于`openai/gpt-oss-120b`模型，我们使用SpecForge(https://github.com/sgl-project/SpecForge)训练了一个EAGLE3草稿模型`lmsys/EAGLE3-gpt-oss-120b-bf16`(https://huggingface.co/lmsys/EAGLE3-gpt-oss-120b-bf16)，这是一个用于推测草稿模型训练的高效框架。我们训练的草稿模型相比NVIDIA的GPT-OSS草稿模型(https://huggingface.co/nvidia/gpt-oss-120b-Eagle3)实现了更高的平均接受长度。

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/0d319353-c3ba-4391-acb5-d0e33a47c672.png)

我们还在H200 TP4上对带有EAGLE3的`openai/gpt-oss-120b`进行了基准测试，在几个标准基准数据集上观察到了有前景的结果：

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/c841d80e-372c-423d-89bb-8d3a9365a041.png)

这实现了：
- 使用`steps=3, topk=1, num_draft_tokens=4`设置的**1.39倍**加速。
- 使用`steps=5, topk=4, num_draft_tokens=8`设置的**1.52倍**加速。

## 支持智能体应用程序 🤖

为了更好地支持智能体工作流，SGLang提供OpenAI Response API支持(https://docs.sglang.ai/basic_usage/gpt_oss.html#responses-api)和原生Chat Completion支持(https://docs.sglang.ai/advanced_features/function_calling.html#)。以下是如何使用SGLang构建简单网络搜索智能体的示例（内置工具需要`python3.12`和`gpt-oss`包，更多设置详情可以在这里找到(https://docs.sglang.ai/basic_usage/gpt_oss.html#responses-api)）。

启动服务器：

```bash
export EXA_API_KEY=YOUR_EXA_KEY
python3 -m sglang.launch_server --port 30000 --model-path openai/gpt-oss-120b --tp 4 --tool-server demo 
```

使用Response API构建网络搜索智能体：

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="EMPTY"
)
response = client.responses.create(
    model="openai/gpt-oss-120b",
    tools=[{"type": "web_search_preview"}],
    input="SGLang今天更新了什么？"
)

print(response.output_text)
```

## 下一步是什么？ 🔮

如果没有SGLang社区的集体努力，Day-0支持和后续优化都不可能实现。感谢SGLang团队、SpecForge团队、FlashInfer团队、Oracle团队、Eigen AI团队、NVIDIA团队和AMD团队一起推动这一进程！

我们将继续推动LLM推理的边界。在我们的路线图上有进一步探索SWA（滑动窗口注意力）优化、AMD AITER集成，以及推测解码的新进展，以提供更大的性能提升。

我们邀请您尝试最新版本的SGLang并分享您的反馈。感谢您成为这一旅程的重要组成部分！

