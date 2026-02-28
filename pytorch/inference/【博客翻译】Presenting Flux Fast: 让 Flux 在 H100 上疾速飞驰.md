> 博客来源：https://pytorch.org/blog/presenting-flux-fast-making-flux-go-brrr-on-h100s/ By Joel Schlosser (Meta), Christian Puhrsch (Meta), and Sayak Paul (Hugging Face)June 25, 2025 。这篇博客介绍了Meta和Hugging Face团队如何使用原生PyTorch代码将Flux.1-Schnell和Flux.1-Dev模型在H100 GPU上实现约2.5倍的加速优化。主要优化技术包括：使用`torch.compile`配合`fullgraph=True`和`max-autotune`模式、合并attention计算的q/k/v投影、采用Flash Attention v3和float8量化、应用特定的Inductor调优标志，以及通过`torch.export`+AOT编译+CUDAGraphs消除框架开销。博客还特别指出了消除CPU-GPU同步点对torch.compile性能的重要性，并展示了在保持图像质量基本不变的前提下显著提升推理性能的效果。公众号翻译此文仅做科普和知识传播，侵删。

# Presenting Flux Fast: 让 Flux 在 H100 上疾速飞驰

在我们之前的文章 diffusion-fast(https://pytorch.org/blog/accelerating-generative-ai-3/) 中，我们展示了如何使用原生 PyTorch 代码将 Stable Diffusion XL (SDXL) pipeline 优化到 3 倍速度。在那时，SDXL 是图像生成领域的开源最先进 pipeline。毫不意外地，自那以后很多事情都发生了变化，可以说 Flux(https://blog.fal.ai/flux-the-largest-open-sourced-text2img-model-now-available-on-fal/) 现在是该领域最强大的开放权重模型之一。

在这篇文章中，我们很兴奋地展示如何使用（主要是）纯 PyTorch 代码和像 H100 这样的强大 GPU 在 Flux.1-Schnell 和 Flux.1-Dev 上实现了约 2.5 倍的加速。

如果您迫不及待想要开始使用代码，可以在这里找到代码仓库(https://github.com/huggingface/flux-fast/)。

## 优化概述

Diffusers 库中提供的 pipeline 尽可能地对 `torch.compile` 友好。这意味着：

- 尽可能避免计算图中断
- 尽可能避免重新编译
- 减少或最小化 CPU<->GPU 同步以降低 inductor 缓存查找开销

因此，这已经为我们提供了一个合理的起点。对于这个项目，我们采用了 diffusion-fast 项目中使用的相同基础原则，并将其应用到 FluxPipeline(https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux)。下面，我们分享我们应用的优化概述（详细信息请参考代码仓库(https://github.com/huggingface/flux-fast/)）：


- `torch.compile` 使用 `fullgraph=True` 和 `max-autotune` 模式，确保使用 CUDAGraphs
- 为 attention 计算合并 q、k、v 投影。这在量化过程中特别有用，因为它增加了维度密度，提高了计算密度
- decoder 输出使用 `torch.channels_last` 内存格式
- Flash Attention v3 (FA3)(https://pytorch.org/blog/flashattention-3/) 配合输入到 `torch.float8_e4m3fn` dtype 的（非缩放）转换
- 通过 torchao 的 `float8_dynamic_activation_float8_weight` 进行动态 float8 激活量化和 Linear 层权重量化
- 一些用于调优此模型上 Inductor 性能的标志：
    - `conv_1x1_as_mm = True`
    - `epilogue_fusion = False`
    - `coordinate_descent_tuning = True`
    - `coordinate_descent_check_all_directions = True`
- `torch.export` + Ahead-of-time Inductor (AOTI) + CUDAGraphs

这些优化中的大部分都是不言自明的，除了这两个：

- Inductor 标志。感兴趣的读者可以查看这篇博客文章(https://pytorch.org/blog/accelerating-generative-ai-3/) 以了解更多详细信息。
- 通过 AoT 编译，我们旨在消除框架开销并获得可以通过 torch.export 导出的编译二进制文件。通过 CUDAGraphs，我们希望实现 kernel 启动的优化。更多详细信息可以在这篇文章中找到(https://pytorch.org/blog/accelerating-generative-ai-4/)。

与 LLM 不同，diffusion 模型主要受计算限制，因此来自 gpt-fast(https://pytorch.org/blog/accelerating-generative-ai-2/) 的优化并不能完全适用于这里。下图显示了每个优化（从左到右递增应用）对 H100 700W GPU 上 Flux.1-Schnell 的影响：

![](https://files.mdnice.com/user/59/481aa91a-36d7-4a3c-b623-1a458162073f.png)

对于 H100 上的 Flux.1-Dev，我们有以下结果：

![](https://files.mdnice.com/user/59/6d16e759-6696-41db-9c4c-343aa5f4f47b.png)

下面是应用不同优化到 Flux.1-Dev 获得的图像视觉对比：

![](https://files.mdnice.com/user/59/3fb20c93-03d0-4221-b37d-54ec87d9e574.png)


需要注意的是，只有 FP8 量化在本质上是有损的，因此对于大多数这些优化，图像质量应该保持相同。然而，在这种情况下，我们看到 FP8 的情况下差异非常微小。

## 关于 CUDA 同步的说明

在我们的研究过程中，我们发现在去噪循环的第一步(https://github.com/huggingface/diffusers/blob/b0f7036d9af75c5df0f39d2d6353964e4c520534/src/diffusers/pipelines/flux/pipeline_flux.py#L900)，存在一个由调度器中的这一步(https://github.com/huggingface/diffusers/blob/b0f7036d9af75c5df0f39d2d6353964e4c520534/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L355)引起的 CPU<->GPU 同步点。我们可以通过在去噪循环开始时添加 `self.scheduler.set_begin_index(0)` 来消除它 (PR)(https://github.com/huggingface/diffusers/pull/11696)。

当使用 torch.compile 时，这实际上会产生更大的影响，因为 CPU 必须等待同步，然后才能执行 Dynamo 缓存查找并在 GPU 上启动指令，而这个缓存查找有点慢。因此，要点是明智的做法是对您的 pipeline 实现进行性能分析，并尽可能消除这些同步以从编译中受益。

## 结论

这篇文章介绍了使用原生 PyTorch 代码为 Hopper 架构优化 Flux 的方法。该方法试图在简洁性和性能之间取得平衡。其他类型的优化也可能是可行的（例如使用融合 MLP kernel 和融合自适应 LayerNorm kernel），但为了简洁起见，我们没有详细介绍它们。

另一个关键点是，具有 Hopper 架构的 GPU 通常成本很高。因此，为了在消费级 GPU 上提供合理的速度-内存权衡，Diffusers 库中也提供了其他（通常与 `torch.compile` 兼容的）选项。我们邀请您在这里(https://huggingface.co/docs/diffusers/main/en/optimization/memory)和这里(https://huggingface.co/docs/diffusers/main/en/optimization/fp16)查看它们。

我们邀请您在其他模型上尝试这些技术并分享结果。祝您优化愉快！






