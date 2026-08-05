> 博客来源：https://pytorch.org/blog/accelerating-generative-ai-3/ By Sayak Paul and Patrick von Platen (Hugging Face 🤗)January 3, 2024 。这篇博客是《加速生成式 AI 第三部分：Diffusion，Fast》，介绍了如何使用纯原生 PyTorch 技术将文本到图像的 diffusion 模型（特别是 Stable Diffusion XL）推理速度提升高达 3 倍的方法。博客详细讲解了五种主要优化技术：使用 bfloat16 降低精度运行（从 7.36 秒降至 4.63 秒）、应用 scaled_dot_product_attention (SDPA) 进行高效 attention 计算（降至 3.31 秒）、使用 torch.compile 编译 UNet 和 VAE 组件（降至 2.54 秒）、结合 q,k,v 投影矩阵进行 attention 计算以及动态 int8 量化（最终降至 2.43 秒）。这些技术都是 PyTorch 原生的，不需要依赖第三方库或 C++ 代码，通过 🤗Diffusers 库仅需几行代码即可实现，同时作者还验证了这些方法在其他 diffusion 模型（如 SSD-1B、Stable Diffusion v1-5、PixArt-Alpha）上的通用性和有效性。公众号翻译此文仅做科普和知识传播，侵删。

# 加速生成式 AI 第三部分：Diffusion，Fast

这篇文章是专注于如何使用纯原生 PyTorch 加速生成式 AI 模型的多部分博客系列的第三部分。我们很兴奋地分享一系列新发布的 PyTorch 性能特性，以及实际示例，来看看我们能将 PyTorch 原生性能推进到什么程度。在第一部分中，我们展示了如何仅使用纯原生 PyTorch 将 Segment Anything 加速超过 8 倍(https://pytorch.org/blog/accelerating-generative-ai/)。在第二部分中，我们展示了如何仅使用原生 PyTorch 优化将 Llama-7B 加速近 10 倍(https://pytorch.org/blog/accelerating-generative-ai-2/)。在这篇博客中，我们将专注于将文本到图像的 diffusion 模型提速高达 3 倍。

> 使用 PyTorch 加速生成式 AI 之 GPT Fast(https://mp.weixin.qq.com/s/wNfpeWxP4HK633RcTBkKyg) 

我们将利用一系列优化技术，包括：
- 使用 bfloat16 精度运行
- scaled_dot_product_attention (SDPA)
- torch.compile
- 结合 q,k,v 投影进行 attention 计算
- 动态 int8 量化

我们将主要关注 Stable Diffusion XL (SDXL)，展示 3 倍的延迟改进。这些技术都是 PyTorch 原生的，这意味着你不需要依赖任何第三方库或任何 C++ 代码就能利用它们。

通过 🤗Diffusers 库启用这些优化只需要几行代码。如果你已经感到兴奋并迫不及待地想要查看代码，请访问配套的代码仓库：https://github.com/huggingface/diffusion-fast。

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/3694fdac-fbae-4fb6-b4d8-dc3490d0a37c.png)

（所讨论的技术不是 SDXL 特有的，可以用来加速其他文本到图像的 diffusion 系统，如稍后所示。）

以下是一些相关主题的博客文章：

- Accelerated Diffusers with PyTorch 2.0(https://pytorch.org/blog/accelerated-diffusers-pt-20/)
- Exploring simple optimizations for SDXL (https://huggingface.co/blog/simple_sdxl_optimizations)
- Accelerated Generative Diffusion Models with PyTorch 2(https://pytorch.org/blog/accelerated-generative-diffusion-models/)

## 环境设置

我们将使用 🤗Diffusers 库(https://github.com/huggingface/diffusers)来演示这些优化及其各自的加速效果。除此之外，我们还将使用以下 PyTorch 原生库和环境：

- Torch nightly (以便受益于高效 attention 的最快内核；2.3.0.dev20231218+cu121)
- 🤗 PEFT (版本：0.7.1)
- torchao (commit SHA: 54bcd5a10d0abbe7b0c045052029257099f83fd9)
- CUDA 12.1

为了更容易重现环境，你也可以参考这个 Dockerfile(https://github.com/huggingface/diffusion-fast/blob/main/Dockerfile)。本文中提供的基准测试数据来自 400W 80GB A100 GPU（时钟频率设置为最大值）。

由于我们这里使用 A100 GPU（Ampere 架构），我们可以指定 `torch.set_float32_matmul_precision("high")` 来受益于 TF32 精度格式。

## 使用降低精度进行推理

在 Diffusers 中运行 SDXL 只需要几行代码：

```python
from diffusers import StableDiffusionXLPipeline

# 加载全精度的 pipeline 并将其模型组件放在 CUDA 上
pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0").to("cuda")

# 使用默认的 attention 处理器（非优化版本）
pipe.unet.set_default_attn_processor()
pipe.vae.set_default_attn_processor()

# 定义生成提示词
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
# 生成图像，使用 30 个推理步骤
image = pipe(prompt, num_inference_steps=30).images[0]
```

但这并不实用，因为生成 30 步的单张图像需要 **7.36 秒**。这是我们的基线，我们将尝试逐步优化。

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/1c4d44b3-a2ac-49a5-9c9e-965de4b08a82.png)

这里，我们使用全精度运行 pipeline。我们可以通过使用降低的精度（如 bfloat16）来立即减少推理时间。此外，现代 GPU 配备了专用核心，可以从降低精度中受益运行加速计算。要使用 bfloat16 精度运行 pipeline 的计算，我们只需要在初始化 pipeline 时指定数据类型：

```python
from diffusers import StableDiffusionXLPipeline
import torch

# 使用 bfloat16 精度加载 pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
	"stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16
).to("cuda")

# 使用默认的 attention 处理器（非优化版本）
pipe.unet.set_default_attn_processor()
pipe.vae.set_default_attn_processor()

# 定义生成提示词
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
# 生成图像，使用 30 个推理步骤
image = pipe(prompt, num_inference_steps=30).images[0]
```

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/2600dcc9-509d-416b-b7bc-8b00099c1e0d.png)

通过使用降低的精度，我们能够将推理延迟从 **7.36 秒**减少到 **4.63 秒**。

**关于使用 bfloat16 的一些说明**

- 使用降低的数值精度（如 float16、bfloat16）进行推理不会影响生成质量，但会显著改善延迟。
- 与 float16 相比，使用 bfloat16 数值精度的好处取决于硬件。现代 GPU 倾向于支持 bfloat16。
- 此外，在我们的实验中，我们发现与 float16 相比，bfloat16 在与量化一起使用时更加稳定。

> （我们后来在 float16 下进行了实验，发现最新版本的 torchao 不会因 float16 而产生数值问题。）

## 使用 SDPA 进行 attention 计算

默认情况下，Diffusers 在使用 PyTorch 2 时会使用 `scaled_dot_product_attention` (SDPA) 来执行 attention 相关的计算。SDPA 提供了更快、更高效的内核来运行密集的 attention 相关操作。要使用 SDPA 运行 pipeline，我们只需不设置任何 attention 处理器即可：

```python
from diffusers import StableDiffusionXLPipeline
import torch

# 使用 bfloat16 精度加载 pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
	"stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16
).to("cuda")

# 不设置 attention 处理器，默认使用 SDPA
# pipe.unet.set_default_attn_processor()  # 注释掉这行以使用 SDPA
# pipe.vae.set_default_attn_processor()   # 注释掉这行以使用 SDPA

# 定义生成提示词
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
# 生成图像，使用 30 个推理步骤
image = pipe(prompt, num_inference_steps=30).images[0]
```

SDPA 带来了很好的提升，从 **4.63 秒**减少到 **3.31 秒**。

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/77e9a3a6-9223-444d-ab7c-85342077d525.png)

## 编译 UNet 和 VAE

我们可以通过使用 `torch.compile` 要求 PyTorch 执行一些低级优化（如算子融合和使用 CUDA graphs 启动更快的内核）。对于 `StableDiffusionXLPipeline`，我们编译去噪器（UNet）和 VAE：

```python
from diffusers import StableDiffusionXLPipeline
import torch

# 使用 bfloat16 精度加载 pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16
).to("cuda")

# 编译 UNet 和 VAE 以获得最大性能
# mode="max-autotune" 使用 CUDA graphs 并针对延迟优化编译图
# fullgraph=True 确保没有图断点，发挥 torch.compile 的最大潜力
pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=True)
pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)

# 定义生成提示词
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# 第一次调用 `pipe` 会很慢（编译时间），后续调用会更快
image = pipe(prompt, num_inference_steps=30).images[0]
```

使用 SDPA attention 并编译 UNet 和 VAE 将延迟从 **3.31 秒**减少到 **2.54 秒**。

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/f26b3c24-99ac-4ad0-a5cb-d8bc6d16dafb.png)


**关于 `torch.compile` 的说明**

`torch.compile` 提供了不同的后端和模式。由于我们的目标是最大推理速度，我们选择使用 inductor 后端的 `"max-autotune"` 模式。`"max-autotune"` 使用 CUDA graphs 并专门针对延迟优化编译图。使用 CUDA graphs 大大减少了启动 GPU 操作的开销。它通过使用一种机制通过单个 CPU 操作启动多个 GPU 操作来节省时间。

指定 `fullgraph` 为 `True` 确保底层模型中没有图断点，确保 `torch.compile` 发挥最大潜力。在我们的情况下，以下编译器标志也很重要，需要明确设置：

```python
# 将 1x1 卷积作为矩阵乘法处理
torch._inductor.config.conv_1x1_as_mm = True
# 启用坐标下降调优
torch._inductor.config.coordinate_descent_tuning = True
# 禁用 epilogue 融合
torch._inductor.config.epilogue_fusion = False
# 检查坐标下降的所有方向
torch._inductor.config.coordinate_descent_check_all_directions = True
```

有关编译器标志的完整列表，请参考此文件(https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py)。

我们还在编译时将 UNet 和 VAE 的内存布局更改为"channels_last"以确保最大速度：

```python
# 使用 channels_last 内存格式以获得更好的性能
pipe.unet.to(memory_format=torch.channels_last)
pipe.vae.to(memory_format=torch.channels_last)
```

在下一节中，我们将展示如何进一步改善延迟。

## 额外的优化技术

### 在 `torch.compile` 期间避免图断点

确保底层模型/方法能够完全编译对性能至关重要（使用 `fullgraph=True` 的 `torch.compile`）。这意味着没有图断点。我们通过改变访问返回变量的方式为 UNet 和 VAE 做到了这一点。考虑以下示例：

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/56d503be-df4d-49df-acdf-cb858f3b6b7f.png)

### 编译后消除 GPU 同步

在迭代反向扩散过程中，每次去噪器预测较少噪声的潜在嵌入后，我们都会调用(https://github.com/huggingface/diffusers/blob/1d686bac8146037e97f3fd8c56e4063230f71751/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L1228) 调度器的 `step()` 方法。在 `step()` 内部，`sigmas` 变量被索引(https://github.com/huggingface/diffusers/blob/1d686bac8146037e97f3fd8c56e4063230f71751/src/diffusers/schedulers/scheduling_euler_discrete.py#L476)。如果 `sigmas` 数组放在 GPU 上，索引会导致 CPU 和 GPU 之间的通信同步。这会导致延迟，当去噪器已经被编译时，这变得更加明显。

但是如果 `sigmas` 数组始终保持在 CPU 上（参考这一行（https://github.com/huggingface/diffusers/blob/35a969d297cba69110d175ee79c59312b9f49e1e/src/diffusers/schedulers/scheduling_euler_discrete.py#L240））， 就不会发生这种同步，从而改善延迟。一般来说，任何 CPU <-> GPU 通信同步都应该没有或保持在最低限度，因为它可能影响推理延迟。

### 为 attention 操作使用组合投影

SDXL 中使用的 UNet 和 VAE 都使用类似 Transformer 的块。Transformer 块由 attention 块和前馈块组成。

在 attention 块中，输入使用三个不同的投影矩阵被投影到三个子空间 – Q、K 和 V。在朴素实现中，这些投影在输入上分别执行。但我们可以水平组合投影矩阵成一个单一矩阵，并一次性执行投影。这增加了输入投影的矩阵乘法的大小，并改善了量化的影响（接下来将讨论）。

在 Diffusers 中启用这种计算只需要一行代码：

```python
# 融合 Q、K、V 投影矩阵以提高计算效率
pipe.fuse_qkv_projections()
```

这将使 UNet 和 VAE 的 attention 操作都利用组合投影。对于Cross attention 层，我们只组合键和值矩阵。要了解更多信息，您可以参考这里的官方文档(https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline.fuse_qkv_projections)。值得注意的是，我们在内部利用(https://github.com/huggingface/diffusers/blob/35a969d297cba69110d175ee79c59312b9f49e1e/src/diffusers/models/attention_processor.py#L1356) PyTorch 的 `scaled_dot_product_attention`。

这些额外的技术将推理延迟从 **2.54 秒**改善到 **2.52 秒**。

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/886d10e2-30c5-4ef2-9c58-25d85578bca4.png)

## 动态 int8 量化

我们选择性地对 UNet 和 VAE 应用动态 int8 量化(https://docs.pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html)。这是因为量化为模型增加了额外的转换开销，希望通过更快的矩阵乘法（动态量化）来弥补。如果矩阵乘法太小，这些技术可能会降低性能。

通过实验，我们发现 UNet 和 VAE 中的某些线性层不能从动态 int8 量化中受益。您可以在这里查看过滤这些层的完整代码(https://github.com/huggingface/diffusion-fast/blob/0f169640b1db106fe6a479f78c1ed3bfaeba3386/utils/pipeline_utils.py#L16)（下面称为 `dynamic_quant_filter_fn`）。

我们利用超轻量级的纯 PyTorch 库 torchao(https://github.com/pytorch/ao) 来使用其用户友好的量化 API：

```python
from torchao.quantization import apply_dynamic_quant

# 对 UNet 应用动态量化，使用过滤函数来选择合适的层
apply_dynamic_quant(pipe.unet, dynamic_quant_filter_fn)
# 对 VAE 应用动态量化，使用过滤函数来选择合适的层
apply_dynamic_quant(pipe.vae, dynamic_quant_filter_fn)
```

由于这种量化支持仅限于线性层，我们还将合适的逐点卷积层转换为线性层以最大化收益。当使用此选项时，我们还指定以下编译器标志：

```python
# 强制融合 int8 矩阵乘法与乘法操作
torch._inductor.config.force_fuse_int_mm_with_mul = True
# 使用混合精度矩阵乘法
torch._inductor.config.use_mixed_mm = True
```

为了防止量化引起的任何数值问题，我们以 bfloat16 格式运行所有内容。

以这种方式应用量化将延迟从 **2.52 秒**改善到 **2.43 秒**。


![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/cf900a3f-024b-437a-bb1f-807433ae4385.png)


## 资源

我们欢迎您查看以下代码库来重现这些数字并将技术扩展到其他文本到图像的扩散系统：

- diffusion-fast（提供重现上述数字和图表的所有代码的仓库）https://github.com/huggingface/diffusion-fast
- torchao 库（https://github.com/pytorch/ao）
- Diffusers 库（https://github.com/huggingface/diffusers）
- PEFT 库（https://github.com/huggingface/peft）

**其他链接**

- SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis（https://huggingface.co/papers/2307.01952）
- Fast diffusion documentation（https://huggingface.co/docs/diffusers/main/en/tutorials/fast_diffusion）


## 其他 pipeline 的改进

我们将这些技术应用于其他 pipeline 来测试我们方法的通用性。以下是我们的发现：

- SSD-1B（https://huggingface.co/segmind/SSD-1B）

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/81e8a539-1a1c-4040-9234-7d8df1c3aac6.png)

- Stable Diffusion v1-5

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/739ab0b9-fd21-4231-bf15-e3e02f5d9227.png)

- PixArt-alpha/PixArt-XL-2-1024-MS（https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS）

值得注意的是，PixArt-Alpha 使用基于 Transformer 的架构作为反向扩散过程的去噪器，而不是 UNet。

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/c57813c7-a5ba-4953-abd1-05b7ddaee6ed.png)

请注意，对于 Stable Diffusion v1-5 和 PixArt-Alpha，我们没有探索应用动态 int8 量化的最佳形状组合标准。通过更好的组合可能获得更好的数字。

总的来说，我们提出的方法在不降低生成质量的情况下相对于基线提供了显著的加速。此外，我们相信这些方法应该与社区中流行的其他优化方法（如 DeepCache(https://github.com/horseee/DeepCache)、Stable Fast(https://github.com/chengzeyi/stable-fast) 等）相辅相成。

## 结论和下一步

在这篇文章中，我们介绍了一系列简单而有效的技术，可以帮助在纯 PyTorch 中改善文本到图像 Diffusion 模型的推理延迟。总结：

- 使用降低的精度来执行我们的计算
- 使用 Scaled-dot product attention 高效运行 attention 块
- 使用"max-autotune"的 `torch.compile` 来改善延迟
- 将不同的投影组合在一起来计算 attention
- 动态 int8 量化

我们相信在如何将量化应用于文本到图像扩散系统方面还有很多值得探索的地方。我们没有详尽地探索 UNet 和 VAE 中哪些层倾向于从动态量化中受益。通过针对量化的更好的层组合，可能有机会进一步加速。

除了以 bfloat16 运行之外，我们保持 SDXL 的文本编码器不变。优化它们也可能导致延迟的改善。

## 致谢

感谢 Ollin Boer Bohan(https://madebyoll.in/)，其 VAE(https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) 在整个基准测试过程中被使用，因为它在降低数值精度下数值更稳定。

感谢来自 Hugging Face 的 Hugo Larcher 在基础设施方面的帮助。


