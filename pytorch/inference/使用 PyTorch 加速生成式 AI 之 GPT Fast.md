> 来源：https://pytorch.org/blog/accelerating-generative-ai-2/ 。这篇博客展示了如何仅使用纯PyTorch来优化LLM推理性能。从基础实现的25.5 tok/s开始，文章通过一系列优化手段，包括使用torch.compile和静态kv-cache减少CPU开销、应用int8权重量化缓解内存带宽瓶颈、使用推测解码技术让小模型预测大模型输出、采用int4量化和GPTQ进一步压缩权重，以及引入张量并行在多GPU上扩展，最终将性能提升了近10倍达到244.7 tok/s。最重要的是，这些优化完全基于PyTorch原生功能实现，不需要额外依赖，整个实现仅用了不到1000行代码，同时保持了代码的简洁性和易用性。

# GPT Fast的几个问题

GPT Fast 的代码很短，然后它应用了`torch.compile`等比较先进的技术，也包括int8/int4 weight only quantize的实现。不过这里面存在几个明显的问题，这是我尝试将GPT Fast的INT8/INT4 weight only quantize代码移植到一个 DiT 模型时发现的。

- 首先，GPT Fast会加载原始的Bfloat16权重，然后进行int8/int4量化，也就是https://github.com/pytorch-labs/gpt-fast/blob/7dd5661e2adf2edd6a1042a2732dcd3a94064ad8/generate.py#L242 这里的`model = simple_quantizer.convert_for_runtime()`，这样很明显如果原始的BF6模型无法放在更小显存的卡中，那么即使是INT8/INT4量化的模型也无法正常加载，因为是在运行时修改的Linear Module。
- 在INT8 weight only的量化实现时，代码如下：https://github.com/pytorch-labs/gpt-fast/blob/7dd5661e2adf2edd6a1042a2732dcd3a94064ad8/quantize.py#L355 。`return F.linear(input, self.weight.to(dtype=input.dtype)) * self.scales` , 实际上这里的 INT8 量化还是直接回退到了BF16的实现，并没有实现gemm和dequantize的kernel融合。在 https://github.com/pytorch-labs/gpt-fast/pull/187 里面提到了`torch.ops.aten._weight_int8pack_mm`可以实现这个功能，不过我尝试运行的时候会报错。
- 运行INT4 weight only的量化时，`torch.ops.aten._weight_int4pack_mm`首先需要sm89或者sm90以上的架构，然后我分别使用了PyTorch nightly和PyTorch 2.4尝试运行，均在kernel内部触发了cuda illegal memory access的错误。
- 由于目前推理框架已经转向VLLM/SGLang等专业框架，可以把GPT Fast当成一个demo来看，它也基本没有继续维护了，不过blog和代码涉及到的技术目前都是推理框架中最主流的技术，大家可以参考。



# 使用 PyTorch 加速生成式 AI 第二部分: GPT Fast

这篇文章是关于如何使用纯原生 PyTorch 加速生成式 AI 模型的多系列博客的第二部分。我们很高兴分享一系列新发布的 PyTorch 性能特性以及实践示例,看看我们能将 PyTorch 原生性能推到多远。在第一部分中,我们展示了如何仅使用纯原生 PyTorch 将 Segment Anything(https://pytorch.org/blog/accelerating-generative-ai/) 加速超过 8 倍。在这篇博客中,我们将重点关注 LLM 优化。

过去一年,生成式 AI 的使用场景呈爆炸式增长。文本生成是一个特别受欢迎的领域,开源项目如 llama.cpp、vLLM 和 MLC-LLM 都有很多创新。

虽然这些项目性能很好,但它们在易用性方面往往需要权衡,比如需要将模型转换为特定格式或构建和部署新的依赖项。这就引出了一个问题:**仅使用纯原生 PyTorch 能让 transformer 推理运行得多快?**

正如我们在最近的 PyTorch 开发者大会(https://www.youtube.com/watch?v=IWpM_9AsC-U)上宣布的那样,PyTorch 团队从头开始编写了一个 LLM,比基线快近 10 倍,且没有精度损失,全部使用原生 PyTorch 优化。我们利用了广泛的优化,包括:

- Torch.compile: PyTorch 模型编译器
- GPU 量化: 通过降低精度运算来加速模型
- 推测解码(https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py#L76): 使用小型"草稿"模型预测大型"目标"模型的输出来加速 LLM
- 张量并行(https://github.com/pytorch-labs/gpt-fast/blob/main/tp.py): 通过在多个设备上运行来加速模型

更好的是,我们可以用**不到 1000 行原生 PyTorch 代码**来实现这一点。

如果这让你兴奋到想直接查看代码,请访问 https://github.com/pytorch-labs/gpt-fast!

![](https://files.mdnice.com/user/59/f3f62980-56b0-4c8a-87b3-172c41b34e3e.jpg)

> 注意:在所有这些基准测试中,我们将重点关注延迟(即 batch size=1)。除非另有说明,所有基准测试都在功率限制为 330W 的 A100-80GB 上运行。

# 起点 (25.5 tok/s)

让我们从一个极其基础和简单的实现开始。

![](https://files.mdnice.com/user/59/20d70458-bd55-43df-a102-dfa9a3d4ee78.png)

遗憾的是,这个性能并不理想。为什么呢?通过查看跟踪信息可以发现答案 - 它严重受到了 **CPU 开销的限制!** 这意味着我们的 CPU 无法足够快地告诉 GPU 该做什么,导致 GPU 无法被充分利用。

![](https://files.mdnice.com/user/59/91f8daab-4e99-4b3a-a3aa-03727800798a.png)

把 GPU 想象成一个拥有大量计算能力的超级工厂。然后,把 CPU 想象成一个在 GPU 之间来回传递指令的信使。记住,在大规模深度学习系统中,GPU 负责完成 100% 的工作!在这样的系统中,CPU 的唯一作用就是告诉 GPU 应该做什么工作。

![](https://files.mdnice.com/user/59/09e1896f-49eb-4187-9b19-36cf5d60897e.jpg)

所以,CPU 跑过来告诉 GPU 执行一个"加法"操作,但当 CPU 能够给 GPU 下一块工作时,GPU 早就完成了前一块工作。

尽管 GPU 需要执行数千次计算而 CPU 只需要做编排工作,但这种情况却出奇地常见!造成这种情况的原因有很多,从 CPU 可能运行的是单线程 Python 到现代 GPU 运算速度实在太快这些因素都有影响。

无论如何,我们现在处于 **CPU 开销限制** 的状态。那么,我们能做什么呢?一种方法是重写我们的实现为 C++,甚至完全抛弃框架,直接写 CUDA。或者……我们可以一次发送更多的工作给 GPU。

![](https://files.mdnice.com/user/59/ee1d10ef-1aea-4ffb-bf7f-365fb95f6dbd.jpg)

通过一次发送大量工作,我们可以让 GPU 忙个不停!尽管在训练时,这可能只是通过增加批量大小来实现,但在推理时,我们该怎么做呢?

进入 torch.compile.

# 第一步: 通过 torch.compile 和静态 kv-cache 减少 CPU 开销 (107.0 tok/s)

`torch.compile` 允许我们将更大的区域捕获到单个编译区域中,特别是当使用 `mode="reduce-overhead"` 运行时,在减少 CPU 开销方面非常有效。在这里,我们还指定 `fullgraph=True`,它会验证模型中没有"图断点"(即 `torch.compile` 无法编译的部分)。换句话说,它确保 `torch.compile` 能发挥最大潜力。

要应用它,我们只需要用它来包装一个函数(或模块)(https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py#L296)。

```python
torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)
```

然而,这里有一些细微之处,使得从将 `torch.compile` 应用于文本生成中获得显著性能提升变得有些复杂。

第一个障碍是 kv-cache。kv-cache 是一种推理时优化,它缓存了之前生成的 token 的激活值(更多细节请参见这里(https://www.dipkumar.dev/becoming-the-unbeatable/posts/gpt-kvcache/))。然而,随着我们生成更多 token,kv-cache 的"逻辑长度"会增长。这有两个问题。一个是每次缓存增长时重新分配(和复制!)缓存的成本很高。另一个是这种动态性使得减少开销变得更加困难,因为我们不再能够利用 cuda graphs 等方法。

为了解决这个问题,我们使用了一个"静态"的 kv-cache(https://github.com/pytorch-labs/gpt-fast/blob/0afae1ace441ce4c5d02ef11a72da28cf7ca4795/generate.py#L154),这意味着我们静态分配 kv-cache 的最大大小,然后在注意力计算部分中屏蔽掉未使用的值。

![](https://files.mdnice.com/user/59/5c2e0df6-da4c-419e-beec-e28a64d15f89.png)

第二个障碍是预填充阶段。Transformer 文本生成可以被视为两个阶段: 1. 预填充阶段,整个 prompt 被处理,2. 解码阶段,每个 token 被自动回归生成。

尽管解码可以完全静态化,一旦 kv-cache 被静态化,预填充阶段仍然需要显著更多的动态性,因为 prompt 长度是可变的。因此,我们实际上需要用不同的编译策略编译这两个阶段。

![](https://files.mdnice.com/user/59/56e613ef-c295-4227-906f-35fb4579971e.png)

尽管这些细节有点复杂,但实际实现并不困难(请参见 gpt-fast)!而且性能提升是显著的。

![](https://files.mdnice.com/user/59/1986feed-a278-42d6-a05e-e590d777b7a2.png)

所有这些加起来,我们的性能提高了 4 倍以上!这种性能提升在处理开销限制的工作负载时通常很常见。

# 附注: torch.compile 如何帮助?

值得分解 `torch.compile` 如何提高性能。有两个主要因素导致 `torch.compile` 的性能。

第一个因素,如上所述,是开销减少。`torch.compile` 通过多种优化方法减少了开销,但其中最有效的一种是 CUDA Graphs。尽管 `torch.compile` 在设置 "reduce-overhead" 时自动应用了这一点,节省了您需要手动编写额外工作量和代码的情况。

第二个因素,是 `torch.compile` 简单地生成更快 kernel。在解码基准测试中,`torch.compile` 实际上从头开始生成每个 kernel,包括矩阵乘法和注意力!而且更酷的是,这些内核实际上比内置替代品(CuBLAS 和 FlashAttention2)更快!

这可能听起来难以置信,考虑到编写高效矩阵乘法/注意力 kernel 有多难,以及有多少人力投入到 CuBLAS 和 FlashAttention 中。然而,这里的关键是 transformer 解码具有非常不寻常的计算属性。特别是,由于 KV-cache,对于 BS=1,每个 transformer 中的矩阵乘法实际上是矩阵向量乘法。

这意味着计算完全受内存带宽限制,因此,它们完全在编译器的范围内。事实上,当我们基准测试 `torch.compile` 的矩阵向量乘法与 CuBLAS 时,我们发现 `torch.compile` 的内核实际上要快得多!

![](https://files.mdnice.com/user/59/06bdc843-643e-4f1f-a620-4f1a7bcdab79.png)

![](https://files.mdnice.com/user/59/2f063659-5bd0-429e-9fb3-9a7fefd64fe0.png)


# 第二步: 通过 int8 权重量化缓解内存带宽瓶颈 (157.4 tok/s)

所以,鉴于我们已经从应用 `torch.compile` 中看到了巨大的性能提升,是否有可能做得更好?一种思考这个问题的方法是计算我们离理论峰值有多近。在这种情况下,最大的瓶颈是加载权重从 GPU 全局内存到寄存器的成本。换句话说,每个前向 Pass 要求我们"接触"GPU 上的每个参数。那么,我们理论上可以多快"接触"模型中的每个参数呢?

![](https://files.mdnice.com/user/59/fbd51862-059b-40f6-8637-89d4efe0a70d.jpg)

为了测量这一点,我们可以使用**模型带宽利用率(MBU)**。这测量了我们推理期间可以使用的内存带宽百分比。

计算它很简单。我们只需将模型的大小(参数数量 * 每个参数的字节数)乘以每秒可以执行的推理次数。然后,我们将这个值除以 GPU 的峰值带宽,以获得 MBU。

![](https://files.mdnice.com/user/59/f42c56c5-9002-4e7b-a99b-f49de14a0e05.png)

例如,对于我们上面的案例,我们有一个 7B 参数的模型。每个参数以 fp16 格式存储(每个参数 2 字节),我们达到了 107 tokens/s 的速度。最后,我们的 A100-80GB 有 2 TB/s 的理论内存带宽。

![](https://files.mdnice.com/user/59/03da0500-4b4a-4d64-9683-145b6fc830cc.png)

将所有这些加在一起,我们得到 **72% MBU!** 这相当不错,考虑到即使只是复制内存也难以突破 85%。

但……这确实意味着我们非常接近理论极限,而且我们显然受限于仅从内存加载权重。无论我们做什么,如果没有以某种方式改变问题陈述,我们可能只能再获得 10% 的性能。

让我们再看一遍上面的等式。我们实际上无法改变模型中的参数数量。我们无法真正改变 GPU 的内存带宽(除非花更多的钱)。但是,我们可以改变每个参数存储的字节数!

![](https://files.mdnice.com/user/59/1334bdd1-f3ee-47d7-a33f-107f722c75e2.png)

因此,我们得出了下一个技术 - int8 量化。这里的想法很简单。如果从内存加载权重是我们的主要瓶颈,为什么不直接使权重变小呢?

![](https://files.mdnice.com/user/59/aee7068e-9c3b-4ffb-b852-08bf5a1a7c6c.png)

请注意,这仅量化权重 - 计算本身仍在 bf16 中进行。这使得这种形式的量化非常容易应用,并且几乎没有精度下降。

此外,`torch.compile` 还可以轻松生成高效的 int8 量化代码。让我们再看一遍上面的基准测试,这次包括了 int8 权重量化。

![](https://files.mdnice.com/user/59/bc4c8d54-a28a-47b8-a314-f71943f30582.png)

![](https://files.mdnice.com/user/59/0fe2087b-28f0-487f-b2de-41b47012bba6.png)

从深蓝色线(torch.compile + int8)可以看出,使用 torch.compile + int8 权重量化时性能有显著提升!此外,浅蓝色线(没有 torch.compile + int8)甚至比 fp16 性能还差!这是因为为了利用 int8 量化的性能优势,我们需要将 kernel 融合。这展示了 `torch.compile` 的一个好处 - 这些 kernel 可以自动为使用者生成!

将 int8 量化应用于我们的模型(https://github.com/pytorch-labs/gpt-fast/blob/main/quantize.py#L314),我们看到了 50% 的性能提升,将我们提升到 157.4 tokens/s!

![](https://files.mdnice.com/user/59/949a1533-dabe-4802-96a3-bcae6d69d2e2.png)

# 第三步: 使用推测解码重新表述问题 (157.4 tok/s)

即使使用了量化技术,我们仍然面临另一个问题。为了生成 100 个 token,我们必须加载我们的权重 100 次。

![](https://files.mdnice.com/user/59/1470fef6-f60a-4ab5-8dec-cab8f47d3607.png)

即使权重被量化,我们仍然必须一遍又一遍地加载权重,每次生成一个 token!有没有办法绕过这个问题?

乍一看,答案似乎是否定的 - 我们的自回归生成有一个严格的序列依赖性。然而,事实证明,通过利用推测解码(https://arxiv.org/abs/2211.17192),我们能够打破这个严格的序列依赖性并获得性能提升!

![](https://files.mdnice.com/user/59/bb1d8bff-1dc4-4298-ae51-25eee1ced7e6.jpg)

想象一下,你有一个高级工程师(称为 Verity),他做出了正确的技术决策,但写代码很慢。然而,你也有一个初级工程师(称为 Drake),他有时会做出错误的技术决策,但写代码比 Verity 快得多(而且便宜得多!)。我们如何利用 Drake(初级工程师)来更快地写代码,同时确保我们仍然做出正确的技术决策?

![](https://files.mdnice.com/user/59/195d67db-d571-4fbe-90e4-82b46e1d3c00.png)

首先,Drake 通过劳动密集型的过程编写代码,并在过程中做出技术决策。接下来,我们将代码交给 Verity 审查。

![](https://files.mdnice.com/user/59/1fa7bb58-90d9-4ccf-a187-c2a9495c8d4c.png)


在审查代码时,Verity 可能会决定 Drake 的前 3 个技术决策是正确的,但最后 2 个需要重做。因此,Drake 回到起点,丢弃他的最后 2 个决策,并从那里重新开始编写代码。

值得注意的是,尽管 Verity(高级工程师)只看过一次代码,但我们能够生成 3 段与她编写的代码完全相同的验证代码!因此,假设 Verity 能够比她自己编写这 3 段代码更快地审查代码,这种方法就会优于直接编写代码。

在 Transformer 推理的上下文中,Verity 将由较大的模型扮演,该模型生成我们任务所需的输出,称为**验证模型**。同样,Drake 将由一个较小的模型扮演,该模型能够比较大的模型更快地生成文本,称为**草稿模型**。因此,我们将使用草稿模型生成 8 个 token,然后并行使用验证模型处理所有 8 个 token,丢弃不匹配的 token。

如上所述,推测解码的一个关键特性是**它不会改变输出质量**。只要生成 token 使用草稿模型 + 验证 token 所需的时间比直接生成这些 token 的时间少,我们就处于领先地位。

使用原生 PyTorch 实现这一点的优点是,这种方法实际上非常容易实现(https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py#L76)!这是整个实现,大约 50 行原生 PyTorch 代码。

![](https://files.mdnice.com/user/59/af0c3077-3f74-49f1-a156-6113e0fdd664.png)


尽管推测解码保证了我们与常规生成相比具有数学上相同的结果,但它确实具有运行时性能取决于生成文本的属性,以及草稿模型和验证模型对齐的程度。例如,当使用 CodeLlama-34B + CodeLlama-7B 运行时,我们能够获得 2x 的性能提升。另一方面,当使用 Llama-7B + TinyLlama-1B 时,我们只能获得大约 1.3x 的性能提升。

# 附注: 在 AMD 上运行

如上所述,解码中的每个 kernel 都是由 torch.compile 从头开始生成的,并转换为 OpenAI Triton。由于 AMD 有 torch.compile 后端(https://pytorch.org/blog/experience-power-pytorch-2.0/) (也有 Triton 后端),我们可以简单地通过所有上述优化……但使用 AMD GPU!使用 int8 量化,我们能够在 MI250x 的一半(即一个 GCD)上实现 102.5 tokens/s!

![](https://files.mdnice.com/user/59/a7742f24-5c3e-4314-8157-ebbdc51dd3fa.png)

# 第四步: 使用 int4 量化和 GPTQ 进一步减小权重大小 (202.1 tok/s)

当然,如果将权重从 16 位降到 8 位可以通过减少需要加载的字节数来提升速度,那么将权重降到 4 位将会带来更大的速度提升!

不幸的是,当将权重降到 4 位时,模型的精度开始成为一个更大的问题。从我们的初步评估来看,虽然使用 int8 权重量化没有明显的精度下降,但使用 int4 权重量化会导致精度下降。

![](https://files.mdnice.com/user/59/7f4d0fd7-76f1-45eb-b346-71736d5dacad.png)

有两种主要方法可以限制 int4 量化的精度下降。

第一种方法是将 scaling factor 变得更细粒度。一种思考 scaling factor 的方法是,当我们有一个量化张量表示时,它是在浮点张量(每个值都有一个 scaling factor)和整数张量(没有值有 scaling factor)之间的滑动尺度。例如,使用 int8 量化,我们每行有一个 scaling factor。然而,如果我们想要更高的精度,我们可以将 scaling factor 改为“每 32 个元素一个 scaling factor”。我们选择 32 作为 group size 以最小化精度下降,这也是社区中的常见选择。

第二种方法使用更高级的量化策略,而不仅仅是四舍五入权重。例如,GPTQ 利用示例数据来更准确地校准权重。在这种情况下,我们在仓库中基于 PyTorch 最近发布的 torch.export(https://pytorch.org/tutorials/intermediate/torch_export_tutorial.html) 原型化了一个 GPTQ 实现。

此外,我们需要融合 int4 反量化与矩阵向量乘法。在这种情况下,torch.compile 不幸地无法从头开始生成这些kernel,所以我们利用一些手写的 CUDA kernel。

这些技术需要一些额外的工作,但将它们结合起来可以获得更好的性能!

![](https://files.mdnice.com/user/59/63ed5e87-2e01-49da-827f-fde87a2c6b4f.png)

# 第五步: 将所有技术结合起来 (244.7 tok/s)

最后,我们可以将所有这些技术结合起来,以获得更好的性能!

![](https://files.mdnice.com/user/59/37ad2b21-7a47-4ba8-8551-1059b1a3740e.png)

# 第六步: 使用张量并行

到目前为止,我们一直在限制自己只在一个 GPU 上最小化延迟。然而,在许多情况下,我们可以访问多个 GPU。这允许我们进一步提高延迟!

为了直观地理解为什么这会允许我们提高延迟,让我们看看之前的 MBU 方程,特别是分母。在多个 GPU 上运行使我们能够访问更多的内存带宽,从而提高潜在的性能。

![](https://files.mdnice.com/user/59/0f73ed4c-4685-4002-8970-5ab7e39bd2cd.png)

至于选择哪种并行策略,请注意,为了减少一个示例的延迟,我们需要能够同时利用多个设备上的内存带宽。这意味着我们需要将一个 token 的处理拆分到多个设备上。换句话说,我们需要使用张量并行。

幸运的是,PyTorch 还提供了低级工具来实现张量并行,这些工具可以与 `torch.compile` 一起使用。我们还在开发更高级别的 API 来表达张量并行,请继续关注!

然而,即使没有更高级别的 API,实现张量并行实际上仍然相当容易。我们的实现只有 150 行代码(https://github.com/pytorch-labs/gpt-fast/blob/main/tp.py),并且不需要任何模型更改。

![](https://files.mdnice.com/user/59/cb084969-fb4e-416c-8d21-e655ba5e72d7.png)

我们仍然能够利用之前提到的所有优化,这些优化都可以与张量并行一起使用。将这些结合起来,我们能够在 int8 量化下以 55 tokens/s 的速度为 Llama-70B 提供服务!

![](https://files.mdnice.com/user/59/9d62629c-d025-4205-8964-ee675a4d9f5b.png)

# 结论

让我们看看我们能够实现什么。

- 简单性: 忽略量化, model.py(https://github.com/pytorch-labs/gpt-fast/blob/main/model.py) (244 LOC) + generate.py(https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py) (371 LOC) + tp.py(https://github.com/pytorch-labs/gpt-fast/blob/main/tp.py) (151 LOC) 总共 766 LOC 来实现快速推理 + 推测解码 + 张量并行。
- 性能: 使用 Llama-7B, 我们能够使用 `torch.compile` + int4 量化 + 推测解码达到 241 tok/s。使用 llama-70B, 我们还能够加入张量并行,达到 80 tok/s。这些都接近或超过了 SOTA 性能!

PyTorch 一直允许简单性、易用性和灵活性。然而,随着 `torch.compile`,我们还可以加入性能!

代码可以在这里找到: https://github.com/pytorch-labs/gpt-fast 。我们希望社区觉得它有用。我们的目标是提供一个库或框架,而不是鼓励用户复制、fork 和修改代码。

# Acknowledgements

我们想感谢开源社区的持续支持,包括:
- Lightning AI 支持 pytorch 和在 flash attention、int8 量化和 LoRA 微调方面的工作。
- GGML 推动了在设备上快速推理 LLMs 的发展。
- Andrej Karpathy 推动了简单、可解释和快速的 LLM 实现。
- MLC-LLM 推动了在异构硬件上的 4 位量化性能。


# 推测解码代码阅读

解读一下推测解码的代码  https://github.com/pytorch-labs/gpt-fast/blob/7dd5661e2adf2edd6a1042a2732dcd3a94064ad8/generate.py#L103 

```python
def speculative_decode(
    model: Transformer,  # 目标模型
    draft_model: Transformer,  # 草稿模型
    cur_token: torch.Tensor,  # 当前token
    input_pos: int,  # 输入位置
    speculate_k: int,  # 推测的token数量
    **sampling_kwargs
) -> torch.Tensor:
    # 获取设备信息
    device = cur_token.device
    # 记录原始输入位置
    orig_input_pos = torch.tensor([input_pos], dtype=torch.int64, device=cur_token.device)
    # 使用草稿模型顺序生成k个token及其概率
    draft_tokens, draft_probs = decode_n_tokens(draft_model, cur_token.view(1, -1), orig_input_pos.clone(), speculate_k, **sampling_kwargs)

    draft_tokens = torch.cat(draft_tokens)
    # 使用目标模型并行推理草稿token
    target_logits = model_forward(
        model,
        torch.cat([cur_token.view(1), draft_tokens]).view(1, -1),
        torch.arange(input_pos, input_pos + speculate_k + 1, device=cur_token.device)
    )
    # 将logits转换为概率分布
    target_probs = logits_to_probs(target_logits[0], **sampling_kwargs)
    draft_probs = torch.stack(draft_probs)
    
    # 计算接受概率
    # q: 目标模型概率, p: 草稿模型概率
    # q >= p: 总是接受草稿token
    # q < p: 以q/p的概率接受草稿token
    p = draft_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    q = target_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    accept_draft_prob = torch.minimum(torch.ones(()), q[:speculate_k]/ p)
    # 找出被拒绝的位置
    rejected_locations = (torch.rand_like(accept_draft_prob) > accept_draft_prob).nonzero()

    if rejected_locations.shape[0] == 0:  # 所有草稿token都被接受
        accept_length = speculate_k + 1
        # 采样最后一个token
        last_token = multinomial_sample_one_no_sync(target_probs[-1])
        # 将最后一个token输入草稿模型
        model_forward(
            draft_model,
            draft_tokens[-1].view(1, -1),
            orig_input_pos + speculate_k,
        )
        return torch.cat([draft_tokens, last_token])
    else:  # 存在被拒绝的token
        # 获取第一个被拒绝位置之前的token数量
        accept_length = rejected_locations[0].item()
        p = draft_probs[accept_length]
        q = target_probs[accept_length]
        # 计算新的概率分布
        new = q - p
        new = torch.where(new > 0, new, 0.0)
        new = new / new.sum()
        # 从新的概率分布中采样下一个token
        next_token = multinomial_sample_one_no_sync(new)
        return torch.cat([draft_tokens[:accept_length], next_token])



@torch.no_grad()
def generate(
    model: Transformer,  # 目标模型
    prompt: torch.Tensor,  # 输入提示
    max_new_tokens: int,  # 最大生成token数
    batch_size: int,  # 批次大小
    *,
    interactive: bool,  # 是否交互模式
    draft_model: Transformer,  # 草稿模型
    speculate_k: Optional[int] = 8,  # 推测token数量
    callback = lambda x: x,  # 回调函数
    **sampling_kwargs  # 采样相关参数
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # 检查是否使用推测解码
    is_speculative = draft_model is not None
    # 计算序列长度
    T = prompt.size(-1)  # 输入序列长度
    T_new = T + max_new_tokens  # 最终序列长度
    # 设置最大序列长度
    if interactive:
        max_seq_length = 350  # 交互模式下固定长度
    else:
        max_seq_length = min(T_new, model.config.block_size)  # 非交互模式取较小值

    # 获取设备和数据类型
    device, dtype = prompt.device, prompt.dtype
    # 如果使用推测解码,增加序列长度以容纳推测token
    max_seq_length = max_seq_length + speculate_k + 1 if is_speculative else max_seq_length
    # 设置模型缓存
    with torch.device(device):
        model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)
        if is_speculative and draft_model is not model:
            draft_model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)

    # 创建输出序列tensor
    empty = torch.empty(batch_size, T_new, dtype=dtype, device=device)
    # 将prompt复制到每个batch
    prompt = prompt.view(1, -1).repeat(batch_size, 1)
    empty[:, :T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    # 使用prefill生成第一个token
    next_token = prefill(model, prompt.view(batch_size, -1), input_pos, **sampling_kwargs).clone()
    if is_speculative:
        prefill(draft_model, prompt.view(batch_size, -1), input_pos, **sampling_kwargs)
    seq[:, T] = next_token.squeeze()

    # 设置输入位置和接受计数器
    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    accept_counts = [0] * (speculate_k + 1)

    # 主生成循环
    if is_speculative:  # 使用推测解码
        input_pos = input_pos.item()  # 转为标量便于推测解码
        while input_pos < T_new - 1:
            cur_token = next_token.view(())

            # 使用推测解码生成下一组token
            next_tokens = speculative_decode(
                model, draft_model, cur_token, input_pos, speculate_k, **sampling_kwargs
            )

            # 更新接受计数
            accept_counts[len(next_tokens) - 1] += 1
            # 计算实际添加的token数量
            num_added = min(T_new - input_pos - 1, len(next_tokens))
            # 将生成的token添加到序列中
            seq[input_pos + 1 : input_pos + num_added + 1] = next_tokens[: num_added]
            # 对每个token调用回调函数
            for i in next_tokens[: num_added,]:
                callback(i)
            # 更新位置和下一个token
            input_pos = input_pos + num_added
            next_token = next_tokens[-1]
    else:  # 不使用推测解码
        # 直接生成所有token
        generated_tokens, _ = decode_n_tokens(model, next_token.view(batch_size, -1), input_pos, max_new_tokens - 1, callback=callback, **sampling_kwargs)
        seq[:, T + 1:] = torch.cat(generated_tokens, dim=-1)

    # 返回生成序列和统计信息
    generate_stats = {
        'accept_counts': accept_counts
    }
    return seq, generate_stats
```

代码整体上比较好懂，不过这行代码不知道有什么实际作用。

```python
for i in next_tokens[: num_added,]:
    callback(i)
```

