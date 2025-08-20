> 博客来源：https://pytorch.org/blog/accelerating-generative-ai-4/ By 
By Yejin Lee, Carole-Jean Wu, Christian Puhrsch, Joel Schlosser, Driss Guessous, Jeffrey Wan, Joe Isaacson, Can Balioglu, Juan Pino 。公众号翻译此文仅做科普和知识传播，侵删。

# 使用PyTorch加速生成式AI第四部分：Seamless M4T，快速优化

这篇文章是专注于如何使用纯原生PyTorch加速生成式AI模型的多系列博客的第四部分。要直接查看代码，请查看我们的github（seamless_communication(https://github.com/facebookresearch/seamless_communication/pull/328), fairseq2(https://github.com/facebookresearch/fairseq2/pull/272)）。我们很兴奋分享一系列新发布的PyTorch性能特性以及实际例子，看看我们能将PyTorch原生性能推进到什么程度。在第一部分中，我们展示了如何仅使用纯原生PyTorch将Segment Anything加速超过8倍(pytorch.org/blog/accelerating-generative-ai/)。在第二部分中，我们展示了如何仅使用原生PyTorch优化将Llama-7B加速近10倍(https://pytorch.org/blog/accelerating-generative-ai-2/)。在第三部分中，我们展示了如何仅使用原生PyTorch优化将text-to-image diffusion模型加速高达3倍(https://pytorch.org/blog/accelerating-generative-ai-3/)。

在这篇博客中，我们将专注于加速FAIR的Seamless M4T-v2模型，通过使用CUDA Graph和原生PyTorch优化，**在不损失精度的情况下实现text decoder模块2倍加速和vocoder模块30倍加速，最终实现端到端推理2.7倍加速**：

- [PyTorch博客翻译 Accelerating Generative AI Part III: Diffusion, Fast](https://mp.weixin.qq.com/s/Sbad9AiMQng3-NZUP5vROA)
- [【博客翻译】Presenting Flux Fast: 让 Flux 在 H100 上疾速飞驰
](https://mp.weixin.qq.com/s/KRKqZdcTjfbAmhIPYDXwTQ)
- [使用 PyTorch 加速生成式 AI 之 GPT Fast](https://mp.weixin.qq.com/s/wNfpeWxP4HK633RcTBkKyg)

## 介绍

Seamless M4T是由FAIR开发的开源基础语音/文本翻译和转录技术。Seamless M4T是一个大规模多语言和多模态机器翻译模型，最新版本(https://github.com/facebookresearch/seamless_communication) (Seamless M4T-v2)于2023年11月30日发布。Seamless M4T-v2的高层模型架构如图1所示。

![](https://files.mdnice.com/user/59/6ddaaa93-81de-48cc-9d68-b59725d80daf.png)

**图1. Seamless M4T-v2的模型架构。**

加速推理延迟对于翻译模型通过更快的跨语言通信改善用户体验至关重要。特别是，batch_size=1通常用于快速翻译，在诸如聊天机器人、语音翻译和实时字幕等应用中延迟非常重要。因此，我们对batch_size=1的推理进行了性能分析，如图2所示，以了解阿姆达尔定律的瓶颈。我们的结果表明，text decoder和vocoder是最耗时的模块，分别占推理时间的61%和23%。

![](https://files.mdnice.com/user/59/a4d7b045-32fd-44cc-924d-86e9b82f0b77.png)

**图2. Text decoder和vocoder是最耗时的模块。在A100 GPU上batch_size=1的英语-西班牙语S2ST（语音到语音文本）任务各模块推理时间的分解。**

为了更仔细地查看text decoder和vocoder的性能瓶颈，我们分析了FLEURS数据集(https://huggingface.co/datasets/google/fleurs)英语-西班牙语翻译示例第8个样本的text decoder和vocoder的GPU trace，如图3所示。结果显示**text decoder和vocoder是严重CPU bound的模块**。我们观察到由CPU开销产生的显著间隔延迟了GPU kernel的启动，导致两个模块的执行时间大幅增加。

![](https://files.mdnice.com/user/59/d48b40bc-4e30-45b6-a95d-deaa2332067e.png)

**(a) Text Decoder的CPU和GPU trace**

![](https://files.mdnice.com/user/59/bfad0a56-77fb-420f-a944-2cc7480e99f9.png)

**图3. Text Decoder和Vocoder是严重CPU bound的模块。FLEURS数据集英语-西班牙语翻译示例第8个样本的(a) Text Decoder (b) Vocoder的CPU和GPU trace。该trace通过在A100 GPU上运行batch_size=1的推理获得。**

基于真实系统性能分析结果，显示在Seamless M4T-v2中text_decoder和vocoder是严重CPU bound的模块，我们为这些模块启用了torch.compile + CUDA Graph。在这篇文章中，我们分享了在batch_size=1推理场景下为每个模块启用torch.compile + CUDA Graph所需的修改、对CUDA Graph的讨论以及下一步计划。

## Torch.compile与CUDA Graph

`torch.compile`是一个PyTorch API，允许用户将PyTorch模型编译成独立的可执行文件或脚本，通常用于通过移除不必要的开销来优化模型性能。

CUDA Graph是NVIDIA提供的一个功能，允许优化CUDA应用程序中的kernel启动。它创建CUDA kernel的执行图，可以在GPU上执行之前由驱动程序进行预处理和优化。使用CUDA Graph的主要优势是它减少了与启动单个kernel相关的开销，因为图可以作为单个单元启动，减少了主机和设备之间的API调用和数据传输次数。这可以带来显著的性能改进，特别是对于具有大量小kernel或重复相同kernel集合多次的应用程序。如果您有兴趣了解更多，请查看这篇论文，它强调了数据在加速计算中的重要作用：《数据在哪里？为什么在没有答案的情况下不能讨论CPU vs GPU性能》(https://ieeexplore.ieee.org/abstract/document/5762730)，作者是我们的Kim Hazelwood！这是NVIDIA大力投资通用GPU（GPGPU）的时候，在深度学习彻底改变计算行业之前！

然而，由于CUDA Graph操作于1）固定的内存指针，2）在编译时记录的固定张量形状，我们为CUDA Graph引入了以下改进，以便在多种输入大小之间重用，防止每次迭代生成CUDA Graph，并让CUDA Graph内部的数据在不同运行之间重用，从而在多个解码步骤中共享KV Cache。

## Text Decoder

Seamless中的Text Decoder是来自NLLB [1]的decoder，执行T2TT（文本到文本翻译）。此外，这个模块是一个CPU bound模型，GPU执行时间不够长，无法隐藏CPU开销，**这是因为自回归生成的特性需要Token的顺序处理**，这限制了在GPU上可以实现的并行性数量。基于这一观察，我们为text decoder启用了torch.compile + CUDA Graph来减少占主导地位的CPU开销，如图4所示。

![](https://files.mdnice.com/user/59/563d6c10-8a99-4fad-b0b0-635e27b5574b.png)

**图4. 启用torch.compile + CUDA Graph后Text Decoder的CPU和GPU trace。**

### 1. 更新和检索KV cache

在推理期间，text decoder有两个计算阶段：消费prompt的预填充阶段和逐个生成输出Token的增量生成阶段。给定足够高的batch size或输入长度，预填充可以并行操作足够多的Token——GPU性能是瓶颈，CPU开销不会显著影响性能。另一方面，增量Token生成总是以序列长度1执行，通常以小batch size（甚至为1）执行，例如用于交互式用例。因此，增量生成可能受到CPU速度的限制，因此是torch.compile + CUDA Graph的良好候选对象。

然而，在增量Token生成阶段，参与注意力计算的key和value的sequence_length维度在每一步都增加1，而query的序列长度始终保持为1。具体来说，key/value是通过将序列长度为1的新计算的key/value附加到迄今为止存储在KV cache中的key/value来生成的。但如上所述，CUDA Graph在编译期间记录所有张量的形状，并使用记录的形状重放。因此，根据这里的出色工作(https://fireworks.ai/blog)，已经进行了一些修改来解决这个问题。

**a)** 我们修改KV-cache处理，使其接受一个CUDA张量中写入新值的索引（即`valid_seq_pos`），而不是Python整数。

![](https://files.mdnice.com/user/59/2b68f138-2694-4b73-a0a0-adca07a3fe43.png)

**图5. 对KV cache `append`和`get`的修改。**

**b)** 我们还修改attention以使用固定形状的key和value在`max_seq_length`上工作。我们只对到当前解码步骤的序列位置（即`valid_seq_pos`）计算softmax。为了掩盖序列位置 > 当前解码步骤（即`valid_seq_pos`），我们创建一个布尔掩码张量（即`mask`），其中序列位置 > `valid_seq_pos`的位置设置为False。

![](https://files.mdnice.com/user/59/0511f0c6-e329-4028-a650-759dc2451ad0.png)

**图6. 生成`valid_seq_pos`和`mask`的辅助函数。**

重要的是要注意，这些修改会增加所需的计算量，因为我们需要计算比必要更多的序列位置（最多到`max_seq_length`）的attention。然而，尽管有这个缺点，我们的结果表明，与标准PyTorch代码相比，torch.compile + CUDA Graph仍然提供了显著的性能优势。

**c)** 由于不同的推理样本具有不同的序列长度，它也为交叉注意力层生成不同形状的输入，这些输入需要投影到key和value。因此，我们填充输入以具有静态形状，并生成填充掩码来掩盖填充的输出。

### 2. 内存指针管理

由于CUDA Graph记录内存指针以及张量的形状，重要的是使不同的推理样本正确引用记录的内存指针（例如，KV cache），以避免为每个推理样本编译CUDA Graph。然而，Seamless代码库的某些部分使不同的推理样本引用不同的内存地址，因此我们进行了修改以改善内存影响。

**e)** Seamless采用beam search作为文本解码策略。在beam search过程中，我们需要为每个增量解码步骤对所有注意力层执行KV cache重新排序，以确保每个选定的beam与相应的KV cache一起执行，如下面的代码片段所示。

![](https://files.mdnice.com/user/59/9577b375-0fbd-4686-af91-780d004c8cb3.png)

**图8. beam search解码策略的KV cache重新排序操作。**

上述代码为`cache_k`和`cache_v`分配新的内存空间并覆盖原始内存指针。因此，我们修改了KV cache重新排序，通过使用`copy_`(https://docs.pytorch.org/docs/stable/generated/torch.Tensor.copy_.html)操作符来保持每个cache的内存指针在编译期间记录的状态。

![](https://files.mdnice.com/user/59/c57d8dfb-9e04-446d-ad57-c545b26927c7.png)

**图9. 使用`copy_`操作符对KV cache进行就地更新**

**f)** 通过修改上述代码为text decoder启用torch.compile + CUDA Graph后，text decoder的开销转移到KV cache重新排序，如图10所示。KV cache重新排序重复调用index_select 96次（假设24个decoder层，其中每层由两种类型的注意力层组成，每层都有key和value的cache）。

![](https://files.mdnice.com/user/59/4b852256-31ea-4ea8-831d-d577c95f435e.png)

**图10. 启用torch.compile + CUDA Graph后Text Decoder的CPU和GPU trace。**

作为加速text decoder的一部分，我们额外对KV cache重新排序应用torch.compile以从融合kernel中受益，如图11所示。注意，我们不能在这里使用CUDA Graph（`mode='max-autotune'`），因为`copy_`操作修改输入，这违反了torch.compile中CUDA graph版本的静态输入要求。

![](https://files.mdnice.com/user/59/1db91ac8-36c1-4a99-a7fd-1e0a63ae7f66.png)

**图11. 对KV Cache重新排序应用torch.compile。**

由于为KV cache重新排序启用torch.compile，原来分别启动的gpu kernel（图12(a)）现在被融合，因此要启动的gpu kernel大幅减少（图12(b)）。

![](https://files.mdnice.com/user/59/f743b725-9f18-4647-913a-4b073a539441.png)

**(a) 启用torch.compile之KV cache重新排序的CPU和GPU trace**

![](https://files.mdnice.com/user/59/dcc077a5-219b-434c-a608-b0ebdd1cba1e.png)

**(b) 启用torch.compile后KV cache重新排序的CPU和GPU trace**

**图12. KV cache重新排序的CPU和GPU trace (a) 启用torch.compile之前和(b) 之后**

## Vocoder

Seamless中的Vocoder是一个HiFi-GAN unit-vocoder，将生成的unit转换为波形输出，其中unit是语音的表示，结合了音素和音节等不同方面，可用于生成人类可听到的声音。Vocoder是一个相对简单的模块，由Conv1d和ConvTranspose1d层组成，是一个CPU bound模块，如图3所示。基于这一观察，我们决定为vocoder启用torch.compile + CUDA Graph来减少不成比例的大CPU开销，如图10所示。但有几个问题需要修复。

![](https://files.mdnice.com/user/59/a1bbf543-63c2-47d9-bb01-e2b8212e23df.png)

**图13. 启用torch.compile + CUDA Graph后Vocoder的CPU和GPU trace。**

**a)** vocoder的输入张量形状在不同的推理样本中是不同的。但由于CUDA Graph记录张量的形状并重放它们，我们必须将输入填充到固定大小的零。由于vocoder仅由Conv1d层组成，我们不需要额外的填充掩码，用零填充就足够了。

**b)** Vocoder由用`torch.nn.utils.weight_norm`包装的conv1d层组成（参见这里(https://github.com/facebookresearch/seamless_communication/blob/main/src/seamless_communication/models/vocoder/hifigan.py#L37-L112)）。然而，直接对Vocoder应用torch.compile会导致如下所示的图中断，这导致次优的性能改进。这个图中断发生在`weight_norm`的PyTorch代码中hook处理部分内部。

```shell
[1/0_2] torch._dynamo.symbolic_convert.__graph_breaks: [DEBUG] Graph break: setattr(UserDefinedObjectVariable) <function Module.__setattr__ at 0x7fac8f483c10> from user code at:
[1/0_2] torch._dynamo.symbolic_convert.__graph_breaks: [DEBUG]   File "/mnt/fsx-home/yejinlee/yejinlee/seamless_communication/src/seamless_communication/models/vocoder/vocoder.py", line 49, in forward
[1/0_2] torch._dynamo.symbolic_convert.__graph_breaks: [DEBUG]     return self.code_generator(x, dur_prediction)  # type: ignore[no-any-return]1/0_2] torch._dynamo.symbolic_convert.__graph_breaks: [DEBUG]   File "/data/home/yejinlee/mambaforge/envs/fairseq2_12.1/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
[1/0_2] torch._dynamo.symbolic_convert.__graph_breaks: [DEBUG]     return forward_call(*args, **kwargs)
[2023-12-13 04:26:16,822] [1/0_2] torch._dynamo.symbolic_convert.__graph_breaks: [DEBUG]   File "/mnt/fsx-home/yejinlee/yejinlee/seamless_communication/src/seamless_communication/models/vocoder/codehifigan.py", line 101, in forward
[1/0_2] torch._dynamo.symbolic_convert.__graph_breaks: [DEBUG]     return super().forward(x)
[1/0_2] torch._dynamo.symbolic_convert.__graph_breaks: [DEBUG]   File "/mnt/fsx-home/yejinlee/yejinlee/seamless_communication/src/seamless_communication/models/vocoder/hifigan.py", line 185, in forward
[1/0_2] torch._dynamo.symbolic_convert.__graph_breaks: [DEBUG]     x = self.ups[i](x)
[1/0_2] torch._dynamo.symbolic_convert.__graph_breaks: [DEBUG]   File "/data/home/yejinlee/mambaforge/envs/fairseq2_12.1/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1550, in _call_impl
[1/0_2] torch._dynamo.symbolic_convert.__graph_breaks: [DEBUG]     args_result = hook(self, args)
[1/0_2] torch._dynamo.symbolic_convert.__graph_breaks: [DEBUG]   File "/data/home/yejinlee/mambaforge/envs/fairseq2_12.1/lib/python3.8/site-packages/torch/nn/utils/weight_norm.py", line 65, in __call__
[1/0_2] torch._dynamo.symbolic_convert.__graph_breaks: [DEBUG]     setattr(module, self.name, self.compute_weight(module))
[1/0_2] torch._dynamo.symbolic_convert.__graph_breaks: [DEBUG] 
```

由于层的权重在推理期间不会改变，我们不需要weight normalization。因此，我们简单地为Vocoder移除了weight normalization，如图14所示，通过利用在Seamless代码库中已提供的`remove_weight_norm`函数（这里(https://github.com/facebookresearch/seamless_communication/blob/main/src/seamless_communication/models/vocoder/hifigan.py#L198-L205)）。

![](https://files.mdnice.com/user/59/5b0e3d10-b44b-497c-b9ab-305665a6f570.png)

**图14. 为Vocoder移除`weight_norm`**

## 性能评估 + CUDA Graph的影响

图15展示了对text decoder和vocoder启用torch.compile(mode="max-autotune") + CUDA Graph的加速结果。我们实现了**text decoder的2倍加速和vocoder的30倍加速，从而实现了端到端推理时间2.7倍加速**。

![](https://files.mdnice.com/user/59/8a18fcd5-a664-4053-ba40-dccd8f96a1a9.png)

**图15. 应用torch.compile和torch.compile + CUDA Graph时text decoder和vocoder的推理时间加速**

我们还报告了使用不带CUDA Graph的torch.compile对text decoder和vocoder的加速，这被torch.compile的API支持（即`torch.compile(mode="max-autotune-no-cudagraphs")`），以识别CUDA Graph对性能的影响。在没有CUDA Graph的情况下，text decoder和vocoder的加速降低到1.17倍和18.4倍。尽管仍然相当显著，但它表明CUDA Graph的重要作用。我们得出结论，Seamless M4T-v2面临大量的CUDA kernel启动时间，特别是当我们使用小batch size（例如1）时，GPU kernel执行时间不足以摄销GPU kernel启动时间。

![](https://files.mdnice.com/user/59/73bd3dd9-9e0e-4cc3-a338-bab3d72f8c78.png)

**图16. 逐步应用torch.compile和CUDA graph的端到端推理加速。a) "Inc. Decoding": 仅对text decoder应用torch.compile b) "Inc. Decoding w/ CUDA Graph": 对text decoder应用torch.compile + CUDA Graph c) "+KV Cache Reordering": 在b)基础上额外对KV cache重新排序操作应用torch.compile d) "+Vocoder": 在c)基础上额外对vocoder应用torch.compile e) "+Vocoder w/ CUDA Graph": 在d)基础上额外对vocoder应用torch.compile + CUDA Graph。**

图16表示对模块应用带和不带CUDA Graph的torch.compile的累积效果。结果表明端到端推理加速有显著改善，证明了这些技术在优化整体延迟方面的有效性。结果，我们在batch_size=1的情况下为Seamless M4T-v2获得了**2.7倍**端到端推理加速。

## 致谢

我们感谢PyTorch团队和Seamless团队在这项工作中给予的巨大支持。







