> 文章来源：https://pytorch.org/blog/pytorch-native-architecture-optimization/

# PyTorch 原生架构优化：torchao

我们很高兴正式推出 torchao，这是一个 PyTorch 原生库，通过利用低位数据类型、量化和稀疏性使模型更快更小。torchao(https://github.com/pytorch/ao) 是一个易于使用的工具包，其中包含了(主要)用易读的 PyTorch 代码编写的技术，涵盖了推理和训练。本文将帮助您选择适合您工作负载的技术。

我们在流行的 GenAI 模型(如 LLama 3 和 Diffusion 模型)上对我们的技术进行了基准测试，发现精度下降很小。除非另有说明，基准测试都是在 A100 80GB GPU 上使用 bf16 运行的。

我们对 llama 3 的主要指标是：

- 使用 autoquant 和 int4 权重量化以及 hqq，Llama 3 8B 推理速度提升 97%
- 在 128K 上下文长度下使用量化 KV 缓存，Llama 3.1 8B 推理的峰值 VRAM 减少 73%
- 在 H100 上使用 float8 训练，Llama 3 70B 预训练速度提升 50%
- 使用 4 位量化优化器，Llama 3 8B 的峰值 VRAM 减少 30%

我们对扩散模型推理的主要指标是：

- 在 H100 上的 flux1.dev 使用 float8 动态量化推理和 float8 row-wise scaling，速度提升 53%
- 使用 int8 动态量化，CogVideoX 的模型 VRAM 减少 50%

下面我们将介绍 torchao 中可用于推理和训练的一些技术。

## 推理

我们的推理量化算法(https://github.com/pytorch/ao/tree/main/torchao/quantization)可以作用于任何包含nn.Linear层的PyTorch模型。可以使用我们的顶层`quantize_` API来选择仅权重量化和动态激活量化，支持各种数据类型和稀疏布局。

```python
from torchao.quantization import (  
    quantize_,  
    int4_weight_only,  
)  
quantize_(model, int4_weight_only())
```

有时量化一个层可能会因为开销而变慢，所以如果您希望我们为您选择模型中每个层的量化方式，您可以改为运行

```python
model = torchao.autoquant(torch.compile(model, mode='max-autotune'))
```

`quantize_` API 有几种不同的选项，具体取决于您的模型是计算受限还是内存受限。

```python
from torchao.quantization import (  
    # Memory bound models  
    int4_weight_only,  
    int8_weight_only,

    # Compute bound models  
    int8_dynamic_activation_int8_semi_sparse_weight,  
    int8_dynamic_activation_int8_weight,  
      
    # Device capability 8.9+  
    float8_weight_only,  
    float8_dynamic_activation_float8_weight,  
)
```

我们在 diffusers-torchao(https://github.com/sayakpaul/diffusers-torchao) 中与 HuggingFace diffusers 团队进行了广泛的扩散模型基准测试，展示了 Flux.1-Dev 上 53.88% 的加速和 CogVideoX-5b 上 27.33% 的加速。

![](https://files.mdnice.com/user/59/931ed0ac-c4f4-4822-88dd-f30939451b91.png)

我们的 API 是可组合的，例如我们通过将稀疏性和量化结合起来，为 ViT-H 推理带来了 5% 的加速(https://github.com/pytorch/ao/tree/main/torchao/sparsity)。


我们还可以量化权重到 int4 和 KV 缓存到 int8，以支持 Llama 3.1 8B 在 128K 上下文长度下运行，占用不到 18.9GB 的 VRAM(https://github.com/pytorch/ao/pull/738)。


![](https://files.mdnice.com/user/59/3597a166-ce69-4bd6-9be2-f7a441b32ed3.png)

## QAT

后训练量化，尤其是在小于 4 位时，可能会遭受严重的精度下降。使用量化感知训练(QAT https://pytorch.org/blog/quantization-aware-training/)，我们设法恢复了 hellaswag 上高达 96% 的精度下降。我们将其集成到 torchtune 中，并提供了一个最小教程(https://github.com/pytorch/ao/tree/main/torchao/quantization/prototype/qat)。

![](https://files.mdnice.com/user/59/f17c22af-d63f-4346-aa36-4e9b8bdbf4cf.png)


## 训练

### 低精度计算和通信

torchao 提供了易于使用的端到端工作流，用于减少训练计算和分布式通信的精度，从 float8 开始，用于 `torch.nn.Linear` 层。以下是将其训练运行中的计算 GEMM 转换为 float8 的一行代码：

```python
from torchao.float8 import convert_to_float8_training  
convert_to_float8_training(model)
```

有关如何使用 float8 将 LLaMa 3 70B 预训练加速高达 1.5 倍的端到端示例，请参阅我们的 README(https://github.com/pytorch/ao/tree/main/torchao/float8)，以及 torchtitan 的博客和 float8 配方(https://github.com/pytorch/torchtitan/blob/main/docs/float8.md)。

### float8 预训练的 LLaMa 3 70B 的性能和精度，vs bfloat16

![](https://files.mdnice.com/user/59/ee0f1ae7-4cc8-491c-8892-18309a346a65.png)

(source: https://dev-discuss.pytorch.org/t/enabling-float8-all-gather-in-fsdp2/2359)

我们正在扩展我们的训练工作流，以支持更多数据类型和布局

- NF4 QLoRA in torchtune(https://pytorch.org/torchtune/main/tutorials/qlora_finetune.html)
- 原型 int8 训练支持(https://github.com/pytorch/ao/pull/748)
- 加速稀疏 2:4 训练(https://pytorch.org/blog/accelerating-neural-network-training/)

### 低精度优化器

受 Bits and Bytes 启发，我们也添加了 8 位和 4 位优化器的原型支持，作为 AdamW 的替代品。

```python
from torchao.prototype.low_bit_optim import AdamW8bit, AdamW4bit  
optim = AdamW8bit(model.parameters())
```

![](https://files.mdnice.com/user/59/d913d879-ecff-4f0c-9c61-a2865fe11224.png)

## 集成

我们一直在积极努力确保 torchao 在开源项目中表现良好。

1. Huggingface transformers 作为推理后端(https://huggingface.co/docs/transformers/main/quantization/torchao)
2. 在 diffusers-torchao(https://github.com/sayakpaul/diffusers-torchao) 中作为加速扩散模型的参考实现
3. 在 HQQ 中用于快速 4 位推理(https://github.com/mobiusml/hqq#faster-inference)
4. 在 torchtune(https://github.com/pytorch/torchtune) 中用于 PyTorch 原生 QLoRA 和 QAT 配方
5. 在 torchchat(https://github.com/pytorch/torchchat) 中用于后训练量化
6. 在 SGLang 中用于 int4 和 int8 后训练量化(https://github.com/sgl-project/sglang/pull/1341)

## 结论

如果您对使您的模型更快更小感兴趣，我们希望您会发现 torchao 有用且易于集成。

```bash
pip install torchao
```

我们有很多激动人心的工作，从低于 4 位，高性能内核，扩展到更多层，扩展类型或粒度，MX 硬件支持，支持更多硬件后端。如果您对上述任何内容感兴趣，可以关注我们的进展：https://github.com/pytorch/ao

如果您对 torchao 感兴趣，我们创建了一个贡献者指南，如果您有任何问题，可以在 discord.gg/gpumode 上加入 #torchao 频道。

## 致谢

我们很幸运能够站在巨人的肩膀上，并与开源社区中的一些最佳人才合作。感谢！

- Bits and Bytes 在低精度优化器和 QLoRA 方面的开创性工作
- Answer.ai 的工程工作，使 FSDP 和 QLoRA 组合
- Mobius Labs 在量化算法和低精度内核方面的可爱交流
- HuggingFace transformers 在战斗测试和集成我们的工作方面的帮助
- HuggingFace diffusers 在广泛的基准测试和最佳实践方面的合作
- torch.compile 使我们能够用纯 PyTorch 编写我们的算法
- GPU MODE 对我们的早期贡献者



