> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 

## CUDA-MODE课程笔记 第11课: Sparsity

> 这节课主要是作者介绍了一下PyTorch团队在Sparsity方向做的一些工作，重点为Sparsity的GPU推理，如果大家对Sparsity感兴趣，想了解一下它在实际工程应用方面的进展可以考虑听一下，如果不感兴趣可以调过，这个技术比较冷门，目前工业界的推理方案集中在量化上面。

### 课程笔记

![](https://files.mdnice.com/user/59/045b9b33-cd43-44a0-a053-57bfcb05bd21.png)

![](https://files.mdnice.com/user/59/1b481816-f9d2-40ae-a50d-e6826ac80cbb.png)


作者的自我介绍，来自PyTorch Core团队，致力于架构优化，量化，Sparsity 方面的工作。特别是过去两年中，研究重点主要集中在生成式AI如LLMs和Vision Transformers上。现在他们的重点是把这些技术引入到GPU上，之前团队主要专注于边缘设备和CPU相关的工作。由于模型规模变得如此庞大，现在必须要在GPU上运行推理。我们希望利用已经训练好的模型，通过移除部分权重或者调整某些权重的数据类型为低比特以牺牲一定的准确度为代价来提升模型的性能。其核心理念在于，如果我们巧妙的恢复精度，那么这种下降是可以量化的。

![](https://files.mdnice.com/user/59/9dcdedc1-3144-48ec-89ce-323163e1c447.png)

Slides里面的流程图展示了Sparsity /剪枝的流程：
- 用户神经网络
- 训练网络直到获得合理解决方案
- 移除部分参数
- 重新训练模型以恢复损失的准确性
- 得到剪枝后的神经网络
- 使用优化的Sparsity kernel运行剪枝后的网络，以加速推理

剪枝包含两个主要部分：
- 准确性：从模型中去除参数
- 性能：如何快速进行与零相乘的运算

然后这个概念可以追溯到 Optimal Brain Damage (Hinton 89) 论文，是一个由来已久的研究领域。

![](https://files.mdnice.com/user/59/e4b1f41e-d63e-463e-9130-1d20af2f758b.png)

理论上，乘以零是非常快的操作，然而，如果计算系统不能识别并优化这些零乘法，实际上并不会节省计算时间。真正的性能提升来自于识别模型中的零参数，并完全跳过与这些零相关的计算。

![](https://files.mdnice.com/user/59/b7161b9f-d7a1-4a1d-b1f0-63f3341a84b1.png)

这张Slides讲到了如何在神经网络中添加零，即如何实现稀疏性。首先，有不同的Sparsity pattern，其次，我们需要灵活性以保证准确性，最后我们也需要结构化的pattern以提高性能。右侧图表展示了不同的稀疏性模式，所有模式都显示了50%的Sparsity 度：
- 非结构化稀疏性（Unstructured Sparsity）：零和非零元素随机分布
- 2:4半结构化稀疏性（2:4 Semi-Structured）：每4个元素中有2个是非零的
- 块稀疏性（Block Sparsity）：以4x4的块为单位，一半的块全为零
- 结构化稀疏性（Structured Sparsity）：按行进行Sparsity 化，每隔一行全为零

不同的Sparsity pattern对准确率的影响也不同，如何在准确率和性能之间进行平衡是我们需要考虑的核心问题。这些问题就是作者在过去几年研究的问题。

![](https://files.mdnice.com/user/59/1f2ad65d-c2a0-435e-bd13-e45f53fcef2c.png)

这张Slides讨论了Sparsity在性能方面的考虑，特别是在张量乘法中的实现。我们使用Sparsity 表示（Sparse representations）和Sparsity kernel（Sparse kernels）以及独立的存储数据结构来完成。下面举了一个COO（Coordinate）表示法的例子，只存储非0元素的坐标和数据，更多的表示方法可以参考 https://pytorch.org/docs/stable/sparse.html 。只有在Sparsity 度超过99%的情况下相比于Dense Matmul才能展现出速度优势。这张slides像是在讨论CPU上的Sparsity。

![](https://files.mdnice.com/user/59/ad14dba0-bcc5-40cc-b841-5ffd28f09394.png)

在GPU上情况更糟糕，Dense Matmul由于并行计算的影响速度很快。非结构化稀疏性虽然很酷且能保持准确性，但在GPU上无法快速运行。GPU基于块操作，而非结构化稀疏性无法形成有结构的块。那么如何在GPU上跑得更快呢？我们可以通过移除整行这种结构化剪枝并重用dense kernels，但这种方法对准确性的影响很大，难以处理。

![](https://files.mdnice.com/user/59/ab69fbcb-6b26-4f9a-8660-869151f13089.png)

这张Slides讲了GPU Sparsity的不同模式和特点：
- 半结构化（Semi-structured）稀疏性 (2:4)：
    - 固定50%的稀疏度，理论上最多可获得2倍加速
    - 相对容易恢复精度（nvidia支持）
- 块稀疏性（Block sparsity）：
    - 基于块大小，在90%稀疏度时可获得约3.4倍加速
    - 需要更高级的算法（如DRESS）来恢复精度

![](https://files.mdnice.com/user/59/a0da8f59-b9fc-4508-aa75-d28cabff3209.png)

这里详细介绍一下Semi-Structured (2:4) Sparsity，也被称为M:N / 细粒度结构化稀疏性，每4个元素中有2个为0。它可以应用于STRIP或TILE模式。右边的图显示我们存储的压缩后的矩阵元素只有原始元素的一半，此外我们有一个2Bit dtype的mask矩阵，这个mask矩阵会应用在Sparse Matmul中，这个已经整合到了PyTorch中，我们可以尝试和使用。对于backend，我们有两种处理方法可以选择。在CutLass中，我们可以按照原始指令进行这个操作，此外还有一个NVIDIA的Sparse处理的专用库cuSPARSELt提供了一些附加功能，使得试用起来速度更快并且更方便。我们已经把这两种处理方法整合到了PyTorch中，如果你在PyTorch中见到cuSPARSELt，那就是和Semi-Structured (2:4) Sparsity相关的。理论上可以有2倍加速，但实际中加速大概是1.6倍左右，这和kernel的实现是有关的，对于不同的矩阵规模加速效果会不同。

![](https://files.mdnice.com/user/59/3a09aa9a-46bd-45f3-8a3c-70351478d6c8.png)

这张slides讲了使用cuSPARSELt库进行GPU稀疏矩阵乘法的过程：
- 初始化：
    - 使用`cusparseLtDenseDescriptorInit()`初始化密集矩阵D和B
    - 使用`cusparseLtStructuredDescriptorInit()`初始化结构化稀疏矩阵A
- 稀疏化和压缩：
    - 对矩阵A进行剪枝(prune)，使用`cusparseLtSpMAPrune()`
    - 将剪枝后的矩阵A压缩，使用`cusparseLtSpMACompress()`
- 计划和执行：
    - 使用`cusparseLtMatmulDescriptorInit()`初始化矩阵乘法描述符
    - 使用`cusparseLtMatmulAlgSelectionInit()`选择算法
    - 使用`cusparseLtMatmulPlanInit()`创建执行计划
    - 使用`cusparseLtMatmul()`执行矩阵乘法 `D = A * B`
- 注意到B可以在迭代过程中改变。

![](https://files.mdnice.com/user/59/5ebba88b-0905-4b80-8014-db541e75fc79.png)

这张slides展示了几种不同模型和技术的端到端(E2E)结果比较：
- 左上角的表格比较了dense FP16和sparse FP16在不同网络和数据集上的性能（这个结果是2022 nvidia paper里的）：
    - 包括ResNet-50、ResNeXt-101、Xception等在ImageNet上的Top-1准确率
    - SSD-RN50和MaskRCNN-RN50在COCO2017上的mAP
    - FairSeq Transformer在EN-DE WMT'14上的BLEU分数
    - BERT-Large在SQuAD v1.1上的F1分数    
    - 结果显示sparse FP16在大多数情况下与dense FP16性能相当。
- 右边两张图展示了最近2年PyTorch团队在Sparisty方面的一些成果。
    - 右上角的柱状图显示了SAM vit_h图像编码器在不同优化技术下的处理速度(img/s)，从FP16到各种优化方法，处理速度逐步提高。
    - 右下角的表格详细列出了SAM vit_h模型在不同优化策略下的性能指标：
       - 包括batch size、每秒处理图像数、峰值内存使用和COCO 2017验证集准确率
        - 优化策略包括FP16、torch.compile、SDPA、INT8量化、动态量化和2:4稀疏化
        - 结果显示，随着优化策略的应用，处理速度提高，内存使用减少，而准确率基本保持不变。
从SAM vit的结果来看，Sparsity是有速度优势的。需要特别指出的是在SAM vit_h上用稀疏化剪枝的方法精度会降低到0.53，作者解释如果直接微调应该能恢复到原始的精度。此外，对于视觉模型来说微调是可以接受的，因为视觉模型一般参数比较少，微调的成本不高。但是对于Sparse GPT来说，为了恢复精度，就只能探索一次性校准的技术了，而不是微调，因为成本太高。

![](https://files.mdnice.com/user/59/60a57992-8803-492e-b427-e327dd0a1a39.png)

这张slides展示了一下在PyTorch中如何对nn.Linear层应用Sparse，代码链接为：https://gist.github.com/jcaip/44376cd69d3a05cbe16610b4379d9b70 。

```python
import torch
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor

# Sparsity helper functions
def apply_fake_sparsity(model):
    """
    This function simulates 2:4 sparsity on all linear layers in a model.
    It uses the torch.ao.pruning flow.
    """
    # torch.ao.pruning flow
    from torch.ao.pruning import WeightNormSparsifier
    sparse_config = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            sparse_config.append({"tensor_fqn": f"{name}.weight"})

    sparsifier = WeightNormSparsifier(sparsity_level=1.0,
                                      sparse_block_shape=(1,4),
                                      zeros_per_block=2)
    sparsifier.prepare(model, sparse_config)
    sparsifier.step()

    sparsifier.step()
    sparsifier.squash_mask()


def apply_sparse(model):
    apply_fake_sparsity(model)
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            mod.weight = torch.nn.Parameter(to_sparse_semi_structured(mod.weight))
```

整体流程为先对模型进行稀疏化（这里使用的是apply_fake_sparsity函数，名字暗示这个是伪稀疏化，应该不能保证模型精度），然后再调用 to_sparse_semi_structured 函数执行真正的权重转换为semi structured sparse tensor。迷惑的点在于，apply_fake_sparsity函数里面执行了2行`sparsifier.step()`，这个可能要深入了解下PyTorch Sparisity才知道了。

此外，目前Torch实现的这种semi structured sparse只支持第一个矩阵为稀疏的，而第二个矩阵必须是的Dense。在PyTorch中Linear的基本形式是xW'，所以不支持对W的稀疏化，但我们可以考虑试用转置来重写它，使它支持对W的稀疏化。如Slides左侧的最后一个公式所示。但这样又明显会对性能产生影响，因为要在GPU上进行内存复制操作，我们可以通过编译器把这个转置融合到后续的操作比如ReLU中，这样整个流程会快得多。但torch.comiple目前好像不支持sparse Matmul之后这种融合。


![](https://files.mdnice.com/user/59/f923d12c-a8fb-4487-b0c2-9bdd225e5ee0.png)

对于Block Sparsity，我们使用 https://github.com/pytorch-labs/superblock 这里的技术来恢复精度。这张Slides对ViT-L的层进行了微基准测试：
- 展示了两个表格，分别对应批量大小为256的MLP 1和MLP 2
- 测试了不同的块大小（8、16、32、64）和稀疏度水平（0.9和0.8）
- 表格中的数值代表处理速度

我们可以发现，随着块大小的增加，性能普遍提高，并且稀疏度水平0.9通常比0.8获得更高的性能数值。此外，端到端（E2E）结果显示在ImageNet上使用ViT-L模型测试时实现了1.44倍的加速，精度下降损失不大。

![](https://files.mdnice.com/user/59/6854c0bb-3371-4084-96a9-ad09c14aee18.png)

作者介绍了一下当前的一些工作，例如把Sparse和Quantization结合，这里碰到了很多问题，不仅涉及到torch.compile，还包括处理fusion操作。关于稀疏化，本质要做的就是在已有权重上添加一些0值，然后进行压缩，接着得到压缩后的表示形式，这都是在离线的状态下完成的。作者展示了一个文档，但是没有公开，截图内容大概如下：

![](https://files.mdnice.com/user/59/e85d27b0-e50d-4622-b883-3a1b008c2e22.png)
![](https://files.mdnice.com/user/59/087a3911-b951-4e47-8894-6fd40c901ad9.png)

这两张截图主要展示了半结构化稀疏性和动态量化两种模型加速技术的计算图和实现方法。
- 半结构化稀疏计算图：
    - 使用2:4稀疏模式压缩权重
    - 离线进行剪枝和压缩，推理时加载压缩后的权重
    - 将密集矩阵乘法(mm)替换为稀疏矩阵乘法内核(cslt_mm)
- Int8动态量化计算图：
    - 对权重和激活都进行量化，转换为int8表示
    - 权重量化可以在推理前进行，激活量化在推理过程中进行
    - 量化过程产生量化参数(w_scales, x_scales)，用于后续的反量化
    - 将普通矩阵乘法(mm)替换为int8矩阵乘法(int_mm)，输出int32张量
    - 添加反量化(dequantize)步骤，将int32输出转回fp16格式

![](https://files.mdnice.com/user/59/7ed40753-4065-421e-a75f-1ecd2e90534f.png)

这张截图主要讲解了如何将半结构化稀疏性和int8动态量化技术结合使用，并展示了benchmark。上面的图展示了结合半结构化稀疏性和int8动态量化的计算图，包括权重剪枝、量化、压缩等步骤。结合的方法为：

- 首先对权重进行剪枝，得到一个稀疏的密集权重。
- 将剪枝后的权重传入量化流程。
- 使用对称（零点保持）量化方法，确保int8表示仍保持2:4稀疏性。
- 使用cslt_compress压缩int8表示，类似fp16情况。
- 使用cslt_int_mm（cuSPARSELt的int8版本）进行矩阵乘法。

> 仅cuSPARSELt v0.5.0支持(i8i8)->i32矩阵乘法。这一功能是与NVIDIA合作开发的。

性能对比表格展示了SAM vit_h模型在不同优化策略下的端到端延迟（e2e latency）：

- 基准版本（bf16编译）：1636.584
- 半结构化稀疏（bf16编译）：1389.318
- 动态量化（int8编译）：1404.085
- 半结构化稀疏 + 动态量化（int8编译）：1370.230

这里的混合应用Sparisty和量化之后延迟下降很少，主要原因是因为没有应用上算子融合技术，应该是目前的torch.compile没支持这种稀疏模式下的算子融合。

![](https://files.mdnice.com/user/59/b763bf45-98ee-4821-959b-f7160530fbe5.png)

![](https://files.mdnice.com/user/59/e81c27a6-1040-4b42-9ed8-88081a104806.png)

![](https://files.mdnice.com/user/59/6030244d-1739-429e-9720-24f2017528fe.png)


这里提到单独使用稀疏性和量化比组合使用它们能获得更多的加速。然后关键的挑战是融合多个操作，减少GPU内存读写，从而节省时间。对于量化，这意味着将反量化操作融入`int_mm` kernel 中。通过融合，我们可以提高速度：减少GPU内存访问以及减少峰值内存使用：避免生成中间fp32张量。当把Sparse和量化进行组合时，无法将反量化操作融入`cuSPARSELt`，因为它是一个外部黑盒，有个解决方法是`cuSPARSELt`支持在矩阵乘法时传入scale向量。我们可以将反量化操作的一个element-wise乘法融入`cuSPARSELt`矩阵乘法中。

Charlie的GPTQ实验表明可以保持在bfloat16范围内。这避免了超出fp16动态范围和精度问题。可以去除反量化操作中的fp32转换，完全在bfloat16内操作。

最后，作者编写了一个融合单个乘法的量化+稀疏SAM代码原型。GitHub链接提供了代码和PR，展示了这种方法的可行性。性能结果上SAM vit_h模型使用半结构化稀疏+动态量化+融合单个乘法（int8编译）的端到端延迟为1278.547。

> 限制：由于cuSPARSELt缺乏(i8i8)->bf16支持，无法验证模型精度。必须使用(i8i8)->fp16 kernel，这会导致之前提到的精度范围问题。

这一节讨论的是作者目前从事的Sparsity + Quantization技术混合使用关于性能的问题。

![](https://files.mdnice.com/user/59/ca1001bb-d263-432e-b305-8d7c0a7a4a07.png)

这张slides展示了混合应用之后精度狂掉，所以上面讲的这一堆目前看来是无法真正应用的。在我们真正做模型推理时，也不会选择Sparse和Quantization这种混合应用，我们简单了解下作者讲的东西就可以了。最终结论就是Sparisty和Quantization结合可以展现出速度优势，然后在精度方面还处于非常初级的实验阶段。

上面Current Work的slides还展示了作者团队在sparse training和剪枝算法方面在做工作，并且欢迎大家参与到这些开源工作。特别是剪枝算法是最关键的点，感觉和LLM的应用关系很大。

![](https://files.mdnice.com/user/59/8b193450-07f3-4ace-b8e7-c2677facd716.png)

![](https://files.mdnice.com/user/59/ca7c6473-eb2b-4078-8408-808d5a983059.png)

![](https://files.mdnice.com/user/59/eee4119b-d5ed-402b-a193-32e28fafdef9.png)

![](https://files.mdnice.com/user/59/7a68e021-f4a1-486e-b14b-0adf7223afbe.png)

![](https://files.mdnice.com/user/59/3369c565-09f0-4902-9928-c56f0bb3e486.png)

![](https://files.mdnice.com/user/59/2d301eac-46aa-4217-8b6a-fd5f11e440b7.png)

![](https://files.mdnice.com/user/59/41af8572-9bc2-4b61-a0b4-5ae73c8cf18d.png)

作者并没有继续深入讲解Sparse Training，只是简单提了一下xformers支持了Semi-Structured (2:4) Sparsity Training，并在ImageNet上做过实验。然后说明了一下Sparse Training和Inference的主要区别以及Sparse Training的一些关键组件，包括需要算得很快的稀疏算子，让我们完成对输入的压缩，需要自定义的torch.autograd.Function实现，需要cuSPARSELt支持转置操作与后续分布式集合通信算子的融合。

### 总结

这节课主要是作者介绍了一下PyTorch团队在Sparsity方向做的一些工作，重点为Sparsity的GPU推理，如果大家对Sparsity感兴趣，想了解一下它在实际工程应用方面的进展可以考虑听一下，如果不感兴趣可以调过，这个技术比较冷门，目前工业界的推理方案集中在量化上面。









