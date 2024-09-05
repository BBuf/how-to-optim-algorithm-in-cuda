> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 

## CUDA-MODE课程笔记 第11课: Sparsity

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

这里详细介绍一下Semi-Structured (2:4) Sparsity，也被称为M:N / 细粒度结构化稀疏性，每4个元素中有2个为0。它可以应用于STRIP或TILE模式。右边的图显示我们存储的压缩后的矩阵元素只有原始元素的一半，此外我们有一个2Bit dtype的mask矩阵，这个mask矩阵会应用在Sparse Matmul中，这个已经整合到了PyTorch中，我们可以尝试和使用。对于backednd，我们有两种处理方法可以选择。在CutLass中，我们可以按照原始指令进行这个操作，此外还有一个NVIDIA的Sparse处理的专用库cuSPARSELt提供了一些附加功能，使得试用起来速度更快并且更方便。我们已经把这两种处理方法整合到了PyTorch中，如果你在PyTorch中见到cuSPARSELt，那就是和Semi-Structured (2:4) Sparsity相关的。






