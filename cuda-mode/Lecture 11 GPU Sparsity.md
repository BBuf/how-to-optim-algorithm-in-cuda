> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 

## CUDA-MODE课程笔记 第11课: Sparsity

### 课程笔记

![](https://files.mdnice.com/user/59/045b9b33-cd43-44a0-a053-57bfcb05bd21.png)

![](https://files.mdnice.com/user/59/1b481816-f9d2-40ae-a50d-e6826ac80cbb.png)


作者的自我介绍，来自PyTorch Core团队，致力于架构优化，量化，稀疏方面的工作。特别是过去两年中，研究重点主要集中在生成式AI如LLMs和Vision Transformers上。现在他们的重点是把这些技术引入到GPU上，之前团队主要专注于边缘设备和CPU相关的工作。由于模型规模变得如此庞大，现在必须要在GPU上运行推理。我们希望利用已经训练好的模型，通过移除部分权重或者调整某些权重的数据类型为低比特以牺牲一定的准确度为代价来提升模型的性能。其核心理念在于，如果我们巧妙的恢复精度，那么这种下降是可以量化的。

![](https://files.mdnice.com/user/59/9dcdedc1-3144-48ec-89ce-323163e1c447.png)

Slides里面的流程图展示了稀疏/剪枝的流程：
- 用户神经网络
- 训练网络直到获得合理解决方案
- 移除部分参数
- 重新训练模型以恢复损失的准确性
- 得到剪枝后的神经网络
- 使用优化的稀疏kernel运行剪枝后的网络，以加速推理

剪枝包含两个主要部分：
- 准确性：从模型中去除参数
- 性能：如何快速进行与零相乘的运算

然后这个概念可以追溯到 Optimal Brain Damage (Hinton 89) 论文，是一个由来已久的研究领域。

![](https://files.mdnice.com/user/59/e4b1f41e-d63e-463e-9130-1d20af2f758b.png)

理论上，乘以零是非常快的操作，然而，如果计算系统不能识别并优化这些零乘法，实际上并不会节省计算时间。真正的性能提升来自于识别模型中的零参数，并完全跳过与这些零相关的计算。

![](https://files.mdnice.com/user/59/b7161b9f-d7a1-4a1d-b1f0-63f3341a84b1.png)

这张Slides讲到了如何在神经网络中添加零，即如何实现稀疏性。首先，有不同的稀疏pattern，其次，我们需要灵活性以保证准确性，最后我们也需要结构化的pattern以提高性能。右侧图表展示了不同的稀疏性模式，所有模式都显示了50%的稀疏度：
- 非结构化稀疏性（Unstructured Sparsity）：零和非零元素随机分布
- 2:4半结构化稀疏性（2:4 Semi-Structured）：每4个元素中有2个是非零的
- 块稀疏性（Block Sparsity）：以4x4的块为单位，一半的块全为零
- 结构化稀疏性（Structured Sparsity）：按行进行稀疏化，每隔一行全为零

不同的稀疏pattern对准确率的影响也不同，如何在准确率和性能之间进行平衡是我们需要考虑的核心问题。这些问题就是作者在过去几年研究的问题。

![](https://files.mdnice.com/user/59/1f2ad65d-c2a0-435e-bd13-e45f53fcef2c.png)









