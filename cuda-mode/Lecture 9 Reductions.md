> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 

## CUDA-MODE课程笔记 第9课: 规约

### 课程笔记

![](https://files.mdnice.com/user/59/b9b77076-7a41-4fea-b371-f9ba927b1219.png)

本节课的题目。

![](https://files.mdnice.com/user/59/cf3e6544-c11a-4a7d-a5d8-b98323488a5e.png)

这节课的内容主要是 Chapter 10 of PMPP book ，Slides里面还给出了本节课的代码所在位置以及如何编译以及使用nsight compute进行profile。

![](https://files.mdnice.com/user/59/fc7532be-2e98-44ed-8e10-5adb8e5aa4e6.png)


这张slides给出了reduction（归约）操作的定义，reduction（归约）是减少输出大小的操作。最典型的reduction操作是将向量（vector）转换为标量（scalar）。然后还列举了一些常见的reduction操作，包括min（最小值），max（最大值），argmax（最大值的索引），argmin（最小值的索引），norm（范数），sum（求和），prod（乘积），mean（平均值），unique（唯一值）。https://github.com/cuda-mode/lectures/blob/main/lecture_009/torch_reductions.py 这个文件展示了一些 reduction 操作的例子。

```python
def reduce(data, identity, op):
    result = identity
    for element in data:
        result = op(result, element)
    return result

# Example usage:

# Summation
data = [1, 2, 3, 4, 5]
print(reduce(data, 0, lambda a, b: a + b))  # Output: 15

# Product
print(reduce(data, 1, lambda a, b: a * b))  # Output: 120

# Maximum
print(reduce(data, float('-inf'), max))  # Output: 5

# Minimum
print(reduce(data, float('inf'), min))  # Output: 1
```

在PyTorch中有一个通用的Recuce算子来做所有的规约操作，所以你不能看到reduce_max这种单独的算子。

![](https://files.mdnice.com/user/59/9d605154-a0a6-42a1-a869-413322881ce1.png)

这张Slides强调了归约操作在深度学习和机器学习中的普遍性，例如：

- Mean/Max pooling（平均/最大池化）：这是在卷积神经网络中常用的操作，用于减少特征图的空间尺寸，提取主要特征。
- Classification: Argmax（分类：最大值索引）：在分类任务中，通常使用argmax来确定最可能的类别。
- Loss calculations（损失计算）：在训练过程中，通常需要计算损失函数，这往往涉及到对多个样本损失的归约操作。
- Softmax normalization（Softmax归一化）：在多分类问题中，Softmax用于将原始的输出分数转换为概率分布，这个过程也涉及归约操作。

![](https://files.mdnice.com/user/59/55b40864-d77a-4a49-8184-3bfa04eb4c93.png)

这张Slides展示了一个PyTorch使用归约操作（在这个case中是求最大值）来处理张量数据的例子。我们可以看到`torch.max`这个算子的实现是在 https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ReduceOps.cpp 这里，并不是一个单独的kernel实现，而是注册为通用的Reduce Op里面的其中一个。

![](https://files.mdnice.com/user/59/70562d90-ae1d-47ac-a3df-e4ab1cd5dd18.png)

这张Slides展示了一个"串行归约"(Serial reduction)的例子，具体是关于最大值(Max)操作的过程。
- 处理方法如下：
    - "Go through elements 1 by 1"（逐个遍历元素）
    - "Compare new number to old max if greater then update"（将新数字与当前最大值比较，如果更大则更新）
- 右侧图示展示了具体的迭代过程：
    - 初始向量(Initial Vector)是 [5, 2, 8, 1]
    - 迭代1：[5]（第一个元素作为初始最大值）
    - 迭代2：[5, 5]（2小于5，最大值保持不变）
    - 迭代3：[5, 5, 8]（8大于5，更新最大值）
    - 迭代4：[5, 5, 8, 8]（1小于8，最大值保持不变）

![](https://files.mdnice.com/user/59/d0d86c02-6f64-4b36-9de1-fcda59989c7c.png)

这张Slides主要展示了数据处理中的两种策略:转换(Transformation)和归约(Reduction)，以及它们各自的线程策略。继续看规约是怎么做的。

![](https://files.mdnice.com/user/59/59d43391-d61f-476f-8e6e-4b0d14b23255.png)

- 这张Slides展示了并行归约（Parallel Reduction）的可视化过程，主要讲解了如何通过并行计算找出一个向量中的最大值。
- 算法步骤：
    - 每一步取一对元素，计算它们的最大值，并将新的最大值存储在新向量中。
    - 重复这个过程，直到向量中只剩下1个元素。
    - 整个过程需要 O(log n) 步完成，其中 n 是初始向量的元素数量。
- Slides右边的图展示了具体的归约过程：
    - 初始向量：[5, 2, 8, 1, 9, 3, 7, 4, 6, 0]
    - 第1步归约：[5, 8, 9, 7, 6]
    - 第2步归约：[8, 9, 7]
    - 第3步归约：[9, 9]
    - 最终步骤：[9]



