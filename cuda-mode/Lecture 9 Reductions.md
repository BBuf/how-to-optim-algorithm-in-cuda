> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 

## CUDA-MODE课程笔记 第9课: 规约（PMPP的第10章）

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
这个算法是后续cuda kernel实现的基础。

![](https://files.mdnice.com/user/59/8e409640-6310-4141-bb5b-82da61621b9e.png)

这张Slides以树的方式可视化了上面的并行Reduction算法，只不过这里是求和而不是求最大值了。这里需要注意的是，浮点数加法不满足交换律，即A=B不等于B+A，使用PyTorch时这一点经常引起混淆。在这个例子中，我们无法控制GPU线程执行的先后顺序，所以无法控制合并两个元素的顺序，这也是不确定性的源头之一。

![](https://files.mdnice.com/user/59/73b82c5d-c169-492d-8a60-673c910c4a0b.png)

PyTorch中使用`torch.use_deterministic_algorithms(True)`来控制使用确定性的算法，但是这种算法一般会降低运行速度。https://github.com/cuda-mode/lectures/blob/main/lecture_009/nondeterminism.py 这个文件举了一个例子说明由于浮点数精度问题导致求和结果的不确定性的问题。

```c++
# We'll use several small numbers that, when added together first, could show a difference
numbers = [1e-20] * 10 + [1e20, -1e20]  # 10 small numbers followed by a large positive and negative number

# Sum the list from left to right
sum_left_to_right_adjusted = sum(numbers)

# Sum the list from right to left
sum_right_to_left_adjusted = sum(reversed(numbers))

# 0.0 9.999999999999997e-20
print(sum_left_to_right_adjusted, sum_right_to_left_adjusted)
```

另外想说明的问题是在《CUDA-MODE课程笔记 第7课: Quantization Cuda vs Triton》讲到过即使是进行INT4/INT8量化时，累加运算往往在更高的精度下运行，其原因在于如果你在float16中累加许多小的数值，最后可能会出现大数吃小数的情况。解决方法有两种：要么使用bf16这种具有更高动态范围的数据类型，要么在float32高精度上进行累加。例如当我们查看Triton矩阵乘法的教程时，我们会发现他的累加器一般都是float32。这个例子的代码为：https://github.com/cuda-mode/lectures/blob/main/lecture_009/accuracy.py

```python
import torch
large_value = torch.tensor([1000.0], dtype=torch.float32)  # Using float32 for initial value

# Define a smaller value that is significant for float32 but not for float16
small_value = torch.tensor([1e-3], dtype=torch.float32)  # Small value in float32

# Add small value to large value in float32
result_float32 = large_value + small_value

# Convert large value to float16 and add the small value (also converted to float16)
result_float16 = large_value.to(torch.float16) + small_value.to(torch.float16)

# Convert results back to float32 for accurate comparison
result_float32 = result_float32.item()
result_float16_converted = result_float16.to(torch.float32).item()

# Print results
# 1000.0009765625 1000.0
print(result_float32, result_float16_converted)
```

![](https://files.mdnice.com/user/59/5e3ef1cb-86f5-4e8f-86ae-d32e5c3c5168.png)

这张Slides建议结合 https://github.com/cuda-mode/lectures/blob/main/lecture_009/simple_reduce.cu 实现代码来看：

```c++
__global__ void SimpleSumReductionKernel(float* input, float* output) {
    unsigned int i = 2 * threadIdx.x;
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0) {
            input[i] += input[i + stride];
        }
        __syncthreads();

    }
    if (threadIdx.x == 0) {
    *output = input[0];
    }
}
```

对于SimpleSumReductionKernel来说，每个线程负责处理两个相邻的元素，这就是为什么Slides中显示8个线程处理16个元素。然后for 循环实现了归约过程，对应图中的每一层。stride 从1开始，每次迭代翻倍，这解释了为什么每次迭代活跃线程数减半。
- 我们可以模拟一下规约的过程：
    - 第一次迭代 (stride = 1): 每个线程将相邻的两个元素相加。
    - 第二次迭代 (stride = 2): 每隔一个线程进行计算，将间隔为2的元素相加。
    - 以此类推，直到最后只有一个线程（线程0）进行最后的加法。
- 此外，cuda代码中 `__syncthreads()` 确保每次迭代后所有线程同步。
- 代码中的 if (threadIdx.x % stride == 0) 条件导致了Slides中提到的线程不活跃问题。
- CUDA中，线程以32个为一组（称为warp）执行。由于这种归约方式，很快就会有整个warp变得不活跃，这就是Slides提到的 "A lot of warps will be inactive" 的原因。
- kernel启动设置：SimpleSumReductionKernel<<<1, size / 2>>>(d_input, d_output); 启动了 size/2 个线程，对应图中的8个线程（假设 size 为16）。
- Slides建议使用 "ncu -set full" 进行性能分析，这可能会揭示更多关于线程和warp效率的详细信息。







