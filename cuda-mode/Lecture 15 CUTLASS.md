> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 

## 第15课: CUTLASS


![](https://files.mdnice.com/user/59/86f297b4-c631-4bdb-a985-554d40fb1d2e.png)

![](https://files.mdnice.com/user/59/4710a35c-674f-4efd-a935-94eb8f2088ef.png)

对CUTLASS库一些作者的展示。

![](https://files.mdnice.com/user/59/b155a86e-6d0a-4f0a-b967-6c2377074a25.png)

演讲者认为CUTLASS的库写得很漂亮和通用，另外CUTLASS也实现了高性能的FlashAttention。演讲者这节课不会提到太多CUTLASS的API和具体函数，方法，而是把重心放在概念介绍上。

![](https://files.mdnice.com/user/59/9fc0c0cf-2c73-476a-b649-c50bb071e975.png)

图中列出了几个CUTLASS的特征指标：

- 使用下划线，如在 some_tensor(2, _, _) 中所示
- local_tile
- local_partition
- partition_D
- partition_S
- _3

这些特征是CUTLASS库中常见的命名约定、函数或变量名。

![](https://files.mdnice.com/user/59/2db040ad-beef-4c75-83f5-295136d0cd34.png)

这张Slides介绍了NVIDIA提供的两大类库:

- 从主机调用的库
    - 例如cuBLAS, cuDNN等
    - 这些库有有限的定制性，能推断是否以及何时使用张量核心
    - 使用这些库时，每次调用都需要在设备和主机之间进行数据传输（除了一些可以进行kernel融合的情况）
- 从设备调用的库
    - 例如CUTLASS, Thrust, CUB
- 特别提到CUTLASS是从设备调用的，具有低级控制，直接暴露张量核心操作
- 开发者可以编写新模型，测试它们是否以及如何能够实现高性能

![](https://files.mdnice.com/user/59/c72dbf8f-a288-4e40-8280-d5a666fd2b4e.png)

为了避免困惑，这张Slides介绍了两个重要的概念：

- [A .. B] 半开区间整数：这表示一个整数范围，包含从A到B的所有整数，但不包括B。例如，[1 .. 5] 表示整数集合 {1, 2, 3, 4}。
- 嵌套元组和mode：这里提到我们将处理嵌套元组（nested tuples）。这种元组中的一个元素（可能本身就是一个元组）被称为"模式"（mode）。特别指出这是CuTe使用的术语。CuTe 是 CUTLASS 库中的一个子组件。它是一个C++模板库，用于简化CUDA编程。CuTe 提供了抽象层，使得开发者可以更容易地编写高效的CUDA代码。

![](https://files.mdnice.com/user/59/e2cce9fa-3c16-47fd-ad81-ca543796d63a.png)

这张Slides用一个树状结构展示了CuTe中张量的组成：

- 最顶层是"tensor"（张量）
- 张量由两个主要部分组成：
    - "engine"（引擎），括号中给出了例子"(e.g. pointer)"，表示引擎可以是指针类型
    - "layout"（布局），进一步分为两个子部分：
        - "shape"（形状）
        - "stride"（步幅）

![](https://files.mdnice.com/user/59/a6c439f5-8237-4a62-9186-4a9cce6d2c52.png)

这张Slides解释了CuTe中的索引（Indexing）的概念：

- 索引公式：
    - 二维索引：(i, j) -> i + jM
    - 三维索引：(i, j, k) -> i + jM + kMN
- 这些公式可以被看作是点积：(i, j, k) · (1, M, MN)
- 红色部分被称为"stride"（步幅）
- 通常，允许的坐标范围被定义为：i, j, k ∈ [0 .. M) × [0 .. N) × [0 .. K) 其中K是尚未显示的某个大小。
- 在CuTe中，允许的输入由"shape" 给出：只需写上限（M, N, K），默认每个分量从零开始。

![](https://files.mdnice.com/user/59/dd941503-5bcf-42bb-9c24-04a9040c9fb2.png)

这张Slides继续解释了CuTe中的索引概念：

- shape（形状）和stride（步幅）的组合被称为layout（布局），用冒号（:）分隔shape和stride。
- 之前讨论的例子可以写成：(M, N, K) : (1, M, MN)，其中：(M, N, K) 是shape，(1, M, MN) 是stride。
- 左侧 (M, N, K) 表示"which inputs are allowed"（允许的输入范围），右侧 (1, M, MN) 表示"how to get from an coordinate to an offset"（如何从坐标得到偏移量）
- 注解："I personally often confuse 'shape' w 'size'"（up主个人经常混淆"shape"和"size"）

![](https://files.mdnice.com/user/59/62007faa-3b63-4eaf-a239-1ec46e1644bc.png)

这张Slides解释了CuTe中的布局（Layouts）概念：

- 可以连接布局，例如：(3 : 1, 2 : 3) = (3, 2) : (1, 3)。这不是关于布局的事实，只是允许我们用两种不同的方式来写同一个布局。
- 可以有嵌套的布局，如 ((3, 4), 2) : ((1, 3), 40)
- 比较 ((3, 4), 2) : ((1, 3), 40) 和 (3, 4, 2) : (1, 3, 40)（其"扁平化"版本）
    - ((3, 4), 2) : ((1, 3), 40) 接受如 ((1, 2), 1) 这样的坐标
    - (3, 4, 2) : (1, 3, 40) 接受如 (1, 2, 1) 这样的坐标
- 形状（shape）和步幅（stride）必须始终具有相同的嵌套结构（在CuTe术语中，它们必须是"一致的"或"congruent"）。


接下来的大量时间其实都在讲PyTorch里面Tensor的shape和stride的关系，由于这部分讲解都是up主手画，所以我简单文字记录下要点：

- 对于一个大张量比如二维张量中的一个小切片张量，相比于大张量来说只有形状（从物理内存角度来说是基础指针初始位置）有变化，Stride不变。
- 实际上这个小切片张量就是Tile，对于矩阵中所有的Tile，他们的Strides和原始张量都相同，但是基础指针位置和Shape不同。
- PyTorch中的`is_contiguous`直观的解释是Tensor底层一维数组元素的存储顺序与Tensor按行优先一维展开的元素顺序是否一致。这个地方推荐大家看 https://zhuanlan.zhihu.com/p/64551412 详细了解。

实际上这部分讨论的就是Tensor的Shape和Stride的概念，以及Tiling后的Sub Tensor的Shape和Stride和原Tensor的关系。

![](https://files.mdnice.com/user/59/bdf56661-c494-4f4f-ad96-4c92519e8455.png)

这张Slides讨论了CUTLASS中的切片（slicing）和平铺（tiling）概念。
- 这个库设计到大量切片（slicing）和平铺（tiling）概念。
- Tiling操作的直观解释：
    - 如果有一个大小为 A × B × C 的张量，用大小为 a × b × c 的张量进行Tiling
    - 结果会得到一个包含两部分的张量：
        - a. "外部部分"，大小为 A/a × B/b × C/c，回答"哪个tile？"的问题
        - b. "内部部分"，大小为 a × b × c，回答"tile中的哪个元素？"的问题
    - 假设所有维度都能被整除
- 将Tiling视为张量上的除法操作：T₁ ⊘ T₂ ↝ T₃
- Tiling的泛化：可以只在特定模式（维度）上进行Tiling，"除数"可能比被除数短，例如，将 A × B × C 用 a × c 在模式1和3上 Tiling：
    - 外部部分：A/a × B × C/c
    - 内部部分：a × c（也可以表示为 a × 1 × c）

![](https://files.mdnice.com/user/59/0bae76a3-ebcc-46a5-aa7a-f8027e83737a.png)


这张Slides讨论了平铺（tiling）操作的扩展概念，不仅限于张量：

- Tiling的应用范围：不仅可以Tiling数据（data）, 还可以Tiling计算资源（compute resources），例如在线程块上Tiling TensorCore 操作。
- Tiling概念的泛化：不再仅仅考虑 tensor ⊘ tensor ↝ tensor，而是更一般化地考虑 shape ⊘ shape ↝ shape。
- Tiling结果的约定：Tiling两个形状的结果有两个模式：内部（inner）和外部（outer），约定内部模式在前，外部模式在后
- Tiling操作的示例：(A, B, C) ⊘ (a, b, c) = ((a, b, c), (A/a, B/b, C/c))，这里，(a, b, c) 是内部模式（表示单个tile的大小），(A/a, B/b, C/c) 是外部模式（表示tile的数量或排列）。
- "Leftover" modes（未被Tiling的维度）按约定放在第二个（外部）模式中。示例：(A, B, C) 在模式1,3上与 (a, c) tiling，结果：((a, c), (A/a, B, C/c))。这里，B 维度作为 "leftover" 模式保留在外部模式中。

![](https://files.mdnice.com/user/59/54318ce7-2021-4b85-b17c-ee5f1077b660.png)

> 这里有个拼写错误，第二行公式的开头的A，B，C应该小写。

- Slides提出问题 "How do strides enter into tiling?"（步幅如何进入Tiling操作？）
- layout ⊘ shape ↝ layout? 这表示将一个布局（layout）与一个形状（shape）进行平铺操作，结果的layout是什么？
- 具体例子：(A, B, C) : (1, A, AB) ⊘ (a, b, c)，左侧 (A, B, C) : (1, A, AB) 是一个布局（layout），右侧 (a, b, c) 是一个形状（shape）
- 结果为：= ((a, b, c) : (?, ?, ?), (A/a, B/b, C/c) : (?, ?, ?))，结果分为内部（inner）和外部（outer）两部分
    - 结果的第一个模式（内部）应该保持与原来相同的布局（前面的文字记录要点里提到了）
    - 外部模式与之前相同，除了步幅未知（用问号表示）

为了说明这个问题，作者画了一张图，这里以（M, N）: (1, M)为例子，也就是一个col major的2D矩阵。

![](https://files.mdnice.com/user/59/cc9712fb-c211-4f78-8ddd-f783a4910e54.png)

当对它进行Tiling的时候我们可以得到图中的三个小块，这些小块的Layout最终表示为：(M/m, N/n) : (m, nM)，其中m, n分别表示tiling的大小。

因此，(A, B, C) : (1, A, AB) ⊘ (a, b, c) =
    ((a, b, c) : (1, A, AB), (A/a, B/b, C/c) : (a, A * b, A * B * c))


上面的公式更新为：

![](https://files.mdnice.com/user/59/8cdc6ef0-847b-4d09-bb49-a4b976ebf7d1.png)

这节课的Slides就讲完了，这节课主要是对CUTLASS里面的2022年底引入的CUTE的一些基础概念进行了讲解。后面up主还挑了一部分cutlass源代码以及简单聊了下CUTLASS的代码目录结构，这部分没有放在Notes里的必要就跳过了。

