> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 

## 第15课: CUTLASS


![](https://files.mdnice.com/user/59/86f297b4-c631-4bdb-a985-554d40fb1d2e.png)

![](https://files.mdnice.com/user/59/4710a35c-674f-4efd-a935-94eb8f2088ef.png)

对CUTLASS库一些作者的展示。

![](https://files.mdnice.com/user/59/b155a86e-6d0a-4f0a-b967-6c2377074a25.png)

演讲者认为CUTLASS的库写得很漂亮和通用，另外CUTLASS也实现了高性能的FlashAttentio。演讲者这节课不会提到太多CUTLASS的API和具体函数，方法，而是把重心放在概念介绍上。

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






