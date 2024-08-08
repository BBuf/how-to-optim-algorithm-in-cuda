> 本次演讲将介绍如何使用CUTLASS 3.x风格的代码在Hopper架构上实现输入为FPA+INTB混合精度矩阵乘法，内容包括：1.使用CuTe进行数据传输。2. FPA+INTB矩阵乘法案例讲解。Slides来自BiliBili NVIDIA英伟达频道 上传的《TensorRT-LLM 中的 Hopper Mixed GEMM 的 CUTLASS 3.x 实现讲解》视频讲解。这里参考视频并更详细记录了每一页Slides的要点，通过这个视频了解下在TRT-LLM中如何使用CUTLASS 3.x来自定义Mixed GEMM算子。我将其作为CUDA-MODE的CUTLASS课程的前置学习内容。

## 总览&目录

![](https://files.mdnice.com/user/59/5fe44715-f36e-4959-85fc-461baa79e254.png)

![](https://files.mdnice.com/user/59/f7e8b02c-29ce-49d5-98a1-de90ab75e53c.png)

这个演讲主要会分成三部分，首先是对CuTe的介绍，然后介以GEMM数据传输为例展示它是如何用Cute来做的。安排这两节是因为是在CUTLASS 3.x的底层实现中，无论你的数据在各个层级的管理，还是真正做一个GEMM运算，都是需要大量的使用到CuTe的API的。不熟悉CUTLASS的开发者初次见到CuTe会比较陌生，所以对其进行介绍。最后一个会具体的去浏览一下Mixed GEMM的CUTLASS 3.x实现。

## CuTe 介绍

![](https://files.mdnice.com/user/59/32dcec1a-3930-42ff-9c13-7632043a7743.png)

![](https://files.mdnice.com/user/59/f47fd7ca-9d1a-4e45-95cc-0f2e92035cf9.png)

CuTe其实就是用来管理CUDA Tensors计算的一个工具库，它最主要的概念就是**Layout**和**Tensor**。其中Layout是由**Shape**和**Stride**这两个概念组成的，可以把它理解为一个函数，作用就是把一个N维的逻辑坐标映射到真实的一维的连续的索引上去。有了这个Layout之后，再把一个真正的内存的指针传给Tensor的模板参数，这就构成了一个真正的Tensor。这里可以发现对于同一个内存指针指向的连续空间，给它不同的Layout就可以得到不同视角的Tensor，这就让CuTe具有了很大的灵活性，让它可以去处理一些复杂的索引问题。

CuTe提供了对Layout的形式代数操作：Layout可以被组合、操作、进行平铺和分区等。

CuTe提供了很多的API，包含一些基本的变换的一些API，这里列出了几个CuTe的操作函数，如get、rank、depth、shape、stride和size等。最后，Slides提到了一些其他相关概念，如Composition（组合）、Complement（补集）、Inverse（逆）、Product（乘积）和Divide（除法）。这张Slides的最后一个链接是CuTe的官方文档有更详细的CuTe的概念以及API介绍。

![](https://files.mdnice.com/user/59/c5902b43-b10b-4a08-9b4c-b672fbb79779.png)

这张Slides讲解了在CUDA张量操作中Layout（布局）的表示方法，主要通过Shape（形状）和Stride（步长）来描述。以下是主要内容：
- 布局表示：通过Shape和Stride来定义多维数组在内存中的排列方式。
- 三个例子展示了不同的布局： 
    - a. 第一个例子：
        - Shape: (2,3)
        - Stride: (1,2)
        - 展示了如何将2x3的矩阵按行优先顺序存储在内存中。
    - b. 第二个例子：
        - Shape: (2,3)
        - Stride: (3,1)
        - 展示了同样的2x3矩阵，但按列优先顺序存储在内存中。
    - c. 第三个例子：
        - Shape: (2,2,2)
        - Stride: (4,1,2)
        - 展示了一个3维数组的布局。
    - 偏移量计算：
        - 使用公式：offset = inner_product(coord, stride)
        - 例如，对于元素f，其逻辑坐标为(0,1,1)，物理偏移量计算为40 + 11 + 2*1 = 3

![](https://files.mdnice.com/user/59/ce79b0c9-0711-420c-956f-741b33f1917a.png)

仍然来看刚才的三维的例子，我们可以把它看成是两个2x2的矩阵，也可以把后面的矩阵放在前面的矩阵的下面。这样我们就得到了一个4x2的二维矩阵，它的shape可以认为是4x2，但是我们这里并不能直接写成4，因为ac/ce/eg之间的距离不是固定的，如果写成4就没办法用一个数字去表示矩阵的Stride。注意到，a和c之间，e和g之间的距离都是4，而a和e，c和g之间的距离都是2。所以我们需要用到一个嵌套的表达。如下图红色部分：

![](https://files.mdnice.com/user/59/28d6e502-0df9-4aae-8e46-dcd211d09d8c.png)

我们需要把这个4x2矩阵的第一维的Shape写成（2，2），Stride写成（4,2），可以参考Slides中的左下角的图进行理解，在水平方向上可以理解为每个元素有2个子元素，每个子元素之间的距离是4，所以第一个Stride就是4，然后在z方向会重复两次，所以第二个Stride就是2。因此，通过嵌套形式的表达我们就可以表达出更复杂的一些Layout的例子。

![](https://files.mdnice.com/user/59/acf8fdfc-4340-466a-abe5-8124914ed7c5.png)

