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

这张Slides讲解了在CUDA张量操作中Layout的表示方法，主要通过Shape（形状）和Stride（步长）来描述。以下是主要内容：
- Layout表示：通过Shape和Stride来定义多维数组在内存中的排列方式。
- 三个例子展示了不同的Layout： 
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
        - 展示了一个3维数组的Layout。
    - 偏移量计算：
        - 使用公式：offset = inner_product(coord, stride)
        - 例如，对于元素f，其逻辑坐标为(0,1,1)，物理偏移量计算为40 + 11 + 2*1 = 3

![](https://files.mdnice.com/user/59/ce79b0c9-0711-420c-956f-741b33f1917a.png)

仍然来看刚才的三维的例子，我们可以把它看成是两个2x2的矩阵，也可以把后面的矩阵放在前面的矩阵的下面。这样我们就得到了一个4x2的二维矩阵，它的shape可以认为是4x2，但是我们这里并不能直接写成4，因为ac/ce/eg之间的距离不是固定的，如果写成4就没办法用一个数字去表示矩阵的Stride。注意到，a和c之间，e和g之间的距离都是4，而a和e，c和g之间的距离都是2。所以我们需要用到一个嵌套的表达。如下图红色部分：

![](https://files.mdnice.com/user/59/28d6e502-0df9-4aae-8e46-dcd211d09d8c.png)

我们需要把这个4x2矩阵的第一维的Shape写成（2，2），Stride写成（4,2），可以参考Slides中的左下角的图进行理解，在水平方向上可以理解为每个元素有2个子元素，每个子元素之间的距离是4，所以第一个Stride就是4，然后在z方向会重复两次，所以第二个Stride就是2。因此，通过嵌套形式的表达我们就可以表达出更复杂的一些Layout的例子。

![](https://files.mdnice.com/user/59/acf8fdfc-4340-466a-abe5-8124914ed7c5.png)

这张Slides展示了多种不同的Layout。每种Layout都有其特定的用途和优势。以下是对每种Layout的简要解释：
- Column-Major (列优先):
    - 形状: (4,8)，步长: (1,4)
    - 数据按列存储，每列连续
- Row-Major (行优先):
    - 形状: (4,8)，步长: (8,1)
    - 数据按行存储，每行连续
- Column-Major Padded (列优先带填充)
    - 形状: (4,8)，步长: (1,5)
    - 类似列优先，但每列之间有额外的填充空间
- Column-Major Interleaved (列优先交错):
    - 形状: (4,(4,2))，步长: (4,(1,16))
    - 数据以2x2的小块为单位进行列优先存储
- Row-Major Pitch-Linear (行优先带间距):
    - 形状: (4,(2,4))，步长: (8,(4,1))
    - 行优先存储，但每行之间可能有额外的间距
- Mixed (混合):
    - 形状: ((2,2),(2,4))，步长: ((1,8),(16,2))
    - 结合了多种Layout特性，形成复杂的嵌套结构

![](https://files.mdnice.com/user/59/600bd772-83fb-4060-a682-8fb6a456c7bd.png)

这张Slides介绍了CUTLASS 3.x中的CuTe的使用示例。主要内容如下：

- 标题为"LAYOUT USAGE EXAMPLE"（Layout使用示例）。
- 定义了一个形状(Shape)为(8, (2, 2))和步长(Stride)为(2, (1, 16))的布局。
- 展示了CuTe中如何使用make_layout函数创建布局，以及如何使用make_tensor函数创建张量。
- 提供了一个8x4的矩阵图示，展示了元素在内存中的排列。
- 解释了逻辑坐标是1D、2D和hD（高维的意思）。
- 给出了几个访问张量元素的例子：
    - A(17) = 18
    - A(1,2) = 18
    - A(1,(0,1)) = 18
- 展示了如何沿逻辑子边界进行切片：
    - A(3,_) = [6,7,22,23]
    - A(5,(_,1)) = [26,27]
- 图中用不同颜色标注了这些访问和切片操作对应的矩阵区域。

![](https://files.mdnice.com/user/59/3df36db2-b182-4b89-8771-777c1d2fcd72.png)

这张Slides可以帮助大家理解为什么我们需要CuTe，在CuTe之前（CUTLASS 2.x）我们实现一个地址准换需要Slides右边展示的这么多代码，我们需要理解每一行代码的作用。有了CuTe之后我们只需要Slides左边的几行代码就可以完成了。而且在CUTLASS 2.x中，定义一个Layout每个都需要有自己的实现，现在我们只需要用Shape和Stride就可以拿到任意想要的Layout。

## GEMM Data Flow with Cute

![](https://files.mdnice.com/user/59/b7bb8422-d3fe-4d21-881c-27b77ce6a9ee.png)

接下来了解一下GEMM中是如何用CuTe做数据的传输的。

![](https://files.mdnice.com/user/59/f74490c3-b3ba-4840-b0f3-344e3cc9a5e5.png)

在讲解GEMM数据传输之前需要讲一下Copy这个API，这个API是用CuTe做数据传输时一定会用到的。左边的API比较简单，把src Tensor和dst Tensor传禁区就可以完成数据拷贝，会自动根据GPU的架构，数据的存储位置去自动选择用UniversalCopy或者SM80_CP_ASYNC_CACHEALWAYS。它只会在这两个里面选择，如果我们想要得到更好的性能，建议使用右边的API。右边的copy API我们需要在第一个参数中显示指定一个copy_atom，它就是CuTe会为各种不同架构中的数据传输指令做一个封装。这里列出了不同架构下的数据传输指令，如果我们想用第二个API需要了解一下每个指令的作用和使用场景。另外，注意到这里copy的都是Tensor，所以我们可以用一些神奇的Layout达到一些除了数据拷贝之外的其它的一些变换的效果。

![](https://files.mdnice.com/user/59/067782a5-6f60-453d-83ba-be9826685b3e.png)

这张Slides举了一个用Copy做矩阵转置的简单例子，虽然不是最佳性能的实现，但可以让大家看到CuTe的魅力。右边的上两个图分别是从逻辑和物理的角度来看矩阵转置在做什么，物理角度来看就是要把abcd...->aeim...的顺序。我们在构造Tensor的时候就可以让iTensor和oTensor的shape都是一样的mxn，只不过在读进iTensor的时候让它以一个Column Major的方式读进来，所以我们构造Stride的时候传入(1, m)，右边的图里把iTensor.layout也画出来了，我们再以Row Major的方式写出去就达到了一个转置的效果，因此oTensor的stride就是(n, 1)。

到了这一步如果不看Tile/Partition相关的代码，我们直接调用一个COPY就完成了转置。然后现在我们希望把它并行起来，要并行起来就需要不同的Block和Thread去负责不同区域的矩阵转置。所以，我们需要继续调用local_tile去给不同的Block分配该负责哪个区域。

