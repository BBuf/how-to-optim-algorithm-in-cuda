> 这个演讲介绍了如何使用CUTLASS 3.x风格的代码在Hopper架构上实现输入为FPA+INTB混合精度矩阵乘法，内容包括：1.使用CuTe进行数据传输。2. FPA+INTB矩阵乘法案例讲解。Slides来自BiliBili NVIDIA英伟达频道 上传的《TensorRT-LLM 中的 Hopper Mixed GEMM 的 CUTLASS 3.x 实现讲解》视频讲解。这里参考视频并更详细记录了每一页Slides的要点，通过这个视频了解下CuTe的基本概念和CuTe实现GEMM的数据流动，以及从更High Level的角度看CUTLASS 3.x是如何实现Mixed GEMM的。

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

到了这一步如果不看Tile/Partition相关的代码，我们直接调用一个COPY就完成了转置。然后现在我们希望把它并行起来，要并行起来就需要不同的Block和Thread去负责不同区域的矩阵转置。所以，我们需要继续调用local_tile去给不同的Block分配该负责哪个区域，local_partion就是给Block里面每个Thread去分配区域。

代码里的local_tile有三个参数，第一个参数就是待划分的Tensor，这里就是iTensor。第二个参数就是希望一个Block的大小是多少，例如这个例子中BLOCK_TILE设成2，这里只是举例子，实际上不可能这么小，只是为了方面画图帮助读者理解，所以一个Block的大小就是2x2的区域。第三个参数会传入当前Block的坐标，这样通过local_tile这个API我们就可以得到一个gI Tensor，它就可以拿到当前Block（Block 0）负责的区域。在右边的图中就是绿色标注出来的元素，也就是左上角的2x2小矩阵。那么我们一共需要4个Block去把这个转置的任务完成。

现在我们得到Block需要的Tile之后，我们就可以把它传入给local_paritition，第一个参数就是local_tile得到的Tensor，第二个参数就是Block内的线程排布的情况，这里以blockDim.x=2，blockDim.y=1举例子，再次强调这里只是为了举例子。这样，我们的一个Tile是2x2，而线程的排布是2x1，那么一个线程就要负责一个1x2的Tile。Slides中图的红色部分就表示第0个Block的第0号线程要负责的数据。然后调用copy就可以做到并行的拷贝了。在这个例子中，0号线程会把offset是0的idata就是a拷贝到odata的offset 0号位置，1号线程会把offset是4的idata就是e拷贝到odata的offset 1号位置。

Slides最下方的表格展示了通过Layout我们还可以做到COPY，BROADCAST，GATHER等等操作。另外，你做这些操作你几乎不用改左边的任何的实现代码，只需要把Layout改成你需要的形式就可以了。


![](https://files.mdnice.com/user/59/81ad28b5-931c-4b90-ad00-0732a478ef55.png)


TiledCopy就是我们用来以Tile为单位的Copy时去构造source/dest tensor需要用到的东西。包括MMA的话，也是不同的MMA实现需要不同的Tile的形式，也需要TiledCopy去做。想构造一个Tiled Copy需要用到make_tiled_copy这个API。第一个参数传的还是Copy_Atom，第二个参数就是Dest Tensor的Stride的Layout，第三个参数是Dest Tensor的Value Layout。

Value Layout不好理解，可以用print_latex把你构造好的一个Tiled Copy打印出来，就是Slides下面的图。最左边的图就是每一个线程在它的Source Tensor里面去负责读取哪些数据，比如在这个例子里面T0需要读取列方向最开始的4个数据，T1是列方向接下来的4个。右边的图就是在Dest Tensor里面每个Thread需要去写哪些数据，在这个例子里和读Tensor的位置是一样的。

通过这个图来理解代码，首先32x8就是说线程在M这个方向是32个线程，然后在K这个方向是8个线程。Value Layout的4x1就是在M方向上要读连续的4个数据，在K方向只需要读一个数据。所以这里构造出来的Tiled Copy它的基本的Copy单位就是一个(32x4, 8x1)=(128, 8)这样的Tile。

拿到Tiled Copy之后首先需要get_slice把当前的线程号传进去，这样会得到一个Thread Copy表示当前线程需要Copy的Tile，然后用它去做partition_S并且把Source Tensor传进去，就可以直接拿到当前线程需要负责拷贝的Source Tensor的数据有哪些。它的Shape就是CPY_M和CPY_K，然后CPY就是我们刚才说的128x8的这个Tile大小，然后CPY_M和CPY_K分别表示它需要在M方向以及K方向做这么多次Copy，才能把gA这个Tensor完整的拷贝过去。同理对于Dest Tensor来说，我们需要调用一个partition_D同样可以得到（CPY, CPY_M, CPY_N）这个shape的Tensor，然后再调用copy这个API就可以了。

最右边的图画了一下Tiled Copy的封装层级，最底层它有Copy_Op和Copy_Traits这两个概念，Copy_Op就是底层的数据传输指令，是PTX的代码，然后Copy_Traits是关于代码的元信息，比如线程的Layou是什么样的。这两个就可以封装层我们最常用的Copy_Atom，CopyAtom再去封装得到TiledCopy，然后我们去划分Source Tensor和Dest Tensor需要通过get_slice拿到ThreadCopy，去做一个划分。

![](https://files.mdnice.com/user/59/60eb07c1-59a3-4998-aa9a-77e8e008fa3e.png)

我们想构造一个Tiled Copy应该怎么设置Thread Layout和Value Layout呢？这个和Copy_Atom的指令是相关的。这里讲一个LDSM的例子展示一下应该怎么设置这个参数。LDSM就是ld.maxtrix这个指令，这个指令就是以一个warp为单位去load 1个/2个/4个 8x8矩阵的指令。这个指令有2种形式，一种是Trans的，一种是非Trans的。在CuTe里面把它封装成LDSM_后缀，LDSM_N代表非Trans的类型。对于非Trans类型可以在这里打印一下Copy_Atom，非Trans表示Source Thread会读取一列连续的数据，然后Dest里的Thread会拿连续的一列里面的2个元素。
对于Trans类型，不同在于Source一个线程仍然是拿8个连续元素，但是Dest这边会把Source这边的一个线程连续的8个元素分给8个不同的线程，而一个线程的元素会来自两个不同的Source的线程。如果使用非Trans类型的话，Layout一定要传一个col-major进来，同理对于Trans类型一定需要一个raw-major的Source Tensor。
 
接下来看Stride/Value Layout的设置，这是一个Warp级别的指令，线程数一定要是32的倍数。我们先看非Trans的情况，Dest Tensor这里是在m方向上4个线程去负责连续的8个数据， 意味着这里的线程Layout一定要是4的倍数，并且M方向一定是取2个连续的数据。另外，对于Thread Layout的另外一个没有标注为红色的数字，我们是可以扩大的然后得到一个更大的Tile。

同理，对于Trans类型，它是8个连续的数据分给8个不同的线程，每个线程拿1个数据，所以在K维度上线程必须是8的倍数，并且K方向的Value Layout必须是1了。然后，M方向的Value Layout值是2，这里Dest Tensor的某个线程在M方向上拿到的2个数据是不连续的2个数据。

> Tiled Copy我听得比较迷糊，建议大家学习下reed佬的CuTe文章。

![](https://files.mdnice.com/user/59/e8aa7932-7945-4a98-ba35-34fac57c1d41.png)

接下来开始看GEMM中的数据传输是怎么做的，我们以矩阵A为例想一下怎么把数据从Global Memory传输到Shared Memory。

首先，我们需要用make_tiled_copy来构造一个Tiled Copy，然后get_slice拿到对应的Thread Copy。接着，我们来构造Copy的Source Tensor，这个时候Soruce Tensor是来自Global Memory的，然后我们需要以Block的形式把它拷贝到Shared Memory，这里就需要用到local_tile这个指令，第一个参数传的就是Global Memory里面的Tensor mA，然后把Block Shape/Thread传进去，Step是<_1, X, _1>{} , 这是因为CUTLASS里面会把Block写成M, N, K三维结构，对于A来说没有N这个维度，所以这里就设置为X，表示这个维度不参与计算。通过这个local_tile我们就得到了gA，它就是当前Block需要负责的一个Source Tensor的表示。它的Shape就是（BLK_M, BLK_K, k），BLK_M和BLK_K就是这个Tile的Shape，k就表示一共有k个这个Shape的Tile要做拷贝。然后构造Dest Tensor的时候，由于Dest Tensor是在Shared Memory上，我们直接用make_tensor就可以了。这里的gA和sA拿到的是当前Block负责的数据区域，我们还需要进一步使用我们刚才获得的Thread Copy，分别用partition_S和partition_D得到当前线程负责的区域，来看Shape的话，对于partition_S得到的就是(ACPY, ACPY_M, ACPY_K, k)，这里的k还是保持不变，即一共有k个Tile，只不过在拷贝当前这个Tile的时候需要以ACPY为单位，然后分别在M和K方向上拷贝这么多次。对于Dest Tensor的话，Shape的最后一个维度就不是k了，是PIPE，因为我们需要用到Pipline，PIPE的意思就是一共有多少个Stage。

![](https://files.mdnice.com/user/59/3a28a727-6902-427e-8eb4-2c4441a9365f.png)

接着是Shared Memory到Register File的拷贝，因为Register File是直接要用来做MMA的，所以数据的排布是不能随便设置的。我们可以直接使用CuTe提供的make_tiled_copy_A这个API来构造Tiled Copy，这样就不需要设置它的Thread Layout和Value Layout了，只需要把我们构造的一个用于做MMA的tile_mma传过去就可以自动为我们计算我们需要用什么样的Layout去做Copy。然后同样还是用get_slice拿到Thread Copy。然后Source Tensor因为它不涉及Register File，所以还是partition_S就可以完成。然后在Dest Tensor涉及到MMA所以有一些不一样，所以我们需要用MMA的get_thread_slice去拿到thread_mma，然后使用partition_fragment_A去拿到MMA视角的当前Thread需要负责的Tensor是什么。最后还需要使用retile_D才可以得到我们的Copy视角下我们需要负责的一个Tensor是什么。最后还是一样的copy完成数据拷贝。


## Mixed GEMM Walk Through

![](https://files.mdnice.com/user/59/3acdc1f3-e425-4380-ae41-3c12c43f66a5.png)

这部分更多的是在Concept级别的事情去怎么做，它是会组合上面提到的CuTe相关的代码以及《TensorRT-LLM中的 Quantization GEMM（Ampere Mixed GEMM）的 CUTLASS 2.x 实现讲解》里面讲到的Fast Convert的相关代码，主要是展示怎么做。

![](https://files.mdnice.com/user/59/a7fdc769-cc83-419f-beca-83c61026c621.png)

这张Slides展示了Hopper上的WGMMA的PTX，定义了Hopper上的WGMMA在做什么。《TensorRT-LLM中的 Quantization GEMM（Ampere Mixed GEMM）的 CUTLASS 2.x 实现讲解》里面介绍到的CUTLASS 2.x的Ampere的这些Tensor Core是同步的。同步意味着输入输出A，B，C都是在寄存器层面发射一条同步指令。Hopper上这个指令变成异步之后它可以接收来自Shared Memory的矩阵A，B了。然后，在Hopper架构中我们仍然没有一种（除了FP8这种指令）FP8和FP16直接计算的指令，所以我们要做的事情和《TensorRT-LLM中的 Quantization GEMM（Ampere Mixed GEMM）的 CUTLASS 2.x 实现讲解》中介绍的差不多。我们的数据是Mixed的，但是我们读上来的数据要做Conversion。然后我们可以把weight权重放到矩阵A那边，然后读出来之后我们做一些Conversion，这个数据留存在寄存器上，然后我们把原来的矩阵A放在矩阵B的位置，这样直接去读就可以了。

然后这张Slides的内容也比较多，看图片比较模糊，这里简要说明下。这张Slides是关于如何实现 Mixed 数据类型的通用矩阵乘加（GEMM）操作，特别是在使用Hopper架构下的异步Warp Group矩阵乘加累积（MMA）操作时。内容涵盖了异步Warp Group级的矩阵乘法和累积操作的执行方法，支持的数据类型和矩阵形状，以及相关的编程指令。具体内容包括：

- 异步Warp Group级的矩阵乘法和累积操作
    - **操作类型**：介绍了两种基本操作，一种是矩阵D作为输入和累积器被禁用的情况，另一种是常规的矩阵乘法和累积。
    - **执行步骤**：描述了执行这些操作的六个步骤，包括：
        - 将矩阵A、B和D加载到寄存器或共享内存中。
        - 执行fence操作，确保寄存器/共享内存中的操作对warp组可见。
            - 执行`wmma.fence`操作，确保所有在warp组内对寄存器或共享内存的写入都已经完成，这是确保数据一致性的关键步骤。
            - 执行`fence.proxy.async`操作，这是一个代理操作，用于在异步代理中使通用代理操作可见。
        - 发起异步矩阵乘法和累积操作。
            - 通过`wmma.mma_async`指令进行异步矩阵乘加操作。这个指令允许在不阻塞其他GPU操作的情况下，进行矩阵乘法和累加运算。
        - 创建wmma组并提交所有之前的操作
            - 使用`wmma.commit_group`指令来创建一个wmma操作组，并提交所有挂起的`wmma.mma_async`操作到这个组中。
        - 等待wmma组操作完成
            - 确保在继续之前，所需的wmma组已经完成所有操作。
        - 完成wmma组操作
            - 一旦wmma组完成，所有的`wmma.mma_async`操作也就全部执行完毕。
- 异步Multiply-and-Accumulate指令
    - `wmma.mma_async`指令：具体介绍了如何使用此指令执行矩阵乘加操作。
    - 语法：提供了不同数据类型（如半精度浮点型）的语法示例。
- 支持的矩阵形状和数据类型
    - 数据类型：列出了支持的多种数据类型，如半精度浮点、整数等。
    - 矩阵形状：详细列出了操作支持的矩阵形状，如16x16x16、32x8x16等，这有助于开发者选择适合其特定应用需求的矩阵形状。


![](https://files.mdnice.com/user/59/18d81d6c-9b73-4301-b225-b99e6698fafa.png)

这张Slides讲解了如何实现混合数据类型的GEMM(通用矩阵乘法)。主要内容包括:
- 我们想要的是:
    - A和B矩阵的数据类型不同，例如A(激活)用FP16/FP8，B(权重)用INT8/INT4/FP8
    - 低精度权重可能有缩放因子或零点
- 我们有：
    - 新引入的异步WGMMA
    - 可以从Smem或RF接收输入矩阵A
    - 只能从Smem接收输入矩阵B
    - 矩阵A和B的数据类型应该相同
- 实现方法:
    - **交换**A和B，让低精度数据始终作为矩阵A
    - 将高精度数据加载到smem (不需要做任何Conversion)
    - 将低精度数据加载到smem (这个是必然要求，我们要做MultiStage，数据一定要从Gloabl中Stream到各个Stage的Shared Memory中)
    - [可选]将缩放因子和零点加载到smem
    - 将低精度数据转换为高精度并保存在**RF**中
    - 触发WGMMA_RS来计算数据


> 补充：
> - WGMMA: Hopper Warp Group MMA (矩阵乘累加操作)
> 解释：这是Hopper架构上的Warp组矩阵乘累加操作
> - WS: Warp Specialized
> 解释：表示warp专业化，即不同的warp执行不同的专门任务
> - SS: Src operator of GMMA are both from SMEM
> 解释：GMMA操作的两个源操作数都来自共享内存(SMEM)
> - RS: Src operator A of GMMA is from RF, Src operator B is from SMEM

下面这张Slides的大部分内容是在《CUTLASS 2.x & CUTLASS 3.x Intro 学习笔记》里面有讲过的，学习之前贴到下面再复习一下再贴这节课的Slides方便大家比较；

![](https://files.mdnice.com/user/59/ee00936a-5a53-43ad-aa69-f10a39863f03.png)

这张Slides介绍了一下Hopper架构上的Warp Specialized GEMM实现，采用了生产者-消费者模型。内容如下：
- 源代码位置：cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized_mixed_input.hpp
- 总体架构分为Producer Warps (TMA Warps) 和 Consumer Warps (TC Warps)，通过共享内存进行数据交换。
- Producer Warps (TMA Warps):
    - 使用CollectiveMma::load(...) & Persistent方法
    - 等待smem_empty barrier
    - 发出TMA指令加载A和B矩阵，更新smem_full barrier
    - 更新传输字节数并到达smem_full barrier
    - 循环K次迭代
- Consumer Warps (TC Warps):
    - 使用CollectiveMma::mma(...) & Persistent方法
    - 等待smem_full barrier
    - 发出WGMMA_SS指令并等待前一个TC工作完成
    - 到达smem_empty barrier
    - 循环K次迭代
    - 使用SWIZZLE将寄存器文件(RF)写入共享内存(SMEM)
    - 发出TMA指令将结果写回全局内存
- 共享内存结构：
    - 包含Mbarrier和Data Buffer两部分
    - 每个stage有两个buffer：Mat A MtilexKtile 和 Mat B NtilexKtile
    - 使用smem_empty和smem_full标志来同步Producer和Consumer
- 执行流程：
    - Producer和Consumer交替工作，通过共享内存和 barrier机制同步
    - 多个stage (0 到 N-1) 用于流水线操作
    - 循环执行直到完成所有tile的计算

![](https://files.mdnice.com/user/59/1c20828d-cc95-46f3-8fe8-62a7a614c025.png)

对比上面的Hopper架构上的Warp Specialized GEMM实现，这里的Consumer Warps少了一个Persistent方法，只关注CollectiveMma::mma(...)。中间的Shared Memory之前是两个Data Type一模一样的，但在这里我们交换了A,B矩阵，并且它们的Data Type是不一样的。所以，这里故意画的矩阵A的这个Buffer的长度要短一点，表示低精度。

- Slides的流程图展示了数据处理的pipeline，包括：
    - 生产者线程束（Producer Warps / TMA Warps）
    - 共享内存（Shared Memory）
    - 消费者线程束（Consumer Warps / TC Warps）
- 生产者线程束的工作流程：
    - 等待smem_empty barrier
    - 发出TMA指令加载A和B矩阵，并更新smem_full barrier
    - 可选：发出TMA指令加载缩放因子和零点
    - 更新传输字节并到达smem_full barrier
- 消费者线程束的工作流程：
    - 等待smem_full barrier
    - 将低精度数据转换为高精度，并保存在RF中
    - 发出WGMMA_RS指令
    - 等待前一个TC工作完成
    - 到达smem_empty barrier

还需要注意下这张Slides里面去掉了K方向的循环，但实际实现中仍然是有的。

![](https://files.mdnice.com/user/59/ae569fb6-ea1e-41e1-a967-6a72c50adf65.png)

![](https://files.mdnice.com/user/59/4358d836-640d-4cda-a561-edde69b5efd7.png)

这两张Slides分别对生产者线程束 (Producer Warps) 和消费者线程束（TC Warps）对应流程的一部分底层代码进行了解释，这些代码比较抽象和复杂，视频里面也没怎么讲，感兴趣的可以去深入CutLass的源代码研究。

![](https://files.mdnice.com/user/59/85d1a1b6-0350-4a9e-8e2a-614877ceaae9.png)

![](https://files.mdnice.com/user/59/4b4cd1cf-e802-4872-9b01-85b17327a820.png)

最后这张Slides是介绍消费者线程束（TC Warps）中将低精度数据转换为高精度并保存在寄存器文件（RF）中的实现细节，以及具体是怎么做的Copy的细节。

## 总结

总的来说，这个演讲是一个针对CUTLASS 3.x版本在Hopper架构上实现混合精度矩阵乘法的技术演讲的总结。主要内容包括:

- CuTe工具库的介绍,其核心概念是Layout和Tensor,可以灵活处理复杂的索引问题。
- 以矩阵转置和GEMM数据传输为例,展示了CuTe在数据操作和并行计算中的强大功能。
- 详细讲解了Hopper架构上异步Warp Group级矩阵乘加操作的执行方法和支持的数据类型。
- 介绍了如何利用异步WGMMA指令以及CuTe实现Mixed GEMM的一些细节。

从这个笔记可以简要了解一些概念，起一个科普和熟悉CuTe相关API的作用，如果要进一步学习则需要继续深入学习CuTe和CutLass，当然这里的内容对学习CutLass也是有帮助的。

