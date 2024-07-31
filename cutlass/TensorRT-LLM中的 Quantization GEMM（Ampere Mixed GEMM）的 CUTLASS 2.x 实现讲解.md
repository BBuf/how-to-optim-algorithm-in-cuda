> 在LLM的推理和部署中，低精度量化对于性能的提升十分关键，本次分享将为大家介绍TRT-LLM中是如何基于CUTLASS 2.x来实现PerChannel/AWQ/SmoothQuant等量化方法在模型推理过程的计算。Slides来自BiliBili NVIDIA英伟达频道 上传的《TensorRT-LLM中的 Quantization GEMM（Ampere Mixed GEMM）的 CUTLASS 2.x 实现讲解》视频讲解。这里参考视频并更详细记录了每一页Slides的要点，通过这个视频了解下在TRT-LLM中如何使用CUTLASS 2.x来自定义Mixed GEMM算子。我将其作为CUDA-MODE的CUTLASS课程的前置学习内容。


## 总览&目录

![](https://files.mdnice.com/user/59/d1896cc0-2892-4068-bbe2-fd164d79270b.png)

![](https://files.mdnice.com/user/59/9c3e97af-8bb8-4118-83e7-7efdeae01247.png)

课程目录如本章Slides所示，作者首先会介绍一下TRT-LLM推理的一些量化方法，然后通过没有代码的方式介绍了一下CUTLASS 2.x的整体流程以及如果我们想实现Mixed GEMM需要做什么修改，最后挑重点讲了一下为什么TRT-LLM里面关于Mixed GEMM在Ampere上要这么设计Weight Layout，主要是从性能以及CUTLASS 2.x本身的限制来考虑的。

## TRT-LLM中的量化

![](https://files.mdnice.com/user/59/0c5006e3-2ebb-4080-998c-cb13f5d80604.png)

在TensorRT中量化方法主要分为2类，一类是Mixed GEMM，也就是Activation和Weight的数据类型是不同的，例如AWQ，GPTQ，PerChannel。另外一类是Universal GEMM，例如SmoothQuant和FP8，它们的Activation和Weight的数据类型是相同的。

![](https://files.mdnice.com/user/59/3dbe4075-dcc5-4551-89d7-a905672f9936.png)

首先来看PerChannel在推理时的计算流程，可以看到它在推理时会先对Weight进行乘Scales的反量化，然后再执行一个正常的GEMM，流程比较简单。

![](https://files.mdnice.com/user/59/23ebc26d-2f36-404e-bcee-a64079d36589.png)

对于AWQ/GPTQ来说，权重的量化不再是PerChannel的而是GroupWise的，也就是在K方向会有GS组Scales和Zeros，例如假设K/GS=128，那就是在K方向有128行的Weight共享一个Scales和Zeros。因此，它和PerChannel的差异就是需要在反量化的时候乘以Scales并加上Zeros。除此之外，AWQ本身需要在Activation计算之前乘以它自己的ActScale。

![](https://files.mdnice.com/user/59/e608aee2-d950-45a7-9d7a-f934af4e42ef.png)

SmoothQuant不需要像之前的Mixed GEMM量化方法在计算GEMM之前做反量化，它的Scale可以在最后输出的时候apply。它前面的计算部分就是普通的Int8 GEMM，然后再输出的时候乘以PerChannelScales和PerTokenScales。

![](https://files.mdnice.com/user/59/9f910db0-111c-4531-a4e8-390f8bb77c8a.png)

这张Slides讨论了使用CUTLASS如何实现不同的量化技术，并指出了它们与常规GEMM（通用矩阵乘法）的区别。主要内容包括：
- PerChannel/AWQ/GPTQ技术：
    - A/B的数据类型不同：A/B数据所需的位宽不同，提出如何使用ld.global.b128来完成这个操作（在算GEMM的时候，我们首先要保证同一个线程或者warp从A，B矩阵加载的元素个数是相同的，因为它们要在K方向进行一个类似于向量点积的运算。假设我们都用128 bit的load，A矩阵假设16bit那么一下load进来了8个元素，但对于B矩阵如果你要load 8个元素，那么只能用ld.global.b32，也就是说无法使用效率更高的ld.global.b128指令。那么我们需要注意怎么调整Layout或者使用其它的方法尽量让B矩阵也用上效率更高位宽的load指令）
    - 需要额外的输入张量：scales/zeros
        - 需要更多的Shared Memory（我们从之前《CUTLASS 2.x & CUTLASS 3.x Intro 学习笔记》可以知道，我们也需要把scales和zeros也放到shared memory里面里用MultiStage让计算和访存Overlap起来）
        - 如何处理分组（group-wise）情况？
    - 在进行矩阵乘法（MMA）之前需要反量化
        - 需要额外的CUDA核心fma指令(在算GEMM前需要把一个低比特的数据反量化回和Activation的数据类型一致)
- SmoothQuant技术：
    - 需要额外的输入张量：PerTokenScales/PerChannelScales
    - 在GEMM计算之后需要应用特定的缩放。（对于SmoothQuant只需要在Epilogue阶段把这两个Tensor load进来乘一下，然后写回global memory）。

## CUTLASS 2.x kernel计算的整体流程

![](https://files.mdnice.com/user/59/2b18e7d6-5b92-424b-818c-0fd8bc287bf7.png)


![](https://files.mdnice.com/user/59/bbc79863-54ec-46a1-b890-2e1b0bcf3698.png)

这张是CUTLASS GEMM的核心概念图。我们以C矩阵为视角，我们把矩阵C切成小块让每个BLOCK去认领一块做计算。接着要指定WARP去做计算，WARP会认领这个小块中的某一块，比如图中Thread Block Tile的绿色块。每个WARP有32个线程，每个线程又应该做哪一部分计算呢？Warp Tile这个图进一步放大细节，其中4个绿色块代表其中一个线程需要负责计算矩阵C的的部分。最后到线程级别，每个线程都有自己的寄存器去负责做自己的工作。再往右就是Epilogue，这个是很多人使用CUTLASS的第一步比如在GEMM后面做一个Activation后处理。最后把数据写回Global Memory完成整个运算过程。分块的关键参数以及Epilogue的操作类型由图上的using语句所指定。

这个图是CUTLASS的概念，但这里还画出了数据流动，数据需要从Global Memory逐级传递的。除了Tiling之外另外一个重要的概念是我们需要把数据尽可能的复用在高级缓存上，享受到更高的带宽，避免频繁的读取global memory的数据。因此，我们要把数据放在Shared Memory, 寄存器上，然后Tensor Core在寄存器上算完后写回Shared Memory（为了合并内存访问，用上更大位宽的load指令），对齐后再从Shared Memory以连续合并缓存的形式写回Global Memory。

在CUTLASS 2.x（类Ampere架构风格）中还提到，除了Tiling和复用内存层级，我们还希望使用软流水的方法把global memory读写latency和计算隐藏起来。也就是做当前计算的时候去读后续计算轮次需要的数据，这个过程可以流水起来。在进入稳定流水之前，也就是图中的Main loop body会有一个建立流水的过程。做了计算预取，当前计算的数据肯定在上几轮计算就取出来了，数据预取叫Prologue，GEMM的整个计算过程分为Prologue, Main Loop和Epilogue部分，使用Tensor Core的GEMM计算发生在Main Loop部分。


![](https://files.mdnice.com/user/59/b34d3200-120d-41b6-9b3d-35472b5cbed4.png)

接下来展示了一下CUTLASS 2.x的流程，我们需要确定thread block怎么分，比如一个thread block它计算多大的一块数据，这块数据位于矩阵里面的哪个位置。首先需要设定一个CTAShapeM和CTAShapeN表示一个线程块需要计算的一个Tile的大小，但是如何确定哪一个thread block计算哪一个块呢？如果ThreadblockSwizzle这个参数为默认值1，那就不会对从thread block到实际计算块的映射有任何改变。那么它的计算就是沿着m的方向，第0个Block去算最左上角这个块，然后沿着m方向顺序排列。如果ThreadblockSwizzle这个参数为2，那么它的映射方式就会变成上面Slides最右边这张图的样子，这样做的好处是什么？我们来了解一下GPU上thread block和实际SM的对应关系，一般会连续发射一系列Block到一系列相邻的SM上，我们知道L2 Cache在所有的SM上都是共享的，所以我们希望尽可能的在不同的SM里面能够更高概率的命中L2 Cache。如果发射的是bid0/1/2/3这4个block，上面Slides中右图的行方向和列方向的Cache命中率会更高，但如果是按照中间图的方式发射，某个放的Cache命中率会很低。总结一下就是让相邻的SM尽可能的去访问一些在空间位置上相邻的数据来提升L2 Cache命中率。这就是BlockSwizzling（块交织）的作用。

![](https://files.mdnice.com/user/59/7f452815-f120-40cc-af0c-c248992ae631.png)

我们在实现CUTLASS GEMM kernel的时候有一个叫做SplitKFactor的参数，这个参数表示它会沿着GEMM的K方向进行切分。例如当SplitKFactor=2的时候就会把K纬度切分成2半，就会在K纬度上产生2个Block，当K维度上blockIdx.z=0的时候会处理前k/2的数据，当blockIdx.z=1的时候会处理后k/2的数据，算完之后，这两个Block就分别只持有最后输出数据的部分和（partial sum）。在Epilogue阶段结束之后会通过一系列的形式把这两个Block里面的数据累加回global memory。这样的形式在CUTLASS的实现中一般有两种，一种叫SplitK Serial，也就是一个串行的形式，会通过信号量或者锁的方式先保证第一个Block写到Global Memory里面去，再去控制第二个Block继续执行这部分，然后把它累加到对应的Global Memory位置。

另外一种形式叫SplitK Parallel，它就是把这两个Block分别写到不同的Buffer里面，然后再用一个reduce的kernel把这两个buffer累加起来。这一页Slides解释了K纬度上是怎么对Block进行划分的。

![](https://files.mdnice.com/user/59/39d27bdc-b3f2-48c5-90e3-172b5b1031a7.png)

然后在确定了Block实际计算哪一个Tile之后，它需要确定在MainLoop中迭代的步数，也就是沿着K方向要迭代多少次。CTAShapeK就决定了在整个GEMM计算过程中沿着K方向循环，它的每一次迭代计算K方向上多少个元素，也就是计算CTAShapeK个元素。然后之前也提到了一个叫Stage Memory的东西，可能会每个Stage就对应了一个CTAShapeK，比如有5个Stage，它就会Prefetch 5个CTAShapeK的数据进来。也就是说它会提前把下面5次计算的数据全部prefetch到Shared Memory里面去。

![](https://files.mdnice.com/user/59/87c9d275-2723-4da2-b466-0b67f4c1c2c3.png)

然后我们需要进一步从BlockShape里面再把WarpShape划分出来，也就是一个Thread Block里面的Warp怎么去排布。这个排布方式在《CUTLASS 2.x & CUTLASS 3.x Intro 学习笔记》也简单介绍了，就是拿MNK三个方向的BlockShape去除上WarpShape就得到了MNK三个方向的warp数。MN方向上的Warp数很好理解，就像刚才的CTAShape一样对应到了最终输出的Thread Block里输出的某一块数据。例如BlockShapeM除以WarpShapeM等于2的话就意味着M方向分成2块，这个方向上的一个Warp计算其中一块输出，N方向也是类似。

但是K方向有一些区别，因为K方向是进行累加的一个方向，因此如果在K方向的WarpShape大于1的话，那么每个Warp只会计算当前这一轮迭代里面的累加和。比如这个BlockShapeK等于64，然后WarpShapeK等于32，K方向上的Warp分别只计算前32个累加和后32个累加和。这样在沿着K方向完成整个迭代之后，K方向的2个Warp就分别也持有一部分的最终的累加和。然后在Epilogue阶段会在把它们写回Shared Memory之后把K方向的两个Warp的数据累加回来。

![](https://files.mdnice.com/user/59/1d4e144a-8428-4668-8f91-fd5528fa1554.png)

前面介绍到了从BlockShape到WarpShape的划分，这里再介绍一下CUTLASS里面的MultiStage是什么。MultiStage就是，首先一个Stage对应着K方向一次迭代的数据，例如Stage数为4的话，prologue就是会把4次迭代的数据给prefetch一下，这个prefetch一般是通过LDGSTS指令(CpAsync)来完成的。这是一个异步的指令，所以提交之后可以等待这个指令的完成。这个指令发射之后就可以开始MainLoop的计算，而MainLoop的计算会等到prologue阶段发射的4个stage里面的第一个stage的数据准备好。这段数据一旦准备好，就意味着这段数据已经从Global Memory拷贝到了Shared Memory，接着下来它就可以在Shared Memory里面把实际的数据load到寄存器里面，接着它就可以再用这段数据去计算Tensor Core指令。这段数据计算完成后就意味着第一个iteration的计算实际都完成了，那它就可以去发射第5个stage的对应的从global到shared memory拷贝的异步指令。这个指令发射完后，就开始真正的第二次迭代。第二次迭代时，它的第二轮的拷贝指令已经在prologue阶段发射了，它只需要等待这个指令完成，完成后就可以拿这段数据去load寄存器然后计算并发射下一条从global到shared memory拷贝的异步指令。

![](https://files.mdnice.com/user/59/38dee903-43cf-4468-a3e2-45a836407892.png)

总结一下就是CUTLASS GEMM在MainLoop阶段沿着整个K方向迭代，首先会把prefetch的数据load到寄存器里面，然后进行Tensor Core的计算，做完计算之后再prefetch下一个stage的数据。这就是整个MainLoop的流程。

![](https://files.mdnice.com/user/59/565674a1-4cf0-400f-970a-b82811ee8e12.png)

MainLoop的计算里面，首先将prefetch的数据load到寄存器里面，然后再从寄存器里面去发起Tensor Core指令的计算，这个过程中实际上也是有一些overlap的。不是把当前需要的数据直接全部从shared memory load出来，然后逐个的去遍历进行MMA（Tensor Core指令）计算。而是像Slides中左图这样，一般TensorCore指令一般是mma.m16n8k16或者mma.m16n8k8这种，那么WarpShape一般来说它的大小是大于等于这个实际的Tensor Core指令shape的，那么就意味着在MNK方向一个warp可能要执行多条TensorCore指令。我们知道在MN方向的Tensor Core指令它对应的确实是一个不同的输出，需要有各自的不同的寄存器去缓存这部分输出。但在K方向是一个累加的方向，所以K方向上无论有多少条Tensor Core指令并不需要去分配额外的寄存器来缓存输出。因此在实际的计算过程中，是拿WarpShapeK去除以InstructionShapeK从而得到在K方向上一个Warp需要发射多少条Tensor Core指令，然后把它变成一个循环，在循环体中每次只会load这一条Tensor Core指令所需的数据到寄存器然后做对应的Tensor Core计算。与此同时，可以并行的去把K方向下一条Tensor Core指令load的数据的指令发射出去。这样就形成了一个从Shared Memory load到寄存器以及Tensor Core指令相互计算之间的Overlap。

![](https://files.mdnice.com/user/59/3634e12e-5d14-4e39-8bc6-632b74713474.png)

完成了刚才的几个步骤之后，基本就是CUTLASS的Prologue和MainLoop的主要部分了。最后还有一个Epilogue计算，Epilogue计算可能有2步，第一步就是把K方向不同warp的数据给累加起来，第二步就是把已经计算完的数据重新从Shared Memory load到寄存器然后执行一些可能存在的Epilogue OP，例如Activation计算，算完后就把数据写回到Global Memory里面。如果SplitKFactor是>1的话，就会用一个锁来控制线后去写入或者是用一个并行的方式写到不同的Buffer，最后Reduction。以上就是整个CUTLASS 2.x kernel计算的整体流程。

![](https://files.mdnice.com/user/59/5824e1d3-1878-4150-8c94-6bbfdc8c2248.png)

这是刚才CUTLASS 2.x kernel计算的整体流程PPT的总结。

## 基于CUTLASS 2.x实现Mixed GEMM

