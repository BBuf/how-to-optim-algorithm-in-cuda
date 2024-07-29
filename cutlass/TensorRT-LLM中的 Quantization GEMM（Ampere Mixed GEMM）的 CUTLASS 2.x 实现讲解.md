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

## 基于CUTLASS 2.x实现Quantization GEMM

![](https://files.mdnice.com/user/59/2b18e7d6-5b92-424b-818c-0fd8bc287bf7.png)


