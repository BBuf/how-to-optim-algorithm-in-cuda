## how-to-optim-algorithm-in-cuda

> 我也维护了一个学习深度学习框架（PyTorch和OneFlow）的仓库 https://github.com/BBuf/how-to-learn-deep-learning-framework 以及一个如何学习深度学习编译器（TVM/MLIR/LLVM）的学习仓库 https://github.com/BBuf/tvm_mlir_learn , 有需要的小伙伴可以**点一点star**

本工程记录如何基于 cuda 优化一些常见的算法。请注意，下面的介绍都分别对应了子目录的代码实现，所以想复现性能的话请查看对应子目录下面的 README 。

> 友情链接：https://github.com/DefTruth/CUDA-Learn-Notes

### 0. **cuda-mode**

- 课程的 Slides 和 脚本：https://github.com/cuda-mode/lectures
- 课程地址：https://www.youtube.com/@CUDAMODE
- 我的课程笔记：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode

一直想系统看一下某个课程系统和科学的学习下 CUDA ，感觉 CUDA-MODE 这个课程能满足我的需求。这个课程是几个 PyTorch 的 Core Dev 搞的，比较系统和专业。不过由于这个课程是 Youtube 上的英语课程，所以要学习和理解这个课程还是需要花不少时间的，我这里记录一下学习这个课程的每一课的笔记，希望可以通过这个笔记帮助对这个课程以及 CUDA 感兴趣的读者更快吸收这个课程的知识。这个课程相比于以前的纯教程更加关注的是我们可以利用 CUDA 做什么事情，而不是让读者陷入到 CUDA 专业术语的细节中，那会非常痛苦。伟大无需多言，感兴趣请阅读本文件夹下的各个课程的学习笔记。


### 1. how-to-compile-pytorch-from-source

记录如何手动编译 PyTorch 源码，学习 PyTorch 的一些 cuda 实现。

### 2. reduce

这里记录学习 NIVDIA 的[reduce优化官方博客](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) 做的笔记。完整实验代码见[这里](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/reduce) , 原理讲解请看：[【BBuf的CUDA笔记】三，reduce优化入门学习笔记](https://zhuanlan.zhihu.com/p/596012674) 。后续又添加了 PyTorch BlockReduce 模板以及在这个模板的基础上额外加了一个数据 Pack ,又获得了一些带宽的提升。详细数据如下：

性能和带宽的测试情况如下 (A100 PCIE 40G)：

![图片](https://user-images.githubusercontent.com/35585791/213908763-480d0c07-5709-4829-9903-db17a0ecca89.png)

### 3. elementwise

将 oneflow 的 elementwise 模板抽出来方便大家使用，这个 elementwise 模板实现了高效的性能和带宽利用率，并且用法非常灵活。完整实验代码见[这里](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/master/elementwise/elementwise.cu) ，原理讲解请看：[【BBuf 的CUDA笔记】一，解析OneFlow Element-Wise 算子实现](https://zhuanlan.zhihu.com/p/591058808) 。这里以逐点乘为例，性能和带宽的测试情况如下 (A100 PCIE 40G)：

|优化手段|数据类型|耗时(us)|带宽利用率|
|--|--|--|--|
|naive elementwise|float|298.46us|85.88%|
|oneflow elementwise|float|284us|89.42%|
|naive elementwise|half|237.28us|52.55%|
|oneflow elementwise|half|140.74us|87.31%|

可以看到无论是性能还是带宽，使用 oneflow 的 elementwise 模板相比于原始实现都有较大提升。

### 4. FastAtomicAdd

实现的脚本是针对half数据类型做向量的内积，用到了atomicAdd，保证数据的长度以及gridsize和blocksize都是完全一致的。一共实现了3个脚本：

1. https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/master/FastAtomicAdd/atomic_add_half.cu 纯half类型的atomicAdd。
2. https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/master/FastAtomicAdd/atomic_add_half_pack2.cu half+pack，最终使用的是half2类型的atomicAdd。
3. https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/master/FastAtomicAdd/fast_atomic_add_half.cu 快速原子加，虽然没有显示的pack，但本质上也是通过对单个half补0使用上了half2的原子加。

性能和带宽的测试情况如下 (A100 PCIE 40G)：

|原子加方式|性能(us)|
|--|--|
|纯half类型|422.36ms|
|pack half2类型|137.02ms|
|fastAtomicAdd|137.01ms|

可以看到使用pack half的方式和直接使用half的fastAtomicAdd方式得到的性能结果一致，均比原始的half的原子加快3-4倍。

### 5. UpsampleNearest2D

upsample_nearest_2d.cu 展示了 oneflow 对 upsample_nearest2d 的前后向的优化 kernel 的用法，性能和带宽的测试情况如下 (A100 PCIE 40G)：

|框架|数据类型|Op类型|带宽利用率|耗时|
|--|--|--|--|--|
| PyTorch | Float32 | UpsampleNearest2D forward | 28.30% | 111.42us |
| PyTorch | Float32 | UpsampleNearest2D backward | 60.16% | 65.12us |
| OneFlow | Float32 |UpsampleNearest2D forward | 52.18% | 61.44us |
| OneFlow | Float32 |UpsampleNearest2D backward | 77.66% | 50.56us |
| PyTorch | Float16 | UpsampleNearest2D forward | 16.99% | 100.38us |
| PyTorch | Float16 | UpsampleNearest2D backward | 31.56% | 57.38us |
| OneFlow | Float16 |UpsampleNearest2D forward | 43.26% | 35.36us |
| OneFlow | Float16 |UpsampleNearest2D backward | 44.82% | 40.26us |

可以看到基于 oneflow upsample_nearest2d 的前后向的优化 kernel 可以获得更好的带宽利用率和性能。注意这里的 profile 使用的是 oneflow 脚本，而不是 upsample_nearest_2d.cu ，详情请看 [UpsampleNearest2D/README.md](UpsampleNearest2D/README.md) 。


### 6. indexing

在 PyTorch 中对 index_add 做了极致的优化，我这里将 [PyTorch 的 index_add 实现](indexing/index_add_cuda_pytorch_impl.cu) 进行了剥离，方便大家应用于其它框架。具体请看 indexing 文件夹的 README 。其中还有和 oneflow 的 index_add 实现的各个 case 的性能比较结果。整体来说 PyTorch 在 index Tensor元素很小，但Tensor很大的情况下有较大的性能提升，其它情况和 OneFlow 基本持平。详情请看 [indexing/README.md](indexing/README.md) 。

### 7. oneflow-cuda-optimize-skills

OneFlow 深度学习框架中基于 cuda 做的优化工作，动态更新中。

### 8. FastTransformer

总结 FastTransformer 相关的 cuda 优化技巧。[README_BERT.md](FastTransformer/README_BERT.md) 总结了 BERT 相关的优化技巧。

### 9. softmax

学习了oneflow的softmax kernel实现以及Faster Transformer softmax kernel的实现，并以个人的角度分别解析了原理和代码实现，最后对性能做一个对比方便大家直观的感受到oneflow softmax kernel相比于FasterTransformer的优越性。

### 10. linear-attention

学习一些 linear attention 的 cuda 优化技巧。

![图片](https://user-images.githubusercontent.com/35585791/221142822-1c2ef670-00e2-4782-98de-d35a4eebd33c.png)

### 11. large-language-model-note

收集了和大语言模型原理，训练，推理，数据标注的相关文章。

### 12. mlsys-paper

前研的大模型训练相关 AI-Infra 论文收集以及阅读笔记。 

### 13. triton

Triton 学习过程中的代码记录和学习笔记。

### 14. meagtron-lm

Meagtron-LM 学习笔记。

### 15. triton-meetup

Triton 中国举办的 Meetup 的slides汇总。点卡这个文件夹也可以找到对应的Meetup的视频回放。

### 16. ptx-isa

对 CUDA PTX ISA 文档的一个翻译和学习。

### 17. pytorch-blog-codes

对 PyTorch 团队发布的 cuda 技术的一些学习笔记。

### 18. cutlass

cutlass 相关的学习笔记。

### 19. cuda-paper

cuda 相关的 paper 的阅读。

### 20. 原创学习笔记

<details>
<summary>点击展开/收起 BBuf 的 CUDA 学习笔记列表</summary>

- [【BBuf的CUDA笔记】一，解析OneFlow Element-Wise 算子实现](https://zhuanlan.zhihu.com/p/591058808)
- [【BBuf的CUDA笔记】二，解析 OneFlow BatchNorm 相关算子实现](https://zhuanlan.zhihu.com/p/593483751)
- [【BBuf的CUDA笔记】三，reduce优化入门学习笔记](https://zhuanlan.zhihu.com/p/596012674)
- [【BBuf的CUDA笔记】四，介绍三个高效实用的CUDA算法实现（OneFlow ElementWise模板，FastAtomicAdd模板，OneFlow UpsampleNearest2d模板）](https://zhuanlan.zhihu.com/p/597435971)
- [【BBuf的CUDA笔记】五，解读 PyTorch index_add 操作涉及的优化技术](https://zhuanlan.zhihu.com/p/599085070)
- [【BBuf的CUDA笔记】六，总结 FasterTransformer Encoder(BERT) 的cuda相关优化技巧](https://zhuanlan.zhihu.com/p/601130731)
- [【BBuf的CUDA笔记】七，总结 FasterTransformer Decoder(GPT) 的cuda相关优化技巧](https://zhuanlan.zhihu.com/p/603611192)
- [【BBuf的CUDA笔记】八，对比学习OneFlow 和 FasterTransformer 的 Softmax Cuda实现](https://zhuanlan.zhihu.com/p/609198294)
- [【BBuf的CUDA笔记】九，使用newbing（chatgpt）解析oneflow softmax相关的fuse优化](https://zhuanlan.zhihu.com/p/615619524)
- [CodeGeeX百亿参数大模型的调优笔记：比FasterTransformer更快的解决方案](https://zhuanlan.zhihu.com/p/617027615)
- [【BBuf的cuda学习笔记十】Megatron-LM的gradient_accumulation_fusion优化](https://mp.weixin.qq.com/s/neP8faIXIvj-XlyFjXjWBg)
- [【BBuf的CUDA笔记】十，Linear Attention的cuda kernel实现解析](https://mp.weixin.qq.com/s/1EPeU5hsOhB7rNAmmXrZRw)
- [【BBuf的CUDA笔记】十一，Linear Attention的cuda kernel实现补档](https://mp.weixin.qq.com/s/qDVKclf_AvpZ5qb2Obf4aA)
- [【BBuf的CUDA笔记】十二，LayerNorm/RMSNorm的重计算实现](https://mp.weixin.qq.com/s/G_XvnB4CeEBWTLNefi0Riw)
- [【BBuf的CUDA笔记】十三，OpenAI Triton 入门笔记一](https://mp.weixin.qq.com/s/RMR_n1n6nBqpdMl6tdd7pQ)
- [【BBuf的CUDA笔记】十四，OpenAI Triton入门笔记二](https://mp.weixin.qq.com/s/ZjADeYg5LCyGaLx0chpSZw)
- [【BBuf的CUDA笔记】十五，OpenAI Triton入门笔记三 FusedAttention](https://mp.weixin.qq.com/s/NKShFDrfDGsb0G6PAkUCGw)
- [AI Infra论文阅读之通过打表得到训练大模型的最佳并行配置](https://mp.weixin.qq.com/s/D-14J482SFQf-zh-EFa-1w)
- [AI Infra论文阅读之将流水线并行气泡几乎降到零（附基于Meagtron-LM的ZB-H1开源代码实现解读）](https://mp.weixin.qq.com/s/PXjYm9dN8C9B8svMQ7nOvw)
- [AI Infra论文阅读之LIGHTSEQ（LLM长文本训练的Infra工作）](https://mp.weixin.qq.com/s/u4gG1WZ73mgH9mEKQQCRww)
- [AI Infra论文阅读之《在LLM训练中减少激活值内存》](https://mp.weixin.qq.com/s/WRUmZT5NIbiHSnNrK1vLOw)
- [系统调优助手，PyTorch Profiler TensorBoard 插件教程](https://mp.weixin.qq.com/s/dG-wlwi8oLg8YMQe_A87qQ)
- [在GPU上加速RWKV6模型的Linear Attention计算](https://mp.weixin.qq.com/s/YXtvafdxB1rVeoy0qJmjyA)
- [flash-linear-attention的fused_recurrent_rwkv6 Triton实现精读](https://mp.weixin.qq.com/s/H6wWBxwIJNCzkIlH_uIuiw)
- [flash-linear-attention中的Chunkwise并行算法的理解](https://mp.weixin.qq.com/s/7utRk157_TFxF8gNRCyIyA)
- [硬件高效的线性注意力机制Gated Linear Attention论文阅读](https://mp.weixin.qq.com/s/IVFeHK1ItPVzttmRRa7ycw)
- [GQA，MLA之外的另一种KV Cache压缩方式：动态内存压缩（DMC）](https://mp.weixin.qq.com/s/5pd4fF14ZUgYeM4UXA7ujQ)
- [vAttention：用于在没有Paged Attention的情况下Serving LLM](https://mp.weixin.qq.com/s/F87-Qoo3xYGbwTTYr68guw)
- [大模型KV Cache节省神器MLA学习笔记（包含推理时的矩阵吸收分析）](https://mp.weixin.qq.com/s/cBMrRUdM1IM0T1ji_ODxng)
- [CUDA-MODE 课程笔记 第一课: 如何在 PyTorch 中 profile CUDA kernels](https://mp.weixin.qq.com/s/owF7AFR61SLrOosUPdZPQQ)
- [CUDA-MODE 第一课课后实战（上）](https://mp.weixin.qq.com/s/9XeJPWUsKTaMU2OdPkL-OQ)
- [CUDA-MODE 第一课课后实战（下）](https://mp.weixin.qq.com/s/FCqnQESCQTtlqCG_BSLulA)
- [CUDA-MODE 课程笔记 第二课: PMPP 书的第1-3章速通](https://mp.weixin.qq.com/s/y0fYn8gUqHqEoRO41ftKnA)
- [CUDA-MODE 课程笔记 第四课: PMPP 书的第4-5章笔记](https://mp.weixin.qq.com/s/P87c8LRJ1CEOOyaQw8L-cA)
- [CUDA-MODE课程笔记 第6课: 如何优化PyTorch中的优化器](https://mp.weixin.qq.com/s/qxPYdGZ71DKVLnnYxmvUVA)
- [CUTLASS 2.x & CUTLASS 3.x Intro 学习笔记](https://mp.weixin.qq.com/s/r9b1dGyOr82ooMl4LD1n_Q)
- [CUDA-MODE课程笔记 第7课: Quantization Cuda vs Triton](https://mp.weixin.qq.com/s/1gCgpp49NF7sDw__EpO-nw)
- [TRT-LLM中的Quantization GEMM（Ampere Mixed GEMM）CUTLASS 2.x 课程学习笔记](https://mp.weixin.qq.com/s/NPytrkchX25YRBc_6Zy6nA)
- [CUDA-MODE课程笔记 第8课: CUDA性能检查清单](https://mp.weixin.qq.com/s/zJLDVF-yjuZ_lMjaCHoS5g)
- [TensorRT-LLM 中的 Hopper Mixed GEMM 的 CUTLASS 3.x 实现讲解](https://mp.weixin.qq.com/s/AntEnjuNqrAnU9pe2rGC6Q)
- [通过微基准测试和指令级分析(Instruction-level Analysis)揭秘英伟达Ampere架构](https://mp.weixin.qq.com/s/lmy6Drqh0LbomcaA19Nf8Q)
- [CUDA-MODE课程笔记 第9课: 归约（也对应PMPP的第10章）](https://mp.weixin.qq.com/s/jdZEPLIzgKm8hilXIUKUww)
- [【翻译】Accelerating Llama3 FP8 Inference with Triton Kernels](https://mp.weixin.qq.com/s/v6Ah4uFtI2zTgiAZ3-mKvw)
- [【PyTorch 奇淫技巧】Python Custom Operators翻译](https://mp.weixin.qq.com/s/1P5gXcDhQxavsgo2IYP6rQ)
- [【翻译】教程：在PyTorch中为CUDA库绑定Python接口](https://mp.weixin.qq.com/s/sgFP59OT-Ex2F9zguSr2Rg)
- [【翻译】教程：CUTLASS中的矩阵转置 (使用CuTe把矩阵转置优化到GPU内存带宽上下限)](https://mp.weixin.qq.com/s/IQaD4Cq0SEVjmus1wB4-cg)
- [CUDA-MODE课程笔记 第11课: Sparsity](https://mp.weixin.qq.com/s/28Ku4_EXm0H-ipJX9LKF6g)
- [【PyTorch 奇淫技巧】Async Checkpoint Save](https://mp.weixin.qq.com/s/DcNjBi_rJKvrU9Ssp8Mo0Q)
- [CUDA-MODE课程笔记 第12课，Flash Attention](https://mp.weixin.qq.com/s/IBeBHO5WlS5BfyL0nZaDHg)
- [【翻译】在 GPU 上如何加速 GPTQ Triton 反量化kernel](https://mp.weixin.qq.com/s/CX6lPJOVYRPlpFS_WbGbmg)
- [基于o1-preview解读 Optimized GPTQ INT4 Dequantization Triton Kernel](https://mp.weixin.qq.com/s/xhCNBjFr6m5hPDPGIhDP7w)
- [【翻译】深入探讨 Hopper TMA 单元在 FP8 GEMM 运算中的应用](https://mp.weixin.qq.com/s/cZRoRq_gzAdA2iaMpZ08VA)
- [【翻译】CUTLASS 教程：掌握 NVIDIA® 张量内存加速器 (TMA)](https://mp.weixin.qq.com/s/0J-JihHhfl77AS2uowA1RA)
- [【PyTorch 奇技淫巧】介绍 depyf：轻松掌握 torch.compile](https://mp.weixin.qq.com/s/Z4VG59ihp_r2H75HLGlMaQ)
- [CUDA-MODE 课程笔记 第13课：Ring Attention](https://mp.weixin.qq.com/s/hvqPhNo3l0tL_-lf978euw)
- [【翻译】torch.compile 的详细示例解析教程](https://mp.weixin.qq.com/s/8FwbaP5q4f_VGWE4vobaMw)
- [【翻译】【PyTorch 奇技淫巧】FlexAttetion 基于Triton打造灵活度拉满的Attention](https://mp.weixin.qq.com/s/KJUk-jmwGPrJvVuLQ44DyQ)
- [Flex Attention API 应用 Notebook 代码速览](https://mp.weixin.qq.com/s/ufOKYJn6z19MreiEk0YAEA)
- [【翻译】CUDA-Free Inference for LLMs](https://mp.weixin.qq.com/s/KlxBzBNxyRBnoEr8qXjgeg)
- [CUDA-MODE 课程笔记 第14课，Triton 实践指南](https://mp.weixin.qq.com/s/bWn4epnUAkHc-7nQGJjpyw)
- [【翻译】使用PyTorch FSDP最大化训练吞吐量](https://mp.weixin.qq.com/s/6wNX38rKcFjxLb4ooYQokw)
- [【翻译】使用PyTorch FSDP和Torch.compile最大化训练吞吐量](https://mp.weixin.qq.com/s/YVVau7boVUEnVB6o_qKORA)
- [【ml-engineering 翻译系列】大模型推理](https://mp.weixin.qq.com/s/9417IxdvNMYThjmaSwPBTw)
- [【ml-engineering 翻译系列】AI系统中的网络概述](https://mp.weixin.qq.com/s/dhspQMOHerIpKESb4IWCgg)
- [【ml-engineering 翻译系列】AI系统中的网络 debug](https://mp.weixin.qq.com/s/sne7cjEnzzSW_5bsAn-P3A)
- [【ml-engineering 翻译系列】AI系统中的网络 benchmark](https://mp.weixin.qq.com/s/FlSkBykNIFXfc6TnqOX25A)
- [【翻译】在FSDP2中开启Float8 All-Gather](https://mp.weixin.qq.com/s/44zFNWr5aVtA3zPtegY9dg)
- [【ml-engineering 翻译系列】训练之模型并行](https://mp.weixin.qq.com/s/VTrTM121jEPGEuFaeIT4Cw)
- [梳理下Flash Attention的dispatch逻辑](https://mp.weixin.qq.com/s/Dcw0F4HpV33Uziy2lvNUeA)
- [【ml-engineering 翻译系列】计算加速器之cpu](https://mp.weixin.qq.com/s/IQd4lz8ebQTrkj_lwDXuSA)
- [CUDA-MODE课程笔记 Lecture 16 通过CUDA C++核心库把llm.c移植为llm.cpp](https://mp.weixin.qq.com/s/ynJwHLH9LFKNBYBBWgU25A)
- [GPU 矩阵乘实际可达最大FLOPS测量工具](https://mp.weixin.qq.com/s/kkIxIUaKtSECMNcvma_ayg)
- [CUDA-MODE 课程笔记 第28课 用在生产环境中的LinkedIn Liger kernel](https://mp.weixin.qq.com/s/Mcmii9XYR7zw2H_DA8IUUQ)
- [RMSNorm的精度陷阱：记一次LLM推理精度调查](https://mp.weixin.qq.com/s/Jag-WRH_2w5-GjTYbRnb-Q)
- [如何正确理解NVIDIA GPU利用率的概念 ](https://mp.weixin.qq.com/s/sYJvdqB9PGhEJphMkuSOzw)
- [CUDA-MODE 课程笔记 第29课 Triton内部机制](https://mp.weixin.qq.com/s/7tfTXaG7D208l_5DzN9hBw)
- [GTX 4090 的 cuda graph 诡异](https://mp.weixin.qq.com/s/SAfnlT4aTd67sRqOAoCxQg)
- [【ml-engineering 翻译系列】计算加速器之gpu](https://mp.weixin.qq.com/s/1B52ORme3s2gzpXPXGNNQw)
- [CUDA-MODE课程笔记 第17课 GPU集合通信(NCCL)](https://mp.weixin.qq.com/s/1QdEJKs4a4u3BepNQ716cQ)
- [Triton Kernel 编译阶段](https://mp.weixin.qq.com/s/dw9bP1ZI__0yrf2_wb6nag)
- [使用torchtune把LLaMa-3.1 8B蒸馏为1B](https://mp.weixin.qq.com/s/TfH9tqNjIdNiIi9iwSdY7w)
- [[分布式训练与TorchTitan] PyTorch中的Async Tensor Parallelism介绍](https://mp.weixin.qq.com/s/Jx4B-sF9dudg7OOT-FbsLg)
- [PyTorch 博客 CUTLASS Ping-Pong GEMM Kernel 简介](https://mp.weixin.qq.com/s/QWS9YEjsbM7hzy5tJm--1g)
- [PyTorch博客 《使用 Triton 加速 2D 动态块量化 Float8 GEMM 简介》](https://mp.weixin.qq.com/s/oK45nVPTctIHW-rXbJ128Q)
- [使用NCU和Cursor Claude-sonnet-3.5写出高效cuda算子的正确姿势](https://mp.weixin.qq.com/s/YEw8JZxn15CfLEnK32Jj-Q)
- [Fused AllGather_MatMul Triton工程实现](https://mp.weixin.qq.com/s/oMkyrelpXjc3-KUQBVx6Tg)
- [MoE之年的总结和MoE 推理优化的一些认识](https://mp.weixin.qq.com/s/RXFmnVI_JIlT0Yo6bN3ZHg)
- [SGLang DP MLA 特性解读](https://mp.weixin.qq.com/s/X2uA507VbQVCv3JIQ8EtPA)
- [Windsurf（可平替 Cursor） 的使用体验和技巧](https://mp.weixin.qq.com/s/3PNaEom76jQ8bdxNtYWkkA)
- [SGLang MLA 实现解析](https://mp.weixin.qq.com/s/wRIjy_HHAH_CeEhkZ_BvNg)
- [详解vLLM和SGLang awq dequantize kernel的魔法](https://mp.weixin.qq.com/s/X9AOH1HGXJ3t0jZ5_hd7Ew)

</details>

### 21. CUDA/大模型 学习资料收集

#### 专栏

- [CUDA编程入门及优化 专栏by jie.hang](https://www.zhihu.com/column/c_1522503697624346624)
- [深入浅出GPU优化 专栏by 有了琦琦的棍子](https://www.zhihu.com/column/c_1437330196193640448)
- [CUDA 编程入门](https://www.zhihu.com/column/c_1699097150611595264)

#### CUDA 相关博客

<details>
<summary>点击展开/收起 CUDA优质博客列表</summary>

- [一文读懂nvidia-smi topo的输出](https://zhuanlan.zhihu.com/p/692947173)
- [如果你是一个C++面试官，你会问哪些问题？](https://www.zhihu.com/question/451327108/answer/3299498791)
- [推理部署工程师面试题库](https://zhuanlan.zhihu.com/p/673046520)
- [[C++特性]对std::move和std::forward的理解](https://zhuanlan.zhihu.com/p/469607144)
- [论文阅读：Mimalloc Free List Sharding in Action](https://zhuanlan.zhihu.com/p/665602526)
- [在 C++ 中，RAII 有哪些妙用？](https://zhuanlan.zhihu.com/p/687230917)
- [AI/HPC面试问题整理](https://zhuanlan.zhihu.com/p/663917237)
- [Roofline Model与深度学习模型的性能分析](https://zhuanlan.zhihu.com/p/34204282)
- [FlashAttention核心逻辑以及V1 V2差异总结](https://zhuanlan.zhihu.com/p/665170554)
- [flash attention 1和flash attention 2算法的python和triton实现](https://zhuanlan.zhihu.com/p/662759306)
- [Flash Attention 推公式](https://zhuanlan.zhihu.com/p/646697716)
- [图解大模型计算加速系列：FlashAttention V1，从硬件到计算逻辑](https://zhuanlan.zhihu.com/p/669926191)
- [flash attention完全解析和CUDA零基础实现](https://zhuanlan.zhihu.com/p/658947627)
- [FlashAttention图解（如何加速Attention）](https://zhuanlan.zhihu.com/p/626079753)
- [FlashAttention:加速计算,节省显存, IO感知的精确注意力](https://zhuanlan.zhihu.com/p/639228219)
- [FlashAttention 反向传播运算推导](https://zhuanlan.zhihu.com/p/631106302)
- [比标准Attention提速5-9倍，大模型都在用的FlashAttention v2来了](https://zhuanlan.zhihu.com/p/644324647)
- [FlashAttention 的速度优化原理是怎样的？](https://www.zhihu.com/question/611236756/answer/3134408839)
- [FlashAttention 的速度优化原理是怎样的？](https://www.zhihu.com/question/611236756/answer/3132304304)
- [FlashAttention2详解（性能比FlashAttention提升200%）](https://zhuanlan.zhihu.com/p/645376942)
- [FlashAttenion-V3: Flash Decoding详解](https://zhuanlan.zhihu.com/p/661478232)
- [速通PageAttention2](https://zhuanlan.zhihu.com/p/671293276)
- [PageAttention代码走读](https://zhuanlan.zhihu.com/p/668736097)
- [大模型推理加速之FlashDecoding++：野生Flash抵达战场](https://zhuanlan.zhihu.com/p/665361668)
- [学习Flash Attention和Flash Decoding的一些思考与疑惑](https://zhuanlan.zhihu.com/p/664704050)
- [大模型推理加速之Flash Decoding：更小子任务提升并行度](https://zhuanlan.zhihu.com/p/664264445)
- [FlashAttention与Multi Query Attention](https://zhuanlan.zhihu.com/p/640312259)
- [动手Attention优化1：Flash Attention 2优化点解析](https://zhuanlan.zhihu.com/p/634427617)
- [Flash Attention推理性能探究](https://zhuanlan.zhihu.com/p/652691133)
- [记录Flash Attention2-对1在GPU并行性和计算量上的一些小优化](https://zhuanlan.zhihu.com/p/650947918)
- [[LLM] FlashAttention 加速attention计算[理论证明｜代码解读]](https://zhuanlan.zhihu.com/p/646084771)
- [FlashAttention核心逻辑以及V1 V2差异总结](https://zhuanlan.zhihu.com/p/665170554)
- [【手撕LLM-FlashAttention】从softmax说起，保姆级超长文！！](https://zhuanlan.zhihu.com/p/663932651)
- [动手Attention优化2：图解基于PTX的Tensor Core矩阵分块乘法实现](https://zhuanlan.zhihu.com/p/650374808)
- [flash attention 的几个要点](https://zhuanlan.zhihu.com/p/663381513)
- [GPU内存(显存)的理解与基本使用](https://zhuanlan.zhihu.com/p/462191421)
- [图文并茂，超详细解读nms cuda拓展源码](https://zhuanlan.zhihu.com/p/466169614)
- [大模型的好伙伴，浅析推理加速引擎FasterTransformer](https://zhuanlan.zhihu.com/p/626008090)
- [LLM Inference CookBook（持续更新）](https://zhuanlan.zhihu.com/p/619596323)
- [NVIDIA的custom allreduce](https://zhuanlan.zhihu.com/p/611229620)
- [[论文速读] FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://zhuanlan.zhihu.com/p/548811565)
- [CUDA随笔之Stream的使用](https://zhuanlan.zhihu.com/p/51402722)
- [简单读读FasterTransformer](https://zhuanlan.zhihu.com/p/589442432)
- [cutlass FusedMultiheadAttention代码解读](https://zhuanlan.zhihu.com/p/600373700)
- [简单谈谈CUDA Reduce](https://zhuanlan.zhihu.com/p/559549740)
- [GridReduce - CUDA Reduce 部分结果归约](https://zhuanlan.zhihu.com/p/635456406)
- [CUTLASS: Fast Linear Algebra in CUDA C++](https://zhuanlan.zhihu.com/p/461060382)
- [cutlass源码导读（1）——API与设计理念](https://zhuanlan.zhihu.com/p/588953452)
- [cutlass源码导读（2）——Gemm的计算流程](https://zhuanlan.zhihu.com/p/592689326)
- [CUDA GroupNorm NHWC优化](https://zhuanlan.zhihu.com/p/596871310)
- [传统 CUDA GEMM 不完全指北](https://zhuanlan.zhihu.com/p/584236348)
- [怎么评估内存带宽的指标，并进行优化?](https://www.zhihu.com/question/424477202/answer/2322341112)
- [TensorRT Diffusion模型优化点](https://zhuanlan.zhihu.com/p/592713879)
- [NVIDIA GPU性能优化基础](https://zhuanlan.zhihu.com/p/577412348)
- [一文理解 PyTorch 中的 SyncBatchNorm](https://zhuanlan.zhihu.com/p/555881100)
- [如何开发机器学习系统：高性能GPU矩阵乘法](https://zhuanlan.zhihu.com/p/531498210)
- [CUDA SGEMM矩阵乘法优化笔记——从入门到cublas](https://zhuanlan.zhihu.com/p/518857175)
- [Dropout算子的bitmask优化](https://zhuanlan.zhihu.com/p/517766170)
- [面向 Tensor Core 的算子自动生成](https://zhuanlan.zhihu.com/p/502935328)
- [PICASSO论文学习](https://zhuanlan.zhihu.com/p/500026086)
- [CUDA翻译：How to Access Global Memory Efficiently in CUDA C/C++ Kernels](https://zhuanlan.zhihu.com/p/473133201)
- [CUDA Pro Tips翻译：Write Flexible Kernels with Grid-Stride Loops](https://zhuanlan.zhihu.com/p/472952257)
- [[施工中] CUDA GEMM 理论性能分析与 kernel 优化](https://zhuanlan.zhihu.com/p/441146275)
- [CUDA Ampere Tensor Core HGEMM 矩阵乘法优化笔记 —— Up To 131 TFLOPS!](https://zhuanlan.zhihu.com/p/555339335)
- [Nvidia Tensor Core-CUDA HGEMM优化进阶](https://zhuanlan.zhihu.com/p/639297098)
- [CUDA C++ Best Practices Guide Release 12.1笔记（一）](https://zhuanlan.zhihu.com/p/636103380)
- [CUDA 矩阵乘法终极优化指南](https://zhuanlan.zhihu.com/p/410278370)
- [如何用CUDA写有CuBLAS 90%性能的GEMM Kernel](https://zhuanlan.zhihu.com/p/631227862)
- [如何理解Nvidia英伟达的Multi-GPU多卡通信框架NCCL？](https://www.zhihu.com/question/63219175/answer/2768301153)
- [如何理解Nvidia英伟达的Multi-GPU多卡通信框架NCCL？](https://www.zhihu.com/question/63219175/answer/206697974)
- [如何理解Nvidia英伟达的Multi-GPU多卡通信框架NCCL？](https://www.zhihu.com/question/63219175/answer/3487108775)
- [使用FasterTransformer实现LLM分布式推理](https://zhuanlan.zhihu.com/p/644322962)
- [细粒度GPU知识点详细总结](https://zhuanlan.zhihu.com/p/349185459)
- [https://siboehm.com/articles/22/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM)
- [【CUDA编程】OneFlow Softmax算子源码解读之BlockSoftmax](https://zhuanlan.zhihu.com/p/646998408)
- [【CUDA编程】OneFlow Softmax 算子源码解读之WarpSoftmax](https://zhuanlan.zhihu.com/p/646994689)
- [【CUDA编程】OneFlow Element-Wise 算子源码解读](https://zhuanlan.zhihu.com/p/646990764)
- [【CUDA编程】Faster Transformer v1.0 源码详解](https://zhuanlan.zhihu.com/p/647012855)
- [【CUDA编程】Faster Transformer v2.0 源码详解](https://zhuanlan.zhihu.com/p/650462095)
- [FasterTransformer Decoding 源码分析(七)-FFNLayer MoE(上篇)](https://zhuanlan.zhihu.com/p/670916589)
- [FasterTransformer Decoding 源码分析(八)-FFNLayer MoE(下篇)](https://zhuanlan.zhihu.com/p/672189305)
- [从roofline模型看CPU矩阵乘法优化](https://zhuanlan.zhihu.com/p/655421318)
- [性能优化的终极手段之 Profile-Guided Optimization (PGO)](https://zhuanlan.zhihu.com/p/652814504)
- [有没有大模型推理加速引擎FasterTransformer入门级教程？](https://www.zhihu.com/question/602468960/answer/3203088852)
- [深入浅出GPU优化系列：gemv优化](https://zhuanlan.zhihu.com/p/494144694)
- [NVIDIA Hopper架构TensorCore分析(4)](https://zhuanlan.zhihu.com/p/654067822)
- [GPU host+device的编译流程](https://zhuanlan.zhihu.com/p/655850951)
- [Tensor Core 优化半精度矩阵乘揭秘](https://zhuanlan.zhihu.com/p/658306956)
- [无痛CUDA实践：μ-CUDA 自动计算图生成](https://zhuanlan.zhihu.com/p/658080362)
- [CUDA（三）：通用矩阵乘法：从入门到熟练](https://zhuanlan.zhihu.com/p/657632577)
- [自己写的CUDA矩阵乘法能优化到多快？](https://www.zhihu.com/question/41060378/answer/2645323107)
- [高效CUDA Scan算法浅析](https://zhuanlan.zhihu.com/p/499963645)
- [一次 CUDA Graph 调试经历](https://zhuanlan.zhihu.com/p/661451140)
- [CUDA中的radix sort算法](https://zhuanlan.zhihu.com/p/488016994)
- [NVIDIA Tensor Core微架构解析](https://zhuanlan.zhihu.com/p/660531822)
- [cutlass cute 101](https://zhuanlan.zhihu.com/p/660379052)
- [在GPU避免分支的方法](https://zhuanlan.zhihu.com/p/143571980)
- [Pytorch-CUDA从入门到放弃（二）](https://zhuanlan.zhihu.com/p/48463543)
- [腾讯机智团队分享--AllReduce算法的前世今生](https://zhuanlan.zhihu.com/p/79030485)
- [cute 之 Layout](https://zhuanlan.zhihu.com/p/661182311)
- [cute Layout 的代数和几何解释](https://zhuanlan.zhihu.com/p/662089556)
- [cute 之 GEMM流水线](https://zhuanlan.zhihu.com/p/665082713)
- [Using CUDA Warp-Level Primitives](https://zhuanlan.zhihu.com/p/664395938)
- [CUDA Pro Tip: Increase Performance with Vectorized Memory Access](https://zhuanlan.zhihu.com/p/666480387)
- [cute 之 简单GEMM实现](https://zhuanlan.zhihu.com/p/667521327)
- [cute 之 MMA抽象](https://zhuanlan.zhihu.com/p/663092747)
- [cute 之 Tensor](https://zhuanlan.zhihu.com/p/663093816)
- [cute Swizzle细谈](https://zhuanlan.zhihu.com/p/684250988)
- [基于 CUTE 的 GEMM 优化【2】—— 高效 GEMM 实现，超越 Cublas 20%](https://zhuanlan.zhihu.com/p/696028389)
- [CUDA单精度矩阵乘法(sgemm)优化笔记](https://zhuanlan.zhihu.com/p/638820727)
- [HPC（高性能计算第一篇） ：一文彻底搞懂并发编程与内存屏障（第一篇）](https://zhuanlan.zhihu.com/p/670350655)
- [GPU CUDA 编程的基本原理是什么? 怎么入门?](https://www.zhihu.com/question/613405221/answer/3129776636)
- [如何入门 OpenAI Triton 编程?](https://www.zhihu.com/question/622685131/answer/3217107882)
- [CUDA（二）：GPU的内存体系及其优化指南](https://zhuanlan.zhihu.com/p/654027980)
- [nvitop: 史上最强GPU性能实时监测工具](https://zhuanlan.zhihu.com/p/614024375)
- [使用Triton在模型中构建自定义算子](https://zhuanlan.zhihu.com/p/670326958)
- [CUDA笔记 内存合并访问](https://zhuanlan.zhihu.com/p/641639133)
- [GPGPU架构，编译器和运行时](https://zhuanlan.zhihu.com/p/592975749)
- [GPGPU的memory 体系理解](https://zhuanlan.zhihu.com/p/658081469)
- [nvlink那些事……](https://zhuanlan.zhihu.com/p/639228770)
- [对NVidia Hopper GH100 的一些理解](https://zhuanlan.zhihu.com/p/486224812)
- [黑科技：用cutlass进行低成本、高性能卷积算子定制开发](https://zhuanlan.zhihu.com/p/258931422)
- [乱谈Triton Ampere WMMA (施工中)](https://zhuanlan.zhihu.com/p/675925978)
- [可能是讲的最清楚的WeightonlyGEMM博客](https://zhuanlan.zhihu.com/p/675427125)
- [GPU 底层机制分析：kernel launch 开销](https://zhuanlan.zhihu.com/p/544492099)
- [GPU内存(显存)的理解与基本使用](https://zhuanlan.zhihu.com/p/462191421)
- [超越AITemplate，打平TensorRT，SD全系列模型加速框架stable-fast隆重登场](https://zhuanlan.zhihu.com/p/669610362)
- [[手把手带你入门CUTLASS系列] 0x00 cutlass基本认知---为什么要用cutlass](https://zhuanlan.zhihu.com/p/677616101)
- [[手把手带你入门CUTLASS系列] 0x02 cutlass 源码分析(一) --- block swizzle 和 tile iterator (附tvm等价code)](https://zhuanlan.zhihu.com/p/679929705)
- [[手把手带你入门CUTLASS系列] 0x03 cutlass 源码分析(二) --- bank conflict free 的shared memory layout (附tvm等价pass)](https://zhuanlan.zhihu.com/p/681966685)
- [[深入分析CUTLASS系列] 0x04 cutlass 源码分析(三) --- 多级流水线(software pipeline)](https://zhuanlan.zhihu.com/p/687397095)
- [[深入分析CUTLASS系列] 0x03 cutlass 源码分析(二) --- bank conflict free 的shared memory layout (附tvm等价pass)](https://zhuanlan.zhihu.com/p/681966685)
- [GPU 内存概念浅析](https://zhuanlan.zhihu.com/p/651179378)
- [NV_GPU tensor core 算力/带宽/编程模型分析](https://zhuanlan.zhihu.com/p/638129792)
- [Nsight Compute - Scheduler Statistics](https://zhuanlan.zhihu.com/p/673770855)
- [NVidia GPU指令集架构-前言](https://zhuanlan.zhihu.com/p/686198447)
- [搞懂 CUDA Shared Memory 上的 bank conflicts 和向量化指令（LDS.128 / float4）的访存特点](https://zhuanlan.zhihu.com/p/690052715)
- [窥探Trition的lower(二)](https://zhuanlan.zhihu.com/p/695255185)
- [窥探Trition的lower(三)](https://zhuanlan.zhihu.com/p/696133729)
- [ops(2)：SoftMax 算子的 CUDA 实现与优化](https://zhuanlan.zhihu.com/p/695307283)
- [cuda学习日记(6) nsight system / nsight compute](https://zhuanlan.zhihu.com/p/640344249)
- [ops(3)：Cross Entropy 的 CUDA 实现](https://zhuanlan.zhihu.com/p/695594396)
- [cuda的ldmatrix指令的详细解释](https://zhuanlan.zhihu.com/p/697228676)
- [揭秘 Tensor Core 底层：如何让AI计算速度飞跃](https://mp.weixin.qq.com/s/UL7CLWp3cmdUgGILr4iVzA)
- [NCCL（NVIDIA Collective Communication Library）的来龙去脉](https://zhuanlan.zhihu.com/p/667221519)
- [ldmatrix与swizzle（笔记）](https://zhuanlan.zhihu.com/p/696231622)
- [GPU上GEMM的边界问题以及优化](https://zhuanlan.zhihu.com/p/699776368)
- [NV Tensor Core and Memory Accelerator 理论分析](https://zhuanlan.zhihu.com/p/601204275)
- [CUTLASS CuTe GEMM细节分析（一）——ldmatrix的选择](https://zhuanlan.zhihu.com/p/702818267)
- [Triton到PTX（1）：Elementwise](https://zhuanlan.zhihu.com/p/699979345)
- [由矩阵乘法边界处理引起的CUDA wmma fragment与原始矩阵元素对应关系探究](https://zhuanlan.zhihu.com/p/703476975)
- [NVIDIA Hopper架构TensorCore分析(4)](https://zhuanlan.zhihu.com/p/654067822)
- [NVidia GPU指令集架构-Load和Cache](https://zhuanlan.zhihu.com/p/692445145)
- [NVidia GPU指令集架构-寄存器](https://zhuanlan.zhihu.com/p/688616037)
- [Async Copy 及 Memory Barrier 指令的功能与实现](https://zhuanlan.zhihu.com/p/685168850)
- [tensorcore中ldmatrix指令的优势是什么？](https://www.zhihu.com/question/600927104/answer/3029266372)
- [使用cutlass cute复现flash attention](https://zhuanlan.zhihu.com/p/696323042)
- [1. Cuda矩阵乘法GeMM性能优化](https://zhuanlan.zhihu.com/p/593462636)
- [一步步优化 GEMM by Tensorcore](https://zhuanlan.zhihu.com/p/638522893)
- [CUTLASS 3.x 异构编程随感](https://zhuanlan.zhihu.com/p/689829403)
- [Triton到PTX（1）：Elementwise](https://zhuanlan.zhihu.com/p/699979345)
- [Triton到SASS（2）：Reduction](https://zhuanlan.zhihu.com/p/703748336)
- [cuda的ldmatrix指令的详细解释](https://zhuanlan.zhihu.com/p/697228676)
- [基于 CuTe 理解 swizzle, LDSM, MMA](https://zhuanlan.zhihu.com/p/934430036)
- [一文读懂nsight system与cuda kernel的时间线分析与可视化](https://zhuanlan.zhihu.com/p/691307737)
- [TileLang: 80行Python kernel代码实现FlashMLA 95%的性能](https://zhuanlan.zhihu.com/p/27965825936)
- [简单CUDA Assembly介绍](https://zhuanlan.zhihu.com/p/27455487044)
- [Deep Gemm 代码浅析](https://zhuanlan.zhihu.com/p/26916462532)
- [如何看懂deepseek ai开源的FlashMLA中的核心cu代码？](https://www.zhihu.com/question/13188512132/answer/113811134716)
- [浅析GEMM优化multistage数怎么算](https://zhuanlan.zhihu.com/p/714353243)
- [DeepSeek: FlashMLA代码解析](https://zhuanlan.zhihu.com/p/26269071923)
- [triton(openai)如何实现splitk和streamk?](https://www.zhihu.com/question/13143162788/answer/108685833211)
- [FlashMLA性能简测](https://zhuanlan.zhihu.com/p/26113545571)
- [DeepSeek-V3/R1 的 Hosting 成本预估](https://zhuanlan.zhihu.com/p/23282743306)
- [实用 Swizzle 教程（一）](https://zhuanlan.zhihu.com/p/20579515046)
- [实用 Swizzle 教程（二）](https://zhuanlan.zhihu.com/p/21142007017)
- [CUDA编程入门之Cooperative Groups(1)](https://zhuanlan.zhihu.com/p/572820342)

</details>

#### 大模型Infra相关博客（DeepSeek，VERL, Megatron-LM, SGLang，vLLM等）

<details>
<summary>点击展开/收起 大模型Infra优质博客列表</summary>

- [Megatron-LM 分布式执行调研](https://strint.notion.site/Megatron-LM-86381cfe51184b9c888be10ee82f3812)
- [BLOOM 训练背后的技术](https://www.cnblogs.com/Matrix_Yao/p/17238627.html)
- [聊聊 PyTorch2.0 中新的Distributed API](https://mp.weixin.qq.com/s/hOOFE_eFD6a8GKTdnRcJXg)
- [聊聊 PyTorch 中新的Distributed API （二）](https://mp.weixin.qq.com/s/zDSuToVMo4iK3sxF662kvg)
- [【LLM】从零开始训练大模型](https://zhuanlan.zhihu.com/p/636270877)
- [在一张 24 GB 的消费级显卡上用 RLHF 微调 20B LLMs](https://www.cnblogs.com/huggingface/p/17245966.html)
- [人手一个ChatGPT！微软DeepSpeed Chat震撼发布，一键RLHF训练千亿级大模型](https://zhuanlan.zhihu.com/p/621379646)
- [大型语言模型(LLM)训练指南🚀](https://zhuanlan.zhihu.com/p/611325149)
- [“StackLLaMA”: 用 RLHF 训练 LLaMA 的手把手教程](https://zhuanlan.zhihu.com/p/626896135)
- [图解大模型训练之：流水线并行（Pipeline Parallelism），以Gpipe为例](https://zhuanlan.zhihu.com/p/613196255)
- [图解大模型训练之：数据并行上篇(DP, DDP与ZeRO)](https://zhuanlan.zhihu.com/p/617133971)
- [图解大模型训练之：数据并行下篇( DeepSpeed ZeRO，零冗余优化)](https://zhuanlan.zhihu.com/p/618865052)
- [图解大模型训练之：张量模型并行(TP)，Megatron-LM](https://zhuanlan.zhihu.com/p/622212228)
- [Megatron-LM 中的 pipeline 并行](https://zhuanlan.zhihu.com/p/432969288)
- [图解大模型系列之：Megatron源码解读1，分布式环境初始化](https://zhuanlan.zhihu.com/p/629121480)
- [图解大模型训练之：Megatron源码解读2，模型并行](https://zhuanlan.zhihu.com/p/634377071)
- [聊聊序列并行Sequence parallelism](https://mp.weixin.qq.com/s/ylScQOpJ1-ufyPK7X6VUjw)
- [深入理解 Megatron-LM（1）基础知识](https://zhuanlan.zhihu.com/p/650234985)
- [深入理解 Megatron-LM（2）原理介绍](https://zhuanlan.zhihu.com/p/650383289)
- [深入理解 Megatron-LM（3）代码结构](https://zhuanlan.zhihu.com/p/650237820)
- [深入理解 Megatron-LM（4）并行设置](https://zhuanlan.zhihu.com/p/650500590)
- [深入理解 Megatron-LM（5）张量并行](https://zhuanlan.zhihu.com/p/650237833)
- [聊聊字节 AML 万卡工作 MegaScale: Scaling Large Language Model Training](https://mp.weixin.qq.com/s/aXsURbHZKzoBw-ChaBnjEQ)
- [深度学习里，模型并行中怎么将模型拆分？](https://www.zhihu.com/question/319355346/answer/2985459442)
- [Transformers DeepSpeed官方文档](https://zhuanlan.zhihu.com/p/621572871)
- [DeepSeek-V3 MTP 工程实现思考](https://zhuanlan.zhihu.com/p/29082207943)
- [DeepSeek V3/R1 推理效率分析（1）：关于DeepSeek V3/R1 Decoding吞吐极限的一些不负责任估计](https://zhuanlan.zhihu.com/p/27292649125)
- [DeepSeek V3/R1 推理效率分析（2）: DeepSeek 满血版逆向工程分析](https://zhuanlan.zhihu.com/p/29841050824)
- [DeepSeek V3/R1 推理效率分析（3）：Decode 配置泛化讨论](https://zhuanlan.zhihu.com/p/29540042383)
- [如何估算不同规格的芯片 EP 部署 Deepseek 的单卡吞吐 V1.0](https://zhuanlan.zhihu.com/p/30471846931)
- [深度解析FlashMLA: 一文读懂大模型加速新利器](https://zhuanlan.zhihu.com/p/27976368445)
- [MoE Inference On AnyScale](https://zhuanlan.zhihu.com/p/28680264165)
- [[AI Infra] VeRL 框架入门&代码带读](https://zhuanlan.zhihu.com/p/27676081245)
- [大模型分布式通信技术博客汇总](https://zhuanlan.zhihu.com/p/30451575581)
- [sglang 源码学习笔记（一）- Cache、Req与Scheduler](https://zhuanlan.zhihu.com/p/17186885141)
- [DualPipe 深入浅出：没有分布式训练基础也能看懂的 DualPipe 全方位讲解](https://zhuanlan.zhihu.com/p/27045651854)
- [DeepSeek MLA引发的一些记忆碎片](https://zhuanlan.zhihu.com/p/25210365944)
- [DeepSeek MLA的序列并行和张量并行](https://zhuanlan.zhihu.com/p/25573883266)
- [SGLang: Triton算子extend_attention/Prefix优化](https://zhuanlan.zhihu.com/p/22996351654)
- [DeepSeek-V3 (671B) 模型参数量分解计算](https://zhuanlan.zhihu.com/p/21455638257)
- [PP->VPP->ZeroBubblePP->deepseekv3 dualPipe，对PP bubble的极致压缩](https://zhuanlan.zhihu.com/p/26559590326)
- [双流并行(DualPipe) 没有双流会更好](https://zhuanlan.zhihu.com/p/26915547331)
- [deepseek 训练 profile data 基础分析](https://zhuanlan.zhihu.com/p/26717172494)
- [Deepseek FlashMLA解析](https://zhuanlan.zhihu.com/p/26262350225)
- [给 Megatron 的长文本训练抓了一个 Bug](https://zhuanlan.zhihu.com/p/26109356836)
- [对DualPipe的一些想法](https://zhuanlan.zhihu.com/p/21525151726)
- [SGLang: Triton算子prefill_attention](https://zhuanlan.zhihu.com/p/19989050229)
- [[CUDA基础]📚CUDA-Learn-Notes: v3.0 大升级-面试刷题不迷路](https://zhuanlan.zhihu.com/p/19862356369)

</details>

## Star History


<a href="https://star-history.com/#BBuf/how-to-optim-algorithm-in-cuda&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=BBuf/how-to-optim-algorithm-in-cuda&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=BBuf/how-to-optim-algorithm-in-cuda&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=BBuf/how-to-optim-algorithm-in-cuda&type=Date" />
  </picture>
</a>