## how-to-optim-algorithm-in-cuda

> 我也维护了一个学习深度学习框架（PyTorch和OneFlow）的仓库 https://github.com/BBuf/how-to-learn-deep-learning-framework 以及一个如何学习深度学习编译器（TVM/MLIR/LLVM）的学习仓库 https://github.com/BBuf/tvm_mlir_learn , 有需要的小伙伴可以**点一点star**

本工程记录如何基于 CUDA 优化一些常见的算法，同时收录了大量 GPU/大模型相关的学习笔记和博客翻译。

> 友情链接：https://github.com/DefTruth/CUDA-Learn-Notes

## 目录结构

```
how-to-optim-algorithm-in-cuda/
├── cuda-kernels/          # 【代码】CUDA 基础算子手写优化实现
│   ├── reduce/
│   ├── softmax/
│   ├── elementwise/
│   ├── gemv/
│   ├── fast-atomic-add/
│   ├── upsample-nearest2d/
│   ├── indexing/
│   └── linear-attention/
├── cuda-mode/             # 【笔记+代码】CUDA-MODE 课程笔记（77+ 讲）
│   ├── code/              #   实验代码（YHs_Sample、cudabmk）
│   ├── slides/            #   课程讲义 PPT/PDF
│   ├── lectures/          #   Lecture 1-77+ 笔记
│   ├── blog-translations/ #   CUDA 博客翻译
│   ├── cute-dsl/          #   CuTe DSL 笔记
│   ├── lei-mao-blogs/     #   Lei Mao CUDA 博客转载
│   ├── practice/          #   课后实战
│   └── tech-notes/        #   GPU 技术专题
├── cutlass/               # 【笔记+代码】CUTLASS / CuTe DSL 学习
│   ├── code/              #   代码（cfx-article-src、cute-examples、swizzle）
│   ├── cute/              #   CuTe Layout 笔记
│   ├── gemm/              #   GEMM 实现解析
│   ├── tma/               #   TMA 教程
│   ├── wgmma/             #   WGMMA 教程
│   ├── swizzle/           #   Swizzle 机制笔记
│   ├── instructions/      #   CUDA 指令笔记
│   └── tutorials/         #   CUTLASS 翻译教程
├── triton/                # 【笔记+代码】Triton 学习
│   ├── code/              #   Python 代码实现
│   └── meetup/            #   Triton 中国 Meetup slides
├── large-language-model/  # 【笔记】大模型推理/训练优化笔记
├── ml-engineering/        # 【笔记】ml-engineering 翻译系列
├── pytorch/               # 【笔记+代码】PyTorch 博客翻译与代码
├── papers/                # 【笔记】论文阅读
│   ├── cuda/
│   └── mlsys/
├── ptx-isa/               # 【笔记+文档】PTX ISA 学习
├── tools/                 # 工具脚本（hfd.sh 等）
└── deprecated/            # 归档：过时/低相关内容
```

---

### 0. **cuda-mode**

- 课程的 Slides 和 脚本：https://github.com/cuda-mode/lectures
- 课程地址：https://www.youtube.com/@CUDAMODE
- 我的课程笔记：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode

一直想系统看一下某个课程系统和科学的学习下 CUDA ，感觉 CUDA-MODE 这个课程能满足我的需求。这个课程是几个 PyTorch 的 Core Dev 搞的，比较系统和专业。不过由于这个课程是 Youtube 上的英语课程，所以要学习和理解这个课程还是需要花不少时间的，我这里记录一下学习这个课程的每一课的笔记，希望可以通过这个笔记帮助对这个课程以及 CUDA 感兴趣的读者更快吸收这个课程的知识。这个课程相比于以前的纯教程更加关注的是我们可以利用 CUDA 做什么事情，而不是让读者陷入到 CUDA 专业术语的细节中，那会非常痛苦。伟大无需多言，感兴趣请阅读本文件夹下的各个课程的学习笔记。

### 1. cuda-kernels（CUDA 基础算子）

各基础算子的 CUDA 优化实现，代码位于 [cuda-kernels/](cuda-kernels/) 目录。

### 2. reduce

这里记录学习 NIVDIA 的[reduce优化官方博客](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) 做的笔记。完整实验代码见[这里](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-kernels/reduce) , 原理讲解请看：[【BBuf的CUDA笔记】三，reduce优化入门学习笔记](https://zhuanlan.zhihu.com/p/596012674) 。后续又添加了 PyTorch BlockReduce 模板以及在这个模板的基础上额外加了一个数据 Pack ,又获得了一些带宽的提升。详细数据如下：

性能和带宽的测试情况如下 (A100 PCIE 40G)：

![图片](https://user-images.githubusercontent.com/35585791/213908763-480d0c07-5709-4829-9903-db17a0ecca89.png)

### 3. elementwise

将 oneflow 的 elementwise 模板抽出来方便大家使用，这个 elementwise 模板实现了高效的性能和带宽利用率，并且用法非常灵活。完整实验代码见[这里](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/master/cuda-kernels/elementwise/elementwise.cu) ，原理讲解请看：[【BBuf 的CUDA笔记】一，解析OneFlow Element-Wise 算子实现](https://zhuanlan.zhihu.com/p/591058808) 。这里以逐点乘为例，性能和带宽的测试情况如下 (A100 PCIE 40G)：

|优化手段|数据类型|耗时(us)|带宽利用率|
|--|--|--|--|
|naive elementwise|float|298.46us|85.88%|
|oneflow elementwise|float|284us|89.42%|
|naive elementwise|half|237.28us|52.55%|
|oneflow elementwise|half|140.74us|87.31%|

可以看到无论是性能还是带宽，使用 oneflow 的 elementwise 模板相比于原始实现都有较大提升。

### 4. FastAtomicAdd

实现的脚本是针对half数据类型做向量的内积，用到了atomicAdd，保证数据的长度以及gridsize和blocksize都是完全一致的。一共实现了3个脚本：

1. https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/master/cuda-kernels/fast-atomic-add/atomic_add_half.cu 纯half类型的atomicAdd。
2. https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/master/cuda-kernels/fast-atomic-add/atomic_add_half_pack2.cu half+pack，最终使用的是half2类型的atomicAdd。
3. https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/master/cuda-kernels/fast-atomic-add/fast_atomic_add_half.cu 快速原子加，虽然没有显示的pack，但本质上也是通过对单个half补0使用上了half2的原子加。

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

可以看到基于 oneflow upsample_nearest2d 的前后向的优化 kernel 可以获得更好的带宽利用率和性能。注意这里的 profile 使用的是 oneflow 脚本，而不是 upsample_nearest_2d.cu ，详情请看 [cuda-kernels/upsample-nearest2d/README.md](cuda-kernels/upsample-nearest2d/README.md) 。


### 6. indexing

在 PyTorch 中对 index_add 做了极致的优化，我这里将 [PyTorch 的 index_add 实现](cuda-kernels/indexing/index_add_cuda_pytorch_impl.cu) 进行了剥离，方便大家应用于其它框架。具体请看 indexing 文件夹的 README 。其中还有和 oneflow 的 index_add 实现的各个 case 的性能比较结果。整体来说 PyTorch 在 index Tensor元素很小，但Tensor很大的情况下有较大的性能提升，其它情况和 OneFlow 基本持平。详情请看 [cuda-kernels/indexing/README.md](cuda-kernels/indexing/README.md) 。

### 7. softmax

学习了oneflow的softmax kernel实现以及Faster Transformer softmax kernel的实现，并以个人的角度分别解析了原理和代码实现，最后对性能做一个对比方便大家直观的感受到oneflow softmax kernel相比于FasterTransformer的优越性。代码位于 [cuda-kernels/softmax/](cuda-kernels/softmax/)。

### 8. linear-attention

学习一些 linear attention 的 cuda 优化技巧。代码位于 [cuda-kernels/linear-attention/](cuda-kernels/linear-attention/)。

![图片](https://user-images.githubusercontent.com/35585791/221142822-1c2ef670-00e2-4782-98de-d35a4eebd33c.png)

### 9. large-language-model

收集了和大语言模型原理，训练，推理优化相关的文章和学习笔记，位于 [large-language-model/](large-language-model/)。

### 10. papers

GPU / AI 系统论文阅读笔记，位于 [papers/](papers/)，分为：
- [papers/cuda/](papers/cuda/)：CUDA 体系结构相关论文
- [papers/mlsys/](papers/mlsys/)：ML 系统（分布式训练、推理）相关论文

### 11. triton

Triton 学习过程中的代码记录和学习笔记，位于 [triton/](triton/)，分为：
- [triton/code/](triton/code/)：Flash Attention、LayerNorm 等 Triton / PyTorch 实现
- [triton/meetup/](triton/meetup/)：Triton 中国 Meetup slides 汇总

### 12. ptx-isa

对 CUDA PTX ISA 文档的翻译和学习，位于 [ptx-isa/](ptx-isa/)。

### 13. pytorch

对 PyTorch 团队发布的 CUDA 技术的学习笔记和博客翻译，位于 [pytorch/](pytorch/)。

### 14. cutlass

CUTLASS / CuTe DSL 相关的学习笔记，位于 [cutlass/](cutlass/)。


## 学习资源

BBuf 公众号笔记文章列表以及 CUDA/大模型 Infra 优质博客资源汇总，见 [RESOURCES.md](RESOURCES.md)。


<a href="https://star-history.com/#BBuf/how-to-optim-algorithm-in-cuda&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=BBuf/how-to-optim-algorithm-in-cuda&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=BBuf/how-to-optim-algorithm-in-cuda&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=BBuf/how-to-optim-algorithm-in-cuda&type=Date" />
  </picture>
</a>