## how-to-optim-algorithm-in-cuda

本工程记录如何基于 cuda 优化一些常见的算法。请注意，下面的介绍都分别对应了子目录的代码实现，所以想复现性能的话请查看对应子目录下面的 README 。

### 1. reduce学习

这里记录学习 NIVDIA 的[reduce优化官方博客](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) 做的笔记。完整实验代码见[这里](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/reduce) , 原理讲解请看：[【BBuf的CUDA笔记】三，reduce优化入门学习笔记](https://zhuanlan.zhihu.com/p/596012674)

性能和带宽的测试情况如下 (A100 PCIE 40G)：

|优化手段|耗时(us)|带宽利用率|加速比|
|--|--|--|--|
|reduce_baseline|990.66us|39.57%|~|
|reduce_v1_interleaved_addressing|479.58us|81.74%|2.06|
|reduce_v2_bank_conflict_free|462.02us|84.81%|2.144|
|reduce_v3_idle_threads_free|244.16us|83.16%|4.057|
|reduce_v4_unroll_last_warp|167.10us|54.10%|5.928|
|reduce_v5_completely_unroll|158.78us|56.94%|6.239|
|reduce_v6_multi_add|105.47us|85.75%|9.392|

### 2. elementwise优化

将 oneflow 的 elementwise 模板抽出来方便大家使用，这个 elementwise 模板实现了高效的性能和带宽利用率，并且用法非常灵活。完整实验代码见[这里](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/master/elementwise/elementwise.cu) ，原理讲解请看：[【BBuf 的CUDA笔记】一，解析OneFlow Element-Wise 算子实现](https://zhuanlan.zhihu.com/p/591058808) 。这里以逐点乘为例，性能和带宽的测试情况如下 (A100 PCIE 40G)：

|优化手段|数据类型|耗时(us)|带宽利用率|
|--|--|--|--|
|naive elementwise|float|298.46us|85.88%|
|oneflow elementwise|float|284us|89.42%|
|naive elementwise|half|237.28us|52.55%|
|oneflow elementwise|half|140.74us|87.31%|

可以看到无论是性能还是带宽，使用 oneflow 的 elementwise 模板相比于原始实现都有较大提升。

### 3. FastAtomicAdd

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

### 4. UpsampleNearest2D

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


### 5. indexing

在 PyTorch 中对 index_add 做了极致的优化，我这里将 [PyTorch 的 index_add 实现](indexing/index_add_cuda_pytorch_impl.cu) 进行了剥离，方便大家应用于其它框架。具体请看 indexing 文件夹的 README 。其中还有和 oneflow 的 index_add 实现的各个 case 的性能比较结果。整体来说 PyTorch 在 index Tensor元素很小，但Tensor很大的情况下有较大的性能提升，其它情况和 OneFlow 基本持平。详情请看 [indexing/README.md](indexing/README.md) 。

### 6. 学习笔记相关博客

- [【BBuf的CUDA笔记】一，解析OneFlow Element-Wise 算子实现](https://zhuanlan.zhihu.com/p/591058808)
- [【BBuf的CUDA笔记】二，解析 OneFlow BatchNorm 相关算子实现](https://zhuanlan.zhihu.com/p/593483751)
- [【BBuf的CUDA笔记】三，reduce优化入门学习笔记](https://zhuanlan.zhihu.com/p/596012674)
- [【BBuf的CUDA笔记】四，介绍三个高效实用的CUDA算法实现（OneFlow ElementWise模板，FastAtomicAdd模板，OneFlow UpsampleNearest2d模板）](https://zhuanlan.zhihu.com/p/597435971)

