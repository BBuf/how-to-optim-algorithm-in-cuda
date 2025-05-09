来自：https://zhuanlan.zhihu.com/p/20579515046

![](https://files.mdnice.com/user/59/17e8530e-73ee-4c95-9048-3a154ead44e8.png)

# 实用 Swizzle 教程（一）

本文实验仓库地址：Chtholly-Boss/swizzle: A practical way of learning Swizzle

## 前言

最近科研工作中需要使用 Tensor Core 进行算子优化，在优化过程中发现了大量的 Bank Conflict
，感到十分苦恼。听说 CUTLASS

提出了一种 Bank Conflict Free 的 Swizzle 技术，感觉很高大上，遂进行了高强度的 RTFM 和博客阅读。 Zhihu 上对于 Swizzle 的讲解已有了不少的优秀的文章，笔者看完后感觉十分长脑子，但在落地的时候我的手跟我说他还没学会，他对以下几点感到十分苦恼：

    他不想使用 CUTLASS 库，因为还不会
    他不知道从哪里写起，大部分博客没代码抄，或者有代码，但调的 CUTLASS

笔者对他感到十分失望，遂经过两三天的实验后，为他写下了本文，试图教会他 **怎么将 Swizzle 技术应用到算子中以消除 Bank Conflict**

## 问题的产生

Swizzle 能解决 Bank Conflict，但哪里来的 Conflict 呢？这就要从我们最开始的目标说起，即 “使用 Tensor Core 进行算子优化”。具体来说，在查阅 CUDA Programming Guide 7.24(https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-matrix-functions) 后我们知道可以通过如下的方式调用 Tensor Core API 完成一次 `(m,n,k) = (16,16,16)` 的 `C = A B^T FP16` 矩阵乘法：

```c++
__device__ void mma_simple(half *a, half *b, half *c) {
    using namespace nvcuda::wmma;
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, half> c_frag;

    load_matrix_sync(a_frag, a, 16);
    load_matrix_sync(b_frag, b, 16);

    fill_fragment(c_frag, 0.0f);

    mma_sync(c_frag, a_frag, b_frag, c_frag);

    store_matrix_sync(c, c_frag, 16, mem_row_major);
}
```


