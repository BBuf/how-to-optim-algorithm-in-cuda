# cute

CuTe Layout 代数与 CuTe 编程模型学习笔记。

## 笔记列表

- [cute笔记.md](cute笔记.md) — CuTe Layout、Tensor、MMA 抽象，GEMM 流水线等核心概念整理

## 参考资源

- [reed 的 CUDA 高性能编程 cutlass 相关专栏](https://zhuanlan.zhihu.com/c_1696937812497235968)
- [CUTLASS CuTe GEMM细节分析（一）——ldmatrix的选择](https://zhuanlan.zhihu.com/p/702818267)

## 对应代码

相关实验代码位于 [../code/cute-examples/](../code/cute-examples/)：

| 文件 | 对应文章 |
|------|---------|
| `vector_add.cu` | reed 的 [cute Tensor](https://zhuanlan.zhihu.com/c_1696937812497235968) 复原实验代码 |
| `gemm-simple.cu` | [cute 之 简单GEMM实现](https://zhuanlan.zhihu.com/p/667521327) 实验代码 |
| `s2r_copy.cu` | [CUTLASS CuTe GEMM细节分析（一）](https://zhuanlan.zhihu.com/p/702818267) 实验代码 |
