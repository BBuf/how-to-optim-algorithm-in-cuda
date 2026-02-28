> 博客来源：https://leimao.github.io/blog/cuBLAS-Transpose-Column-Major-Relationship/ ，来自Lei Mao，已获得作者转载授权。后续会转载一些Lei Mao的CUDA相关Blog，也是一个完整的专栏，Blog会从稍早一些的CUDA架构到当前最新的CUDA架构，也会包含实用工程技巧，底层指令分析，Cutlass分析等等多个课题，是一个时间线十分明确的专栏。

# cuBLAS GEMM API 在列主序和行主序矩阵中的使用方法

## 引言

cuBLAS GEMM API 对输入和输出矩阵的存储格式有非常严格的要求。如果所有矩阵都以列主序格式存储，cuBLAS GEMM API 可以直接使用。但如果某些矩阵以行主序格式存储，为这种矩阵乘法设置 cuBLAS GEMM API 的参数可能会出错。

在这篇博客文章中，我们将讨论矩阵转置与列主序存储之间的关系，以及在不同情况下应该如何使用 cuBLAS GEMM API。

## cuBLAS GEMM

### cuBLAS GEMM API

cuBLAS 单精度 GEMM API 声明如下。

```c++
cublasStatus_t cublasSgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float *alpha,
                           const float *A, int lda,
                           const float *B, int ldb,
                           const float *beta,
                           float *C, int ldc)
```

### 列主序矩阵的 cuBLAS GEMM

此函数执行通用矩阵-矩阵乘法

$$C = \alpha \text{op}(A)\text{op}(B) + \beta C$$

其中 $\alpha$ 和 $\beta$ 是标量，$A$、$B$ 和 $C$ 是以列主序格式存储的矩阵，$\text{op}(A)$ 的维度为 $m \times k$，$\text{op}(B)$ 的维度为 $k \times n$，$C$ 的维度为 $m \times n$。对于矩阵 $A$：

![](https://files.mdnice.com/user/59/04b8b317-d67b-4a0b-a338-745c9b441793.png)

### cuBLAS GEMM 和行主序矩阵

但是如果某些矩阵以行主序格式存储会怎样呢？让我们看几个例子。

假设 $m' \times k'$ 矩阵 $A'$ 以行主序格式存储，而 $k' \times n'$ 矩阵 $B'$ 和 $m' \times n'$ 矩阵 $C'$ 以列主序格式存储。$A'$ 的转置，即 $k' \times m'$ 矩阵 $A'^T$，以列主序格式存储，等价于原始的以行主序格式存储的 $A'$。但为了使用 cuBLAS 执行通用矩阵-矩阵乘法，$A'^T$ 必须转置为 $A'$。在这种情况下，**transa = CUBLAS_OP_T**，**transb = CUBLAS_OP_N**，$m = m'$，$n = n'$，$k = k'$，$A = A'$，$B = B'$，$C = C'$。

假设 $m' \times k'$ 矩阵 $A'$ 和 $k' \times n'$ 矩阵 $B'$ 以列主序格式存储，而 $m' \times n'$ 矩阵 $C'$ 以行主序格式存储。在这种情况下，无法通过 cuBLAS API 对 $C'$ 进行转置。

我们注意到，可以在执行通用矩阵-矩阵乘法之前先对公式中的矩阵 $C$ 进行转置。

$$C'^T = \alpha(\text{op}(A)\text{op}(B))^T + \beta C'^T$$
$$= \alpha\text{op}(B)^T \text{op}(A)^T + \beta C'^T$$
$$= \alpha\text{op}(B^T) \text{op}(A^T) + \beta C'^T$$

因此，如果 $B^T$、$A^T$ 和 $C^T$ 以列主序格式存储，我们仍然可以使用现有的 cuBLAS API 执行通用矩阵-矩阵乘法。

在这种情况下，$C'$ 的转置，即 $n' \times m'$ 矩阵 $C'^T$，以列主序格式存储，等价于原始的以行主序格式存储的 $C'$。此外，矩阵 $A'$ 和 $B'$ 也必须转置。$A'$ 的转置，即 $k' \times m'$ 矩阵 $A'^T$，以行主序格式存储，等价于原始的以列主序格式存储的 $A'$。$B'$ 的转置，即 $n' \times k'$ 矩阵 $B'^T$，以行主序格式存储，等价于原始的以列主序格式存储的 $B'$。在这种情况下，**transa = CUBLAS_OP_T**，**transb = CUBLAS_OP_T**，$m = n'$，$n = m'$，$k = k'$，$A = B'$，$B = A'$，$C = C'$。

## 结论

假设我们想要使用 cuBLAS API 执行矩阵乘法 $C' = \alpha A'B' + \beta C'$，其中 $A'$、$B'$ 和 $C'$ 是形状分别为 $m' \times k'$、$k' \times n'$ 和 $m' \times n'$ 的矩阵。下表总结了矩阵 $A'$、$B'$ 和 $C'$ 的转置与列主序存储之间的关系，以及应该如何使用 cuBLAS API。

| $m' \times k'$ 矩阵 $A'$ | $k' \times n'$ 矩阵 $B'$ | $m' \times n'$ 矩阵 $C'$ | **transa** | **transb** | **m** | **n** | **k** | **A** | **B** | **C** |
|---|---|---|---|---|---|---|---|---|---|---|
| 列主序 | 列主序 | 列主序 | CUBLAS_OP_N | CUBLAS_OP_N | $m'$ | $n'$ | $k'$ | $A'$ | $B'$ | $C'$ |
| 行主序 | 列主序 | 列主序 | CUBLAS_OP_T | CUBLAS_OP_N | $m'$ | $n'$ | $k'$ | $A'$ | $B'$ | $C'$ |
| 列主序 | 行主序 | 列主序 | CUBLAS_OP_N | CUBLAS_OP_T | $m'$ | $n'$ | $k'$ | $A'$ | $B'$ | $C'$ |
| 行主序 | 行主序 | 列主序 | CUBLAS_OP_T | CUBLAS_OP_T | $m'$ | $n'$ | $k'$ | $A'$ | $B'$ | $C'$ |
| 列主序 | 列主序 | 行主序 | CUBLAS_OP_T | CUBLAS_OP_T | $n'$ | $m'$ | $k'$ | $B'$ | $A'$ | $C'$ |
| 行主序 | 列主序 | 行主序 | CUBLAS_OP_T | CUBLAS_OP_N | $n'$ | $m'$ | $k'$ | $B'$ | $A'$ | $C'$ |
| 列主序 | 行主序 | 行主序 | CUBLAS_OP_N | CUBLAS_OP_T | $n'$ | $m'$ | $k'$ | $B'$ | $A'$ | $C'$ |
| 行主序 | 行主序 | 行主序 | CUBLAS_OP_N | CUBLAS_OP_N | $n'$ | $m'$ | $k'$ | $B'$ | $A'$ | $C'$ |


## References

- `cublas<t>gemm()`(https://docs.nvidia.com/cuda/archive/12.6.2/cublas/#cublas-t-gemm)
- `cuBLAS GEMM Transpose and Column-Major Application`(https://github.com/leimao/CUDA-GEMM-Optimization/blob/747e49b418bca18557581969c78333443055fd43/include/profile_utils.cuh#L113)

