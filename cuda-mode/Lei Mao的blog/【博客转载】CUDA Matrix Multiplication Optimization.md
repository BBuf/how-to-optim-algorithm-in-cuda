> 博客来源：https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/ ，来自Lei Mao，已获得作者转载授权。后续会转载一些Lei Mao的CUDA相关Blog，也是一个完整的专栏，Blog会从稍早一些的CUDA架构到当前最新的CUDA架构，也会包含实用工程技巧，底层指令分析，Cutlass分析等等多个课题，是一个时间线十分明确的专栏。

# CUDA 矩阵乘法优化

## 介绍

通用矩阵乘法（GEMM）是线性代数中的基本运算。它也是许多科学计算应用中非常重要的运算，如机器学习和深度学习。

在本文中，我们将讨论如何使用 CUDA 优化 NVIDIA GPU 上 FP32 GEMM 的性能，以及如何使用 NVIDIA Tensor Core 将 FP32 GEMM 优化扩展到 FP16 GEMM。

## 通用矩阵乘法

GEMM 运算计算 $D = AB + C$，其中 $D \in \mathbb{R}^{m \times n}$，$A \in \mathbb{R}^{m \times k}$，$B \in \mathbb{R}^{k \times n}$，$C \in \mathbb{R}^{m \times n}$。在计算机程序中，通常 $A$ 和 $B$ 是常量输入矩阵，$C$ 将被输出矩阵 $D$ 覆盖。

在我们的实现中，我们假设所有矩阵 $A$、$B$、$C$ 和 $D$ 都以行主序存储在内存中，对于 FP32 矩阵，主维度填充为 64 字节，对于 FP16 矩阵，主维度填充为 32 字节。

## 具有非合并内存访问的朴素实现

朴素实现是使用 2D 块，其中每个线程负责计算输出矩阵的一个元素。具体来说，对于全局线程索引为 $(t_m, t_n)$ 的每个线程，其中 $t_m \in [1, m]$ 和 $t_n \in [1, n]$，它计算

$$D_{t_m,t_n} = \sum_{t_k=1}^{k} A_{t_m,t_k} B_{t_k,t_n} + C_{t_m,t_n}.$$

以下代码片段显示了朴素实现。

```c++
template <typename T>
__global__ void gemm_v00(size_t m, size_t n, size_t k, T alpha, T const* A,
                         size_t lda, T const* B, size_t ldb, T beta, T* C,
                         size_t ldc)
{
    // Compute the row and column of C that this thread is responsible for.
    size_t const C_row_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_col_idx{blockIdx.y * blockDim.y + threadIdx.y};

    // Each thread compute
    // C[C_row_idx, C_col_idx] = alpha * A[C_row_idx, :] * B[:, C_col_idx] +
    // beta * C[C_row_idx, C_col_idx].
    if (C_row_idx < m && C_col_idx < n)
    {
        T sum{static_cast<T>(0)};
        for (size_t k_idx{0U}; k_idx < k; ++k_idx)
        {
            sum += A[C_row_idx * lda + k_idx] * B[k_idx * ldb + C_col_idx];
        }
        C[C_row_idx * ldc + C_col_idx] =
            alpha * sum + beta * C[C_row_idx * ldc + C_col_idx];
    }
}

template <typename T>
void launch_gemm_kernel_v00(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream)
{
    dim3 const block_dim{32U, 32U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(m) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(n) + block_dim.y - 1U) / block_dim.y, 1U};
    gemm_v00<T><<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B,
                                                     ldb, *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}
```
然而，除了朴素算法的其他缺点之外，这个实现还有一个主要问题，即读取和写入全局内存时的非合并内存访问。在我们的实现中，由于疏忽使用快速线程索引来索引矩阵 $A$ 和 $C$ 的行，同一 warp 中的线程从以行主序存储在内存中的矩阵 $A$ 的同一列读取元素，由于读取完全不连续，导致非合并内存访问。当 warp 覆写矩阵 $C$ 的元素时也会发生同样的问题。同一 warp 中的线程读取矩阵 $B$ 的同一元素，导致广播内存访问，这不受疏忽的影响。

这个 FP32 GEMM 实现的性能在 NVIDIA GeForce RTX 3090 GPU 上仅为 0.27 TFLOPS，性能非常差。

## 具有合并内存访问的朴素实现

解决非合并内存访问的方法是使用快速线程索引来索引以行主序存储在内存中的矩阵的行，这样同一 warp 中的线程读取或覆写矩阵同一行的元素时是合并的。在我们的实现中，我们只需要在内核函数中交换快速线程索引和慢速线程索引。

以下代码片段显示了具有合并内存访问的朴素实现。

```c++
template <typename T>
__global__ void gemm_v01(size_t m, size_t n, size_t k, T alpha, T const* A,
                         size_t lda, T const* B, size_t ldb, T beta, T* C,
                         size_t ldc)
{
    // Compute the row and column of C that this thread is responsible for.
    size_t const C_col_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_row_idx{blockIdx.y * blockDim.y + threadIdx.y};

    // Each thread compute
    // C[C_row_idx, C_col_idx] = alpha * A[C_row_idx, :] * B[:, C_col_idx] +
    // beta * C[C_row_idx, C_col_idx].
    if (C_row_idx < m && C_col_idx < n)
    {
        T sum{static_cast<T>(0)};
        for (size_t k_idx{0U}; k_idx < k; ++k_idx)
        {
            sum += A[C_row_idx * lda + k_idx] * B[k_idx * ldb + C_col_idx];
        }
        C[C_row_idx * ldc + C_col_idx] =
            alpha * sum + beta * C[C_row_idx * ldc + C_col_idx];
    }
}

template <typename T>
void launch_gemm_kernel_v01(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream)
{
    dim3 const block_dim{32U, 32U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(m) + block_dim.y - 1U) / block_dim.y, 1U};
    gemm_v01<T><<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B,
                                                     ldb, *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}
```
现在，由于这个修复，同一warp中的线程从以行主序存储在内存中的矩阵$B$的同一行读取元素，导致合并内存访问。当warp覆写矩阵$C$的元素时也会发生同样的情况。同一warp中的线程读取矩阵$A$的同一元素，导致广播内存访问。因此，这个实现应该比非合并内存访问的实现性能好得多。

这个FP32 GEMM实现的性能在NVIDIA GeForce RTX 3090 GPU上变成了1.72 TFLOPS，比之前的实现好得多。然而，考虑到GPU的理论峰值性能是35.58 TFLOPS，这个实现的性能仍然很差。

## 具有2D块分块的实现

由于前面的实现频繁访问全局内存，GEMM实现变成了内存受限的。由于访问共享内存比访问全局内存快得多，为了提高性能，我们可以使用共享内存来缓存输入矩阵$A$和$B$以实现数据重用。

然而，由于共享内存大小有限，我们无法在共享内存中缓存整个输入矩阵$A$和$B$。相反，我们可以在共享内存中缓存$A$和$B$的2D分块，并使用2D分块来计算输出矩阵$D$的2D分块。然后，我们可以将$A$和$B$的下一个2D分块加载到共享内存中，并计算$D$的下一个2D分块。

数学上，给定一个GEMM运算$D = AB + C$，其中$D \in \mathbb{R}^{m \times n}$，$A \in \mathbb{R}^{m \times k}$，$B \in \mathbb{R}^{k \times n}$，$C \in \mathbb{R}^{m \times n}$，矩阵可以被分成更小的矩阵。

$$A = \begin{bmatrix}
A_{1,1}^{d_m \times d_{bk}} & A_{1,2}^{d_m \times d_{bk}} & \cdots & A_{1,k/d_{bk}}^{d_m \times d_{bk}} \\
A_{2,1}^{d_m \times d_{bk}} & A_{2,2}^{d_m \times d_{bk}} & \cdots & A_{2,k/d_{bk}}^{d_m \times d_{bk}} \\
\vdots & \vdots & \ddots & \vdots \\
A_{m/d_m,1}^{d_m \times d_{bk}} & A_{m/d_m,2}^{d_m \times d_{bk}} & \cdots & A_{m/d_m,k/d_{bk}}^{d_m \times d_{bk}}
\end{bmatrix}$$

$$B = \begin{bmatrix}
B_{1,1}^{d_{bk} \times d_n} & B_{1,2}^{d_{bk} \times d_n} & \cdots & B_{1,n/d_n}^{d_{bk} \times d_n} \\
B_{2,1}^{d_{bk} \times d_n} & B_{2,2}^{d_{bk} \times d_n} & \cdots & B_{2,n/d_n}^{d_{bk} \times d_n} \\
\vdots & \vdots & \ddots & \vdots \\
B_{k/d_{bk},1}^{d_{bk} \times d_n} & B_{k/d_{bk},2}^{d_{bk} \times d_n} & \cdots & B_{k/d_{bk},n/d_n}^{d_{bk} \times d_n}
\end{bmatrix}$$

$$C = \begin{bmatrix}
C_{1,1}^{d_m \times d_n} & C_{1,2}^{d_m \times d_n} & \cdots & C_{1,n/d_n}^{d_m \times d_n} \\
C_{2,1}^{d_m \times d_n} & C_{2,2}^{d_m \times d_n} & \cdots & C_{2,n/d_n}^{d_m \times d_n} \\
\vdots & \vdots & \ddots & \vdots \\
C_{m/d_m,1}^{d_m \times d_n} & C_{m/d_m,2}^{d_m \times d_n} & \cdots & C_{m/d_m,n/d_n}^{d_m \times d_n}
\end{bmatrix}$$

$$D = \begin{bmatrix}
D_{1,1}^{d_m \times d_n} & D_{1,2}^{d_m \times d_n} & \cdots & D_{1,n/d_n}^{d_m \times d_n} \\
D_{2,1}^{d_m \times d_n} & D_{2,2}^{d_m \times d_n} & \cdots & D_{2,n/d_n}^{d_m \times d_n} \\
\vdots & \vdots & \ddots & \vdots \\
D_{m/d_m,1}^{d_m \times d_n} & D_{m/d_m,2}^{d_m \times d_n} & \cdots & D_{m/d_m,n/d_n}^{d_m \times d_n}
\end{bmatrix}$$

每个$D$中的小矩阵都是通过多个小矩阵乘法和累加计算的。

$$D_{b_m,b_n}^{d_m \times d_n} = \sum_{b_k=1}^{k/d_{bk}} A_{b_m,b_k}^{d_m \times d_{bk}} B_{b_k,b_n}^{d_{bk} \times d_n} + C_{b_m,b_n}^{d_m \times d_n}$$
在这个实现中，每个具有块索引$(b_m, b_n)$的2D块，其中$b_m \in [1, m/d_{bm}]$和$b_n \in [1, n/d_{bn}]$，负责计算一个小矩阵$D_{b_m,b_n}^{d_m \times d_n}$。共享内存用于缓存大小分别为$d_{bm} \times d_{bk}$和$d_{bk} \times d_{bn}$的$A$和$B$的2D分块。$A$的2D分块由$(b_m, b_k)$索引，其中$b_m \in [1, m/d_{bm}]$和$b_k \in [1, k/d_{bk}]$。$B$的2D分块由$(b_k, b_n)$索引，其中$b_k \in [1, k/d_{bk}]$和$b_n \in [1, n/d_{bn}]$。缓存和小矩阵乘法计算过程重复$k/d_{bk}$次，直到整个小矩阵$D_{b_m,b_n}^{d_m \times d_n}$被累加完成。

与之前的实现类似，每个块需要$d_{bm} \times d_{bn}$个线程来计算小矩阵$D_{b_m,b_n}^{d_m \times d_n}$，每个具有块线程索引$(t_m, t_n)$的线程，其中$t_m \in [1, d_{bm}]$和$t_n \in [1, d_{bn}]$，负责计算小矩阵的一个元素。

$$\left(D_{b_m,b_n}^{d_m \times d_n}\right)_{t_m,t_n} = \left(\sum_{b_k=1}^{k/d_{bk}} A_{b_m,b_k}^{d_m \times d_{bk}} B_{b_k,b_n}^{d_{bk} \times d_n} + C_{b_m,b_n}^{d_m \times d_n}\right)_{t_m,t_n}$$

$$= \sum_{b_k=1}^{k/d_{bk}} \left(A_{b_m,b_k}^{d_m \times d_{bk}} B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{t_m,t_n} + \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{t_m,t_n}$$

$$= \sum_{b_k=1}^{k/d_{bk}} \left(\sum_{t_k=1}^{d_{bk}} \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{t_m,t_k} \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{t_k,t_n}\right) + \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{t_m,t_n}$$

下面的代码片段展示了使用2D块分块的实现。

```c++
template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS, size_t BLOCK_TILE_SKEW_SIZE_X = 0U, size_t BLOCK_TILE_SKEW_SIZE_K = 0U>
__device__ void load_data_to_shared_memory(T const* A, size_t lda,
                                           T const* B, size_t ldb,
                                           T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K + BLOCK_TILE_SKEW_SIZE_K],
                                           T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X],
                                           size_t thread_block_tile_idx,
                                           size_t thread_linear_idx,
                                           size_t m, size_t n,
                                           size_t k)
{
    // 将DRAM中的矩阵A数据加载到共享内存中的A_thread_block_tile
#pragma unroll
    for (size_t load_idx{0U};
         load_idx <
         (BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K + NUM_THREADS - 1U) /
             NUM_THREADS;
         ++load_idx)
    {
        // 计算在共享内存分块中的行列索引
        size_t const A_thread_block_tile_row_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) /
            BLOCK_TILE_SIZE_K};
        size_t const A_thread_block_tile_col_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) %
            BLOCK_TILE_SIZE_K};
        // 计算在全局矩阵A中的行列索引
        size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
                               A_thread_block_tile_row_idx};
        size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                               A_thread_block_tile_col_idx};

        // 边界检查可能会在一定程度上降低内核性能
        // 但是能够保证内核对所有不同GEMM配置的正确性
        T val{static_cast<T>(0)};
        if (A_row_idx < m && A_col_idx < k)
        {
            val = A[A_row_idx * lda + A_col_idx];
        }
        // 这个if语句会降低内核性能
        // 在主机代码中添加静态断言来保证这个if条件总是为true
        static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS ==
                      0U);
        // if (A_thread_block_tile_row_idx < BLOCK_TILE_SIZE_Y &&
        //     A_thread_block_tile_col_idx < BLOCK_TILE_SIZE_K)
        // {
        //     A_thread_block_tile[A_thread_block_tile_row_idx]
        //                        [A_thread_block_tile_col_idx] = val;
        // }
        A_thread_block_tile[A_thread_block_tile_row_idx]
                           [A_thread_block_tile_col_idx] = val;
    }
// 将DRAM中的矩阵B数据加载到共享内存中的B_thread_block_tile
#pragma unroll
    for (size_t load_idx{0U};
         load_idx <
         (BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X + NUM_THREADS - 1U) /
             NUM_THREADS;
         ++load_idx)
    {
        // 计算在共享内存分块中的行列索引
        size_t const B_thread_block_tile_row_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) /
            BLOCK_TILE_SIZE_X};
        size_t const B_thread_block_tile_col_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) %
            BLOCK_TILE_SIZE_X};
        // 计算在全局矩阵B中的行列索引
        size_t const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                               B_thread_block_tile_row_idx};
        size_t const B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X +
                               B_thread_block_tile_col_idx};

        // 边界检查可能会在一定程度上降低内核性能
        // 但是能够保证内核对所有不同GEMM配置的正确性
        T val{static_cast<T>(0)};
        if (B_row_idx < k && B_col_idx < n)
        {
            val = B[B_row_idx * ldb + B_col_idx];
        }
        // 这个if语句会降低内核性能
        // 在主机代码中添加静态断言来保证这个if条件总是为true
        static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS ==
                      0U);
        // if (B_thread_block_tile_row_idx < BLOCK_TILE_SIZE_K &&
        //     B_thread_block_tile_col_idx < BLOCK_TILE_SIZE_X)
        // {
        //     B_thread_block_tile[B_thread_block_tile_row_idx]
        //                        [B_thread_block_tile_col_idx] = val;
        // }
        B_thread_block_tile[B_thread_block_tile_row_idx]
                           [B_thread_block_tile_col_idx] = val;
    }
}

template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K>
__global__ void gemm_v02(size_t m, size_t n, size_t k, T alpha, T const* A,
                         size_t lda, T const* B, size_t ldb, T beta, T* C,
                         size_t ldc)
{
    // 避免使用blockDim.x * blockDim.y作为每个块的线程数
    // 因为它是一个运行时常数，编译器无法基于它优化循环展开
    // 改为使用编译时常数
    constexpr size_t NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};
    size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};

    // 计算此线程负责的矩阵C的行列索引
    size_t const C_col_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_row_idx{blockIdx.y * blockDim.y + threadIdx.y};

    // 在共享内存中缓存A和B的分块以实现数据重用
    __shared__ T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    // 计算需要处理的线程块分块数量
    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
                                        BLOCK_TILE_SIZE_K};

    // 累加和初始化为0
    T sum{static_cast<T>(0)};
    // 遍历每个线程块分块
    for (size_t thread_block_tile_idx{0U};
         thread_block_tile_idx < num_thread_block_tiles;
         ++thread_block_tile_idx)
    {
        // 加载当前分块的数据到共享内存
        load_data_to_shared_memory<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                                   BLOCK_TILE_SIZE_K, NUM_THREADS>(
            A, lda, B, ldb, A_thread_block_tile, B_thread_block_tile,
            thread_block_tile_idx, thread_linear_idx, m, n, k);
        __syncthreads();

        // 计算当前分块的矩阵乘法
#pragma unroll
        for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
        {
            // 这样做会导致性能达到2 TOPS
            // 假设blockDim.x = blockDim.y = 32
            // 实际上，对于一个warp，在一次迭代中，我们从A_thread_block_tile的
            // 共享内存同一位置读取值，导致广播，我们也从B_thread_block_tile
            // 读取32个没有bank冲突的值。即使如此，所有值都必须从共享内存读取，
            // 结果是共享内存指令运行得非常密集，只是为了使用简单的算术指令
            // 计算少量值，这是不高效的
            sum += A_thread_block_tile[threadIdx.y][k_i] *
                   B_thread_block_tile[k_i][threadIdx.x];
        }
        __syncthreads();
    }
    // 将最终结果写入输出矩阵C（边界检查）
    if (C_row_idx < m && C_col_idx < n)
    {
        C[C_row_idx * ldc + C_col_idx] =
            alpha * sum + beta * C[C_row_idx * ldc + C_col_idx];
    }
}

template <typename T>
void launch_gemm_kernel_v02(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream)
{
    // 可以随意调整块分块大小
    // 算法正确性应该始终得到保证
    constexpr unsigned int BLOCK_TILE_SIZE_X{32U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{32U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{32U};
    constexpr unsigned int NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};
    // 静态断言确保分块大小与线程数兼容
    static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS == 0U);
    static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS == 0U);
    // 设置块维度
    dim3 const block_dim{BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, 1U};
    // 设置网格维度
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(m) + block_dim.y - 1U) / block_dim.y, 1U};
    // 启动GEMM内核
    gemm_v02<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B, ldb,
                                              *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}
```

这个FP32 GEMM实现的性能在NVIDIA GeForce RTX 3090 GPU上变成了2.66 TFLOPS，比之前的实现好得多。然而，它仍然远远低于GPU的理论峰值性能。

这个实现的问题是共享内存被频繁访问。即使访问共享内存比访问全局内存快得多，共享内存指令运行得非常密集，只是为了使用简单的算术指令计算少量值，这是不高效的。因此，这个实现的性能仍然受限于内存带宽，这次来自共享内存。

## 具有2D块分块和1D线程分块的实现

为了进一步提高性能，我们可以通过进一步缓存输入矩阵$A$和$B$的更小的分块到线程的寄存器中来缓解共享内存带宽问题。这次，每个线程负责计算输出矩阵$D$的一个小分块，而不是一个单个元素。因为寄存器是最快的访问方式，这个实现的性能应该比之前的实现好得多。

我们首先只缓存矩阵$B$的数据从共享内存到寄存器。每个具有块线程索引$(t_m, t_n)$的线程，其中$t_m \in [1, d_{bm}/d_{tm}]$和$t_n \in [1, d_{bn}]$，现在负责计算小矩阵的$d_{tm}$个元素，其中$d_{tm}$是线程分块大小。

$$\left(D_{b_m,b_n}^{d_m \times d_n}\right)_{t_m:t_m+d_{tm},t_n} = \left(\sum_{b_k=1}^{k/d_{bk}} A_{b_m,b_k}^{d_m \times d_{bk}} B_{b_k,b_n}^{d_{bk} \times d_n} + C_{b_m,b_n}^{d_m \times d_n}\right)_{t_m:t_m+d_{tm},t_n}$$

$$= \sum_{b_k=1}^{k/d_{bk}} \left(A_{b_m,b_k}^{d_m \times d_{bk}} B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{t_m:t_m+d_{tm},t_n} + \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{t_m:t_m+d_{tm},t_n}$$

$$= \sum_{b_k=1}^{k/d_{bk}} \left(\sum_{t_k=1}^{d_{bk}} \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{t_m:t_m+d_{tm},t_k} \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{t_k,t_n}\right) + \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{t_m:t_m+d_{tm},t_n}$$

在我们之前的实现中没有线程分块，为了计算小矩阵的一个元素，我们需要从共享内存中缓存的矩阵$A$中读取$d_{bk}$个值，从共享内存中缓存的矩阵$B$中读取$d_{bk}$个值。总共需要从共享内存中读取$2d_k$个值。

现在，有了1D线程分块，为了计算小矩阵的$d_{tm}$个元素，我们只需要从共享内存中缓存的矩阵$A$中读取$d_{bk} \times d_{tm}$个值，从共享内存中缓存的矩阵$B$中读取$d_{bk}$个值。具体来说，在每个内层循环中，$\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{t_k,t_n}$被缓存到寄存器中以被重复使用$d_{tm}$次。总共需要从共享内存中读取$d_{bk} \times d_{tm} + d_{bk}$个值。平均来说，为了计算小矩阵的一个元素，我们需要从共享内存中读取$d_{bk} + d_{bk}/d_{tm}$个值。

因为 $d_{bk} + d_{bk}/d_{tm} < 2d_k$，共享内存访问频率降低，共享内存带宽问题得到缓解。

下面的代码片段展示了使用2D块分块和1D线程分块的实现。

```c++
template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t THREAD_TILE_SIZE_Y>
__global__ void gemm_v03(size_t m, size_t n, size_t k, T alpha, T const* A,
                         size_t lda, T const* B, size_t ldb, T beta, T* C,
                         size_t ldc)
{
    // Avoid using blockDim.x * blockDim.y as the number of threads per block.
    // Because it is a runtime constant and the compiler cannot optimize the
    // loop unrolling based on that.
    // Use a compile time constant instead.
    constexpr size_t NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y /
                                 THREAD_TILE_SIZE_Y};
    size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};

    // Cache a tile of A and B in shared memory for data reuse.
    __shared__ T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
                                        BLOCK_TILE_SIZE_K};

    // Each thread in the block processes BLOCK_TILE_SIZE_Y output values.
    // Specifically, these values corresponds to
    // C[blockIdx.y * BLOCK_TILE_SIZE_Y + threadIdx.x / BLOCK_TILE_SIZE_X *
    // THREAD_TILE_SIZE_Y : blockIdx.y * BLOCK_TILE_SIZE_Y + (threadIdx.x /
    // BLOCK_TILE_SIZE_X + 1) * THREAD_TILE_SIZE_Y][blockIdx.x *
    // BLOCK_TILE_SIZE_X + threadIdx.x % BLOCK_TILE_SIZE_X]
    T C_thread_results[THREAD_TILE_SIZE_Y] = {static_cast<T>(0)};

    for (size_t thread_block_tile_idx{0U};
         thread_block_tile_idx < num_thread_block_tiles;
         ++thread_block_tile_idx)
    {
        load_data_to_shared_memory<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                                   BLOCK_TILE_SIZE_K, NUM_THREADS>(
            A, lda, B, ldb, A_thread_block_tile, B_thread_block_tile,
            thread_block_tile_idx, thread_linear_idx, m, n, k);
        __syncthreads();

#pragma unroll
        for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
        {
            size_t const B_thread_block_tile_row_idx{k_i};
            // B_val is cached in the register to alleviate the pressure on the
            // shared memory access.
            T const B_val{
                B_thread_block_tile[B_thread_block_tile_row_idx]
                                   [thread_linear_idx % BLOCK_TILE_SIZE_X]};
#pragma unroll
            for (size_t thread_tile_row_idx{0U};
                 thread_tile_row_idx < THREAD_TILE_SIZE_Y;
                 ++thread_tile_row_idx)
            {
                size_t const A_thread_block_tile_row_idx{
                    thread_linear_idx / BLOCK_TILE_SIZE_X * THREAD_TILE_SIZE_Y +
                    thread_tile_row_idx};
                size_t const A_thread_block_tile_col_idx{k_i};
                T const A_val{A_thread_block_tile[A_thread_block_tile_row_idx]
                                                 [A_thread_block_tile_col_idx]};
                C_thread_results[thread_tile_row_idx] += A_val * B_val;
            }
        }
        __syncthreads();
    }

// Write the results to DRAM.
#pragma unroll
    for (size_t thread_tile_row_idx{0U};
         thread_tile_row_idx < THREAD_TILE_SIZE_Y; ++thread_tile_row_idx)
    {
        size_t const C_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
                               thread_linear_idx / BLOCK_TILE_SIZE_X *
                                   THREAD_TILE_SIZE_Y +
                               thread_tile_row_idx};
        size_t const C_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X +
                               thread_linear_idx % BLOCK_TILE_SIZE_X};
        if (C_row_idx < m && C_col_idx < n)
        {
            C[C_row_idx * ldc + C_col_idx] =
                alpha * C_thread_results[thread_tile_row_idx] +
                beta * C[C_row_idx * ldc + C_col_idx];
        }
    }
}

template <typename T>
void launch_gemm_kernel_v03(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream)
{
    // Feel free to play with the block tile sizes.
    // The algorithm correctness should always be guaranteed.
    constexpr unsigned int BLOCK_TILE_SIZE_X{64U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{64U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{8U};
    // Each thread computes THREAD_TILE_SIZE_Y values of C.
    constexpr unsigned int THREAD_TILE_SIZE_Y{8U};
    constexpr unsigned int NUM_THREADS_PER_BLOCK{
        BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y / THREAD_TILE_SIZE_Y};
    static_assert(BLOCK_TILE_SIZE_Y % THREAD_TILE_SIZE_Y == 0U);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_K == 0U);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_X == 0U);
    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) /
            BLOCK_TILE_SIZE_X,
        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) /
            BLOCK_TILE_SIZE_Y,
        1U};
    gemm_v03<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
             THREAD_TILE_SIZE_Y><<<grid_dim, block_dim, 0U, stream>>>(
        m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}
```

The performance of this FP32 GEMM implementation becomes 8.91 TFLOPS on an NVIDIA GeForce RTX 3090 GPU. It seems that we have been making good progress.


## Implementation with 2D Block Tiling and 2D Thread Tiling

If the number of registers is not a bottleneck for the performance, we can further improve the performance by caching the data of both matrix $A$ and $B$ from the shared memory to the registers. Each thread with block thread index $(t_m, t_n)$, where $t_m \in [1, d_{bm}/d_{tm}]$ and $t_n \in [1, d_{bn}/d_{tn}]$, is now responsible for computing $d_{tm} \times d_{tn}$ elements of the small matrix, where $d_{tm}$ and $d_{tn}$ are the thread tile sizes for the row and column, respectively.

$$\left(D_{b_m,b_n}^{d_m \times d_n}\right)_{t_m:t_m+d_{tm},t_n:t_n+d_{tn}} = \left(\sum_{b_k=1}^{k/d_{bk}} A_{b_m,b_k}^{d_m \times d_{bk}} B_{b_k,b_n}^{d_{bk} \times d_n} + C_{b_m,b_n}^{d_m \times d_n}\right)_{t_m:t_m+d_{tm},t_n:t_n+d_{tn}}$$

$$= \sum_{b_k=1}^{k/d_{bk}} \left(A_{b_m,b_k}^{d_m \times d_{bk}} B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{t_m:t_m+d_{tm},t_n:t_n+d_{tn}} + \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{t_m:t_m+d_{tm},t_n:t_n+d_{tn}}$$

$$= \sum_{b_k=1}^{k/d_{bk}} \left(\sum_{t_k=1}^{d_{bk}} \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{t_m:t_m+d_{tm},t_k} \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{t_k,t_n:t_n+d_{tn}}\right) + \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{t_m:t_m+d_{tm},t_n:t_n+d_{tn}}$$

In our previous implementation with 1D thread tiling, to compute one element of the small matrix, we need to read $d_{bk} + d_{bk}/d_{tm}$ values from the shared memory on average.

Now, with 2D thread tiling, to compute $d_{tm} \times d_{tn}$ elements of the small matrix, we only need to read $d_{bk} \times (d_{tm} + d_{tn})$ values from the shared memory. Specifically, in each inner loop, $\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{t_m:t_m+d_{tm},t_k}$ and $\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{t_k,t_n:t_n+d_{tn}}$ are cached in the register to be reused for computing the matrix multiplication $\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{t_m:t_m+d_{tm},t_k} \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{t_k,t_n:t_n+d_{tn}}$. In total, we need to read $d_{bk} \times (d_{tm} + d_{tn})$ values from the shared memory. On average, to compute one element of the small matrix, we need to read $d_{bk}/d_{tm} + d_{bk}/d_{tn}$ values from the shared memory.

Because $d_{bk}/d_{tm} + d_{bk}/d_{tn} < d_{bk} + d_{bk}/d_{tm}$, the shared memory is accessed even less frequently and the shared memory bandwidth problem is further alleviated.

There is an alternative way to describe the 2D thread tiling implementation.

Mathematically, given a matrix multiplication and accumulation operation $D_{b_m,b_n}^{d_m \times d_n} = \sum_{b_k=1}^{k/d_{bk}} A_{b_m,b_k}^{d_m \times d_{bk}} B_{b_k,b_n}^{d_{bk} \times d_n} + C_{b_m,b_n}^{d_m \times d_n}$, where $D_{b_m,b_n} \in \mathbb{R}^{d_m \times d_n}$, $A_{b_m,b_k} \in \mathbb{R}^{d_m \times d_{bk}}$, $B_{b_k,b_n} \in \mathbb{R}^{d_{bk} \times d_n}$, $C_{b_m,b_n} \in \mathbb{R}^{d_m \times d_n}$, the matrices could be divided into smaller matrices.

$$A_{b_m,b_k}^{d_m \times d_{bk}} = \begin{bmatrix}
\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{1,1}^{d_{tm} \times d_{tk}} & \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{1,2}^{d_{tm} \times d_{tk}} & \cdots & \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{1,d_{bk}/d_{tk}}^{d_{tm} \times d_{tk}} \\
\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{2,1}^{d_{tm} \times d_{tk}} & \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{2,2}^{d_{tm} \times d_{tk}} & \cdots & \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{2,d_{bk}/d_{tk}}^{d_{tm} \times d_{tk}} \\
\vdots & \vdots & \ddots & \vdots \\
\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{d_m/d_{tm},1}^{d_{tm} \times d_{tk}} & \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{d_m/d_{tm},2}^{d_{tm} \times d_{tk}} & \cdots & \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{d_m/d_{tm},d_{bk}/d_{tk}}^{d_{tm} \times d_{tk}}
\end{bmatrix}$$

$$B_{b_k,b_n}^{d_{bk} \times d_n} = \begin{bmatrix}
\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{1,1}^{d_{tk} \times d_{tn}} & \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{1,2}^{d_{tk} \times d_{tn}} & \cdots & \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{1,d_n/d_{tn}}^{d_{tk} \times d_{tn}} \\
\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{2,1}^{d_{tk} \times d_{tn}} & \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{2,2}^{d_{tk} \times d_{tn}} & \cdots & \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{2,d_n/d_{tn}}^{d_{tk} \times d_{tn}} \\
\vdots & \vdots & \ddots & \vdots \\
\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{d_{bk}/d_{tk},1}^{d_{tk} \times d_{tn}} & \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{d_{bk}/d_{tk},2}^{d_{tk} \times d_{tn}} & \cdots & \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{d_{bk}/d_{tk},d_n/d_{tn}}^{d_{tk} \times d_{tn}}
\end{bmatrix}$$

$$C_{b_m,b_n}^{d_m \times d_n} = \begin{bmatrix}
\left(C_{b_m,b_n}^{d_m \times d_n}\right)_{1,1}^{d_{tm} \times d_{tn}} & \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{1,2}^{d_{tm} \times d_{tn}} & \cdots & \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{1,d_n/d_{tn}}^{d_{tm} \times d_{tn}} \\
\left(C_{b_m,b_n}^{d_m \times d_n}\right)_{2,1}^{d_{tm} \times d_{tn}} & \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{2,2}^{d_{tm} \times d_{tn}} & \cdots & \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{2,d_n/d_{tn}}^{d_{tm} \times d_{tn}} \\
\vdots & \vdots & \ddots & \vdots \\
\left(C_{b_m,b_n}^{d_m \times d_n}\right)_{d_m/d_{tm},1}^{d_{tm} \times d_{tn}} & \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{d_m/d_{tm},2}^{d_{tm} \times d_{tn}} & \cdots & \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{d_m/d_{tm},d_n/d_{tn}}^{d_{tm} \times d_{tn}}
\end{bmatrix}$$

$$D_{b_m,b_n}^{d_m \times d_n} = \begin{bmatrix}
\left(D_{b_m,b_n}^{d_m \times d_n}\right)_{1,1}^{d_{tm} \times d_{tn}} & \left(D_{b_m,b_n}^{d_m \times d_n}\right)_{1,2}^{d_{tm} \times d_{tn}} & \cdots & \left(D_{b_m,b_n}^{d_m \times d_n}\right)_{1,d_n/d_{tn}}^{d_{tm} \times d_{tn}} \\
\left(D_{b_m,b_n}^{d_m \times d_n}\right)_{2,1}^{d_{tm} \times d_{tn}} & \left(D_{b_m,b_n}^{d_m \times d_n}\right)_{2,2}^{d_{tm} \times d_{tn}} & \cdots & \left(D_{b_m,b_n}^{d_m \times d_n}\right)_{2,d_n/d_{tn}}^{d_{tm} \times d_{tn}} \\
\vdots & \vdots & \ddots & \vdots \\
\left(D_{b_m,b_n}^{d_m \times d_n}\right)_{d_m/d_{tm},1}^{d_{tm} \times d_{tn}} & \left(D_{b_m,b_n}^{d_m \times d_n}\right)_{d_m/d_{tm},2}^{d_{tm} \times d_{tn}} & \cdots & \left(D_{b_m,b_n}^{d_m \times d_n}\right)_{d_m/d_{tm},d_n/d_{tn}}^{d_{tm} \times d_{tn}}
\end{bmatrix}$$

Each small matrix in $D_{b_m,b_n}^{d_m \times d_n}$ is computed as multiple small matrix multiplications and accumulations.

$$\left(D_{b_m,b_n}^{d_m \times d_n}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}} = \left(\sum_{b_k=1}^{k/d_{bk}} A_{b_m,b_k}^{d_m \times d_{bk}} B_{b_k,b_n}^{d_{bk} \times d_n} + C_{b_m,b_n}^{d_m \times d_n}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}}$$

$$= \sum_{b_k=1}^{k/d_{bk}} \left(A_{b_m,b_k}^{d_m \times d_{bk}} B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}} + \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}}$$

$$= \sum_{b_k=1}^{k/d_{bk}} \left(\sum_{t_k=1}^{d_{bk}/d_{tk}} \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{t_m,t_k}^{d_{tm} \times d_{tk}} \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{t_k,t_n}^{d_{tk} \times d_{tn}}\right) + \left(\left(C_{b_m,b_n}^{d_m \times d_n}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}}$$

综合起来，线程分块 $\left(\left(D_{b_m,b_n}^{d_m \times d_n}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}}$ 可以按如下方式计算。

$$\left(\left(D_{b_m,b_n}^{d_m \times d_n}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}} = \left(\sum_{b_k=1}^{k/d_{bk}} \left(\sum_{w_k=1}^{d_{bk}/d_{wk}} \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}} \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right) + \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}}$$

$$= \sum_{b_k=1}^{k/d_{bk}} \left(\sum_{w_k=1}^{d_{bk}/d_{wk}} \left(\sum_{t_k=1}^{d_{wk}/d_{tk}} \left(\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{t_m,t_k}^{d_{tm} \times d_{tk}} \left(\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{t_k,t_n}^{d_{tk} \times d_{tn}}\right) + \left(\left(C_{b_m,b_n}^{d_m \times d_n}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}}\right)_{t_m,t_n}$$

在此实现中，我们设置 $d_{wk} = d_{tk}$ 以使线程分块算法更简单。

以下代码片段显示了使用2D块分块和2D warp分块和2D线程分块和向量化内存访问的实现。

```c++
// GEMM kernel v04.
// Coalesced read and write from global memory.
template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t THREAD_TILE_SIZE_X,
          size_t THREAD_TILE_SIZE_Y>
__global__ void gemm_v04(size_t m, size_t n, size_t k, T alpha, T const* A,
                         size_t lda, T const* B, size_t ldb, T beta, T* C,
                         size_t ldc)
{
    // Avoid using blockDim.x * blockDim.y as the number of threads per block.
    // Because it is a runtime constant and the compiler cannot optimize the
    // loop unrolling based on that.
    // Use a compile time constant instead.
    constexpr size_t NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y /
                                 (THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y)};
    size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};

    // Cache a tile of A and B in shared memory for data reuse.
    __shared__ T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
                                        BLOCK_TILE_SIZE_K};

    // Each thread in the block processes BLOCK_TILE_SIZE_Y output values.
    // Specifically, these values corresponds to
    // C[blockIdx.y * BLOCK_TILE_SIZE_Y + threadIdx.x / BLOCK_TILE_SIZE_X *
    // THREAD_TILE_SIZE_Y : blockIdx.y * BLOCK_TILE_SIZE_Y + (threadIdx.x /
    // BLOCK_TILE_SIZE_X + 1) * THREAD_TILE_SIZE_Y][blockIdx.x *
    // BLOCK_TILE_SIZE_X + threadIdx.x % BLOCK_TILE_SIZE_X *
    // THREAD_TILE_SIZE_X : blockIdx.x * BLOCK_TILE_SIZE_X + (threadIdx.x %
    // BLOCK_TILE_SIZE_X + 1) * THREAD_TILE_SIZE_X]
    T C_thread_results[THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X] = {
        static_cast<T>(0)};
    // A_vals is cached in the register.
    T A_vals[THREAD_TILE_SIZE_Y] = {static_cast<T>(0)};
    // B_vals is cached in the register.
    T B_vals[THREAD_TILE_SIZE_X] = {static_cast<T>(0)};

    for (size_t thread_block_tile_idx{0U};
         thread_block_tile_idx < num_thread_block_tiles;
         ++thread_block_tile_idx)
    {

        load_data_to_shared_memory<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                                   BLOCK_TILE_SIZE_K, NUM_THREADS>(
            A, lda, B, ldb, A_thread_block_tile, B_thread_block_tile,
            thread_block_tile_idx, thread_linear_idx, m, n, k);
        __syncthreads();

#pragma unroll
        for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
        {
            size_t const A_thread_block_tile_row_idx{
                thread_linear_idx / (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) *
                THREAD_TILE_SIZE_Y};
            size_t const A_thread_block_tile_col_idx{k_i};

#pragma unroll
            for (size_t thread_tile_row_idx{0U};
                 thread_tile_row_idx < THREAD_TILE_SIZE_Y;
                 ++thread_tile_row_idx)
            {
                // There will be shared memory bank conflicts accessing the
                // values from A_thread_block_tile. We can do it better by
                // transposing the A_thread_block_tile when we load the data
                // from DRAM.
                A_vals[thread_tile_row_idx] =
                    A_thread_block_tile[A_thread_block_tile_row_idx +
                                        thread_tile_row_idx]
                                       [A_thread_block_tile_col_idx];
            }

            size_t const B_thread_block_tile_row_idx{k_i};
            size_t const B_thread_block_tile_col_idx{
                thread_linear_idx % (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) *
                THREAD_TILE_SIZE_X};
#pragma unroll
            for (size_t thread_tile_col_idx{0U};
                 thread_tile_col_idx < THREAD_TILE_SIZE_X;
                 ++thread_tile_col_idx)
            {
                B_vals[thread_tile_col_idx] =
                    B_thread_block_tile[B_thread_block_tile_row_idx]
                                       [B_thread_block_tile_col_idx +
                                        thread_tile_col_idx];
            }

            for (size_t thread_tile_row_idx{0U};
                 thread_tile_row_idx < THREAD_TILE_SIZE_Y;
                 ++thread_tile_row_idx)
            {
                for (size_t thread_tile_col_idx{0U};
                     thread_tile_col_idx < THREAD_TILE_SIZE_X;
                     ++thread_tile_col_idx)
                {
                    C_thread_results[thread_tile_row_idx]
                                    [thread_tile_col_idx] +=
                        A_vals[thread_tile_row_idx] *
                        B_vals[thread_tile_col_idx];
                }
            }
        }
        __syncthreads();
    }

    // Write the results to DRAM.
    for (size_t thread_tile_row_idx{0U};
         thread_tile_row_idx < THREAD_TILE_SIZE_Y; ++thread_tile_row_idx)
    {
        for (size_t thread_tile_col_idx{0U};
             thread_tile_col_idx < THREAD_TILE_SIZE_X; ++thread_tile_col_idx)
        {
            size_t const C_row_idx{
                blockIdx.y * BLOCK_TILE_SIZE_Y +
                threadIdx.x / (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) *
                    THREAD_TILE_SIZE_Y +
                thread_tile_row_idx};
            size_t const C_col_idx{
                blockIdx.x * BLOCK_TILE_SIZE_X +
                threadIdx.x % (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) *
                    THREAD_TILE_SIZE_X +
                thread_tile_col_idx};
            if (C_row_idx < m && C_col_idx < n)
            {
                C[C_row_idx * ldc + C_col_idx] =
                    alpha * C_thread_results[thread_tile_row_idx]
                                            [thread_tile_col_idx] +
                    beta * C[C_row_idx * ldc + C_col_idx];
            }
        }
    }
}

template <typename T>
void launch_gemm_kernel_v04(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream)
{
    // Feel free to play with the block tile sizes.
    // The algorithm correctness should always be guaranteed.
    constexpr unsigned int BLOCK_TILE_SIZE_X{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{16U};
    // Each thread computes THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y values of C.
    constexpr unsigned int THREAD_TILE_SIZE_X{8U};
    constexpr unsigned int THREAD_TILE_SIZE_Y{8U};
    constexpr unsigned int NUM_THREADS_PER_BLOCK{
        BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y /
        (THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y)};
    static_assert(BLOCK_TILE_SIZE_X % THREAD_TILE_SIZE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_Y % THREAD_TILE_SIZE_Y == 0U);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_K == 0U);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_X == 0U);
    static_assert(
        BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS_PER_BLOCK == 0U);
    static_assert(
        BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS_PER_BLOCK == 0U);
    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) /
            BLOCK_TILE_SIZE_X,
        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) /
            BLOCK_TILE_SIZE_Y,
        1U};
    gemm_v04<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
             THREAD_TILE_SIZE_X, THREAD_TILE_SIZE_Y>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B, ldb,
                                              *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}
```



除了使用向量化内存访问加载数据外，其余内核与之前使用2D块分块和2D线程分块的实现相同。然而，在我们的使用案例中，向量化内存访问存在一个在之前实现中不存在的注意事项。当我们将数据从全局内存加载到共享内存，并将数据从共享内存加载到寄存器时，考虑到矩阵是2D的，我们需要确保向量化内存访问数据类型的数据对齐是正确的。否则，将会发生未定义行为。例如，如果我们使用 $int4$ 作为向量化内存访问数据类型，我们需要确保数据对齐是16字节的倍数。这就是为什么我们必须填充矩阵 $A$ 和矩阵 $B$ 在全局内存中的前导维度，并且共享内存维度必须仔细选择的原因。

此FP32 GEMM实现的性能在NVIDIA GeForce RTX 3090 GPU上达到19.66 TFLOPS。


## Implementation with 2D Block Tiling and 2D Thread Tiling and Vectorized Memory Access

In my previous article “CUDA Vectorized Memory Access”, I showed how to use vectorized memory access to improve the performance of a trivial memory copy kernel. Vectorized memory access reduces the number of memory transactions and therefore improves the memory bandwidth utilization. The same trick can be applied to this GEMM kernel to accelerate the data loading from global memory to the shared memory and the data loading from the shared memory to the registers.

In the previous implementation, to compute matrix multiplication, each thread would have to read a column of matrix $A$ and a row of matrix $B$ from the shared memory and cache them in the registers. Because reading the data from a column of matrix $A$ would prevent vectorized memory access, we would like to transpose the matrix $A$ when loading the data from global memory to the shared memory, so that each thread can access a row of transposed matrix $A$ and a row of matrix $B$ from the shared memory in a vectorized fashion and cache them in the registers.

The following code snippet shows the implementation with 2D block tiling and 2D thread tiling and vectorized memory access.

```c++
template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS, size_t BLOCK_TILE_SKEW_SIZE_X = 0U, size_t BLOCK_TILE_SKEW_SIZE_Y = 0U, typename VECTOR_TYPE = int4>
__device__ void load_data_to_shared_memory_transposed_vectorized(T const* A, size_t lda,
                                           T const* B, size_t ldb,
                                           T A_thread_block_tile_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y + BLOCK_TILE_SKEW_SIZE_Y],
                                           T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X],
                                           size_t thread_block_tile_idx,
                                           size_t thread_linear_idx,
                                           size_t m, size_t n,
                                           size_t k)
{
    constexpr size_t NUM_VECTOR_UNITS{sizeof(VECTOR_TYPE) / sizeof(T)};
    static_assert(sizeof(VECTOR_TYPE) % sizeof(T) == 0U);
    static_assert(BLOCK_TILE_SIZE_K % NUM_VECTOR_UNITS == 0U);
    static_assert(BLOCK_TILE_SIZE_X % NUM_VECTOR_UNITS == 0U);
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_K{BLOCK_TILE_SIZE_K /
                                                  NUM_VECTOR_UNITS};
    static_assert(BLOCK_TILE_SIZE_K % NUM_VECTOR_UNITS == 0U);
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_X{BLOCK_TILE_SIZE_X /
                                                  NUM_VECTOR_UNITS};
    static_assert(BLOCK_TILE_SIZE_X % NUM_VECTOR_UNITS == 0U);

    // The skew size could affect the data alignment in shared memory when we use vectorized load.
    // We need to make sure the data alignment is correct.
    static_assert((BLOCK_TILE_SIZE_Y) * sizeof(T) % sizeof(VECTOR_TYPE) == 0U);
    static_assert((BLOCK_TILE_SIZE_X) * sizeof(T) % sizeof(VECTOR_TYPE) == 0U);
    static_assert((BLOCK_TILE_SIZE_Y + BLOCK_TILE_SKEW_SIZE_Y) * sizeof(T) % sizeof(VECTOR_TYPE) == 0U);
    static_assert((BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X) * sizeof(T) % sizeof(VECTOR_TYPE) == 0U);

// Load data from A on DRAM to A_thread_block_tile on shared memory.
#pragma unroll
    for (size_t load_idx{0U};
            load_idx < (BLOCK_TILE_SIZE_Y * VECTORIZED_BLOCK_TILE_SIZE_K +
                        NUM_THREADS - 1U) /
                        NUM_THREADS;
            ++load_idx)
    {
        size_t const A_thread_block_tile_row_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) /
            VECTORIZED_BLOCK_TILE_SIZE_K};
        size_t const A_thread_block_tile_col_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) %
            VECTORIZED_BLOCK_TILE_SIZE_K * NUM_VECTOR_UNITS};
        size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
                                A_thread_block_tile_row_idx};
        size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                                A_thread_block_tile_col_idx};

        // These boundary checks might slow down the kernel to some extent.
        // But they guarantee the correctness of the kernel for all
        // different GEMM configurations.
        int4 A_row_vector_vals{0, 0, 0, 0};
        if (A_row_idx < m && A_col_idx < k)
        {
            A_row_vector_vals = *reinterpret_cast<int4 const*>(
                &A[A_row_idx * lda + A_col_idx]);
        }
        if (A_col_idx + NUM_VECTOR_UNITS > k)
        {
            // Number of invalid elements in the last vector.
            size_t const num_invalid_elements{A_col_idx + NUM_VECTOR_UNITS -
                                                k};
            // Mask out the invalid elements.
            T* const A_row_vector_vals_ptr{
                reinterpret_cast<T*>(&A_row_vector_vals)};
            for (size_t i{0U}; i < num_invalid_elements; ++i)
            {
                A_row_vector_vals_ptr[NUM_VECTOR_UNITS - 1U - i] =
                    static_cast<T>(0);
            }
        }
        // If this is true, the following if can be removed.
        // static_assert(VECTORIZED_BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y %
        // NUM_THREADS ==
        //               0U);
        if (A_thread_block_tile_row_idx < BLOCK_TILE_SIZE_Y &&
            A_thread_block_tile_col_idx < BLOCK_TILE_SIZE_K)
        {
            for (size_t i{0U}; i < NUM_VECTOR_UNITS; ++i)
            {
                A_thread_block_tile_transposed
                    [A_thread_block_tile_col_idx + i]
                    [A_thread_block_tile_row_idx] =
                        reinterpret_cast<T const*>(&A_row_vector_vals)[i];
            }
        }
    }
// Load data from B on DRAM to B_thread_block_tile on shared memory.
#pragma unroll
    for (size_t load_idx{0U};
            load_idx < (BLOCK_TILE_SIZE_K * VECTORIZED_BLOCK_TILE_SIZE_X +
                        NUM_THREADS - 1U) /
                        NUM_THREADS;
            ++load_idx)
    {
        size_t const B_thread_block_tile_row_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) /
            VECTORIZED_BLOCK_TILE_SIZE_X};
        size_t const B_thread_block_tile_col_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) %
            VECTORIZED_BLOCK_TILE_SIZE_X * NUM_VECTOR_UNITS};
        size_t const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                                B_thread_block_tile_row_idx};
        size_t const B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X +
                                B_thread_block_tile_col_idx};

        // These boundary checks might slow down the kernel to some extent.
        // But they guarantee the correctness of the kernel for all
        // different GEMM configurations.
        int4 B_row_vector_vals{0, 0, 0, 0};
        if (B_row_idx < k && B_col_idx < n)
        {
            B_row_vector_vals = *reinterpret_cast<int4 const*>(
                &B[B_row_idx * ldb + B_col_idx]);
        }
        if (B_col_idx + NUM_VECTOR_UNITS > n)
        {
            // Number of invalid elements in the last vector.
            size_t const num_invalid_elements{B_col_idx + NUM_VECTOR_UNITS -
                                                n};
            // Mask out the invalid elements.
            T* const B_row_vector_vals_ptr{
                reinterpret_cast<T*>(&B_row_vector_vals)};
            for (size_t i{0U}; i < num_invalid_elements; ++i)
            {
                B_row_vector_vals_ptr[NUM_VECTOR_UNITS - 1U - i] =
                    static_cast<T>(0);
            }
        }
        // If this is true, the following if can be removed.
        // static_assert(VECTORIZED_BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K %
        // NUM_THREADS ==
        //               0U);
        if (B_thread_block_tile_row_idx < BLOCK_TILE_SIZE_K &&
            B_thread_block_tile_col_idx < BLOCK_TILE_SIZE_X)
        {
            *reinterpret_cast<int4*>(
                &B_thread_block_tile[B_thread_block_tile_row_idx]
                                    [B_thread_block_tile_col_idx]) =
                B_row_vector_vals;
        }
    }
}

// GEMM kernel v05.
// Coalesced read and write from global memory.
template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t THREAD_TILE_SIZE_X,
          size_t THREAD_TILE_SIZE_Y>
__global__ void gemm_v05_vectorized(size_t m, size_t n, size_t k, T alpha,
                                    T const* A, size_t lda, T const* B,
                                    size_t ldb, T beta, T* C, size_t ldc)
{
    // Avoid using blockDim.x * blockDim.y as the number of threads per block.
    // Because it is a runtime constant and the compiler cannot optimize the
    // loop unrolling based on that.
    // Use a compile time constant instead.
    constexpr size_t NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y /
                                 (THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y)};
    size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};

    // Cache a tile of A and B in shared memory for data reuse.
    __shared__ T
        A_thread_block_tile_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
                                        BLOCK_TILE_SIZE_K};

    // Each thread in the block processes BLOCK_TILE_SIZE_Y output values.
    // Specifically, these values corresponds to
    // C[blockIdx.y * BLOCK_TILE_SIZE_Y + threadIdx.x / BLOCK_TILE_SIZE_X *
    // THREAD_TILE_SIZE_Y : blockIdx.y * BLOCK_TILE_SIZE_Y + (threadIdx.x /
    // BLOCK_TILE_SIZE_X + 1) * THREAD_TILE_SIZE_Y][blockIdx.x *
    // BLOCK_TILE_SIZE_X + threadIdx.x % BLOCK_TILE_SIZE_X *
    // THREAD_TILE_SIZE_X : blockIdx.x * BLOCK_TILE_SIZE_X + (threadIdx.x %
    // BLOCK_TILE_SIZE_X + 1) * THREAD_TILE_SIZE_X]
    T C_thread_results[THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X] = {
        static_cast<T>(0)};
    // A_vals is cached in the register.
    T A_vals[THREAD_TILE_SIZE_Y] = {static_cast<T>(0)};
    // B_vals is cached in the register.
    T B_vals[THREAD_TILE_SIZE_X] = {static_cast<T>(0)};

    constexpr size_t NUM_VECTOR_UNITS{sizeof(int4) / sizeof(T)};
    static_assert(sizeof(int4) % sizeof(T) == 0U);
    static_assert(BLOCK_TILE_SIZE_K % NUM_VECTOR_UNITS == 0U);
    static_assert(BLOCK_TILE_SIZE_X % NUM_VECTOR_UNITS == 0U);
    constexpr size_t VECTORIZED_THREAD_TILE_SIZE_X{THREAD_TILE_SIZE_X /
                                                   NUM_VECTOR_UNITS};
    static_assert(THREAD_TILE_SIZE_X % NUM_VECTOR_UNITS == 0U);

    for (size_t thread_block_tile_idx{0U};
         thread_block_tile_idx < num_thread_block_tiles;
         ++thread_block_tile_idx)
    {
        load_data_to_shared_memory_transposed_vectorized<
            T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
            NUM_THREADS>(A, lda, B, ldb, A_thread_block_tile_transposed,
                         B_thread_block_tile, thread_block_tile_idx,
                         thread_linear_idx, m, n, k);
        __syncthreads();

#pragma unroll
        for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
        {
            size_t const A_thread_block_tile_row_idx{
                thread_linear_idx / (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) *
                THREAD_TILE_SIZE_Y};
            size_t const A_thread_block_tile_col_idx{k_i};

#pragma unroll
            for (size_t thread_tile_row_idx{0U};
                 thread_tile_row_idx < THREAD_TILE_SIZE_Y;
                 ++thread_tile_row_idx)
            {
                A_vals[thread_tile_row_idx] =
                    A_thread_block_tile_transposed[A_thread_block_tile_col_idx]
                                                  [A_thread_block_tile_row_idx +
                                                   thread_tile_row_idx];
            }

            size_t const B_thread_block_tile_row_idx{k_i};
            size_t const B_thread_block_tile_col_idx{
                thread_linear_idx % (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) *
                THREAD_TILE_SIZE_X};
// Although the read from A_thread_block_tile cannot be vectorized, the read
// from B_thread_block_tile can be vectorized.
#pragma unroll
            for (size_t thread_tile_col_vector_idx{0U};
                 thread_tile_col_vector_idx < VECTORIZED_THREAD_TILE_SIZE_X;
                 ++thread_tile_col_vector_idx)
            {
                *reinterpret_cast<int4*>(
                    &B_vals[thread_tile_col_vector_idx * NUM_VECTOR_UNITS]) =
                    *reinterpret_cast<int4 const*>(
                        &B_thread_block_tile[B_thread_block_tile_row_idx]
                                            [B_thread_block_tile_col_idx +
                                             thread_tile_col_vector_idx *
                                                 NUM_VECTOR_UNITS]);
            }

            for (size_t thread_tile_row_idx{0U};
                 thread_tile_row_idx < THREAD_TILE_SIZE_Y;
                 ++thread_tile_row_idx)
            {
                for (size_t thread_tile_col_idx{0U};
                     thread_tile_col_idx < THREAD_TILE_SIZE_X;
                     ++thread_tile_col_idx)
                {
                    C_thread_results[thread_tile_row_idx]
                                    [thread_tile_col_idx] +=
                        A_vals[thread_tile_row_idx] *
                        B_vals[thread_tile_col_idx];
                }
            }
        }
        __syncthreads();
    }

    // Vectorized writing the results to DRAM.
    for (size_t thread_tile_row_idx{0U};
         thread_tile_row_idx < THREAD_TILE_SIZE_Y; ++thread_tile_row_idx)
    {
        for (size_t thread_tile_col_vector_idx{0U};
             thread_tile_col_vector_idx < VECTORIZED_THREAD_TILE_SIZE_X;
             ++thread_tile_col_vector_idx)
        {
            size_t const C_row_idx{
                blockIdx.y * BLOCK_TILE_SIZE_Y +
                thread_linear_idx / (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) *
                    THREAD_TILE_SIZE_Y +
                thread_tile_row_idx};
            size_t const C_col_idx{
                blockIdx.x * BLOCK_TILE_SIZE_X +
                thread_linear_idx % (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) *
                    THREAD_TILE_SIZE_X +
                thread_tile_col_vector_idx * NUM_VECTOR_UNITS};
            // Vectorized read from C.
            int4 C_row_vector_vals{*reinterpret_cast<int4 const*>(
                &C[C_row_idx * ldc + C_col_idx])};
            // Vectorized read from C_thread_results.
            int4 const C_thread_results_row_vector_vals{
                *reinterpret_cast<int4 const*>(
                    &C_thread_results[thread_tile_row_idx]
                                     [thread_tile_col_vector_idx *
                                      NUM_VECTOR_UNITS])};
            // Update the values in C_row_vector_vals
            for (size_t i{0U}; i < NUM_VECTOR_UNITS; ++i)
            {
                reinterpret_cast<T*>(&C_row_vector_vals)[i] =
                    alpha * reinterpret_cast<T const*>(
                                &C_thread_results_row_vector_vals)[i] +
                    beta * reinterpret_cast<T const*>(&C_row_vector_vals)[i];
            }
            // Vectorized write to C.
            if (C_row_idx < m && C_col_idx < n)
            {
                // No need to mask out the out-of-bound invalid elements,
                // because the row of C matrix is 32-byte aligned.
                *reinterpret_cast<int4*>(&C[C_row_idx * ldc + C_col_idx]) =
                    C_row_vector_vals;
            }
        }
    }
}

template <typename T>
void launch_gemm_kernel_v05_vectorized(size_t m, size_t n, size_t k,
                                       T const* alpha, T const* A, size_t lda,
                                       T const* B, size_t ldb, T const* beta,
                                       T* C, size_t ldc, cudaStream_t stream)
{
    // Feel free to play with the block tile sizes.
    // The algorithm correctness should always be guaranteed.
    constexpr unsigned int BLOCK_TILE_SIZE_X{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{16U};
    // Each thread computes THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y values of C.
    constexpr unsigned int THREAD_TILE_SIZE_X{8U};
    constexpr unsigned int THREAD_TILE_SIZE_Y{8U};
    constexpr unsigned int NUM_THREADS_PER_BLOCK{
        BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y /
        (THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y)};
    static_assert(BLOCK_TILE_SIZE_X % THREAD_TILE_SIZE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_Y % THREAD_TILE_SIZE_Y == 0U);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_K == 0U);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_X == 0U);
    static_assert(
        BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS_PER_BLOCK == 0U);
    static_assert(
        BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS_PER_BLOCK == 0U);
    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) /
            BLOCK_TILE_SIZE_X,
        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) /
            BLOCK_TILE_SIZE_Y,
        1U};
    gemm_v05_vectorized<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                        BLOCK_TILE_SIZE_K, THREAD_TILE_SIZE_X,
                        THREAD_TILE_SIZE_Y>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B, ldb,
                                              *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}
```


Except the data loading using vectorized memory access, the rest of the kernel is the same as the previous implementation with 2D block tiling and 2D thread tiling. There is, however, a caveat for vectorized memory access in our use case which does not exist in the previous implementation. When we load the data from global memory to the shared memory and load the data from the shared memory to the registers, considering the matrices are 2D, we need to make sure the data alignment is correct for the vectorized memory access data type. Otherwise, undefined behavior will happen. For example, if we use int4 as the vectorized memory access data type, we need to make sure the data alignment is a multiple of 16 bytes. This is why we will have to pad the leading dimension of the matrix $A$ and matrix $B$ in the global memory and the shared memory dimensions have to be carefully chosen.

The performance of this FP32 GEMM implementation becomes 19.66 TFLOPS on an NVIDIA GeForce RTX 3090 GPU.


## 使用2D块分块和2D Warp分块和2D线程分块和向量化内存访问的实现

在CUDA编程模型中，warp由32个线程组成，是调度和执行的最小单位。当warp中的线程访问共享内存的同一个bank时，可能会发生[共享内存bank冲突](https://leimao.github.io/blog/CUDA-Shared-Memory-Bank-Conflicts/)。在我们之前的实现中，由于GEMM CUDA内核不是以warp为中心的方式组织的，如何避免共享内存bank冲突并不明显。

在这个实现中，我们将以warp为中心的方式组织GEMM CUDA内核，并使用2D warp分块和2D线程分块，以便可以更容易地预期和优化共享内存bank冲突。

理解warp分块几乎与理解线程分块完全相同。

数学上，给定矩阵乘法和累加操作 $D_{b_m,b_n}^{d_m \times d_n} = \sum_{b_k=1}^{k/d_{bk}} A_{b_m,b_k}^{d_m \times d_{bk}} B_{b_k,b_n}^{d_{bk} \times d_n} + C_{b_m,b_n}^{d_m \times d_n}$，其中 $D_{b_m,b_n} \in \mathbb{R}^{d_m \times d_n}$，$A_{b_m,b_k} \in \mathbb{R}^{d_m \times d_{bk}}$，$B_{b_k,b_n} \in \mathbb{R}^{d_{bk} \times d_n}$，$C_{b_m,b_n} \in \mathbb{R}^{d_m \times d_n}$，矩阵可以被分成更小的矩阵。

$$A_{b_m,b_k}^{d_m \times d_{bk}} = \begin{bmatrix}
\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{1,1}^{d_{wm} \times d_{wk}} & \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{1,2}^{d_{wm} \times d_{wk}} & \cdots & \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{1,d_{bk}/d_{wk}}^{d_{wm} \times d_{wk}} \\
\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{2,1}^{d_{wm} \times d_{wk}} & \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{2,2}^{d_{wm} \times d_{wk}} & \cdots & \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{2,d_{bk}/d_{wk}}^{d_{wm} \times d_{wk}} \\
\vdots & \vdots & \ddots & \vdots \\
\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{d_m/d_{wm},1}^{d_{wm} \times d_{wk}} & \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{d_m/d_{wm},2}^{d_{wm} \times d_{wk}} & \cdots & \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{d_m/d_{wm},d_{bk}/d_{wk}}^{d_{wm} \times d_{wk}}
\end{bmatrix}$$

$$B_{b_k,b_n}^{d_{bk} \times d_n} = \begin{bmatrix}
\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{1,1}^{d_{wk} \times d_{wn}} & \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{1,2}^{d_{wk} \times d_{wn}} & \cdots & \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{1,d_n/d_{wn}}^{d_{wk} \times d_{wn}} \\
\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{2,1}^{d_{wk} \times d_{wn}} & \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{2,2}^{d_{wk} \times d_{wn}} & \cdots & \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{2,d_n/d_{wn}}^{d_{wk} \times d_{wn}} \\
\vdots & \vdots & \ddots & \vdots \\
\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{d_{bk}/d_{wk},1}^{d_{wk} \times d_{wn}} & \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{d_{bk}/d_{wk},2}^{d_{wk} \times d_{wn}} & \cdots & \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{d_{bk}/d_{wk},d_n/d_{wn}}^{d_{wk} \times d_{wn}}
\end{bmatrix}$$

$$C_{b_m,b_n}^{d_m \times d_n} = \begin{bmatrix}
\left(C_{b_m,b_n}^{d_m \times d_n}\right)_{1,1}^{d_{wm} \times d_{wn}} & \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{1,2}^{d_{wm} \times d_{wn}} & \cdots & \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{1,d_n/d_{wn}}^{d_{wm} \times d_{wn}} \\
\left(C_{b_m,b_n}^{d_m \times d_n}\right)_{2,1}^{d_{wm} \times d_{wn}} & \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{2,2}^{d_{wm} \times d_{wn}} & \cdots & \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{2,d_n/d_{wn}}^{d_{wm} \times d_{wn}} \\
\vdots & \vdots & \ddots & \vdots \\
\left(C_{b_m,b_n}^{d_m \times d_n}\right)_{d_m/d_{wm},1}^{d_{wm} \times d_{wn}} & \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{d_m/d_{wm},2}^{d_{wm} \times d_{wn}} & \cdots & \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{d_m/d_{wm},d_n/d_{wn}}^{d_{wm} \times d_{wn}}
\end{bmatrix}$$

$$D_{b_m,b_n}^{d_m \times d_n} = \begin{bmatrix}
\left(D_{b_m,b_n}^{d_m \times d_n}\right)_{1,1}^{d_{wm} \times d_{wn}} & \left(D_{b_m,b_n}^{d_m \times d_n}\right)_{1,2}^{d_{wm} \times d_{wn}} & \cdots & \left(D_{b_m,b_n}^{d_m \times d_n}\right)_{1,d_n/d_{wn}}^{d_{wm} \times d_{wn}} \\
\left(D_{b_m,b_n}^{d_m \times d_n}\right)_{2,1}^{d_{wm} \times d_{wn}} & \left(D_{b_m,b_n}^{d_m \times d_n}\right)_{2,2}^{d_{wm} \times d_{wn}} & \cdots & \left(D_{b_m,b_n}^{d_m \times d_n}\right)_{2,d_n/d_{wn}}^{d_{wm} \times d_{wn}} \\
\vdots & \vdots & \ddots & \vdots \\
\left(D_{b_m,b_n}^{d_m \times d_n}\right)_{d_m/d_{wm},1}^{d_{wm} \times d_{wn}} & \left(D_{b_m,b_n}^{d_m \times d_n}\right)_{d_m/d_{wm},2}^{d_{wm} \times d_{wn}} & \cdots & \left(D_{b_m,b_n}^{d_m \times d_n}\right)_{d_m/d_{wm},d_n/d_{wn}}^{d_{wm} \times d_{wn}}
\end{bmatrix}$$

$D_{b_m,b_n}^{d_m \times d_n}$ 中的每个小矩阵都作为多个小矩阵乘法和累加来计算。

$$\left(D_{b_m,b_n}^{d_m \times d_n}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}} = \left(\sum_{b_k=1}^{k/d_{bk}} A_{b_m,b_k}^{d_m \times d_{bk}} B_{b_k,b_n}^{d_{bk} \times d_n} + C_{b_m,b_n}^{d_m \times d_n}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}$$

$$= \sum_{b_k=1}^{k/d_{bk}} \left(A_{b_m,b_k}^{d_m \times d_{bk}} B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}} + \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}$$

$$= \sum_{b_k=1}^{k/d_{bk}} \left(\sum_{w_k=1}^{d_{bk}/d_{wk}} \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}} \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right) + \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}$$

每个具有块warp索引 $(w_m, w_n)$ 的warp，其中 $w_m \in [1, d_m/d_{wm}]$ 和 $w_n \in [1, d_n/d_{wn}]$，在具有块索引 $(b_m, b_n)$ 的块中，其中 $b_m \in [1, m/d_m]$ 和 $b_n \in [1, n/d_n]$，负责计算一个小矩阵乘法和累加 $\left(D_{b_m,b_n}^{d_m \times d_n}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}$。

到目前为止，一切看起来都与2D线程分块的数学描述相同，只是线程索引和线程分块大小被warp索引和warp分块大小所替代。

剩下的问题是如何使用warp中具有块warp索引 $(w_m, w_n)$ 的所有32个线程来计算 $\left(\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}} \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}$。做到这一点并没有唯一的方法。我们选择的方式是使用2D线程分块。假设warp中行的线程数为 $m_t$，列的线程数为 $n_t$，我们必须有 $m_t \times n_t = 32$。warp中的每个线程应该负责计算 $(d_{wm}/m_t) \times (d_{wn}/n_t)$ 个输出矩阵的值。我们接着将线程分块大小设置为行为 $d_{tm}$，列为 $d_{tn}$，使得 $(d_{wm}/m_t) \bmod d_{tm} = 0$ 和 $(d_{wn}/n_t) \bmod d_{tn} = 0$。warp中的每个线程将必须计算 $((d_{wm}/m_t)/d_{tm}) \times ((d_{wn}/n_t)/d_{tn})$ 个大小为 $d_{tm} \times d_{tn}$ 的输出矩阵 $\left(\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}} \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}$ 的块。

假设线程分块索引为 $(t_m, t_n)$，其中 $t_m \in [1, d_{wm}/d_{tm}]$ 和 $t_n \in [1, d_{wn}/d_{tn}]$。负责计算该分块的线程具有warp线程索引 $(t_m \bmod m_t, t_n \bmod n_t)$。因为矩阵 $\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}$ 可以沿行维度分成 $d_{wm}/d_{tm}$ 个片段，矩阵 $\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}$ 可以沿列维度分成 $d_{wn}/d_{tn}$ 个片段。我们有

$$\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}} = \begin{bmatrix}
\left(\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{1,1}^{d_{tm} \times d_{tk}} & \left(\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{1,2}^{d_{tm} \times d_{tk}} & \cdots & \left(\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{1,d_{wk}/d_{tk}}^{d_{tm} \times d_{tk}} \\
\left(\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{2,1}^{d_{tm} \times d_{tk}} & \left(\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{2,2}^{d_{tm} \times d_{tk}} & \cdots & \left(\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{2,d_{wk}/d_{tk}}^{d_{tm} \times d_{tk}} \\
\vdots & \vdots & \ddots & \vdots \\
\left(\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{d_{wm}/d_{tm},1}^{d_{tm} \times d_{tk}} & \left(\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{d_{wm}/d_{tm},2}^{d_{tm} \times d_{tk}} & \cdots & \left(\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{d_{wm}/d_{tm},d_{wk}/d_{tk}}^{d_{tm} \times d_{tk}}
\end{bmatrix}$$

$$\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}} = \begin{bmatrix}
\left(\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{1,1}^{d_{tk} \times d_{tn}} & \left(\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{1,2}^{d_{tk} \times d_{tn}} & \cdots & \left(\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{1,d_{wn}/d_{tn}}^{d_{tk} \times d_{tn}} \\
\left(\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{2,1}^{d_{tk} \times d_{tn}} & \left(\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{2,2}^{d_{tk} \times d_{tn}} & \cdots & \left(\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{2,d_{wn}/d_{tn}}^{d_{tk} \times d_{tn}} \\
\vdots & \vdots & \ddots & \vdots \\
\left(\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{d_{wk}/d_{tk},1}^{d_{tk} \times d_{tn}} & \left(\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{d_{wk}/d_{tk},2}^{d_{tk} \times d_{tn}} & \cdots & \left(\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{d_{wk}/d_{tk},d_{wn}/d_{tn}}^{d_{tk} \times d_{tn}}
\end{bmatrix}$$

每个具有warp线程索引 $(t_m \bmod m_t, t_n \bmod n_t)$ 的线程负责计算一个小矩阵乘法和累加 $\left(\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}} \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}}$。

线程分块 $\left(\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}} \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}}$ 可以按如下方式计算。

$$\left(\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}} \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}} = \sum_{t_k=1}^{d_{wk}/d_{tk}} \left(\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{t_m,t_k}^{d_{tm} \times d_{tk}} \left(\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{t_k,t_n}^{d_{tk} \times d_{tn}}$$

综合起来，线程分块 $\left(\left(D_{b_m,b_n}^{d_m \times d_n}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}}$ 可以按如下方式计算。

$$\left(\left(D_{b_m,b_n}^{d_m \times d_n}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}} = \left(\sum_{b_k=1}^{k/d_{bk}} \left(\sum_{w_k=1}^{d_{bk}/d_{wk}} \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}} \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right) + \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}}$$

$$= \sum_{b_k=1}^{k/d_{bk}} \left(\sum_{w_k=1}^{d_{bk}/d_{wk}} \left(\sum_{t_k=1}^{d_{wk}/d_{tk}} \left(\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{t_m,t_k}^{d_{tm} \times d_{tk}} \left(\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{t_k,t_n}^{d_{tk} \times d_{tn}}\right)\right) + \left(\left(C_{b_m,b_n}^{d_m \times d_n}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}}$$

在此实现中，我们设置 $d_{wk} = d_{tk}$ 以使线程分块算法更简单。


The following code snippet shows the implementation with 2D block tiling and 2D warp tiling and 2D thread tiling and vectorized memory access.

```c++
// GEMM kernel v06.
// Each thread in the block processes THREAD_TILE_SIZE_Y *
// THREAD_TILE_SIZE_X output values. Number of threads BLOCK_TILE_SIZE_Y *
// BLOCK_TILE_SIZE_X / (THREAD_TILE_SIZE_Y * THREAD_TILE_SIZE_X)
template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t WARP_TILE_SIZE_X,
          size_t WARP_TILE_SIZE_Y, size_t THREAD_TILE_SIZE_X,
          size_t THREAD_TILE_SIZE_Y, size_t NUM_THREADS_PER_WARP_X,
          size_t NUM_THREADS_PER_WARP_Y>
__global__ void gemm_v06_vectorized(size_t m, size_t n, size_t k, T alpha,
                                    T const* A, size_t lda, T const* B,
                                    size_t ldb, T beta, T* C, size_t ldc)
{
    static_assert(NUM_THREADS_PER_WARP_X * NUM_THREADS_PER_WARP_Y == 32U);
    constexpr size_t NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
    static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0U);
    constexpr size_t NUM_WARPS_Y{BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y};
    static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0U);
    constexpr unsigned int NUM_THREAD_TILES_PER_WARP_X{
        WARP_TILE_SIZE_X / (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X)};
    constexpr unsigned int NUM_THREAD_TILES_PER_WARP_Y{
        WARP_TILE_SIZE_Y / (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y)};
    static_assert(
        WARP_TILE_SIZE_X % (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X) == 0U);
    static_assert(
        WARP_TILE_SIZE_Y % (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y) == 0U);

    constexpr unsigned int NUM_THREADS_X{NUM_WARPS_X * NUM_THREADS_PER_WARP_X};
    constexpr unsigned int NUM_THREADS_Y{NUM_WARPS_Y * NUM_THREADS_PER_WARP_Y};
    // Avoid using blockDim.x * blockDim.y as the number of threads per block.
    // Because it is a runtime constant and the compiler cannot optimize the
    // loop unrolling based on that.
    // Use a compile time constant instead.
    constexpr size_t NUM_THREADS{NUM_THREADS_X * NUM_THREADS_Y};

    // Cache a tile of A and B in shared memory for data reuse.
    __shared__ T
        A_thread_block_tile_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    // A_vals is cached in the register.
    T A_vals[NUM_THREAD_TILES_PER_WARP_Y][THREAD_TILE_SIZE_Y] = {
        static_cast<T>(0)};
    // B_vals is cached in the register.
    T B_vals[NUM_THREAD_TILES_PER_WARP_X][THREAD_TILE_SIZE_X] = {
        static_cast<T>(0)};

    size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};
    size_t const warp_linear_idx{thread_linear_idx / 32U};
    size_t const warp_row_idx{warp_linear_idx / NUM_WARPS_X};
    size_t const warp_col_idx{warp_linear_idx % NUM_WARPS_X};
    size_t const thread_linear_idx_in_warp{thread_linear_idx % 32U};
    size_t const thread_linear_row_idx_in_warp{thread_linear_idx_in_warp /
                                               NUM_THREADS_PER_WARP_X};
    size_t const thread_linear_col_idx_in_warp{thread_linear_idx_in_warp %
                                               NUM_THREADS_PER_WARP_X};

    // Number of outer loops to perform the sum of inner products.
    // C_thread_block_tile =
    // \sigma_{thread_block_tile_idx=0}^{num_thread_block_tiles-1} A[:,
    // thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
    // B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :]
    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
                                        BLOCK_TILE_SIZE_K};
    // Each thread in the block processes NUM_THREAD_TILES_PER_WARP_Y *
    // NUM_THREAD_TILES_PER_WARP_X * THREAD_TILE_SIZE_Y *
    // THREAD_TILE_SIZE_X output values.
    T C_thread_results[NUM_THREAD_TILES_PER_WARP_Y][NUM_THREAD_TILES_PER_WARP_X]
                      [THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X] = {
                          static_cast<T>(0)};

    constexpr size_t NUM_VECTOR_UNITS{sizeof(int4) / sizeof(T)};
    static_assert(sizeof(int4) % sizeof(T) == 0U);
    static_assert(BLOCK_TILE_SIZE_K % NUM_VECTOR_UNITS == 0U);
    static_assert(BLOCK_TILE_SIZE_X % NUM_VECTOR_UNITS == 0U);
    constexpr size_t VECTORIZED_THREAD_TILE_SIZE_X{THREAD_TILE_SIZE_X /
                                                   NUM_VECTOR_UNITS};
    static_assert(THREAD_TILE_SIZE_X % NUM_VECTOR_UNITS == 0U);
    constexpr size_t VECTORIZED_THREAD_TILE_SIZE_Y{THREAD_TILE_SIZE_Y /
                                                   NUM_VECTOR_UNITS};
    static_assert(THREAD_TILE_SIZE_Y % NUM_VECTOR_UNITS == 0U);

    for (size_t thread_block_tile_idx{0U};
         thread_block_tile_idx < num_thread_block_tiles;
         ++thread_block_tile_idx)
    {
        load_data_to_shared_memory_transposed_vectorized<
            T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
            NUM_THREADS>(A, lda, B, ldb, A_thread_block_tile_transposed,
                         B_thread_block_tile, thread_block_tile_idx,
                         thread_linear_idx, m, n, k);
        __syncthreads();

// Perform A[:, thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
// B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :] where A[:,
// thread_block_tile_idx:BLOCK_TILE_SIZE_K] and
// B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :] are cached in the
// shared memory as A_thread_block_tile and B_thread_block_tile,
// respectively. This inner product is further decomposed to
// BLOCK_TILE_SIZE_K outer products. A_thread_block_tile *
// B_thread_block_tile = \sigma_{k_i=0}^{BLOCK_TILE_SIZE_K-1}
// A_thread_block_tile[:, k_i] @ B_thread_block_tile[k_i, :] Note that
// both A_thread_block_tile and B_thread_block_tile can be cached in the
// register.
#pragma unroll
        for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
        {
#pragma unroll
            for (size_t thread_tile_repeat_row_idx{0U};
                 thread_tile_repeat_row_idx < NUM_THREAD_TILES_PER_WARP_Y;
                 ++thread_tile_repeat_row_idx)
            {
                size_t const A_thread_block_tile_row_idx{
                    warp_row_idx * WARP_TILE_SIZE_Y +
                    thread_tile_repeat_row_idx *
                        (WARP_TILE_SIZE_Y / NUM_THREAD_TILES_PER_WARP_Y) +
                    thread_linear_row_idx_in_warp * THREAD_TILE_SIZE_Y};
                size_t const A_thread_block_tile_col_idx{k_i};
#pragma unroll
                for (size_t thread_tile_y_vector_idx{0U};
                     thread_tile_y_vector_idx < VECTORIZED_THREAD_TILE_SIZE_Y;
                     ++thread_tile_y_vector_idx)
                {
                    *reinterpret_cast<int4*>(
                        &A_vals[thread_tile_repeat_row_idx]
                               [thread_tile_y_vector_idx * NUM_VECTOR_UNITS]) =
                        *reinterpret_cast<int4 const*>(
                            &A_thread_block_tile_transposed
                                [A_thread_block_tile_col_idx]
                                [A_thread_block_tile_row_idx +
                                 thread_tile_y_vector_idx * NUM_VECTOR_UNITS]);
                }
            }
#pragma unroll
            for (size_t thread_tile_repeat_col_idx{0U};
                 thread_tile_repeat_col_idx < NUM_THREAD_TILES_PER_WARP_X;
                 ++thread_tile_repeat_col_idx)
            {
                size_t const B_thread_block_tile_row_idx{k_i};
                size_t const B_thread_block_tile_col_idx{
                    warp_col_idx * WARP_TILE_SIZE_X +
                    thread_tile_repeat_col_idx *
                        (WARP_TILE_SIZE_X / NUM_THREAD_TILES_PER_WARP_X) +
                    thread_linear_col_idx_in_warp * THREAD_TILE_SIZE_X};
#pragma unroll
                for (size_t thread_tile_x_vector_idx{0U};
                     thread_tile_x_vector_idx < VECTORIZED_THREAD_TILE_SIZE_X;
                     ++thread_tile_x_vector_idx)
                {
                    *reinterpret_cast<int4*>(
                        &B_vals[thread_tile_repeat_col_idx]
                               [thread_tile_x_vector_idx * NUM_VECTOR_UNITS]) =
                        *reinterpret_cast<int4 const*>(
                            &B_thread_block_tile[B_thread_block_tile_row_idx]
                                                [B_thread_block_tile_col_idx +
                                                 thread_tile_x_vector_idx *
                                                     NUM_VECTOR_UNITS]);
                }
            }

// Compute NUM_THREAD_TILES_PER_WARP_Y * NUM_THREAD_TILES_PER_WARP_X outer
// products.
#pragma unroll
            for (size_t thread_tile_repeat_row_idx{0U};
                 thread_tile_repeat_row_idx < NUM_THREAD_TILES_PER_WARP_Y;
                 ++thread_tile_repeat_row_idx)
            {
#pragma unroll
                for (size_t thread_tile_repeat_col_idx{0U};
                     thread_tile_repeat_col_idx < NUM_THREAD_TILES_PER_WARP_X;
                     ++thread_tile_repeat_col_idx)
                {
#pragma unroll
                    for (size_t thread_tile_y_idx{0U};
                         thread_tile_y_idx < THREAD_TILE_SIZE_Y;
                         ++thread_tile_y_idx)
                    {
#pragma unroll
                        for (size_t thread_tile_x_idx{0U};
                             thread_tile_x_idx < THREAD_TILE_SIZE_X;
                             ++thread_tile_x_idx)
                        {
                            C_thread_results[thread_tile_repeat_row_idx]
                                            [thread_tile_repeat_col_idx]
                                            [thread_tile_y_idx]
                                            [thread_tile_x_idx] +=
                                A_vals[thread_tile_repeat_row_idx]
                                      [thread_tile_y_idx] *
                                B_vals[thread_tile_repeat_col_idx]
                                      [thread_tile_x_idx];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

// Write the results to DRAM.
#pragma unroll
    for (size_t thread_tile_repeat_row_idx{0U};
         thread_tile_repeat_row_idx < NUM_THREAD_TILES_PER_WARP_Y;
         ++thread_tile_repeat_row_idx)
    {
#pragma unroll
        for (size_t thread_tile_repeat_col_idx{0U};
             thread_tile_repeat_col_idx < NUM_THREAD_TILES_PER_WARP_X;
             ++thread_tile_repeat_col_idx)
        {
#pragma unroll
            for (size_t thread_tile_y_idx{0U};
                 thread_tile_y_idx < THREAD_TILE_SIZE_Y; ++thread_tile_y_idx)
            {
#pragma unroll
                for (size_t thread_tile_x_vector_idx{0U};
                     thread_tile_x_vector_idx < VECTORIZED_THREAD_TILE_SIZE_X;
                     ++thread_tile_x_vector_idx)
                {
                    size_t const C_row_idx{
                        blockIdx.y * BLOCK_TILE_SIZE_Y +
                        warp_row_idx * WARP_TILE_SIZE_Y +
                        thread_tile_repeat_row_idx *
                            (WARP_TILE_SIZE_Y / NUM_THREAD_TILES_PER_WARP_Y) +
                        thread_linear_row_idx_in_warp * THREAD_TILE_SIZE_Y +
                        thread_tile_y_idx};
                    size_t const C_col_idx{
                        blockIdx.x * BLOCK_TILE_SIZE_X +
                        warp_col_idx * WARP_TILE_SIZE_X +
                        thread_tile_repeat_col_idx *
                            (WARP_TILE_SIZE_X / NUM_THREAD_TILES_PER_WARP_X) +
                        thread_linear_col_idx_in_warp * THREAD_TILE_SIZE_X +
                        thread_tile_x_vector_idx * NUM_VECTOR_UNITS};

                    if (C_row_idx < m && C_col_idx < n)
                    {
                        int4 C_vals{*reinterpret_cast<int4 const*>(
                            &C[C_row_idx * ldc + C_col_idx])};
#pragma unroll
                        for (size_t i{0U}; i < NUM_VECTOR_UNITS; ++i)
                        {
                            reinterpret_cast<T*>(&C_vals)[i] =
                                alpha *
                                    C_thread_results[thread_tile_repeat_row_idx]
                                                    [thread_tile_repeat_col_idx]
                                                    [thread_tile_y_idx]
                                                    [thread_tile_x_vector_idx *
                                                         NUM_VECTOR_UNITS +
                                                     i] +
                                beta * reinterpret_cast<T const*>(&C_vals)[i];
                        }
                        *reinterpret_cast<int4*>(
                            &C[C_row_idx * ldc + C_col_idx]) = C_vals;
                    }
                }
            }
        }
    }
}

template <typename T>
void launch_gemm_kernel_v06_vectorized(size_t m, size_t n, size_t k,
                                       T const* alpha, T const* A, size_t lda,
                                       T const* B, size_t ldb, T const* beta,
                                       T* C, size_t ldc, cudaStream_t stream)
{
    // Feel free to play with the block tile sizes.
    // The algorithm correctness should always be guaranteed.
    constexpr unsigned int BLOCK_TILE_SIZE_X{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{16U};

    constexpr unsigned int WARP_TILE_SIZE_X{32U};
    constexpr unsigned int WARP_TILE_SIZE_Y{64U};
    constexpr unsigned int NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
    constexpr unsigned int NUM_WARPS_Y{BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y};
    static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0U);

    constexpr unsigned int THREAD_TILE_SIZE_X{8U};
    constexpr unsigned int THREAD_TILE_SIZE_Y{8U};

    constexpr unsigned int NUM_THREADS_PER_WARP_X{4U};
    constexpr unsigned int NUM_THREADS_PER_WARP_Y{8U};
    static_assert(NUM_THREADS_PER_WARP_X * NUM_THREADS_PER_WARP_Y == 32U);
    static_assert(
        WARP_TILE_SIZE_X % (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X) == 0U);
    static_assert(
        WARP_TILE_SIZE_Y % (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y) == 0U);

    constexpr unsigned int NUM_THREADS_X{NUM_WARPS_X * NUM_THREADS_PER_WARP_X};
    constexpr unsigned int NUM_THREADS_Y{NUM_WARPS_Y * NUM_THREADS_PER_WARP_Y};

    constexpr unsigned int NUM_THREADS_PER_BLOCK{NUM_THREADS_X * NUM_THREADS_Y};

    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) /
            BLOCK_TILE_SIZE_X,
        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) /
            BLOCK_TILE_SIZE_Y,
        1U};
    gemm_v06_vectorized<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                        BLOCK_TILE_SIZE_K, WARP_TILE_SIZE_X, WARP_TILE_SIZE_Y,
                        THREAD_TILE_SIZE_X, THREAD_TILE_SIZE_Y,
                        NUM_THREADS_PER_WARP_X, NUM_THREADS_PER_WARP_Y>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B, ldb,
                                              *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}
```

The performance of this FP32 GEMM implementation becomes 20.16 TFLOPS on an NVIDIA GeForce RTX 3090 GPU. Comparing to the cuBLAS FP32 GEMM performance, which is 24.59 TFLOPS, this implementation has been optimized reasonably well.


## Implementation with 2D Block Tiling and 2D Warp Tiling and Tensor Core and Vectorized Memory Access

Because we have already organized the GEMM CUDA kernel in a warp-centric way, and NVIDIA Tensor Core instructions are interfaced at the warp level, it is then very straightforward to utilize NVIDIA Tensor Core(https://leimao.github.io/blog/NVIDIA-Tensor-Core-Programming/) WMMA APIs to further accelerate the GEMM computation. Because the NVIDIA Tensor Core does not support IEEE FP32 computation, we will make this CUDA kernel to run FP16 GEMM instead.

Comparing to the implementation with 2D block tiling and 2D warp tiling and 2D thread tiling and vectorized memory access, the implementation with 2D block tiling and 2D warp tiling and Tensor Core and vectorized memory access is simpler because the thread tiling process is abstracted away by the NVIDIA Tensor Core warp-level WMMA APIs.


从数学角度来看，给定矩阵乘法和累加操作：$D_{b_m,b_n}^{d_{bm} \times d_{bn}} = \sum_{b_k=1}^{k/d_{bk}} A_{b_m,b_k}^{d_{bm} \times d_{bk}} B_{b_k,b_n}^{d_{bk} \times d_{bn}} + C_{b_m,b_n}^{d_{bm} \times d_{bn}}$，其中 $D_{b_m,b_n} \in \mathbb{R}^{d_{bm} \times d_{bn}}$，$A_{b_m,b_k} \in \mathbb{R}^{d_{bm} \times d_{bk}}$，$B_{b_k,b_n} \in \mathbb{R}^{d_{bk} \times d_{bn}}$，$C_{b_m,b_n} \in \mathbb{R}^{d_{bm} \times d_{bn}}$，这些矩阵可以被分割成更小的矩阵。

$$A_{b_m,b_k}^{d_{bm} \times d_{bk}} = \begin{bmatrix}
\left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{1,1} & \left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{1,2} & \cdots & \left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{1,d_{bk}/d_{wk}} \\
\left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{2,1} & \left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{2,2} & \cdots & \left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{2,d_{bk}/d_{wk}} \\
\vdots & \vdots & \ddots & \vdots \\
\left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{d_{bm}/d_{wm},1} & \left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{d_{bm}/d_{wm},2} & \cdots & \left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{d_{bm}/d_{wm},d_{bk}/d_{wk}}
\end{bmatrix}$$

$$B_{b_k,b_n}^{d_{bk} \times d_{bn}} = \begin{bmatrix}
\left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{1,1} & \left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{1,2} & \cdots & \left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{1,d_{bn}/d_{wn}} \\
\left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{2,1} & \left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{2,2} & \cdots & \left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{2,d_{bn}/d_{wn}} \\
\vdots & \vdots & \ddots & \vdots \\
\left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{d_{bk}/d_{wk},1} & \left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{d_{bk}/d_{wk},2} & \cdots & \left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{d_{bk}/d_{wk},d_{bn}/d_{wn}}
\end{bmatrix}$$

$$C_{b_m,b_n}^{d_{bm} \times d_{bn}} = \begin{bmatrix}
\left(C_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{1,1} & \left(C_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{1,2} & \cdots & \left(C_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{1,d_{bn}/d_{wn}} \\
\left(C_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{2,1} & \left(C_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{2,2} & \cdots & \left(C_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{2,d_{bn}/d_{wn}} \\
\vdots & \vdots & \ddots & \vdots \\
\left(C_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{d_{bm}/d_{wm},1} & \left(C_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{d_{bm}/d_{wm},2} & \cdots & \left(C_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{d_{bm}/d_{wm},d_{bn}/d_{wn}}
\end{bmatrix}$$

$$D_{b_m,b_n}^{d_{bm} \times d_{bn}} = \begin{bmatrix}
\left(D_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{1,1} & \left(D_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{1,2} & \cdots & \left(D_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{1,d_{bn}/d_{wn}} \\
\left(D_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{2,1} & \left(D_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{2,2} & \cdots & \left(D_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{2,d_{bn}/d_{wn}} \\
\vdots & \vdots & \ddots & \vdots \\
\left(D_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{d_{bm}/d_{wm},1} & \left(D_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{d_{bm}/d_{wm},2} & \cdots & \left(D_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{d_{bm}/d_{wm},d_{bn}/d_{wn}}
\end{bmatrix}$$

$D_{b_m,b_n}^{d_{bm} \times d_{bn}}$ 中的每个小矩阵都是通过多个小矩阵乘法和累加计算得出的。

$$\left(D_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}} = \left(\sum_{b_k=1}^{k/d_{bk}} A_{b_m,b_k}^{d_{bm} \times d_{bk}} B_{b_k,b_n}^{d_{bk} \times d_{bn}} + C_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}$$

$$= \sum_{b_k=1}^{k/d_{bk}} \left(A_{b_m,b_k}^{d_{bm} \times d_{bk}} B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}} + \left(C_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}$$

$$= \sum_{b_k=1}^{k/d_{bk}} \left(\sum_{w_k=1}^{d_{bk}/d_{wk}} \left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}} \left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right) + \left(C_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}$$

每个具有块warp索引 $(w_m, w_n)$ 的warp，其中 $w_m \in [1, d_{bm}/d_{wm}]$ 和 $w_n \in [1, d_{bn}/d_{wn}]$，在具有块索引 $(b_m, b_n)$ 的块中，其中 $b_m \in [1, m/d_{bm}]$ 和 $b_n \in [1, n/d_{bn}]$，负责计算一个小矩阵乘法和累加 $\left(D_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}$。

假设Tensor Core WMMA GEMM的大小是 $d_{tm} \times d_{tn} \times d_{tk}$。因为矩阵 $\left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}$ 可以沿行维度分割为 $d_{wm}/d_{tm}$ 个片段，矩阵 $\left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}$ 可以沿列维度分割为 $d_{wn}/d_{tn}$ 个片段。我们有：

$$\left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}} = \begin{bmatrix}
\left(\left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{1,1}^{d_{tm} \times d_{tk}} & \left(\left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{1,2}^{d_{tm} \times d_{tk}} & \cdots & \left(\left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{1,d_{wk}/d_{tk}}^{d_{tm} \times d_{tk}} \\
\left(\left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{2,1}^{d_{tm} \times d_{tk}} & \left(\left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{2,2}^{d_{tm} \times d_{tk}} & \cdots & \left(\left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{2,d_{wk}/d_{tk}}^{d_{tm} \times d_{tk}} \\
\vdots & \vdots & \ddots & \vdots \\
\left(\left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{d_{wm}/d_{tm},1}^{d_{tm} \times d_{tk}} & \left(\left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{d_{wm}/d_{tm},2}^{d_{tm} \times d_{tk}} & \cdots & \left(\left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{d_{wm}/d_{tm},d_{wk}/d_{tk}}^{d_{tm} \times d_{tk}}
\end{bmatrix}$$

$$\left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}} = \begin{bmatrix}
\left(\left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{1,1}^{d_{tk} \times d_{tn}} & \left(\left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{1,2}^{d_{tk} \times d_{tn}} & \cdots & \left(\left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{1,d_{wn}/d_{tn}}^{d_{tk} \times d_{tn}} \\
\left(\left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{2,1}^{d_{tk} \times d_{tn}} & \left(\left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{2,2}^{d_{tk} \times d_{tn}} & \cdots & \left(\left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{2,d_{wn}/d_{tn}}^{d_{tk} \times d_{tn}} \\
\vdots & \vdots & \ddots & \vdots \\
\left(\left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{d_{wk}/d_{tk},1}^{d_{tk} \times d_{tn}} & \left(\left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{d_{wk}/d_{tk},2}^{d_{tk} \times d_{tn}} & \cdots & \left(\left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{d_{wk}/d_{tk},d_{wn}/d_{tn}}^{d_{tk} \times d_{tn}}
\end{bmatrix}$$

每个warp不是调用线程级指令，而是调用WMMA warp级Tensor Core来计算所有的 $\left(\left(D_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}}$ 迭代计算 $\left(D_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}$。

$$\left(\left(D_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}} = \left(\sum_{b_k=1}^{k/d_{bk}} \left(\sum_{w_k=1}^{d_{bk}/d_{wk}} \left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}} \left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right) + \left(C_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}}$$

$$= \sum_{b_k=1}^{k/d_{bk}} \left(\sum_{w_k=1}^{d_{bk}/d_{wk}} \left(\sum_{t_k=1}^{d_{wk}/d_{tk}} \left(\left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{t_m,t_k}^{d_{tm} \times d_{tk}} \left(\left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{t_k,t_n}^{d_{tk} \times d_{tn}}\right)\right) + \left(\left(C_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}}$$

在这个实现中，由于WMMA Tensor Core API的限制，$d_{tm} = 16$，$d_{tn} = 16$，$d_{tk} = 16$。

下面的代码片段展示了使用2D块分块、2D warp分块、Tensor Core和向量化内存访问的实现。

```c++
// GEMM kernel v07.
// Each thread in the block processes THREAD_TILE_SIZE_Y *
// THREAD_TILE_SIZE_X output values. Number of threads BLOCK_TILE_SIZE_Y *
// BLOCK_TILE_SIZE_X / (THREAD_TILE_SIZE_Y * THREAD_TILE_SIZE_X)
template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t BLOCK_TILE_SKEW_SIZE_X,
          size_t BLOCK_TILE_SKEW_SIZE_Y, size_t WARP_TILE_SIZE_X,
          size_t WARP_TILE_SIZE_Y, size_t WMMA_TILE_SIZE_X,
          size_t WMMA_TILE_SIZE_Y, size_t WMMA_TILE_SIZE_K, size_t NUM_THREADS>
__global__ void gemm_v07_vectorized(size_t m, size_t n, size_t k, T alpha,
                                    T const* A, size_t lda, T const* B,
                                    size_t ldb, T beta, T* C, size_t ldc)
{
    constexpr size_t NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
    static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0U);

    // Cache a tile of A and B in shared memory for data reuse.
    __shared__ T A_thread_block_tile_transposed[BLOCK_TILE_SIZE_K]
                                               [BLOCK_TILE_SIZE_Y +
                                                BLOCK_TILE_SKEW_SIZE_Y];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X +
                                                        BLOCK_TILE_SKEW_SIZE_X];

    constexpr size_t NUM_WMMA_TILES_X{WARP_TILE_SIZE_X / WMMA_TILE_SIZE_X};
    static_assert(WARP_TILE_SIZE_X % WMMA_TILE_SIZE_X == 0U);
    constexpr size_t NUM_WMMA_TILES_Y{WARP_TILE_SIZE_Y / WMMA_TILE_SIZE_Y};
    static_assert(WARP_TILE_SIZE_Y % WMMA_TILE_SIZE_Y == 0U);
    constexpr size_t NUM_WMMA_TILES_K{BLOCK_TILE_SIZE_K / WMMA_TILE_SIZE_K};
    static_assert(BLOCK_TILE_SIZE_K % WMMA_TILE_SIZE_K == 0U);

    // Declare the fragments.
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_TILE_SIZE_Y,
                           WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_K, T,
                           nvcuda::wmma::col_major>
        a_frags[NUM_WMMA_TILES_Y];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_TILE_SIZE_Y,
                           WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_K, T,
                           nvcuda::wmma::row_major>
        b_frags[NUM_WMMA_TILES_X];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_SIZE_Y,
                           WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_K, T>
        acc_frags[NUM_WMMA_TILES_Y][NUM_WMMA_TILES_X];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_SIZE_Y,
                           WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_K, T>
        c_frag;

// Make sure the accumulator starts from 0.
#pragma unroll
    for (size_t wmma_tile_row_idx{0U}; wmma_tile_row_idx < NUM_WMMA_TILES_Y;
         ++wmma_tile_row_idx)
    {
        for (size_t wmma_tile_col_idx{0U}; wmma_tile_col_idx < NUM_WMMA_TILES_X;
             ++wmma_tile_col_idx)
        {
            nvcuda::wmma::fill_fragment(
                acc_frags[wmma_tile_row_idx][wmma_tile_col_idx],
                static_cast<T>(0));
        }
    }

    size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};
    size_t const warp_linear_idx{thread_linear_idx / 32U};
    size_t const warp_row_idx{warp_linear_idx / NUM_WARPS_X};
    size_t const warp_col_idx{warp_linear_idx % NUM_WARPS_X};

    // Number of outer loops to perform the sum of inner products.
    // C_thread_block_tile =
    // \sigma_{thread_block_tile_idx=0}^{num_thread_block_tiles-1} A[:,
    // thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
    // B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :]
    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
                                        BLOCK_TILE_SIZE_K};

    for (size_t thread_block_tile_idx{0U};
         thread_block_tile_idx < num_thread_block_tiles;
         ++thread_block_tile_idx)
    {
        load_data_to_shared_memory_transposed_vectorized<
            T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
            NUM_THREADS, BLOCK_TILE_SKEW_SIZE_X, BLOCK_TILE_SKEW_SIZE_Y>(
            A, lda, B, ldb, A_thread_block_tile_transposed, B_thread_block_tile,
            thread_block_tile_idx, thread_linear_idx, m, n, k);
        __syncthreads();

// Perform A[:, thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
// B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :] where A[:,
// thread_block_tile_idx:BLOCK_TILE_SIZE_K] and
// B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :] are cached in the
// shared memory as A_thread_block_tile and B_thread_block_tile,
// respectively. This inner product is further decomposed to
// BLOCK_TILE_SIZE_K outer products. A_thread_block_tile *
// B_thread_block_tile = \sigma_{k_i=0}^{BLOCK_TILE_SIZE_K-1}
// A_thread_block_tile[:, k_i] @ B_thread_block_tile[k_i, :] Note that
// both A_thread_block_tile and B_thread_block_tile can be cached in the
// register.
#pragma unroll
        for (size_t k_i{0U}; k_i < NUM_WMMA_TILES_K; ++k_i)
        {
#pragma unroll
            for (size_t wmma_tile_row_idx{0U};
                 wmma_tile_row_idx < NUM_WMMA_TILES_Y; ++wmma_tile_row_idx)
            {
                nvcuda::wmma::load_matrix_sync(
                    a_frags[wmma_tile_row_idx],
                    &A_thread_block_tile_transposed[k_i * WMMA_TILE_SIZE_K]
                                                   [warp_row_idx *
                                                        WARP_TILE_SIZE_Y +
                                                    wmma_tile_row_idx *
                                                        WMMA_TILE_SIZE_Y],
                    BLOCK_TILE_SIZE_Y + BLOCK_TILE_SKEW_SIZE_Y);
#pragma unroll
                for (size_t wmma_tile_col_idx{0U};
                     wmma_tile_col_idx < NUM_WMMA_TILES_X; ++wmma_tile_col_idx)
                {
                    // These loads are extremely slow somehow, which affects the
                    // performance a lot. Load the fragment from shared memory.
                    nvcuda::wmma::load_matrix_sync(
                        b_frags[wmma_tile_col_idx],
                        &B_thread_block_tile[k_i * WMMA_TILE_SIZE_K]
                                            [warp_col_idx * WARP_TILE_SIZE_X +
                                             wmma_tile_col_idx *
                                                 WMMA_TILE_SIZE_Y],
                        BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X);

                    // Perform the matrix multiplication.
                    nvcuda::wmma::mma_sync(
                        acc_frags[wmma_tile_row_idx][wmma_tile_col_idx],
                        a_frags[wmma_tile_row_idx], b_frags[wmma_tile_col_idx],
                        acc_frags[wmma_tile_row_idx][wmma_tile_col_idx]);
                }
            }
        }
        __syncthreads();
    }

// Write the results to DRAM.
#pragma unroll
    for (size_t wmma_tile_row_idx{0U}; wmma_tile_row_idx < NUM_WMMA_TILES_Y;
         ++wmma_tile_row_idx)
    {
#pragma unroll
        for (size_t wmma_tile_col_idx{0U}; wmma_tile_col_idx < NUM_WMMA_TILES_X;
             ++wmma_tile_col_idx)
        {
            // Load the fragment from shared memory.
            nvcuda::wmma::load_matrix_sync(
                c_frag,
                &C[(blockIdx.y * BLOCK_TILE_SIZE_Y +
                    warp_row_idx * WARP_TILE_SIZE_Y +
                    wmma_tile_row_idx * WMMA_TILE_SIZE_Y) *
                       n +
                   blockIdx.x * BLOCK_TILE_SIZE_X +
                   warp_col_idx * WARP_TILE_SIZE_X +
                   wmma_tile_col_idx * WMMA_TILE_SIZE_X],
                n, nvcuda::wmma::mem_row_major);
            // Perform scaling and addition.
            for (size_t i{0}; i < c_frag.num_elements; ++i)
            {
                c_frag.x[i] =
                    alpha *
                        acc_frags[wmma_tile_row_idx][wmma_tile_col_idx].x[i] +
                    beta * c_frag.x[i];
            }
            // Store the fragment back to shared memory.
            nvcuda::wmma::store_matrix_sync(
                &C[(blockIdx.y * BLOCK_TILE_SIZE_Y +
                    warp_row_idx * WARP_TILE_SIZE_Y +
                    wmma_tile_row_idx * WMMA_TILE_SIZE_Y) *
                       n +
                   blockIdx.x * BLOCK_TILE_SIZE_X +
                   warp_col_idx * WARP_TILE_SIZE_X +
                   wmma_tile_col_idx * WMMA_TILE_SIZE_X],
                c_frag, n, nvcuda::wmma::mem_row_major);
        }
    }
}

template <typename T>
void launch_gemm_kernel_v07_vectorized(size_t m, size_t n, size_t k,
                                       T const* alpha, T const* A, size_t lda,
                                       T const* B, size_t ldb, T const* beta,
                                       T* C, size_t ldc, cudaStream_t stream)
{
    // Feel free to play with the block tile sizes.
    // The algorithm correctness should always be guaranteed.
    constexpr unsigned int BLOCK_TILE_SIZE_X{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{16U};

    // The skew size is used to avoid bank conflicts in shared memory.
    constexpr size_t BLOCK_TILE_SKEW_SIZE_X{16U};
    constexpr size_t BLOCK_TILE_SKEW_SIZE_Y{16U};

    constexpr unsigned int WARP_TILE_SIZE_X{32U};
    constexpr unsigned int WARP_TILE_SIZE_Y{64U};
    constexpr unsigned int NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
    constexpr unsigned int NUM_WARPS_Y{BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y};
    static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0U);

    constexpr unsigned int WMMA_TILE_SIZE_X{16U};
    constexpr unsigned int WMMA_TILE_SIZE_Y{16U};
    constexpr unsigned int WMMA_TILE_SIZE_K{16U};

    constexpr unsigned int NUM_THREADS_PER_BLOCK{NUM_WARPS_X * NUM_WARPS_Y *
                                                 32U};

    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) /
            BLOCK_TILE_SIZE_X,
        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) /
            BLOCK_TILE_SIZE_Y,
        1U};
    gemm_v07_vectorized<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                        BLOCK_TILE_SIZE_K, BLOCK_TILE_SKEW_SIZE_X,
                        BLOCK_TILE_SKEW_SIZE_Y, WARP_TILE_SIZE_X,
                        WARP_TILE_SIZE_Y, WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_Y,
                        WMMA_TILE_SIZE_K, NUM_THREADS_PER_BLOCK>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B, ldb,
                                              *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}
```

Because the fundamental WMMA size is $16\times 16 \times 16$ , all the 32 threads in the same warp has to synergistically access the shared memory where the WMMA fragment is cached. It is then very possible that the shared memory bank conflicts will happen. To avoid the shared memory bank conflicts, we will have to pad the shared memory size to make sure the shared memory bank conflicts will not happen. This is why we have to use the skew size to pad the shared memory size at the leading dimension.

The performance of this FP16 GEMM implementation becomes 46.78 TFLOPS on an NVIDIA GeForce RTX 3090 GPU. Comparing to the cuBLAS FP16 GEMM performance, which is 138.95 TFLOPS, this implementation only achieves 33.7% of the cuBLAS FP16 GEMM performance. We will leave the performance optimization of this implementation as a future work.

## Conclusions

The optimizations we performed on the GEMM CUDA kernels mainly follow the diagrams in the article “CUTLASS: Fast Linear Algebra in CUDA C++”.(https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)

![](https://files.mdnice.com/user/59/5681a7ca-2fad-47d3-8068-c0f94e329955.png)

With the optimization techniques, such as 2D block tiling, 2D warp tiling, 2D thread tiling, and vectorized memory access, we can achieve 20.16 TFLOPS FP32 GEMM performance on an NVIDIA GeForce RTX 3090 GPU, which is 80% - 90% of the cuBLAS FP32 GEMM performance.

## Source Code

The source code of the GEMM CUDA kernels can be found in my GitHub repository “CUDA GEMM Optimization”(https://github.com/leimao/CUDA-GEMM-Optimization/).

## References

- CUTLASS: Fast Linear Algebra in CUDA C++(https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)
- CUDA GEMM Optimization - GitHub(https://github.com/leimao/CUDA-GEMM-Optimization/)
- CUDA Matrix Multiplication(https://leimao.github.io/blog/CUDA-Matrix-Multiplication/)
- CUDA Vectorized Memory Access(https://leimao.github.io/blog/CUDA-Vectorized-Memory-Access/)
- CUDA Data Alignment(https://leimao.github.io/blog/CUDA-Data-Alignment/)
- CUDA Shared Memory Bank(https://leimao.github.io/blog/CUDA-Shared-Memory-Bank/)
- NVIDIA Tensor Core Programming(https://leimao.github.io/blog/NVIDIA-Tensor-Core-Programming/)
- CUDA Tensor Core GEMM(https://github.com/NVIDIA/cuda-samples/blob/e8568c417356f7e66bb9b7130d6be7e55324a519/Samples/3_CUDA_Features/cudaTensorCoreGemm/cudaTensorCoreGemm.cu)
- How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog(https://siboehm.com/articles/22/CUDA-MMM)





