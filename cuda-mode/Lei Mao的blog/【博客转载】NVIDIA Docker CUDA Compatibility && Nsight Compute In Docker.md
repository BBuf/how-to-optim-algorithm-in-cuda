> 博客来源：https://leimao.github.io/blog/NVIDIA-Docker-CUDA-Compatibility/ ，来自Lei Mao，已获得作者转载授权。后续会转载一些Lei Mao的CUDA相关Blog，也是一个完整的专栏，Blog会从稍早一些的CUDA架构到当前最新的CUDA架构，也会包含实用工程技巧，底层指令分析，Cutlass分析等等多个课题，是一个时间线十分明确的专栏。

# NVIDIA Docker CUDA 兼容性

## 简介

NVIDIA NGC CUDA Docker 容器是开发和部署 CUDA 应用程序的极为便利的工具。它们允许我们在安装了 Docker 的平台上运行几乎任何 CUDA 运行时库和 CUDA 应用程序，使代码具有可移植性和可重现性。正因为如此，我一直在使用 NVIDIA NGC CUDA Docker 容器进行所有的 CUDA 开发和部署工作。在我的个人电脑上，我只安装了 NVIDIA CUDA 驱动程序和 Docker。当我需要使用 CUDA 运行时库时，我只需拉取 NVIDIA NGC CUDA Docker 容器并运行它。我从未在个人电脑上安装过 CUDA 运行时库。

然而，最近我在使用 NVIDIA NGC CUDA Docker 容器时遇到了一些奇怪的问题。经过一些调查，我发现这些问题是由 Docker 容器内的 CUDA 运行时库版本与宿主机上的 CUDA 驱动程序版本不兼容引起的。在这篇博客中，我将分享我的经历，并解释为什么我们应该尽量使用与 Docker 容器内 CUDA 运行时库版本相同的宿主机 CUDA 驱动程序版本。

## NVIDIA Docker 兼容性引起的奇怪问题

最近，我在使用 NVIDIA NGC CUDA Docker 容器(https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda)在 Ubuntu 上实现一些 CUDA  kernel时遇到了一些无法解释的问题。

例如，对于某些 `for` 循环，如果我使用 `#pragma unroll` 来展开循环体，CUDA  kernel在安装了 GV100 GPU 的机器上不会产生正确的结果。然而，在安装了 RTX 3090 GPU 的另一台机器上，使用 `#pragma unroll` 的相同代码工作正常。两台机器都运行相同版本的 Ubuntu，并使用相同版本的 NVIDIA NGC CUDA Docker 容器 `nvcr.io/nvidia/cuda:12.0.1-devel-ubuntu22.04`。

此外，以下两个 `if` 语句是完全等价的。然而，使用第一个 `if` 语句的 CUDA  kernel在两台机器上都产生正确的结果，而使用第二个 `if` 语句的 CUDA  kernel在安装了 GV100 GPU 的机器上产生错误的结果。

```c++
size_t m, n, C_row_idx, C_col_idx, i;
// The following conditions have no numerical overflow.
if (C_row_idx < m && C_col_idx + i < n) // This worked.
{
    // Some math.
}
if (C_row_idx < m && C_col_idx < n && C_col_idx + i < n) // This failed.
{
    // The same math.
}
```

这些问题似乎表明 CUDA 编译器存在严重问题。然而，我不相信 CUDA 编译器会有如此天真的错误。

经过一些调查，我发现尽管两台机器运行相同版本的 Ubuntu 并使用相同版本的 NVIDIA NGC CUDA Docker 容器，但 CUDA 驱动程序版本不同。安装了 GV100 GPU 的机器在宿主机上有 `Driver Version: 470.223.02 CUDA Version: 11.4`，而安装了 RTX 3090 GPU 的机器在宿主机上有 `Driver Version: 525.147.05 CUDA Version: 12.0`。所以我们在安装了 GV100 GPU 的机器上运行的是 NVIDIA NGC CUDA Docker 容器 `nvcr.io/nvidia/cuda:12.0.1-devel-ubuntu22.04`，它安装了 12.0.1 版本的 CUDA 运行时库，但运行在 11.4 版本的 CUDA 驱动程序上。这就是为什么某些 CUDA  kernel在安装了 GV100 GPU 的机器上行为不正确的原因。

因此，我在安装了 GV100 GPU 的机器上改用 NVIDIA NGC CUDA Docker 容器 `nvcr.io/nvidia/cuda:11.4.3-devel-ubuntu20.04`。这次，之前有问题的 CUDA  kernel都完美地工作了。

## 结论

CUDA 向后和向前兼容性(leimao.github.io/blog/CUDA-Compatibility/)是很棒的功能，允许我们在 Docker 容器内运行几乎任何 CUDA 运行时库和 CUDA 应用程序。然而，这并不意味着我们应该假设使用 NVIDIA NGC CUDA Docker 容器总是能够工作，而不仔细检查宿主机上的 CUDA 驱动程序版本。我们应该尽量使用与 Docker 容器内 CUDA 运行时库版本相同的宿主机 CUDA 驱动程序版本。否则，我们可能会遇到一些难以解释的奇怪问题。

## 参考资料

- CUDA Compatibility(https://leimao.github.io/blog/CUDA-Compatibility/)
- NVIDIA NGC CUDA Docker containers(https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda)


------------------------------------------------------------------
------------------------------------------------------------------
------------------------------------------------------------------

# Docker 中的 Nsight Compute

## 简介

NVIDIA Nsight Compute 是 CUDA 的交互式分析器，通过用户界面和命令行工具提供详细的性能指标和 API 调试。用户可以运行引导分析并通过可定制的数据驱动用户界面比较结果，以及在自己的工作流程中后处理和分析结果。

在这篇博客文章中，我想讨论如何在 Docker 容器中安装和使用 Nsight Compute，以便我们可以在任何安装了 Docker 的地方使用它及其 GUI。

## Nsight Compute

### 构建 Docker 镜像

可以在 Docker 镜像中安装 Nsight Compute 并在任何地方使用它。构建 Nsight Compute 的 Dockerfile 如下所示。

```c++
FROM nvcr.io/nvidia/cuda:12.0.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        dbus \
        fontconfig \
        gnupg \
        libasound2 \
        libfreetype6 \
        libglib2.0-0 \
        libnss3 \
        libsqlite3-0 \
        libx11-xcb1 \
        libxcb-glx0 \
        libxcb-xkb1 \
        libxcomposite1 \
        libxcursor1 \
        libxdamage1 \
        libxi6 \
        libxml2 \
        libxrandr2 \
        libxrender1 \
        libxtst6 \
        libgl1-mesa-glx \
        libxkbfile-dev \
        openssh-client \
        wget \
        xcb \
        xkb-data && \
    apt-get clean

# QT6 is required for the Nsight Compute UI.
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        qt6-base-dev && \
    apt-get clean

```

要构建 Docker 镜像，请运行以下命令。

```shell
$ docker build -f nsight-compute.Dockerfile --no-cache --tag nsight-compute:12.0.1 .
```

### 上传 Docker 镜像

要上传 Docker 镜像，请运行以下命令。

```shell
$ docker tag nsight-compute:12.0.1 leimao/nsight-compute:12.0.1
$ docker push leimao/nsight-compute:12.0.1
```

### 拉取 Docker 镜像

要拉取 Docker 镜像，请运行以下命令。

```shell
$ docker pull leimao/nsight-compute:12.0.1
$ docker tag leimao/nsight-compute:12.0.1 nsight-compute:12.0.1
```

### 运行 Docker 容器

要运行 Docker 容器，请运行以下命令。

```shell
$ xhost +
$ docker run -it --rm --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --cap-add=SYS_ADMIN --security-opt seccomp=unconfined -v $(pwd):/mnt -w /mnt --network host nsight-compute:12.0.1
$ xhost -
```

### 运行 Nsight Compute

要运行带有 GUI 的 Nsight Compute，请运行以下命令。

```shell
$ ncu-ui
```

我们现在可以从 Docker 容器、通过 Docker 挂载从 Docker 本地主机机器以及从远程主机（如远程工作站或嵌入式设备）分析应用程序。

## 示例

### 非合并内存访问 VS 合并内存访问

在这个示例中，我们实现了一个在 GPU 上执行矩阵乘法的朴素 GEMM  kernel。这个 kernel是朴素的，因为它没有使用任何高级技术，如共享内存平铺。我们的原始目标是以合并的方式读写全局内存。然而，在 kernel的第一个版本 `gemm_non_coalesced` 中，我们创建了一个错误，交换了输出矩阵的行和列索引。因此， kernel以非合并的方式读写全局内存。在 kernel的第二个版本 `gemm_coalesced` 中，我们修复了错误，并以合并的方式读写全局内存。我们使用 Nsight Compute 分析了两个 kernel，并比较了两个 kernel之间的性能差异。

```c++
#include <iostream>

#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t err, const char* const func, const char* const file,
                const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() check_cuda_last(__FILE__, __LINE__)
void check_cuda_last(const char* const file, const int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// 非合并的全局内存读写访问
template <typename T>
__global__ void gemm_non_coalesced(size_t m, size_t n, size_t k, T alpha,
                                   T const* A, size_t lda, T const* B,
                                   size_t ldb, T beta, T* C, size_t ldc)
{
    // 计算当前线程负责的C矩阵的行和列索引
    // 注意：这里故意交换了行列索引，导致非合并访问
    size_t const C_row_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_col_idx{blockIdx.y * blockDim.y + threadIdx.y};

    // 每个线程计算
    // C[C_row_idx, C_col_idx] = alpha * A[C_row_idx, :] * B[:, C_col_idx] +
    // beta * C[C_row_idx, C_col_idx]
    if (C_row_idx < m && C_col_idx < n)
    {
        T sum{static_cast<T>(0)};
        // 计算点积：A的第C_row_idx行与B的第C_col_idx列
        for (size_t k_idx{0U}; k_idx < k; ++k_idx)
        {
            sum += A[C_row_idx * lda + k_idx] * B[k_idx * ldb + C_col_idx];
        }
        // 应用alpha和beta系数，更新C矩阵
        C[C_row_idx * ldc + C_col_idx] =
            alpha * sum + beta * C[C_row_idx * ldc + C_col_idx];
    }
}

// 启动非合并内存访问的GEMM kernel
template <typename T>
void launch_gemm_kernel_non_coalesced(size_t m, size_t n, size_t k,
                                      T const* alpha, T const* A, size_t lda,
                                      T const* B, size_t ldb, T const* beta,
                                      T* C, size_t ldc, cudaStream_t stream)
{
    // 定义线程块大小：32x8的线程块
    dim3 const block_dim{32U, 8U, 1U};
    // 计算网格大小，确保覆盖整个矩阵
    dim3 const grid_dim{
        (static_cast<unsigned int>(m) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(n) + block_dim.y - 1U) / block_dim.y, 1U};
    // 启动kernel
    gemm_non_coalesced<T><<<grid_dim, block_dim, 0U, stream>>>(
        m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}

// 合并的全局内存读写访问
template <typename T>
__global__ void gemm_coalesced(size_t m, size_t n, size_t k, T alpha,
                               T const* A, size_t lda, T const* B, size_t ldb,
                               T beta, T* C, size_t ldc)
{
    // 计算当前线程负责的C矩阵的行和列索引
    // 注意：这里正确地分配了行列索引，实现合并访问
    size_t const C_col_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_row_idx{blockIdx.y * blockDim.y + threadIdx.y};

    // 每个线程计算
    // C[C_row_idx, C_col_idx] = alpha * A[C_row_idx, :] * B[:, C_col_idx] +
    // beta * C[C_row_idx, C_col_idx]
    if (C_row_idx < m && C_col_idx < n)
    {
        T sum{static_cast<T>(0)};
        // 计算点积：A的第C_row_idx行与B的第C_col_idx列
        for (size_t k_idx{0U}; k_idx < k; ++k_idx)
        {
            sum += A[C_row_idx * lda + k_idx] * B[k_idx * ldb + C_col_idx];
        }
        // 应用alpha和beta系数，更新C矩阵
        C[C_row_idx * ldc + C_col_idx] =
            alpha * sum + beta * C[C_row_idx * ldc + C_col_idx];
    }
}

// 启动合并内存访问的GEMM kernel
template <typename T>
void launch_gemm_kernel_coalesced(size_t m, size_t n, size_t k, T const* alpha,
                                  T const* A, size_t lda, T const* B,
                                  size_t ldb, T const* beta, T* C, size_t ldc,
                                  cudaStream_t stream)
{
    // 定义线程块大小：32x8的线程块
    dim3 const block_dim{32U, 8U, 1U};
    // 计算网格大小，注意这里的维度分配与非合并版本不同
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(m) + block_dim.y - 1U) / block_dim.y, 1U};
    // 启动kernel
    gemm_coalesced<T><<<grid_dim, block_dim, 0U, stream>>>(
        m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}

// CPU版本的GEMM实现，用于验证GPU结果的正确性
template <typename T>
void gemm_cpu(size_t m, size_t n, size_t k, T alpha, T const* A, size_t lda,
              T const* B, size_t ldb, T beta, T* C, size_t ldc)
{
    // 三重循环实现矩阵乘法：C = alpha * A * B + beta * C
    for (size_t i{0U}; i < m; ++i)
    {
        for (size_t j{0U}; j < n; ++j)
        {
            T sum{static_cast<T>(0)};
            // 计算A的第i行与B的第j列的点积
            for (size_t k_idx{0U}; k_idx < k; ++k_idx)
            {
                sum += A[i * lda + k_idx] * B[k_idx * ldb + j];
            }
            // 应用alpha和beta系数
            C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
        }
    }
}

// 验证GPU计算结果与CPU参考结果的一致性
template <typename T>
void verify_outputs(size_t m, size_t n, size_t ldc, T const* C, T const* C_ref,
                    T abs_error_tol)
{
    // 逐元素比较GPU结果和CPU参考结果
    for (size_t i{0U}; i < m; ++i)
    {
        for (size_t j{0U}; j < n; ++j)
        {
            T const abs_error{std::abs(C[i * ldc + j] - C_ref[i * ldc + j])};
            // 如果误差超过容忍度，报告错误并退出
            if (abs_error > abs_error_tol)
            {
                std::cerr << "Error: i = " << i << ", j = " << j
                          << ", abs_error = " << abs_error << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
    }
}

int main()
{
    // 定义矩阵维度和GEMM参数
    size_t const m{1024U};  // A矩阵的行数，C矩阵的行数
    size_t const n{1024U};  // B矩阵的列数，C矩阵的列数
    size_t const k{1024U};  // A矩阵的列数，B矩阵的行数
    float const alpha{1.0f}; // 缩放因子alpha
    float const beta{0.0f};  // 缩放因子beta
    float const abs_error_tol{1e-5f}; // 误差容忍度

    // 定义矩阵的leading dimension（行优先存储）
    size_t const lda{k};  // A矩阵的leading dimension
    size_t const ldb{n};  // B矩阵的leading dimension
    size_t const ldc{n};  // C矩阵的leading dimension

    // 创建CUDA流
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // 在主机上分配内存
    float* A_host{nullptr};
    float* B_host{nullptr};
    float* C_host{nullptr};
    float* C_host_from_device{nullptr};
    CHECK_CUDA_ERROR(cudaMallocHost(&A_host, m * lda * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost(&B_host, k * ldb * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost(&C_host, m * ldc * sizeof(float)));
    CHECK_CUDA_ERROR(
        cudaMallocHost(&C_host_from_device, m * ldc * sizeof(float)));

    // 初始化矩阵A和B
    for (size_t i{0U}; i < m; ++i)
    {
        for (size_t j{0U}; j < k; ++j)
        {
            A_host[i * lda + j] = static_cast<float>(i + j);
        }
    }
    for (size_t i{0U}; i < k; ++i)
    {
        for (size_t j{0U}; j < n; ++j)
        {
            B_host[i * ldb + j] = static_cast<float>(i + j);
        }
    }

    // 在设备上分配内存
    float* A_device{nullptr};
    float* B_device{nullptr};
    float* C_device{nullptr};
    CHECK_CUDA_ERROR(cudaMalloc(&A_device, m * lda * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&B_device, k * ldb * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&C_device, m * ldc * sizeof(float)));

    // 将矩阵A和B从主机复制到设备
    CHECK_CUDA_ERROR(cudaMemcpy(A_device, A_host, m * lda * sizeof(float),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(B_device, B_host, k * ldb * sizeof(float),
                                cudaMemcpyHostToDevice));

    // 运行CPU版本，生成参考结果
    gemm_cpu(m, n, k, alpha, A_host, lda, B_host, ldb, beta, C_host, ldc);

    // 启动非合并内存访问的kernel
    launch_gemm_kernel_non_coalesced(m, n, k, &alpha, A_device, lda, B_device,
                                     ldb, &beta, C_device, ldc, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // 将结果从设备复制到主机
    CHECK_CUDA_ERROR(cudaMemcpy(C_host_from_device, C_device,
                                m * ldc * sizeof(float),
                                cudaMemcpyDeviceToHost));

    // 验证非合并版本的结果
    verify_outputs(m, n, ldc, C_host_from_device, C_host, abs_error_tol);

    // 启动合并内存访问的kernel
    launch_gemm_kernel_coalesced(m, n, k, &alpha, A_device, lda, B_device, ldb,
                                 &beta, C_device, ldc, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // 将结果从设备复制到主机
    CHECK_CUDA_ERROR(cudaMemcpy(C_host_from_device, C_device,
                                m * ldc * sizeof(float),
                                cudaMemcpyDeviceToHost));

    // 验证合并版本的结果
    verify_outputs(m, n, ldc, C_host_from_device, C_host, abs_error_tol);

    // 释放设备内存
    CHECK_CUDA_ERROR(cudaFree(A_device));
    CHECK_CUDA_ERROR(cudaFree(B_device));
    CHECK_CUDA_ERROR(cudaFree(C_device));
    // 释放主机内存
    CHECK_CUDA_ERROR(cudaFreeHost(A_host));
    CHECK_CUDA_ERROR(cudaFreeHost(B_host));
    CHECK_CUDA_ERROR(cudaFreeHost(C_host));
    CHECK_CUDA_ERROR(cudaFreeHost(C_host_from_device));
}
```

要使用 `nvcc` 构建示例并使用 Nsight Compute 分析示例，请运行以下命令。


```shell
$ nvcc gemm_naive.cu -o gemm_naive
$ ncu --set full -f -o gemm_naive gemm_naive
```

要使用 Nsight Compute GUI 查看分析结果，请运行以下命令。

```shell
$ ncu-ui
```

![Nsight Compute Naive GEMM Profile Details](https://files.mdnice.com/user/59/864f5055-572a-4552-90e0-5fa82532a052.jpg)

从 Nsight Compute 分析详情中，我们可以看到 kernel的第一个版本 `gemm_non_coalesced` 花费了 9.85 毫秒，而 kernel的第二个版本 `gemm_coalesced` 花费了 1.20 毫秒。尽管这两个 kernel都没有得到很好的优化，但 Nsight Compute 发现了两个 kernel的很多问题。具体来说， kernel `gemm_non_coalesced` 非常受内存限制，Nsight Compute 告诉我们"This kernel has non-coalesced global accesses resulting in a total of 941339104 excessive sectors (85% of the total 1109395408 sectors)"。例如，对于 kernel `gemm_non_coalesced`，L1/TEX 缓存统计显示全局加载每个请求需要 16.51 个扇区（为了计算一个乘积，warp 中的每个线程从不同的扇区加载，一个 warp 每个请求从矩阵 A 的缓冲区加载 32 × 1 个扇区，warp 中的每个线程从相同的扇区加载，一个 warp 每个请求从矩阵 B 的缓冲区加载 1 个扇区，平均 $\frac{32 \times 1 + 1}{2} = 16.5$），全局存储每个请求需要 32 个扇区（$\frac{32 \times 1}{1} = 32$），而对于修复了全局内存合并访问问题的 kernel `gemm_coalesced`，全局加载每个请求需要 2.5 个扇区（为了计算一个乘积，warp 中每 32/4 个线程从相同的扇区加载，一个 warp 每个请求从矩阵 A 的缓冲区加载 32 × 4/32 个扇区，warp 中的每个线程从相同的扇区加载，一个 warp 每个请求从矩阵 B 的缓冲区加载 1 个扇区，平均 $\frac{32 \times 4/32 + 1}{2} = 2.5$），全局存储每个请求需要 4 个扇区（$\frac{32 \times 4/32}{1} = 4$）。一个扇区的大小是 32 字节。

> 扇区（Sector）是GPU全局内存访问的基本单位，每个扇区的大小是32字节。当GPU从全局内存中读取或写入数据时，即使只需要4字节的数据，GPU也必须读取/写入整个32字节的扇区。

## GitHub

所有的 Dockerfile 和示例都可以在 GitHub(https://github.com/leimao/Nsight-Compute-Docker-Image) 上找到。

## 参考资料


- Nsight Compute Docker Image(https://github.com/leimao/Nsight-Compute-Docker-Image)


