> 本文由 @Simon V(https://github.com/simveit) 授权转载和翻译并发表到本公众号。原始地址为：https://veitner.bearblog.dev/tma-introduction/

# TMA简介

2025年4月27日

高效地将数据从全局内存传输到共享内存以及反之亦然通常是CUDA应用程序的瓶颈。因此，我们应充分利用这种操作的最快机制。TMA是Hopper GPU中的一个新特性，其主要目标是提供一种高效的从全局内存到共享内存的数据传输机制，用于多维数组。在本篇博客中，我们将解释如何使用TMA对2D数组进行操作。

为了专注于基础知识，我们将执行一个非常简单的操作：给定一个矩阵`M`，我们希望对矩阵中的每个元素加`1`。

## 创建张量映射

为了利用2D TMA操作，我们需要一个`TensorMap`。下面我们展示了如何创建它：

```c++
int *d;
CHECK_CUDA_ERROR(cudaMalloc(&d, SIZE));
CHECK_CUDA_ERROR(cudaMemcpy(d, h_in, SIZE, cudaMemcpyHostToDevice));
void *tensor_ptr = (void *)d;

CUtensorMap tensor_map{};
// rank是数组的维度数。
constexpr uint32_t rank = 2;
uint64_t size[rank] = {GMEM_WIDTH, GMEM_HEIGHT};
// stride是从一行的第一个元素到下一行第一个元素所需遍历的字节数。它必须是16的倍数。
uint64_t stride[rank - 1] = {GMEM_WIDTH * sizeof(int)};
// box_size是用作TMA传输目标的共享内存缓冲区的大小。
uint32_t box_size[rank] = {SMEM_WIDTH, SMEM_HEIGHT};
// 元素之间的距离，以sizeof(element)为单位。例如，stride为2可以用来只加载复值张量的实部。
uint32_t elem_stride[rank] = {1, 1};

// 创建张量描述符。
CUresult res = cuTensorMapEncodeTiled(
  &tensor_map,  // CUtensorMap *tensorMap,
  CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
  rank,         // cuuint32_t tensorRank,
  tensor_ptr,   // void *globalAddress,
  size,         // const cuuint64_t *globalDim,
  stride,       // const cuuint64_t *globalStrides,
  box_size,     // const cuuint32_t *boxDim,
  elem_stride,  // const cuuint32_t *elementStrides,
  // 交错模式可用于加速加载小于4字节长的值。
  CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
  // 交织可用于避免共享内存bank冲突。
  CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
  // L2提升可用于将缓存策略的效果扩展到更广泛的L2缓存行集合。
  CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
  // TMA传输会将任何超出边界的元素设置为零。
  CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

assert(res == CUDA_SUCCESS);
```

注释取自NVIDIA文档(https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-data-copies-using-the-tensor-memory-accelerator-tma)。如代码所示，我们基本上为GMEM和SMEM设置配置。我们考虑矩阵`d`为`行主序`布局，并且不在此阶段执行任何swizzling。

## Kernel

下面的kernel改编自NVIDIA文档。我们可以看到TMA的基本用法。

```c++
template <int BLOCK_SIZE>
__global__ void kernel(const __grid_constant__ CUtensorMap tensor_map) {
  // 批量张量操作的目标共享内存缓冲区应该是128字节对齐的。
  __shared__ alignas(1024) int smem_buffer[BLOCK_SIZE * BLOCK_SIZE];

  // GMEM中左上角Tile的坐标。
  int x = blockIdx.x * BLOCK_SIZE;
  int y = blockIdx.y * BLOCK_SIZE;

  int col = threadIdx.x % BLOCK_SIZE;
  int row = threadIdx.x / BLOCK_SIZE;

// 使用参与屏障的线程数量初始化共享内存屏障。
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;

  if (threadIdx.x == 0) {
    // 初始化屏障。块中所有`blockDim.x`线程都参与。
    init(&bar, blockDim.x);
    // 使初始化的屏障在异步代理中可见。
    cde::fence_proxy_async_shared_cta();
  }
  // 同步线程，使初始化的屏障对所有线程可见。
  __syncthreads();

  barrier::arrival_token token;
  if (threadIdx.x == 0) {
    // 启动批量张量复制。
    cde::cp_async_bulk_tensor_2d_global_to_shared(&smem_buffer, &tensor_map, x,
                                                  y, bar);
    // 到达屏障并告知预期传入的字节数。
    token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
  } else {
    // 其他线程只需到达。
    token = bar.arrive();
  }
  // 等待数据到达。
  bar.wait(std::move(token));

  // 在共享内存中符号性地修改一个值。
  smem_buffer[row * BLOCK_SIZE + col] += 1;

  // 等待共享内存写入对TMA引擎可见。
  cde::fence_proxy_async_shared_cta();
  __syncthreads();
  // 同步线程后，所有线程的写入对TMA引擎可见。

  // 启动TMA传输，将共享内存复制到全局内存
  if (threadIdx.x == 0) {
    cde::cp_async_bulk_tensor_2d_shared_to_global(&tensor_map, x, y,
                                                  &smem_buffer);
    // 等待TMA传输完成读取共享内存。
    // 从上一个批量复制操作创建一个"批量异步组"。
    cde::cp_async_bulk_commit_group();
    // 等待组完成从共享内存的读取。
    cde::cp_async_bulk_wait_group_read<0>();
  }

  // 销毁屏障。这会使屏障的内存区域无效。如果kernel中还有进一步的计算，
  // 这允许重用共享内存屏障的内存位置。
  if (threadIdx.x == 0) {
    (&bar)->~barrier();
  }
}
```

在

```c++
cde::cp_async_bulk_tensor_2d_global_to_shared(&smem_buffer, &tensor_map, x,
                                                  y, bar);
```

我们将我们的矩阵Tile复制到共享内存中。这里的x和y对应于矩阵中左上角Tile的坐标。然后我们在共享内存中执行一个简单的操作（为每个元素加1）并将其传回全局内存。


## Swizzling

在共享内存中我们可能会遇到bank冲突。关于这个主题的更多背景知识，你可以阅读NVIDIA文档(https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor)中的专门章节。

我们可以在张量映射中定义交织模式来避免bank冲突。TMA将以交织方式传输数组，因此我们需要使用交织后的索引来正确地索引共享内存中的列。kernel的调整如下：

```c++
template <int BLOCK_SIZE>
__global__ void kernel(const __grid_constant__ CUtensorMap tensor_map) {
  // 批量张量操作的目标共享内存缓冲区应该是128字节对齐的。
  __shared__ alignas(256) int smem_buffer[BLOCK_SIZE * BLOCK_SIZE];

  // GMEM中左上角Tile的坐标。
  int x = blockIdx.x * BLOCK_SIZE;
  int y = blockIdx.y * BLOCK_SIZE;

  int col = threadIdx.x % BLOCK_SIZE;
  int row = threadIdx.x / BLOCK_SIZE;
  int col_swizzle = (row % 2) ^ col;
// 使用参与屏障的线程数量初始化共享内存屏障。
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;

  if (threadIdx.x == 0) {
    // 初始化屏障。块中所有`blockDim.x`线程都参与。
    init(&bar, blockDim.x);
    // 使初始化的屏障在异步代理中可见。
    cde::fence_proxy_async_shared_cta();
  }
  // 同步线程，使初始化的屏障对所有线程可见。
  __syncthreads();

  barrier::arrival_token token;
  if (threadIdx.x == 0) {
    // 启动批量张量复制。
    cde::cp_async_bulk_tensor_2d_global_to_shared(&smem_buffer, &tensor_map, x,
                                                  y, bar);
    // 到达屏障并告知预期传入的字节数。
    token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
  } else {
    // 其他线程只需到达。
    token = bar.arrive();
  }
  // 等待数据到达。
  bar.wait(std::move(token));

  // 在共享内存中符号性地修改一个值。
  smem_buffer[row * BLOCK_SIZE + col_swizzle] += 1;

  // 等待共享内存写入对TMA引擎可见。
  cde::fence_proxy_async_shared_cta();
  __syncthreads();
  // 同步线程后，所有线程的写入对TMA引擎可见。

  // 启动TMA传输，将共享内存复制到全局内存
  if (threadIdx.x == 0) {
    cde::cp_async_bulk_tensor_2d_shared_to_global(&tensor_map, x, y,
                                                  &smem_buffer);
    // 等待TMA传输完成读取共享内存。
    // 从上一个批量复制操作创建一个"批量异步组"。
    cde::cp_async_bulk_commit_group();
    // 等待组完成从共享内存的读取。
    cde::cp_async_bulk_wait_group_read<0>();
  }

  // 销毁屏障。这会使屏障的内存区域无效。如果kernel中还有进一步的计算，
  // 这允许重用共享内存屏障的内存位置。
  if (threadIdx.x == 0) {
    (&bar)->~barrier();
  }
}
```

并且在张量映射中我们需要调整交织设置

```c++
int *d;
  CHECK_CUDA_ERROR(cudaMalloc(&d, SIZE));
  CHECK_CUDA_ERROR(cudaMemcpy(d, h_in, SIZE, cudaMemcpyHostToDevice));
  void *tensor_ptr = (void *)d;

  CUtensorMap tensor_map{};
  // rank是数组的维度数。
  constexpr uint32_t rank = 2;
  uint64_t size[rank] = {GMEM_WIDTH, GMEM_HEIGHT};
  // stride是从一行的第一个元素到下一行第一个元素所需遍历的字节数。它必须是16的倍数。
  uint64_t stride[rank - 1] = {GMEM_WIDTH * sizeof(int)};
  // box_size是用作TMA传输目标的共享内存缓冲区的大小。
  uint32_t box_size[rank] = {SMEM_WIDTH, SMEM_HEIGHT};
  // 元素之间的距离，以sizeof(element)为单位。例如，stride为2可以用来只加载复值张量的实部。
  uint32_t elem_stride[rank] = {1, 1};

  // 创建张量描述符。
  CUresult res = cuTensorMapEncodeTiled(
      &tensor_map,  // CUtensorMap *tensorMap,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
      rank,         // cuuint32_t tensorRank,
      tensor_ptr,   // void *globalAddress,
      size,         // const cuuint64_t *globalDim,
      stride,       // const cuuint64_t *globalStrides,
      box_size,     // const cuuint32_t *boxDim,
      elem_stride,  // const cuuint32_t *elementStrides,
      // 交错模式可用于加速加载小于4字节长的值。
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      // 交织可用于避免共享内存bank冲突。
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B,
      // L2提升可用于将缓存策略的效果扩展到更广泛的L2缓存行集合。
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      // TMA传输会将任何超出边界的元素设置为零。
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  assert(res == CUDA_SUCCESS);
```

## 结论

以上我们展示了TMA在Hopper架构中的一个非常简单的应用。此外，交织相当复杂，为了更好地理解它，进一步实验并深入研究这个主题会很有帮助。如果有人有好的学习资源建议，我很乐意听取。你可以在Github(https://github.com/simveit/tma_intro)上找到代码。








