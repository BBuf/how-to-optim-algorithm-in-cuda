> 本文由 @Simon V(https://github.com/simveit) 授权转载和翻译并发表到本公众号。

# 让向量求和变得非常快

06 Apr, 2025

在这篇博客文章中，我们将简要描述如何为向量归约任务实现最先进的性能，即我们的程序应该执行以下操作：给定一个向量v，返回v中所有元素的和。我们将假设向量很大，即它包含`N = 1 << 30 = 2^30`个元素。

## 基准实现

```c++
template <unsigned int threadsPerBlock>
__global__ void kernel_0(const int *d_in, int *d_out, size_t N) {
  extern __shared__ int sums[threadsPerBlock];
  int sum = 0;
  const int tid = threadIdx.x;
  const int global_tid = blockIdx.x * threadsPerBlock + tid;
  const int threads_in_grid = threadsPerBlock * gridDim.x;

  for (int i = global_tid; i < N; i += threads_in_grid) {
    sum += d_in[i];
  }
  sums[tid] = sum;
  __syncthreads();

  for (int activeThreads = threadsPerBlock >> 1; activeThreads;
       activeThreads >>= 1) {
    if (tid < activeThreads) {
      sums[tid] += sums[tid + activeThreads];
    }
    __syncthreads();
  }

  if (tid == 0) {
    d_out[blockIdx.x] = sums[tid];
  }
}

template <int threadsPerBlock>
void kernel_0_launch(const int *d_in, int *d_first, int *d_out, size_t N) {
  const int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  kernel_0<threadsPerBlock><<<numBlocks, threadsPerBlock>>>(
      d_in, d_first, N);
  kernel_0<threadsPerBlock><<<1, threadsPerBlock>>>(
      d_first, d_out, numBlocks);
}
```

我们的基准实现是一个简单的 Two Pass 方法。我们启动`kernel_0`两次。第一次我们将每个块中的元素归约为和，然后再将这些和相加。算法的工作原理如下：

1. 我们将当前线程对应的向量元素存储在共享内存中。
2. 然后我们将线程块分成两半。我们将左半部分第一个线程的结果与右半部分线程的结果累加，以此类推。
3. 我们同步，即等待所有线程完成操作，即左半部分包含每个线程的累加结果。
4. 我们忽略上面的右半部分，继续上述过程，直到到达左侧的第一个线程。这样我们就得到了每个块中的和。然后5. 我们将这视为另一个向量，并在单个块中归约这个向量以获得总和。

这种方法在H100 GPU上实现了`639.103 GB/s`的带宽，相当于可能带宽的`19.3668%`。

## 使用warp

在GPU中，一个warp由32个线程组成。我们可以单独处理每个块中的第一个warp，并使用更高效的`__syncwarp();`来强制同步。请注意，在博客文章的初始版本中，我假设warp中的所有线程都以同步方式执行。事实证明，对于较新的架构来说这是错误的，虽然大多数情况下我们会得到正确的结果，但它可能导致竞态条件，这可以通过在编译的 kernel上运行`compute-sanitizer --tool racecheck`来发现。幸运的是，使用`__syncwarp();`只会在带宽上损失约`1GB/s`。感谢Pauleonix(https://github.com/pauleonix)指出这一点！

```c++
template <unsigned int threadsPerBlock>
__global__ void kernel_1(const int *d_in, int *d_out, size_t N) {
  extern __shared__ int sums[threadsPerBlock];
  int sum = 0;
  const int tid = threadIdx.x;
  const int global_tid = blockIdx.x * threadsPerBlock + tid;
  const int threads_in_grid = threadsPerBlock * gridDim.x;

  for (int i = global_tid; i < N; i += threads_in_grid) {
    sum += d_in[i];
  }
  sums[tid] = sum;
  __syncthreads();

#pragma unroll
  for (int activeThreads = threadsPerBlock >> 1; activeThreads > 32;
       activeThreads >>= 1) {
    if (tid < activeThreads) {
      sums[tid] += sums[tid + activeThreads];
    }
    __syncthreads();
  }

  volatile int *volatile_sums = sums;
#pragma unroll
  for (int activeThreads = 32; activeThreads; activeThreads >>= 1) {
    if (tid < activeThreads) {
      volatile_sums[tid] += volatile_sums[tid + activeThreads];
    }
    __syncwarp();
  }

  if (tid == 0) {
    d_out[blockIdx.x] = volatile_sums[tid];
  }
}

template <int threadsPerBlock>
void kernel_1_launch(const int *d_in, int *d_first, int *d_out, size_t N) {
  const int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  kernel_1<threadsPerBlock><<<numBlocks, threadsPerBlock>>>(d_in, d_first, N);
  kernel_1<threadsPerBlock><<<1, threadsPerBlock>>>(d_first, d_out, numBlocks);
}
```

这使我们的性能略微提升到661.203 GB/s，相当于20.0365%的利用率。

## One Pass实现

我们可以使用`atomicAdd`在CUDA中跨块将结果添加到内存位置。我们可以用它来实现一个简单的One Pass Kernel。

```c++
template <unsigned int threadsPerBlock>
__global__ void kernel_2(const int *d_in, int *d_out, size_t N) {
  extern __shared__ int sums[threadsPerBlock];
  int sum = 0;
  const int tid = threadIdx.x;
  const int global_tid = blockIdx.x * threadsPerBlock + tid;

  if (global_tid == 0) {
    *d_out = 0;
  }

  if (global_tid < N) {
    sum += d_in[global_tid];
  }
  sums[tid] = sum;
  __syncthreads();

#pragma unroll
  for (int activeThreads = threadsPerBlock >> 1; activeThreads > 32;
       activeThreads >>= 1) {
    if (tid < activeThreads) {
      sums[tid] += sums[tid + activeThreads];
    }
    __syncthreads();
  }

  volatile int *volatile_sums = sums;
#pragma unroll
  for (int activeThreads = 32; activeThreads; activeThreads >>= 1) {
    if (tid < activeThreads) {
      volatile_sums[tid] += volatile_sums[tid + activeThreads];
    }
    __syncwarp();
  }

  if (tid == 0) {
    atomicAdd(d_out, volatile_sums[tid]);
  }
}

template <int threadsPerBlock>
void kernel_2_launch(const int *d_in, int *d_out, size_t N) {
  const int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  kernel_2<threadsPerBlock><<<numBlocks, threadsPerBlock>>>(d_in, d_out, N);
}
```

这使我们的性能提升到`859.534 GB/s`，相当于`26.0465%`的利用率。

## 增加算术强度

如果你看一下上面的 kernel，很明显每个线程只是访问向量中的相应元素并将其写入共享内存。我们可以通过让每个线程处理一批元素来做得更好。

```c++
template <unsigned int threadsPerBlock, unsigned int batchSize>
__global__ void kernel_3(const int *d_in, int *d_out, size_t N) {
  extern __shared__ int sums[threadsPerBlock];
  int sum = 0;
  const int tid = threadIdx.x;  
  const int global_tid = blockIdx.x * threadsPerBlock + tid;
  const int threads_in_grid = threadsPerBlock * gridDim.x;

  if (global_tid == 0) {
    *d_out = 0;
  }

  if (global_tid < N) {
#pragma unroll
    for (int j = 0; j < batchSize; j++) {
      if (global_tid * batchSize + j < N) {
        sum += d_in[global_tid * batchSize + j];
      }
    }
  }
  sums[tid] = sum;
  __syncthreads();

#pragma unroll
  for (int activeThreads = threadsPerBlock >> 1; activeThreads > 32;
       activeThreads >>= 1) {
    if (tid < activeThreads) {
      sums[tid] += sums[tid + activeThreads];
    }
    __syncthreads();
  }

  volatile int *volatile_sums = sums;
#pragma unroll
  for (int activeThreads = 32; activeThreads; activeThreads >>= 1) {
    if (tid < activeThreads) {
      volatile_sums[tid] += volatile_sums[tid + activeThreads];
    }
    __syncwarp();
  }

  if (tid == 0) {
    atomicAdd(d_out, volatile_sums[tid]);
  }
}

template <int threadsPerBlock, int batchSize>
void kernel_3_launch(const int *d_in, int *d_out, size_t N) {
  const int numBlocks = (N + threadsPerBlock * batchSize - 1) /
                        (threadsPerBlock * batchSize);
  kernel_3<threadsPerBlock, batchSize><<<numBlocks, threadsPerBlock>>>(d_in,
                                                                       d_out, N);
}
```

如我们所见，我们现在启动的块更少了，这是因为每个线程现在处理`Batchsize`个元素。这增加了每个批次的工作量，并大幅提升了性能！使用这种方法，我们可以获得`3228.5 GB/s`的带宽，非常接近物理最大值，利用率为`97.8334%`。

## 向量化加载

CUDA提供了用户向量化数据类型`int4`。我们可以用它来更高效地加载数据。

```c++
template <unsigned int threadsPerBlock, unsigned int batchSize>
__global__ void kernel_4(const int4 *d_in, int *d_out, size_t N) {
  extern __shared__ int sums[threadsPerBlock];
  int sum = 0;
  const int tid = threadIdx.x;  
  const int global_tid = blockIdx.x * threadsPerBlock + tid;
  const int threads_in_grid = threadsPerBlock * gridDim.x;

  if (global_tid == 0) {
    *d_out = 0;
  }

  if (global_tid < N) {
#pragma unroll
    for (int i = 0; i < batchSize >> 2; i++) {
      const int4 val = d_in[global_tid * (batchSize >> 2) + i];
      if (global_tid * batchSize + i * 4 < N) {
        sum += val.x + val.y + val.z + val.w;
      }
    }
  }
  sums[tid] = sum;
  __syncthreads();

#pragma unroll
  for (int activeThreads = threadsPerBlock >> 1; activeThreads > 32;
       activeThreads >>= 1) {
    if (tid < activeThreads) {
      sums[tid] += sums[tid + activeThreads];
    }
    __syncthreads();
  }

  volatile int *volatile_sums = sums;
#pragma unroll
  for (int activeThreads = 32; activeThreads; activeThreads >>= 1) {
    if (tid < activeThreads) {
      volatile_sums[tid] += volatile_sums[tid + activeThreads];
    }
    __syncwarp();
  }

  if (tid == 0) {
    atomicAdd(d_out, volatile_sums[tid]);
  }
}

template <int threadsPerBlock, int batchSize>
void kernel_4_launch(const int *d_in, int *d_out, size_t N) {
  const int numBlocks = (N + threadsPerBlock * batchSize - 1) /
                        (threadsPerBlock * batchSize);
  const int4 *d_in_cast = reinterpret_cast<const int4 *>(d_in);
  kernel_4<threadsPerBlock, batchSize><<<numBlocks, threadsPerBlock>>>(d_in_cast,
                                                                       d_out, N);
}
```

这比上面的版本有微小的改进，达到`3231.9 GB/s`，相当于`97.9364%`的利用率。

## 基准测试NVIDIA库

我们可以按如下方式对上述操作的NVIDIA原生实现进行基准测试：

```c++
void kernel_5_launch(const int *d_in, int *d_out, size_t N) {
  void* d_temp = nullptr;
  size_t temp_storage = 0;

  // First call to determine temporary storage size
  cub::DeviceReduce::Sum(d_temp, temp_storage, d_in, d_out, N);
  
  // Allocate temporary storage
  assert(temp_storage > 0);
  cudaMalloc(&d_temp, temp_storage);

  cub::DeviceReduce::Sum(d_temp, temp_storage, d_in, d_out, N);
}
```

这给我们带来了`3191.42 GB/s`的带宽和`96.7097%`的利用率。这意味着使用我们的方法，我们在所选问题大小（N = 1 << 30）和硬件（H100）上超越了NVIDIA的实现。

## 参考文献

这篇博客文章受到了CUDA手册(https://www.cudahandbook.com/)中关于归约的讨论的启发。批处理的思路来自fast.cu (https://github.com/pranjalssh/fast.cu/blob/main/sum.cu)仓库以及用于基准测试cub库的代码。那里采用的一些方法应该能够进一步提高我们 kernel的性能，但我选择在一个对初学者来说仍然容易理解的地方停止。我强烈建议查看这个仓库及其作者关于编写高性能CUDA kernel的宝贵见解的博客文章。

你可以在这个仓库(https://github.com/simveit/effective_reduction)中重现实验并找到我的代码。我在H100和`CUDA 12.8`的docker镜像上运行了实验。





















