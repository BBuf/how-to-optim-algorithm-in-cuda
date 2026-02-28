> 本文由 @Simon V(https://github.com/simveit) 授权转载和翻译并发表到本公众号。原始地址为：https://veitner.bearblog.dev/making-prefix-sum-really-fast/

# 让前缀和变得更快

2025年4月13日

在这篇博文中,我们将展示如何优化分块前缀和操作。分块前缀和的工作原理如下:

给定一个向量 `v`,我们将该向量分成若干块。在每个块内部执行前缀和运算。一个简单的例子,假设每个块包含4个元素,输入向量为 `v = [0, 1, 2, 3, 4, 5, 6, 7]`,那么输出向量将是 `o = [0, 1, 3, 6, 4, 9, 15, 22]`

这是GPU中前缀和操作的一个基本构建块。

## 算法

请参考幻灯片21(https://safari.ethz.ch/projects_and_seminars/fall2022/lib/exe/fetch.php?media=p_s-hetsys-fs2022-meeting10-aftermeeting.pdf)来理解该算法。这张图片清晰地描述了我们要执行的算法。该算法包含多个阶段。在每个阶段,我们将两个相加元素之间的步长增加两倍。在最后一个阶段结束后,我们得到累积和向量作为结果。

## 朴素基线实现

```c++
template <int threadsPerBlock, int numElements>
__global__ void kernel_0(int *input, int *output) {
  const int tid = threadIdx.x;
  const int gtid = blockIdx.x * threadsPerBlock + tid;

  output[gtid] = input[gtid];
  __syncthreads();

#pragma unroll
  for (unsigned int offset = 1; offset <= threadsPerBlock / 2; offset <<= 1) {
    int tmp;
    if (tid >= offset) {
      tmp = output[gtid - offset];
    }
    __syncthreads();

    if (tid >= offset && gtid < numElements) {
      output[gtid] += tmp;
    }
    __syncthreads();
  }
}

template <int threadsPerBlock, int numElements>
void launch_kernel_0(int *input, int *output) {
  const int numBlocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  kernel_0<threadsPerBlock, numElements>
      <<<numBlocks, threadsPerBlock>>>(input, output);
}
```

该算法实现了上图所示的功能。在每个阶段,我们将偏移量增加两倍。我们会一直这样做,直到到达线程块的中间位置,并且累积元素之间的距离达到线程块大小的一半。注意,我们需要使用 `__syncthreads` 来避免竞争条件。如果没有这两个同步屏障,可能会出现两个数组元素同时被读写的情况。

使用一个简单的CPU实现来检查程序的正确性是一个很好的做法。

```c++
emplate <int threadsPerBlock, int numElements>
void cpu_scan(int *input, int *output) {
  output[0] = input[0];
  for (int i = 1; i < numElements; i++) {
    if (!((i & (threadsPerBlock - 1)) == 0)) {
      output[i] = input[i] + output[i - 1];
    } else {
      output[i] = input[i];
    }
  }
}
```

该算法给出了正确的结果。不幸的是,它的性能并不理想。这是由于频繁访问全局内存。我们可以通过计算带宽来衡量性能。我们进行了N次读写操作,其中`N = 1 << 30 = 2**30`。上述kernel的测量性能如下:

```c++
Bandwidth: 823.944 GB/s
Efficiency: 0.24968
```

## 使用共享内存

```c++
template <int threadsPerBlock, int numElements>
__global__ void kernel_1(int *input, int *output) {
  extern __shared__ int buffer[threadsPerBlock];

  const int tid = threadIdx.x;
  const int gtid = blockIdx.x * threadsPerBlock + tid;

  buffer[tid] = input[gtid];
  __syncthreads();

#pragma unroll
  for (unsigned int offset = 1; offset <= threadsPerBlock / 2; offset <<= 1) {
    int tmp;
    if (tid >= offset) {
      tmp = buffer[tid - offset];
    }
    __syncthreads();

    if (tid >= offset && gtid < numElements) {
      buffer[tid] += tmp;
    }
    __syncthreads();
  }

  if (gtid < numElements) {
    output[gtid] = buffer[tid];
  }
}

template <int threadsPerBlock, int numElements>
void launch_kernel_1(int *input, int *output) {
  const int numBlocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  kernel_1<threadsPerBlock, numElements>
      <<<numBlocks, threadsPerBlock>>>(input, output);
}
```

这个kernel与上面的kernel非常相似。主要区别在于这里我们使用共享内存。如果我们要频繁访问元素,共享内存比全局内存便宜得多。性能如下:

```c++
Bandwidth: 1288.72 GB/s
Efficiency: 0.390522
```

## 使用双缓冲区

```c++
template <int threadsPerBlock, int numElements>
__global__ void kernel_2(int *input, int *output) {
  __shared__ int _buffer_one[threadsPerBlock];
  __shared__ int _buffer_two[threadsPerBlock];

  const int tid = threadIdx.x;
  const int gtid = blockIdx.x * threadsPerBlock + tid;

  int *buffer_one = _buffer_one;
  int *buffer_two = _buffer_two;

  buffer_one[tid] = input[gtid];
  __syncthreads();

#pragma unroll
  for (unsigned int offset = 1; offset <= threadsPerBlock / 2; offset <<= 1) {
    if (tid >= offset) {
      buffer_two[tid] = buffer_one[tid] + buffer_one[tid - offset];
    } else {
      buffer_two[tid] = buffer_one[tid];
    }
    __syncthreads();

    int *tmp = buffer_one;
    buffer_one = buffer_two;
    buffer_two = tmp;
  }

  if (gtid < numElements) {
    output[gtid] = buffer_one[tid];
  }
}

template <int threadsPerBlock, int numElements>
void launch_kernel_2(int *input, int *output) {
  const int numBlocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  kernel_2<threadsPerBlock, numElements>
      <<<numBlocks, threadsPerBlock>>>(input, output);
}
```

这个kernel使用双缓冲区。我们在共享内存中初始化两个数组。我们在每个阶段交换缓冲区。这个方法的优点是我们可以节省一个同步屏障。这是因为我们现在有两个数组可以访问,因此可以确保不会出现竞争条件。这个kernel的性能如下:

```c++
Bandwidth: 1616.71 GB/s
Efficiency: 0.489913
```

## 使用warp原语

CUDA提供了warp原语。其中一个warp原语是`__shfl_up_sync`，它非常适合我们的操作，因为它精确地执行了上面图片中的操作。你可以在这篇博客文章中了解更多关于它的信息(https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)。我们可以使用它来进一步加速kernel的性能:

```c++
#define WARP_SIZE 32
#define LOG_WARP_SIZE 5
#define WARP_MASK (WARP_SIZE - 1)
__device__ inline int lane_id(void) { return threadIdx.x & WARP_MASK; }
__device__ inline int warp_id(void) { return threadIdx.x >> LOG_WARP_SIZE; }
// Warp scan
__device__ __forceinline__ int warp_scan(int val) {
  int x = val;
#pragma unroll
  for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
    int y = __shfl_up_sync(0xffffffff, x, offset);
    if (lane_id() >= offset) x += y;
  }
  return x - val;
}

template <int threadsPerBlock>
__device__ int block_scan(int in) {
  __shared__ int sdata[threadsPerBlock >> LOG_WARP_SIZE];
  // A. Exclusive scan within each warp
  int warpPrefix = warp_scan(in);
  // B. Store in shared memory
  if (lane_id() == WARP_SIZE - 1) sdata[warp_id()] = warpPrefix + in;
  __syncthreads();
  // C. One warp scans in shared memory
  if (threadIdx.x < WARP_SIZE)
    sdata[threadIdx.x] = warp_scan(sdata[threadIdx.x]);
  __syncthreads();
  // D. Each thread calculates its final value
  int thread_out_element = warpPrefix + sdata[warp_id()];
  return thread_out_element;
}

template <int threadsPerBlock, int numElements>
__global__ void kernel_3(int *input, int *output) {
  int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  int val = input[gtid];
  int result = block_scan<threadsPerBlock>(val);
  if (gtid < numElements) {
    output[gtid] = result + val;
  }
}

template <int threadsPerBlock, int numElements>
void launch_kernel_3(int *input, int *output) {
  const int numBlocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  kernel_3<threadsPerBlock, numElements>
      <<<numBlocks, threadsPerBlock>>>(input, output);
}
```

这个kernel的性能进一步提高,因为我们使用warp原语非常高效地执行了warp级别的 reductions。有关详细解释,请参阅以下视频(https://www.youtube.com/watch?v=SG0gvcbf2eo)。我们的新性能是:

```c++
Bandwidth: 1976.42 GB/s
Efficiency: 0.598916
```

## 增加每个线程的工作量

上述kernel(或至少它们的变体)是众所周知的,你可以在网上找到许多关于它们的解释。最后一个kernel没有很好地记录(我在网上没有找到任何关于这个简单技术的参考资料)但它是接近峰值性能的关键。

注意:这种技术被称为线程粗化(thread coarsening),这是GPU mode discord服务器上的ngc92指出的。你可以在PPMP书籍中了解更多相关内容(https://www.sciencedirect.com/science/article/abs/pii/B9780323912310000227)。

```c++
template <int threadsPerBlock, int numElements, int batchSize>
__global__ void kernel_4(int *input, int *output) {
  int reductions[batchSize];
  int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  int total_sum = 0;
#pragma unroll
  for (int i = 0; i < batchSize; i++) {
    const int idx = gtid * batchSize + i;
    if (idx < numElements) {
      total_sum += input[idx];
      reductions[i] = total_sum;
    }
  }
  int reduced_total_sum = block_scan<threadsPerBlock>(total_sum);
#pragma unroll
  for (int i = 0; i < batchSize; i++) {
    const int idx = gtid * batchSize + i;
    if (idx < numElements) {
      output[idx] = reduced_total_sum + reductions[i];
    }
  }
}

template <int threadsPerBlock, int numElements, int batchSize>
void launch_kernel_4(int *input, int *output) {
  const int numBlocks = (numElements + threadsPerBlock * batchSize - 1) /
                        (threadsPerBlock * batchSize);
  kernel_4<threadsPerBlock, numElements, batchSize>
      <<<numBlocks, threadsPerBlock>>>(input, output);
}
```

`block_scan`在这里与之前相同。不同的是,我们现在用每个线程处理多个元素。

调整验证函数以使用不同的batchSize:

```c++
template <int threadsPerBlock, int numElements, int batchSize>
void cpu_scan(int *input, int *output) {
  output[0] = input[0];
  for (int i = 1; i < numElements; i++) {
    if (!((i % (threadsPerBlock * batchSize)) == 0)) {
      output[i] = input[i] + output[i - 1];
    } else {
      output[i] = input[i];
    }
  }
}
```

显示我们的结果仍然是正确的(注意,如果batchSize是2^n的形式,我们可以使用上面的位操作来执行取模运算)。

我们通过首先对属于当前线程的元素执行简单的顺序扫描来实现这一点。然后我们将块对这些总和进行扫描。之后,我们通过将reduced的总和添加到reduce的部分来写入输出。这个过程与上面的warp扫描层次结构以及我们为完整前缀和执行的操作类似。我再次参考上面的讲座以获得更详细的解释。最终kernel的性能如下:

```shell
Bandwidth: 3056.53 GB/s
Efficiency: 0.926221
```

我们可以通过调整块和batchsize来进一步挤压GPU的性能,但我在这里停止,以保持博客文章简洁。例如,我们也可以使用`int4`来在加载批量数据时使用更少的指令,尽管在我的实验中,这并没有对性能产生巨大影响。如果你有进一步的技巧来提高性能,请告诉我!我希望你喜欢阅读这篇博客文章。上面提到的讲座(https://www.youtube.com/watch?v=SG0gvcbf2eo)非常有帮助,让我更好地理解了前缀和。你可以在Linkedin(https://www.linkedin.com/in/simon-veitner-174a681b6/)上联系我,进一步讨论CUDA。我很想听听你的看法!所有代码都可以在我的github repo(https://github.com/simveit/effective_scan)上找到。



