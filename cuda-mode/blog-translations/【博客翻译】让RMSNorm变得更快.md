> 本文由 @Simon V(https://github.com/simveit) 授权转载和翻译并发表到本公众号。原始地址为：https://veitner.bearblog.dev/making-rmsnorm-really-fast/

# 让RMSNorm变得更快

2025年4月18日

RMS Norm是一个在现代LLMs中常用的操作。给定一个向量$v$，它的RMS Norm计算方式为$v_i = \frac{v_i}{RMS(v)} \cdot w_i$，其中$w_i$是权重，且$RMS(v) = \sqrt{\epsilon + \frac{1}{N}\sum_{i=1,...,N}v_i^2}$。在这篇博文中，我们要计算矩阵$V = [v_1,...,v_{numToken}]$中每一行的RMS Norm，其中$v_i = [x_1,...,x_{hiddenDim}]$，给定权重$w = [w_1,...,w_{hiddenDim}]$。

## 顺序实现

检查我们的kernel的正确性需要一个基本的顺序实现作为参考。下面是我们使用的简单版本。

```c++
template <int numTokens, int hiddenDim>
void launchRmsNormCpu(float *x, float *w, float eps, float *y) {
  float rms;
  for (int token = 0; token < numTokens; token++) {
    rms = 0;
    for (int hidden = 0; hidden < hiddenDim; hidden++) {
      rms += x[token * hiddenDim + hidden] * x[token * hiddenDim + hidden];
    }
    rms = sqrt(rms / hiddenDim + eps);
    for (int hidden = 0; hidden < hiddenDim; hidden++) {
      y[token * hiddenDim + hidden] =
          x[token * hiddenDim + hidden] / rms * w[hidden];
    }
  }
}

```

## 如何并行化?

我们的并行化尝试非常简单。每个block处理一个token。如果block中的线程数小于隐藏维度的大小,每个线程就需要处理多个元素。然后我们执行一个简单的归约操作,计算RMS Norm并写入输出。如果你对归约操作不熟悉,请参考我之前[关于归约的博文](https://mp.weixin.qq.com/s/RklG6tmJnzPbIWxVBKDgLg) 。

## Naive kernel

A naive solution in CUDA is as follows.

```c++
template <int hiddenDim, int threadsPerBlock>
__global__ void rmsNormKernelNaive(float *x, float *w, float eps, float *y) {
  __shared__ float squaredPerThread[threadsPerBlock];
  __shared__ float rms_;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  float sum = 0.0f;

  for (int i = tid; i < hiddenDim; i += threadsPerBlock) {
    float x_ = x[bid * hiddenDim + i];
    sum += x_ * x_;
  }
  squaredPerThread[tid] = sum;
  __syncthreads();

  for (int activeThreads = threadsPerBlock / 2; activeThreads > 0;
       activeThreads >>= 1) {
    if (tid < activeThreads) {
      squaredPerThread[tid] += squaredPerThread[tid + activeThreads];
    }
    __syncthreads();
  }

  if (tid == 0) {
    rms_ = rsqrtf(squaredPerThread[tid] / hiddenDim + eps);
  }
  __syncthreads();

  for (int i = tid; i < hiddenDim; i += threadsPerBlock) {
    y[bid * hiddenDim + i] = x[bid * hiddenDim + i] * rms_ * w[i];
  }
}

template <int numTokens, int hiddenDim, int threadsPerBlock>
void launchRmsNormNaive(float *x, float *w, float eps, float *y) {
  rmsNormKernelNaive<hiddenDim, threadsPerBlock>
      <<<numTokens, threadsPerBlock>>>(x, w, eps, y);
}
```

`x` 跨内存访问一次,`w` 跨内存访问一次,`y` 跨内存访问一次。对于 `numTokens = 1 << 18` 和 `hiddenDim = 1 << 12` 的情况,`w` 的影响可以忽略不计,我们可以按如下方式计算带宽:

```c++
const size_t size = numTokens * hiddenDim * sizeof(float);
size_t numCrossMemoryBound = 2 * size;
float latency = time / numRounds;
float bandwidth = (numCrossMemoryBound / latency) / 1e6;
```

上述kernel的结果如下:

```shell
Latency = 2.84878 ms
Bandwidth = 3015.3 GB/s
% of max = 91.3727 %
```

## 使用共享内存

正如我们在上面看到的,我们频繁地访问`x`中的元素。我们可以使用共享内存来加快内存访问。

```c++
template <int hiddenDim, int threadsPerBlock>
__global__ void rmsNormKernelSmem(float *x, float *w, float eps, float *y) {
  __shared__ float squaredPerThread[threadsPerBlock];
  __shared__ float xShared[hiddenDim];
  __shared__ float rms_;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  float sum = 0.0f;

  for (int i = tid; i < hiddenDim; i += threadsPerBlock) {
    int index = bid * hiddenDim + i;
    float x_ = x[index];
    xShared[i] = x_;
    sum += x_ * x_;
  }
  squaredPerThread[tid] = sum;
  __syncthreads();

  for (int activeThreads = threadsPerBlock / 2; activeThreads > 0;
       activeThreads >>= 1) {
    if (tid < activeThreads) {
      squaredPerThread[tid] += squaredPerThread[tid + activeThreads];
    }
    __syncthreads();
  }

  if (tid == 0) {
    rms_ = rsqrtf(squaredPerThread[tid] / hiddenDim + eps);
  }
  __syncthreads();

  for (int i = tid; i < hiddenDim; i += threadsPerBlock) {
    float val = xShared[i] * rms_ * w[i];
    y[bid * hiddenDim + i] = val;
  }
}

template <int numTokens, int hiddenDim, int threadsPerBlock>
void launchRmsNormSmem(float *x, float *w, float eps, float *y) {
  rmsNormKernelSmem<hiddenDim, threadsPerBlock>
      <<<numTokens, threadsPerBlock>>>(x, w, eps, y);
}
```

上述kernel的结果如下:

```shell
Latency = 2.82101 ms
Bandwidth = 3044.99 GB/s
% of max = 92.2723 %
```

## 使用warp

类似我们在[前缀和操作](https://mp.weixin.qq.com/s/aKBwPEBEsxbLXJc_CKtl-A)中应用的技术,我们也可以这样做:

- 在每个warp中进行归约
- 使用一个warp归约这个数组以获得最终的归约结果。这个过程的代码如下:

```c++
#define WARP_SIZE 32

__device__ float warpReduce(float x) {
  float val = x;
  for (int activeThreads = WARP_SIZE >> 1; activeThreads > 0;
       activeThreads >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, activeThreads);
  }
  return val;
}

template <int hiddenDim, int threadsPerBlock>
__global__ void rmsNormKernelWarp(float *x, float *w, float eps, float *y) {
  __shared__ float squaredPerThread[threadsPerBlock];
  __shared__ float xShared[hiddenDim];
  __shared__ float sumPerWarp[WARP_SIZE];
  __shared__ float rms_;

  const int tid = threadIdx.x;
  const int laneId = tid & 31;
  const int warpId = tid >> 5;
  const int warpsPerBlock = threadsPerBlock >> 5;

  const int bid = blockIdx.x;
  float sum = 0.0f;

  for (int i = tid; i < hiddenDim; i += threadsPerBlock) {
    float x_ = x[bid * hiddenDim + i];
    xShared[i] = x_;
    sum += x_ * x_;
  }
  squaredPerThread[tid] = sum;
  __syncthreads();

  float warpSum = warpReduce(squaredPerThread[tid]);
  if (laneId == 0) {
    sumPerWarp[warpId] = warpSum;
  }
  __syncthreads();

  if (tid < WARP_SIZE) {
    sumPerWarp[tid] = warpReduce(tid < warpsPerBlock ? sumPerWarp[tid] : 0);
    if (tid == 0) {
      rms_ = rsqrtf(sumPerWarp[tid] / hiddenDim + eps);
    }
  }
  __syncthreads();

  for (int i = tid; i < hiddenDim; i += threadsPerBlock) {
    y[bid * hiddenDim + i] = xShared[i] * rms_ * w[i];
  }
}

template <int numTokens, int hiddenDim, int threadsPerBlock>
void launchRmsNormWarp(float *x, float *w, float eps, float *y) {
  rmsNormKernelWarp<hiddenDim, threadsPerBlock>
      <<<numTokens, threadsPerBlock>>>(x, w, eps, y);
}
```

上述kernel的结果如下:

```shell
Latency = 2.82263 ms
Bandwidth = 3043.23 GB/s
% of max = 92.2192 %
```

最初我预计这个会更快,但事实并非如此。

## 向量化加载和存储

如果我们对上述kernel进行性能分析,可以看到内存加载和存储消耗了最多的指令。我们可以使用CUDA的float4数据类型来向量化加载和存储操作来优化这一点。

对于共享内存的方法,代码如下所示:

```c++
template <int hiddenDim, int threadsPerBlock>
__global__ void rmsNormKernelSmemFloat4(float4 *x, float4 *w, float eps,
                                        float4 *y) {
  __shared__ float squaredPerThread[threadsPerBlock];
  __shared__ float4 xShared[hiddenDim >> 2];
  __shared__ float rms_;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  float sum = 0.0f;

  for (int i = tid; i < hiddenDim >> 2; i += threadsPerBlock) {
    int index = bid * (hiddenDim >> 2) + i;
    float4 x_ = x[index];
    xShared[i] = x_;
    sum += (x_.x * x_.x) + (x_.y * x_.y) + (x_.z * x_.z) + (x_.w * x_.w);
  }
  squaredPerThread[tid] = sum;
  __syncthreads();

  for (int activeThreads = threadsPerBlock >> 1; activeThreads > 0;
       activeThreads >>= 1) {
    if (tid < activeThreads) {
      squaredPerThread[tid] += squaredPerThread[tid + activeThreads];
    }
    __syncthreads();
  }

  if (tid == 0) {
    rms_ = rsqrtf(squaredPerThread[tid] / hiddenDim + eps);
  }
  __syncthreads();

  for (int i = tid; i < hiddenDim >> 2; i += threadsPerBlock) {
    float4 w_ = w[i];
    float4 x_ = xShared[i];
    float4 val = make_float4(x_.x * rms_ * w_.x, x_.y * rms_ * w_.y,
                             x_.z * rms_ * w_.z, x_.w * rms_ * w_.w);
    y[bid * (hiddenDim >> 2) + i] = val;
  }
}

template <int numTokens, int hiddenDim, int threadsPerBlock>
void launchRmsNormSmemFloat4(float *x, float *w, float eps, float *y) {
  float4 *x_ = reinterpret_cast<float4 *>(x);
  float4 *w_ = reinterpret_cast<float4 *>(w);
  float4 *y_ = reinterpret_cast<float4 *>(y);
  rmsNormKernelSmemFloat4<hiddenDim, threadsPerBlock>
      <<<numTokens, threadsPerBlock>>>(x_, w_, eps, y_);
}
```

上述kernel的结果如下:

```shell
Latency = 2.80455 ms
Bandwidth = 3062.86 GB/s
% of max = 92.8139 %
```

类似地,我们也可以对warp kernel进行优化:

```c++
#define WARP_SIZE 32

__device__ float warpReduce(float x) {
  float val = x;
  for (int activeThreads = WARP_SIZE >> 1; activeThreads > 0;
       activeThreads >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, activeThreads);
  }
  return val;
}

template <int hiddenDim, int threadsPerBlock>
__global__ void rmsNormKernelWarpFloat4(float4 *x, float4 *w, float eps,
                                        float4 *y) {
  __shared__ float squaredPerThread[threadsPerBlock];
  __shared__ float4 xShared[hiddenDim >> 2];
  __shared__ float sumPerWarp[WARP_SIZE];
  __shared__ float rms_;

  const int tid = threadIdx.x;
  const int laneId = tid & 31;
  const int warpId = tid >> 5;
  const int warpsPerBlock = threadsPerBlock >> 5;

  const int bid = blockIdx.x;
  float sum = 0.0f;

  for (int i = tid; i < hiddenDim >> 2; i += threadsPerBlock) {
    int index = bid * (hiddenDim >> 2) + i;
    float4 x_ = x[index];
    xShared[i] = x_;
    sum += (x_.x * x_.x) + (x_.y * x_.y) + (x_.z * x_.z) + (x_.w * x_.w);
  }
  squaredPerThread[tid] = sum;
  __syncthreads();

  float warpSum = warpReduce(squaredPerThread[tid]);
  if (laneId == 0) {
    sumPerWarp[warpId] = warpSum;
  }
  __syncthreads();

  if (tid < WARP_SIZE) {
    sumPerWarp[tid] = warpReduce(tid < warpsPerBlock ? sumPerWarp[tid] : 0);
    if (tid == 0) {
      rms_ = rsqrtf(sumPerWarp[tid] / hiddenDim + eps);
    }
  }
  __syncthreads();

  for (int i = tid; i < hiddenDim >> 2; i += threadsPerBlock) {
    float4 w_ = w[i];
    float4 x_ = xShared[i];
    float4 val = make_float4(x_.x * rms_ * w_.x, x_.y * rms_ * w_.y,
                             x_.z * rms_ * w_.z, x_.w * rms_ * w_.w);
    y[bid * (hiddenDim >> 2) + i] = val;
  }
}

template <int numTokens, int hiddenDim, int threadsPerBlock>
void launchRmsNormWarpFloat4(float *x, float *w, float eps, float *y) {
  float4 *x_ = reinterpret_cast<float4 *>(x);
  float4 *w_ = reinterpret_cast<float4 *>(w);
  float4 *y_ = reinterpret_cast<float4 *>(y);

  rmsNormKernelWarpFloat4<hiddenDim, threadsPerBlock>
      <<<numTokens, threadsPerBlock>>>(x_, w_, eps, y_);
}
```

上述kernel的结果如下:

```shell
Latency = 2.80475 ms
Bandwidth = 3062.63 GB/s
% of max = 92.8071 %
```

## 结论

我们看到,如果我们了解Reduction的工作原理,实现高性能的`RMSNorm`操作kernel并不困难。如果你发现了进一步的优化机会,我很乐意听取你的意见。让我感到惊讶的一点是,使用`#pragma unroll`并没有对性能产生积极影响。如果你喜欢这篇博文,我很乐意在LinkedIn(https://www.linkedin.com/in/simon-veitner-174a681b6/)上与你联系,交流关于CUDA或其他机器学习系统的想法。上述结果的所有复现代码都可以在我的Github(https://github.com/simveit/effective_rms_norm)上找到。




