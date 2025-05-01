> 本文由 @Simon V(https://github.com/simveit) 授权转载和翻译并发表到本公众号。原始地址为：https://veitner.bearblog.dev/making-prefix-sum-really-fast/

# Making prefix sum really fast

13 Apr, 2025

In this blogpost we want to show how to optimize blockwise prefix sum operation. Blockwise prefix sum does the following:

Given a vector: `v` we divide that vector into blocks. Inside each block we perform than prefix sum. A simple example with blocks which consists of 4 elements would be `v = [0, 1, 2, 3, 4, 5, 6, 7]` That would than return the vector `o = [0, 1, 3, 6, 4, 9, 15, 22]`

This is an essential building block of prefix sum operation in GPUs.

## The algorithm

Please see slide 21(https://safari.ethz.ch/projects_and_seminars/fall2022/lib/exe/fetch.php?media=p_s-hetsys-fs2022-meeting10-aftermeeting.pdf) to understand the algorithm. This picture describes the algorithm we want to perform very clearly. We have multiple stages. At each stage we increase the stride between two elements that get added by a factor of two. We obtain the cumulative sum vector as a result after the last stage.

## Naive baseline

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

This algorithm implements the above picture. At each stage we increase the offset by a factor of two. We will do that until we arrive at the middle of the thread block and and have distance of half of threadblock size between accumulated elements. Note that we need the `__syncthreads` to avoid a race condition. If we wouldn't have these two barriers it could happen that two array elements would get used for read and write at the same time.

It is good practice to check the correctness of our program with a simple cpu implementation

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

The above kernel gives the correct result. Unfortunately it doesn't perform very good. That is due to the frequent accesses to global memory. The bandwidth can be calculated using that we make N read and write operation where we used `N = 1 << 30 = 2**30`. The measured performance for the above kernel is:

```c++
Bandwidth: 823.944 GB/s
Efficiency: 0.24968
```

## Using Shared memory

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

This kernel is very similar to the one above. The main difference is that here we use shared memory. Shared memory is good if we want to frequently access elements because it's much cheaper to read an element from shared than from global memory. The performance is as follows:

```c++
Bandwidth: 1288.72 GB/s
Efficiency: 0.390522
```

## Using double buffer

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

This kernel uses a double buffer. We initialise two arrays in shared memory. We than swap out the buffers at each stage. This has the big advantage that we can save one barrier. This is because we now have two arrays we access and therefore can be sure we don't get a race condition. The performance of this kernel is

```c++
Bandwidth: 1616.71 GB/s
Efficiency: 0.489913
```

## Using warp primitives.

CUDA offers warp primitives. One of these warp primitives is called `__shfl_up_sync` and it is very well suited for our operation because it does exactly carry out the operation pictured out above. You can read more about it in this blogpost(https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/). We can use that to further speed up the performance of our kernel:

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

This gives a further boost in performance because we can perform the warp wise reductions very efficiently using the warp primitives. For a detailed explanation please see the following lecture(https://www.youtube.com/watch?v=SG0gvcbf2eo) on youtube. Our new performance is:

```c++
Bandwidth: 1976.42 GB/s
Efficiency: 0.598916
```

## Increase workload per thread

The above kernels (or at least variants of them) are well known and you can find many explanations of them on the internet. The last kernel is not well documented (I didn't find any references on this simple technique in the internet) but essential to get close to peak performance.

Note: This technique is called thread coarsening as ngc92 from GPU mode discord server pointed out to me. You can read more about it in the PPMP book(https://www.sciencedirect.com/science/article/abs/pii/B9780323912310000227).

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

The `block_scan` stays the same here as before. The difference is that now we process multiple elements with each thread.

Adjusting the verification function to

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

shows that our result is still correct (Note that iff batchSize is of form 2^n we can use the bit operation from above to perform modulo).

We do this by first performing a simple sequential scan on the elements belonging to current thread. We'll than let the block perform scan on these total sums. Afterwards we write the outputs by adding the reduced total sum to the reduction part. This procedure is similar to the above hierarchy of the warp scan as well as the the operations we have to carry out for a full prefix sum. I again refer to the lecture above for more detailed explanation on what we mean by hierarchy. The performance of the final kernel is:

```shell
Bandwidth: 3056.53 GB/s
Efficiency: 0.926221
```

We can probably squeeze even more out of the GPU to reach near to 100% performance by tuning the block- and batchsize but I stop at this point to keep the blogpost concise. For example we could also use `int4` to use less instructions when loading the batched data, though in my experiments that didn't have a huge effect on performance. If you have further techniques to improve the performance please let me know! I hope you enjoyed reading this blogpost. The above mentioned lecture(https://www.youtube.com/watch?v=SG0gvcbf2eo) was very helpful in getting a better understanding for prefix sum. You can contact me on Linkedin(https://www.linkedin.com/in/simon-veitner-174a681b6/) for further discussion on CUDA. I am interested to learn your perspective! All the code can be found on my github repo(https://github.com/simveit/effective_scan).



