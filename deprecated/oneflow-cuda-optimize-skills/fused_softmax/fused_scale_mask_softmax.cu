#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <math_constants.h>
using namespace std;

// softmax
// source from https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/cuda/softmax.cuh

#define CUDA_CHECK()  if( (cudaPeekAtLastError()) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__-1); exit(-1);}

constexpr int kWarpSize = 32;

template<typename T>
struct SumOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
};

template<typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return max(a, b); }
};

// 每个 Warp 处理一行或两行元素，每行的Reduce操作 需要做 Warp 内的 Reduce 操作，
// 我们实现 WarpAllReduce 来完成 Warp 内各线程间的求 Global Max 和 Global Sum 操作，
// WarpAllReduce 是利用Warp级别原语 __shfl_xor_sync 实现的，代码如下。
template<template<typename> class ReductionOp, typename T, int thread_group_width = kWarpSize>
__inline__ __device__ T WarpAllReduce(T val) {
  for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
    val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

// BlockReduce 使用 cub 进行实现
template<template<typename> class ReductionOp, typename T, int block_size>
__inline__ __device__ T BlockAllReduce(T val) {
  typedef cub::BlockReduce<T, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T result_broadcast;
  T result = BlockReduce(temp_storage).Reduce(val, ReductionOp<T>());
  if (threadIdx.x == 0) { result_broadcast = result; }
  __syncthreads();
  return result_broadcast;
}

// 定义各种数据类型下面的 inf 值
template<typename T>
__inline__ __device__ T Inf();

template<>
__inline__ __device__ float Inf<float>() {
  return CUDART_INF_F;
}

template<>
__inline__ __device__ double Inf<double>() {
  return CUDART_INF;
}

// 定义 exp 函数，OF_SOFTMAX_USE_FAST_MATH 这个宏表示开启 FAST_MATH 的 exp
template<typename T>
__inline__ __device__ T Exp(T x);

template<>
__inline__ __device__ float Exp<float>(float x) {
#ifdef OF_SOFTMAX_USE_FAST_MATH
  return __expf(x);
#else
  return exp(x);
#endif
}

template<>
__inline__ __device__ double Exp<double>(double x) {
  return exp(x);
}

// 定义 div 函数，OF_SOFTMAX_USE_FAST_MATH 这个宏表示开启 FAST_MATH 的 div
template<typename T>
__inline__ __device__ T Div(T a, T b);

template<>
__inline__ __device__ float Div<float>(float a, float b) {
#ifdef OF_SOFTMAX_USE_FAST_MATH
  return __fdividef(a, b);
#else
  return a / b;
#endif
}

template<>
__inline__ __device__ double Div<double>(double a, double b) {
  return a / b;
}

// 定义 log 函数，OF_SOFTMAX_USE_FAST_MATH 这个宏表示开启 FAST_MATH 的 log
template<typename T>
__inline__ __device__ T Log(T x);

template<>
__inline__ __device__ float Log<float>(float x) {
#ifdef OF_SOFTMAX_USE_FAST_MATH
  return __logf(x);
#else
  return log(x);
#endif
}
template<>
__inline__ __device__ double Log<double>(double x) {
  return log(x);
}

// 对于 cuda kernel 来说，启动多少个线程块（grid_size）来做计算？
// 具体可以参考俊丞大佬这篇 [如何设置CUDA Kernel中的grid_size和block_size？ ](https://mp.weixin.qq.com/s/1_ao9xM6Qk3JaavptChXew)
inline cudaError_t GetNumBlocks(int64_t block_size, int64_t max_blocks, int64_t waves,
                                int* num_blocks) {
  int dev;
  {
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) { return err; }
  }
  int sm_count;
  {
    cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) { return err; }
  }
  int tpm;
  {
    cudaError_t err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    if (err != cudaSuccess) { return err; }
  }
  *num_blocks =
      std::max<int>(1, std::min<int64_t>(max_blocks, sm_count * tpm / block_size * waves));
  return cudaSuccess;
}

// 定义默认的计算类型，一般的数据类型进行 softmax 计算时计算类型就是自己，
// 而对于 half（或者bfp16） 来说我们往往需要将计算类型 fallback 到 float 来避免精度损失
template<typename T>
struct DefaultComputeType {
  using type = T;
};

template<>
struct DefaultComputeType<half> {
  using type = float;
};

#if CUDA_VERSION >= 11000
template<>
struct DefaultComputeType<nv_bfloat16> {
  using type = float;
};
#endif  // CUDA_VERSION >= 11000

// GetPackType 结构体中使用了 std::aligned_storage 先声明了一个内存对齐的数据类型 type ，
// 注意这个 type 的内存长度为 pack_size * sizeof(T) 。然后这里的 T 是我们需要进行 Pack 
// 的数据类型，而 pack_size 则表示我们需要 Pack 的元素个数。接下来我们看到 Pack 联合体中
// 声明了 storage 和 elem 两个数组，它们共用同一段对齐的内存。然后 Pack 联合体的入口有一个
// 检查: static_assert(sizeof(PackType<T, pack_size>) == sizeof(T) * pack_size, ""); 
// 这是用来判断我们之前声明的 type 的内存长度是否符合预期。
template<typename T, int N>
struct GetPackType {
  using type = typename std::aligned_storage<N * sizeof(T), N * sizeof(T)>::type;
};

template<typename T, int N>
using PackType = typename GetPackType<T, N>::type;

template<typename T, int N>
union Pack {
  static_assert(sizeof(PackType<T, N>) == sizeof(T) * N, "");
  __device__ Pack() {
    // do nothing
  }
  PackType<T, N> storage;
  T elem[N];
};

// 下面分别定义了两个代表输入输出的数据结构
template<typename SRC, typename DST>
struct DirectLoad {
  DirectLoad(const SRC* src, int64_t row_size) : src(src), row_size(row_size) {}
  template<int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) const {
    Pack<SRC, N> pack;
    const int64_t offset = (row * row_size + col) / N;
    pack.storage = *(reinterpret_cast<const PackType<SRC, N>*>(src) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) { dst[i] = static_cast<DST>(pack.elem[i]); }
  }
  const SRC* src;
  int64_t row_size;
};

template<typename SRC, typename DST>
struct DirectStore {
  DirectStore(DST* dst, int64_t row_size) : dst(dst), row_size(row_size) {}
  template<int N>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    Pack<DST, N> pack;
    const int64_t offset = (row * row_size + col) / N;
#pragma unroll
    for (int i = 0; i < N; ++i) { pack.elem[i] = static_cast<DST>(src[i]); }
    *(reinterpret_cast<PackType<DST, N>*>(dst) + offset) = pack.storage;
  }
  DST* dst;
  int64_t row_size;
};

// 
enum class Algorithm {
  kSoftmax = 0,
  kLogSoftmax = 1,
};

// 使用 `load.template load<pack_size>(ptr, row_id, col_id);`
// 和`store.template store<pack_size>(ptr, row_id, col_id);` 进行读取和写入
// 使用LOAD和STORE有两个好处：1、可以在CUDA Kernel中只关心计算类型ComputeType，而不用关心具体的数据类型T。
// 2、只需要加几行代码就可以快速支持Softmax和其他Kernel Fuse，减少带宽需求，提升整体性能。
// 普通的SoftmaxKernel直接使用DirectLoad和DirectStore，FusedScaleSoftmaxKernel如FusedScaleSoftmaxDropoutKernel
// 只需要定义一个ScaleLoad结构和一个DropoutStore结构用于对输入x做Scale预处理和对输出y做Dropout后处理。
// ComputeType代表计算类型。pack_size代表向量化访存操作的pack元素的个数，我们将几个元素pack起来读写，提升带宽利用率。
// cols_per_thread代表每个线程处理的元素个数。thread_group_width代表处理元素的线程组的宽度，
// 当cols > pack_size * warp_size时，thread_group_width就是warp_size，即32。
// 当cols < pack_size * warp_size时，就根据cols大小用1/2个warp或1/4个warp来处理每行的元素。
// 采用更小的thread_group_width后，WarpAllReduce需要执行的轮次也相应减少。
// rows_per_access代表每个线程组一次处理的行数，当cols较小，thread_group_width不是warp_size 32时，
// 若rows能被2整除，我们就让每个线程处理2行来增加指令并行度，从而提升性能。
// padding代表当前是否做了padding，若cols不是warp_size的整数倍，我们会把它padding到最近的整数倍处理。
// algorithm代表使用的算法，可选项有Algorithm::kSoftmax或Algorithm::kLogSoftmax。

// CUDA Kernel执行的主体循环逻辑如下，首先根据 num_cols信息算出每个线程要处理的cols_per_thread，
// 每个线程分配`rows_per_access * cols_per_thread`大小的寄存器，将输入x读到寄存器中，后续计算均从寄存器中读取。
template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int cols_per_thread,
         int thread_group_width, int rows_per_access, bool padding, Algorithm algorithm>
__global__ void SoftmaxWarpImpl(LOAD load, STORE store, const int64_t rows, const int64_t cols) {
  // 我们需要保证每个线程处理的元素个数可以被 pack_size 整除
  static_assert(cols_per_thread % pack_size == 0, "");
  // 处理元素的线程组的宽度需要小于等于kWarpSize，并且需要被kWarpSize整除
  static_assert(thread_group_width <= kWarpSize, "");
  static_assert(kWarpSize % thread_group_width == 0, "");
  // 每个线程处理的 pack 后的元素个数
  constexpr int num_packs = cols_per_thread / pack_size;
  // 需要保证 cols <= 每个线程处理的元素个数 * 处理元素的线程组的宽度 ，因为这个地方是每个 Warp 处理一行或两行元素
  assert(cols <= cols_per_thread  * thread_group_width);
  // 开一块共享内存，行数为每个线程组一次处理的行数，列数为每个线程处理的元素个数
  ComputeType buf[rows_per_access][cols_per_thread];

  // int grid_dim_x;
  // dim3 block_dim(thread_group_width, thread_groups_per_block);
  // 从下面启动 SoftmaxWarpImpl 的参数来看，这里使用的是一维的 grid，二维的 block，
  // 并且 block 的长度为处理元素的线程组的宽度（warp_size），block 的宽度为每个 block 的线程组的个数
  // 注意启动 kernel 时每个 block 的总线程数是 128 ，如果 thread_group_width = 32
  // 那么 thread_groups_per_block = 128 / 32 = 4
  // 获取全局的线程组id
  const int global_thread_group_id = blockIdx.x * blockDim.y + threadIdx.y;
  // 获取全局线程组的数量
  const int num_global_thread_group = gridDim.x * blockDim.y;
  // lane id，表示当前 thread 在当前 lane 中的索引，注意 threadIdx.x 在这个 kernel 里面是不可能超越一个 warp (32) 的
  // 所以这里省掉了取模
  const int lane_id = threadIdx.x;
  // step 表示循环计数器大小为 全局线程组的数量 * 每个线程组一次处理的行数
  const int64_t step = num_global_thread_group * rows_per_access;
  // for 循环的开始为 row = 全局的线程组id * 每个线程组一次处理的行数，结束为总行数
  for (int64_t row = global_thread_group_id * rows_per_access; row < rows; row += step) {
    // 开辟一块共享内存记录当前线程组处理的每一行的最大值
    ComputeType thread_max[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      // 把 thread_max[row_id] 初始化为 -inf
      thread_max[row_id] = -Inf<ComputeType>();
      // 获取第 row_id 行的共享内存数据
      ComputeType* row_buf = buf[row_id];
#pragma unroll
      for (int pack_id = 0; pack_id < num_packs; ++pack_id) {
        // pack的偏移量
        const int pack_offset = pack_id * pack_size;
        const int col = (pack_id * thread_group_width + lane_id) * pack_size;
        if (!padding || col < cols) {
          load.template load<pack_size>(row_buf + pack_offset, row + row_id, col);
#pragma unroll
          for (int i = 0; i < pack_size; ++i) {
            thread_max[row_id] = max(thread_max[row_id], row_buf[pack_offset + i]);
          }
        } else {
#pragma unroll
          for (int i = 0; i < pack_size; ++i) { row_buf[pack_offset + i] = -Inf<ComputeType>(); }
        }
      }
    }
    // 开辟一块共享内存记录属于同一个warp的线程组的每一行的最大值，也就是需要进行一次warpReduce max
    ComputeType warp_max[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      warp_max[row_id] = WarpAllReduce<MaxOp, ComputeType, thread_group_width>(thread_max[row_id]);
    }
    // 开辟一块共享内存记录当前线程组处理的每一行的sum
    ComputeType thread_sum[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      thread_sum[row_id] = 0;
      ComputeType* row_buf = buf[row_id];
#pragma unroll
      for (int i = 0; i < cols_per_thread; ++i) {
        if (algorithm == Algorithm::kSoftmax) {
          row_buf[i] = Exp(row_buf[i] - warp_max[row_id]);
          thread_sum[row_id] += row_buf[i];
        } else if (algorithm == Algorithm::kLogSoftmax) {
          row_buf[i] -= warp_max[row_id];
          thread_sum[row_id] += Exp(row_buf[i]);
        } else {
          __trap(); // 内核的执行被中止并在主机程序中引发中断。
        }
      }
    }
    // 开辟一块共享内存记录属于同一个warp的线程组的每一行的sum (注意这里考虑了指数运算的安全性，
    // 实际上求的是sum{exp(x_i-max{x_i})})，也就是需要进行一次warpReduce sum
    ComputeType warp_sum[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      warp_sum[row_id] = WarpAllReduce<SumOp, ComputeType, thread_group_width>(thread_sum[row_id]);
    }
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      ComputeType* row_buf = buf[row_id];
#pragma unroll
      for (int i = 0; i < cols_per_thread; ++i) {
        if (algorithm == Algorithm::kSoftmax) {
          row_buf[i] = Div(row_buf[i], warp_sum[row_id]);
        } else if (algorithm == Algorithm::kLogSoftmax) {
          row_buf[i] -= Log(warp_sum[row_id]);
        } else {
          __trap();// 内核的执行被中止并在主机程序中引发中断。
        }
      }
#pragma unroll
      for (int i = 0; i < num_packs; ++i) {
        const int col = (i * thread_group_width + lane_id) * pack_size;
        if (!padding || col < cols) {
          store.template store<pack_size>(row_buf + i * pack_size, row + row_id, col);
        }
      }
    }
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int cols_per_thread,
         int thread_group_width, int rows_per_access, bool padding, Algorithm algorithm>
inline cudaError_t LaunchSoftmaxWarpImpl(cudaStream_t stream, LOAD load, STORE store,
                                         const int64_t rows, const int64_t cols) {
  // 每个 Block 里面设置 128 个线程
  constexpr int block_size = 128;
  // GPU 一次可以调度 SM 数量 * 每个 SM 最大 block 数个 block，
  // 因为每个 block 的计算量相等，所以所有 SM 应几乎同时完成这些 block 的计算，
  // 然后处理下一批，这其中的每一批被称之为一个 wave。
  // grid_size 设置为可以满足足够多的 wave，也就是这里定义的 waves
  constexpr int waves = 32;
  // block_size 需要整除处理元素的线程组的宽度
  static_assert(block_size % thread_group_width == 0, "");
  // 每个 block 的线程组的个数，如果这里 thread_group_width = 32，那么 thread_groups_per_block = 4
  constexpr int thread_groups_per_block = block_size / thread_group_width;
  dim3 block_dim(thread_group_width, thread_groups_per_block);
  // 根据数据的大小计算最多设置多少个 block ，就是数据的行数 / 每个线程组一次处理的行数
  const int64_t num_blocks =
      (rows / rows_per_access + thread_groups_per_block - 1) / thread_groups_per_block;
  // 根据上述的设置以及硬件本身的限制计算最终启动的 block 数
  int grid_dim_x;
  {
    cudaError_t err = GetNumBlocks(block_size, num_blocks, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  SoftmaxWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread, thread_group_width,
                  rows_per_access, padding, algorithm>
      <<<grid_dim_x, block_dim, 0, stream>>>(load, store, rows, cols);
  return cudaPeekAtLastError();
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int cols_per_thread,
         int thread_group_width, int rows_per_access, Algorithm algorithm>
inline cudaError_t DispatchSoftmaxWarpImplPadding(cudaStream_t stream, LOAD load, STORE store,
                                                  const int64_t rows, const int64_t cols) {
  // 如果每个线程处理的元素个数 * 处理元素的线程组的宽度(warp_size)和cols相等，就不需要padding
  if (cols == cols_per_thread * thread_group_width) {
    return LaunchSoftmaxWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread,
                                 thread_group_width, rows_per_access, false, algorithm>(
        stream, load, store, rows, cols);
  } else {
    return LaunchSoftmaxWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread,
                                 thread_group_width, rows_per_access, true, algorithm>(
        stream, load, store, rows, cols);
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, Algorithm algorithm>
typename std::enable_if<pack_size == 1, cudaError_t>::type DispatchSoftmaxWarpImplCols(
    cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols) {
  if (cols <= 0) { return cudaErrorInvalidValue; }
// 这里是对比数据的列数(cols)和线程组的宽度*pack_size，如果满足条件就dispatch到对应的线程组宽度(warp_size)，
// 注意这里是一个线程处理 pack_size 个元素
#define DEFINE_ONE_ELIF(thread_group_width)                                                        \
  else if (cols <= (thread_group_width)*pack_size) {                                               \
    if (rows % 2 == 0) {                                                                           \
      return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size,        \
                                            thread_group_width, 2, algorithm>(stream, load, store, \
                                                                              rows, cols);         \
    } else {                                                                                       \
      return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size,        \
                                            thread_group_width, 1, algorithm>(stream, load, store, \
                                                                              rows, cols);         \
    }                                                                                              \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
// 如果上面的条件都不满足，那么直接一个线程处理col个元素，而不是pack_size个元素
#define DEFINE_ONE_ELIF(col)                                                                      \
  else if (cols <= (col)*kWarpSize) {                                                             \
    return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, col, kWarpSize, 1, \
                                          algorithm>(stream, load, store, rows, cols);            \
  }
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(3)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(5)
  DEFINE_ONE_ELIF(6)
  DEFINE_ONE_ELIF(7)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(9)
  DEFINE_ONE_ELIF(10)
  DEFINE_ONE_ELIF(11)
  DEFINE_ONE_ELIF(12)
  DEFINE_ONE_ELIF(13)
  DEFINE_ONE_ELIF(14)
  DEFINE_ONE_ELIF(15)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(17)
  DEFINE_ONE_ELIF(18)
  DEFINE_ONE_ELIF(19)
  DEFINE_ONE_ELIF(20)
  DEFINE_ONE_ELIF(21)
  DEFINE_ONE_ELIF(22)
  DEFINE_ONE_ELIF(23)
  DEFINE_ONE_ELIF(24)
  DEFINE_ONE_ELIF(25)
  DEFINE_ONE_ELIF(26)
  DEFINE_ONE_ELIF(27)
  DEFINE_ONE_ELIF(28)
  DEFINE_ONE_ELIF(29)
  DEFINE_ONE_ELIF(30)
  DEFINE_ONE_ELIF(31)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
  else {
    return cudaErrorInvalidValue;
  }
}

// 同理，dispatch pack_size==2情况下的kernel
template<typename LOAD, typename STORE, typename ComputeType, int pack_size, Algorithm algorithm>
typename std::enable_if<pack_size == 2, cudaError_t>::type DispatchSoftmaxWarpImplCols(
    cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols) {
  if (cols <= 0) { return cudaErrorInvalidValue; }
#define DEFINE_ONE_ELIF(thread_group_width)                                                        \
  else if (cols <= (thread_group_width)*pack_size) {                                               \
    if (rows % 2 == 0) {                                                                           \
      return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size,        \
                                            thread_group_width, 2, algorithm>(stream, load, store, \
                                                                              rows, cols);         \
    } else {                                                                                       \
      return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size,        \
                                            thread_group_width, 1, algorithm>(stream, load, store, \
                                                                              rows, cols);         \
    }                                                                                              \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                                      \
  else if (cols <= (col)*kWarpSize) {                                                             \
    return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, col, kWarpSize, 1, \
                                          algorithm>(stream, load, store, rows, cols);            \
  }
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(6)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(10)
  DEFINE_ONE_ELIF(12)
  DEFINE_ONE_ELIF(14)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(18)
  DEFINE_ONE_ELIF(20)
  DEFINE_ONE_ELIF(22)
  DEFINE_ONE_ELIF(24)
  DEFINE_ONE_ELIF(26)
  DEFINE_ONE_ELIF(28)
  DEFINE_ONE_ELIF(30)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
  else {
    return cudaErrorInvalidValue;
  }
}

template<typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
struct DispatchSoftmaxWarpImplPackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                         const int64_t cols) {
    if (cols % 2 == 0) {
      return DispatchSoftmaxWarpImplCols<LOAD, STORE, ComputeType, 2, algorithm>(stream, load,
                                                                                 store, rows, cols);
    } else {
      return DispatchSoftmaxWarpImplCols<LOAD, STORE, ComputeType, 1, algorithm>(stream, load,
                                                                                 store, rows, cols);
    }
  }
};

// 每个Warp处理一行或两行元素时最原始的dispatch接口
template<typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
inline cudaError_t DispatchSoftmaxWarpImpl(cudaStream_t stream, LOAD load, STORE store,
                                           const int64_t rows, const int64_t cols) {
  return DispatchSoftmaxWarpImplPackSize<LOAD, STORE, ComputeType, algorithm>()(stream, load, store,
                                                                                rows, cols);
}

// 一个 Block 处理一行元素， 利用 BlockAllReduce 完成 Warp 内各线程间的求 Global Max 和 Global Sum 操作。
// BlockAllReduce 是借助 Cub 的 BlockReduce 方法实现的。

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size,
         Algorithm algorithm>
__global__ void SoftmaxBlockSMemImpl(LOAD load, STORE store, const int64_t rows,
                                     const int64_t cols) {
  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  auto* buf = reinterpret_cast<ComputeType*>(shared_buf);
  const int tid = threadIdx.x;
  assert(cols % pack_size == 0);
  const int num_packs = cols / pack_size;
  // 一个 Block 处理一行元素
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    // 当前线程的最大值初始化为 -inf
    ComputeType thread_max = -Inf<ComputeType>();
    // 以向量化的方式加载一行数据，然后执行pack reduce操作
    // 这里的 pack reduce操作我在 https://zhuanlan.zhihu.com/p/596012674 最后一节也有介绍
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        buf[i * num_packs + pack_id] = pack[i];
        thread_max = max(thread_max, pack[i]);
      }
    }
    // 执行block reduce获取当前行（由一个 Block 进行处理）的最大值
    const ComputeType row_max = BlockAllReduce<MaxOp, ComputeType, block_size>(thread_max);
    ComputeType thread_sum = 0;
    for (int col = tid; col < cols; col += block_size) {
      if (algorithm == Algorithm::kSoftmax) {
        const ComputeType exp_x = Exp(buf[col] - row_max);
        buf[col] = exp_x;
        thread_sum += exp_x;
      } else {
        const ComputeType x = buf[col] - row_max;
        buf[col] = x;
        thread_sum += Exp(x);
      }
    }
    // 同理，获得当前行的sum
    const ComputeType row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);
    // 计算结果并写回到全局内存中
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        if (algorithm == Algorithm::kSoftmax) {
          pack[i] = Div(buf[i * num_packs + pack_id], row_sum);
        } else if (algorithm == Algorithm::kLogSoftmax) {
          pack[i] = buf[i * num_packs + pack_id] - Log(row_sum);
        } else {
          __trap();
        }
      }
      store.template store<pack_size>(pack, row, pack_id * pack_size);
    }
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size,
         Algorithm algorithm>
inline cudaError_t LaunchSoftmaxBlockSMemImpl(cudaStream_t stream, LOAD load, STORE store, int smem,
                                              const int64_t rows, const int64_t cols) {
  constexpr int waves = 32;
  int grid_dim_x;
  {
    cudaError_t err = GetNumBlocks(block_size, rows, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size, algorithm>
      <<<grid_dim_x, block_size, smem, stream>>>(load, store, rows, cols);
  return cudaPeekAtLastError();
}

// 执行的主体循环逻辑如下，根据 num_cols算出需要的 Shared Memory 大小作为 Launch Kernel 参数，
// 借助 Shared Memory 保存输入，后续计算直接从 Shared Memory 读取。
// 由于 SM 内的 Shared Memory 资源同样有限，因此当 num_cols超过一定范围，kernel 启动时申请 Shared Memory 超过最大限制，
// 就会出现无法启动的问题，因此，仅在调用 cudaOccupancyMaxActiveBlocksPerMultiprocessor 返回值大于0时采用 Shared Memory 方案。
// 此外，需要注意的是，由于 Block 内线程要做同步，当 SM 中正在调度执行的一个 Block 到达同步点时，SM 内可执行 Warp 逐渐减少，
// 若同时执行的 Block 只有一个，则 SM 中可同时执行的 Warp 会在此时逐渐降成0，会导致计算资源空闲，造成浪费，若此时同时有其他 Block 在执行，
// 则在一个 Block 到达同步点时仍然有其他 Block 可以执行。当 block_size 越小时，SM 可同时调度的 Block 越多，因此在这种情况下 block_size 越小越好。
// 但是当在调大 block_size，SM 能同时调度的 Block 数不变的情况下，block_size 应该是越大越好，越大就有越好的并行度。
// 因此代码中在选择 block_size 时，对不同 block_size 都计算了 cudaOccupancyMaxActiveBlocksPerMultiprocessor，若结果相同，使用较大的 block_size。
template<typename LOAD, typename STORE, typename ComputeType, int pack_size, Algorithm algorithm>
inline cudaError_t TryDispatchSoftmaxBlockSMemImplBlockSize(cudaStream_t stream, LOAD load,
                                                            STORE store, const int64_t rows,
                                                            const int64_t cols, bool* success) {
  // 设置4个不同的block_size
  constexpr int block_size_conf_1 = 128;
  constexpr int block_size_conf_2 = 256;
  constexpr int block_size_conf_3 = 512;
  constexpr int block_size_conf_4 = 1024;
  // 计算第二种方案需要的共享内存大小
  const size_t smem = cols * sizeof(ComputeType);
  int max_active_blocks_conf_1;
  {
    // 占用计算器API cudaOccupancyMaxActiveBlocksPerMultiprocessor可以根据 kernel 的 block 大小和共享内存使用情况提供占用率预测。
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_1,
        SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_1, algorithm>,
        block_size_conf_1, smem);
    if (err != cudaSuccess) { return err; }
  }
  if (max_active_blocks_conf_1 <= 0) {
    *success = false;
    return cudaSuccess;
  }
  int max_active_blocks_conf_4;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_4,
        SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_4, algorithm>,
        block_size_conf_4, smem);
    if (err != cudaSuccess) { return err; }
  }
  if (max_active_blocks_conf_4 == max_active_blocks_conf_1) {
    *success = true;
    return LaunchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_4,
                                      algorithm>(stream, load, store, smem, rows, cols);
  }
  int max_active_blocks_conf_3;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_3,
        SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_3, algorithm>,
        block_size_conf_3, smem);
    if (err != cudaSuccess) { return err; }
  }
  if (max_active_blocks_conf_3 == max_active_blocks_conf_1) {
    *success = true;
    return LaunchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_3,
                                      algorithm>(stream, load, store, smem, rows, cols);
  }
  int max_active_blocks_conf_2;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_2,
        SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_2, algorithm>,
        block_size_conf_2, smem);
    if (err != cudaSuccess) { return err; }
  }
  if (max_active_blocks_conf_2 == max_active_blocks_conf_1) {
    *success = true;
    return LaunchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_2,
                                      algorithm>(stream, load, store, smem, rows, cols);
  }
  *success = true;
  return LaunchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_1,
                                    algorithm>(stream, load, store, smem, rows, cols);
}

template<typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
struct TryDispatchSoftmaxBlockSMemImplPackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                         const int64_t cols, bool* success) {
    if (cols % 2 == 0) {
      return TryDispatchSoftmaxBlockSMemImplBlockSize<LOAD, STORE, ComputeType, 2, algorithm>(
          stream, load, store, rows, cols, success);
    } else {
      return TryDispatchSoftmaxBlockSMemImplBlockSize<LOAD, STORE, ComputeType, 1, algorithm>(
          stream, load, store, rows, cols, success);
    }
  }
};

template<typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
inline cudaError_t TryDispatchSoftmaxBlockSMemImpl(cudaStream_t stream, LOAD load, STORE store,
                                                   const int64_t rows, const int64_t cols,
                                                   bool* success) {
  return TryDispatchSoftmaxBlockSMemImplPackSize<LOAD, STORE, ComputeType, algorithm>()(
      stream, load, store, rows, cols, success);
}

// 和实现2一样，仍然是一个 Block 处理一行元素，
// 不同的是，不再用 Shared Memory 缓存输入x，
// 而是在每次计算时重新读输入 x，
// 这种实现没有最大 num_cols的限制，可以支持任意大小。
// 此外，需要注意的是，在这种实现中，block_size 应该设越大越好，
// block_size 越大，SM 中能同时并行执行的 Block 数就越少，
// 对 cache 的需求就越少，就有更多机会命中 Cache，
// 多次读x不会多次访问 Global Memory，因此在实际测试中，
// 在能利用 Cache 情况下，有效带宽不会因为读3次x而降低几倍。
template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size,
         Algorithm algorithm>
__global__ void SoftmaxBlockUncachedImpl(LOAD load, STORE store, const int64_t rows,
                                         const int64_t cols) {
  const int tid = threadIdx.x;
  assert(cols % pack_size == 0);
  const int num_packs = cols / pack_size;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    ComputeType thread_max = -Inf<ComputeType>();
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) { thread_max = max(thread_max, pack[i]); }
    }
    const ComputeType row_max = BlockAllReduce<MaxOp, ComputeType, block_size>(thread_max);
    ComputeType thread_sum = 0;
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) { thread_sum += Exp(pack[i] - row_max); }
    }
    const ComputeType row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        if (algorithm == Algorithm::kSoftmax) {
          pack[i] = Div(Exp(pack[i] - row_max), row_sum);
        } else if (algorithm == Algorithm::kLogSoftmax) {
          pack[i] = (pack[i] - row_max) - Log(row_sum);
        } else {
          __trap();
        }
      }
      store.template store<pack_size>(pack, row, pack_id * pack_size);
    }
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, Algorithm algorithm>
inline cudaError_t LaunchSoftmaxBlockUncachedImpl(cudaStream_t stream, LOAD load, STORE store,
                                                  const int64_t rows, const int64_t cols) {
  // 每个 Block 使用 1024 个线程
  constexpr int block_size = 1024;
  // waves 需要满足32组
  constexpr int waves = 32;
  // 根据 BlockSize 以及硬件参数计算 Block 数量
  int grid_dim_x;
  {
    cudaError_t err = GetNumBlocks(block_size, rows, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  // 启动第三者实现的 cuda kernel
  SoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, pack_size, block_size, algorithm>
      <<<grid_dim_x, block_size, 0, stream>>>(load, store, rows, cols);
  return cudaPeekAtLastError();
}

template<typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
struct DispatchSoftmaxBlockUncachedImplPackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                         const int64_t cols) {
    // 如果 cols % 2 == 0，就执行 pack2，否则不 pack
    if (cols % 2 == 0) {
      return LaunchSoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, 2, algorithm>(
          stream, load, store, rows, cols);
    } else {
      return LaunchSoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, 1, algorithm>(
          stream, load, store, rows, cols);
    }
  }
};

template<typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
inline cudaError_t DispatchSoftmaxBlockUncachedImpl(cudaStream_t stream, LOAD load, STORE store,
                                                    const int64_t rows, const int64_t cols) {
  return DispatchSoftmaxBlockUncachedImplPackSize<LOAD, STORE, ComputeType, algorithm>()(
      stream, load, store, rows, cols);
}

template<typename LOAD, typename STORE, typename ComputeType>
inline typename std::enable_if<!std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchSoftmax(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                const int64_t cols) {
  if (cols < 1024) {
    return DispatchSoftmaxWarpImpl<LOAD, STORE, ComputeType, Algorithm::kSoftmax>(
        stream, load, store, rows, cols);
  } else {
    bool dispatch_smem_impl_success;
    {
      cudaError_t err =
          TryDispatchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, Algorithm::kSoftmax>(
              stream, load, store, rows, cols, &dispatch_smem_impl_success);
      if (err != cudaSuccess) { return err; }
    }
    if (!dispatch_smem_impl_success) {
      return DispatchSoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, Algorithm::kSoftmax>(
          stream, load, store, rows, cols);
    }
    return cudaSuccess;
  }
}

template<typename LOAD, typename STORE, typename ComputeType>
inline typename std::enable_if<std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchSoftmax(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                const int64_t cols) {
  return DispatchSoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, Algorithm::kSoftmax>(
      stream, load, store, rows, cols);
}

template<typename LOAD, typename STORE, typename ComputeType>
inline typename std::enable_if<!std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchLogSoftmax(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                   const int64_t cols) {
  if (cols <= 1024) {
    return DispatchSoftmaxWarpImpl<LOAD, STORE, ComputeType, Algorithm::kLogSoftmax>(
        stream, load, store, rows, cols);
  } else {
    bool dispatch_smem_impl_success;
    {
      cudaError_t err =
          TryDispatchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, Algorithm::kLogSoftmax>(
              stream, load, store, rows, cols, &dispatch_smem_impl_success);
      if (err != cudaSuccess) { return err; }
    }
    if (!dispatch_smem_impl_success) {
      return DispatchSoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, Algorithm::kLogSoftmax>(
          stream, load, store, rows, cols);
    }
    return cudaSuccess;
  }
}

template<typename LOAD, typename STORE, typename ComputeType>
inline typename std::enable_if<std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchLogSoftmax(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                   const int64_t cols) {
  return DispatchSoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, Algorithm::kLogSoftmax>(
      stream, load, store, rows, cols);
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         int cols_per_thread, int thread_group_width, int rows_per_access, bool padding,
         Algorithm algorithm>
__global__ void SoftmaxGradWarpImpl(LOAD_Y load_y, LOAD_DY load_dy, STORE store, const int64_t rows,
                                    const int64_t cols) {
  static_assert(cols_per_thread % pack_size == 0, "");
  constexpr int pack_per_thread = cols_per_thread / pack_size;
  assert(cols <= cols_per_thread * thread_group_width);
  static_assert(thread_group_width <= kWarpSize, "");
  static_assert(kWarpSize % thread_group_width == 0, "");
  ComputeType y_buf[rows_per_access][cols_per_thread];
  ComputeType dy_buf[rows_per_access][cols_per_thread];
  const int global_thread_group_id = blockIdx.x * blockDim.y + threadIdx.y;
  const int num_global_thread_group = gridDim.x * blockDim.y;
  const int lane_id = threadIdx.x;
  const int64_t step = num_global_thread_group * rows_per_access;
  for (int64_t row = global_thread_group_id * rows_per_access; row < rows; row += step) {
    ComputeType thread_sum[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      thread_sum[row_id] = 0;
      ComputeType* row_y_buf = y_buf[row_id];
      ComputeType* row_dy_buf = dy_buf[row_id];
#pragma unroll
      for (int pack_id = 0; pack_id < pack_per_thread; ++pack_id) {
        const int pack_offset = pack_id * pack_size;
        const int col = (pack_id * thread_group_width + lane_id) * pack_size;
        if (!padding || col < cols) {
          load_y.template load<pack_size>(row_y_buf + pack_offset, row + row_id, col);
          load_dy.template load<pack_size>(row_dy_buf + pack_offset, row + row_id, col);
#pragma unroll
          for (int i = 0; i < pack_size; ++i) {
            if (algorithm == Algorithm::kSoftmax) {
              thread_sum[row_id] += row_y_buf[pack_offset + i] * row_dy_buf[pack_offset + i];
            } else if (algorithm == Algorithm::kLogSoftmax) {
              thread_sum[row_id] += row_dy_buf[pack_offset + i];
            } else {
              __trap();
            }
          }
        }
      }
    }
    ComputeType warp_sum[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      warp_sum[row_id] = WarpAllReduce<SumOp, ComputeType, thread_group_width>(thread_sum[row_id]);
    }
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      ComputeType* row_y_buf = y_buf[row_id];
      ComputeType* row_dy_buf = dy_buf[row_id];
#pragma unroll
      for (int pack_id = 0; pack_id < pack_per_thread; ++pack_id) {
        const int pack_offset = pack_id * pack_size;
        const int col = (pack_id * thread_group_width + lane_id) * pack_size;
        if (!padding || col < cols) {
          for (int i = 0; i < pack_size; ++i) {
            if (algorithm == Algorithm::kSoftmax) {
              row_dy_buf[pack_offset + i] =
                  (row_dy_buf[pack_offset + i] - warp_sum[row_id]) * row_y_buf[pack_offset + i];
            } else if (algorithm == Algorithm::kLogSoftmax) {
              row_dy_buf[pack_offset + i] -= Exp(row_y_buf[pack_offset + i]) * warp_sum[row_id];
            } else {
              __trap();
            }
          }
          store.template store<pack_size>(row_dy_buf + pack_offset, row + row_id, col);
        }
      }
    }
  }
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         int cols_per_thread, int thread_group_width, int rows_per_access, bool padding,
         Algorithm algorithm>
inline cudaError_t LaunchSoftmaxGradWarpImpl(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy,
                                             STORE store, const int64_t rows, const int64_t cols) {
  constexpr int block_size = 128;
  constexpr int waves = 32;
  static_assert(block_size % thread_group_width == 0, "");
  constexpr int thread_groups_per_block = block_size / thread_group_width;
  dim3 block_dim(thread_group_width, thread_groups_per_block);
  const int64_t num_blocks =
      (rows / rows_per_access + thread_groups_per_block - 1) / thread_groups_per_block;
  int grid_dim_x;
  {
    cudaError_t err = GetNumBlocks(block_size, num_blocks, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  SoftmaxGradWarpImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, cols_per_thread,
                      thread_group_width, rows_per_access, padding, algorithm>
      <<<grid_dim_x, block_dim, 0, stream>>>(load_y, load_dy, store, rows, cols);
  return cudaPeekAtLastError();
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         int cols_per_thread, int thread_group_width, int rows_per_access, Algorithm algorithm>
inline cudaError_t DispatchSoftmaxGradWarpImplPadding(cudaStream_t stream, LOAD_Y load_y,
                                                      LOAD_DY load_dy, STORE store,
                                                      const int64_t rows, const int64_t cols) {
  if (cols == cols_per_thread * thread_group_width) {
    return LaunchSoftmaxGradWarpImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size,
                                     cols_per_thread, thread_group_width, rows_per_access, false,
                                     algorithm>(stream, load_y, load_dy, store, rows, cols);
  } else {
    return LaunchSoftmaxGradWarpImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size,
                                     cols_per_thread, thread_group_width, rows_per_access, true,
                                     algorithm>(stream, load_y, load_dy, store, rows, cols);
  }
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         Algorithm algorithm>
typename std::enable_if<pack_size == 1, cudaError_t>::type DispatchSoftmaxGradWarpImplCols(
    cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store, const int64_t rows,
    const int64_t cols) {
  if (cols <= 0) { return cudaErrorInvalidValue; }
#define DEFINE_ONE_ELIF(thread_group_width)                                                     \
  else if (cols <= (thread_group_width)*pack_size) {                                            \
    if (rows % 2 == 0) {                                                                        \
      return DispatchSoftmaxGradWarpImplPadding<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, \
                                                pack_size, thread_group_width, 2, algorithm>(   \
          stream, load_y, load_dy, store, rows, cols);                                          \
    } else {                                                                                    \
      return DispatchSoftmaxGradWarpImplPadding<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, \
                                                pack_size, thread_group_width, 1, algorithm>(   \
          stream, load_y, load_dy, store, rows, cols);                                          \
    }                                                                                           \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                                       \
  else if (cols <= (col)*kWarpSize) {                                                              \
    return DispatchSoftmaxGradWarpImplPadding<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, col, \
                                              kWarpSize, 1, algorithm>(stream, load_y, load_dy,    \
                                                                       store, rows, cols);         \
  }
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(3)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(5)
  DEFINE_ONE_ELIF(6)
  DEFINE_ONE_ELIF(7)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(9)
  DEFINE_ONE_ELIF(10)
  DEFINE_ONE_ELIF(11)
  DEFINE_ONE_ELIF(12)
  DEFINE_ONE_ELIF(13)
  DEFINE_ONE_ELIF(14)
  DEFINE_ONE_ELIF(15)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(17)
  DEFINE_ONE_ELIF(18)
  DEFINE_ONE_ELIF(19)
  DEFINE_ONE_ELIF(20)
  DEFINE_ONE_ELIF(21)
  DEFINE_ONE_ELIF(22)
  DEFINE_ONE_ELIF(23)
  DEFINE_ONE_ELIF(24)
  DEFINE_ONE_ELIF(25)
  DEFINE_ONE_ELIF(26)
  DEFINE_ONE_ELIF(27)
  DEFINE_ONE_ELIF(28)
  DEFINE_ONE_ELIF(29)
  DEFINE_ONE_ELIF(30)
  DEFINE_ONE_ELIF(31)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
  else {
    return cudaErrorInvalidValue;
  }
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         Algorithm algorithm>
typename std::enable_if<pack_size == 2, cudaError_t>::type DispatchSoftmaxGradWarpImplCols(
    cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store, const int64_t rows,
    const int64_t cols) {
  if (cols <= 0) { return cudaErrorInvalidValue; }
#define DEFINE_ONE_ELIF(thread_group_width)                                                     \
  else if (cols <= (thread_group_width)*pack_size) {                                            \
    if (rows % 2 == 0) {                                                                        \
      return DispatchSoftmaxGradWarpImplPadding<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, \
                                                pack_size, thread_group_width, 2, algorithm>(   \
          stream, load_y, load_dy, store, rows, cols);                                          \
    } else {                                                                                    \
      return DispatchSoftmaxGradWarpImplPadding<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, \
                                                pack_size, thread_group_width, 1, algorithm>(   \
          stream, load_y, load_dy, store, rows, cols);                                          \
    }                                                                                           \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                                       \
  else if (cols <= (col)*kWarpSize) {                                                              \
    return DispatchSoftmaxGradWarpImplPadding<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, col, \
                                              kWarpSize, 1, algorithm>(stream, load_y, load_dy,    \
                                                                       store, rows, cols);         \
  }
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(6)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(10)
  DEFINE_ONE_ELIF(12)
  DEFINE_ONE_ELIF(14)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(18)
  DEFINE_ONE_ELIF(20)
  DEFINE_ONE_ELIF(22)
  DEFINE_ONE_ELIF(24)
  DEFINE_ONE_ELIF(26)
  DEFINE_ONE_ELIF(28)
  DEFINE_ONE_ELIF(30)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
  else {
    return cudaErrorInvalidValue;
  }
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType,
         Algorithm algorithm>
struct DispatchSoftmaxGradWarpImplPackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store,
                         const int64_t rows, const int64_t cols) {
    if (cols % 2 == 0) {
      return DispatchSoftmaxGradWarpImplCols<LOAD_Y, LOAD_DY, STORE, ComputeType, 2, algorithm>(
          stream, load_y, load_dy, store, rows, cols);
    } else {
      return DispatchSoftmaxGradWarpImplCols<LOAD_Y, LOAD_DY, STORE, ComputeType, 1, algorithm>(
          stream, load_y, load_dy, store, rows, cols);
    }
  }
};

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType,
         Algorithm algorithm>
inline cudaError_t DispatchSoftmaxGradWarpImpl(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy,
                                               STORE store, const int64_t rows,
                                               const int64_t cols) {
  return DispatchSoftmaxGradWarpImplPackSize<LOAD_Y, LOAD_DY, STORE, ComputeType, algorithm>()(
      stream, load_y, load_dy, store, rows, cols);
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         int block_size, Algorithm algorithm>
__global__ void SoftmaxGradBlockSMemImpl(LOAD_Y load_y, LOAD_DY load_dy, STORE store,
                                         const int64_t rows, const int64_t cols) {
  extern __shared__ __align__(sizeof(double)) unsigned char grad_shared_buf[];
  auto* y_buf = reinterpret_cast<ComputeType*>(grad_shared_buf);
  auto* dy_buf = y_buf + cols;
  const int tid = threadIdx.x;
  assert(cols % pack_size == 0);
  const int num_packs = cols / pack_size;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    ComputeType thread_sum = 0;
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType y_pack[pack_size];
      ComputeType dy_pack[pack_size];
      load_y.template load<pack_size>(y_pack, row, pack_id * pack_size);
      load_dy.template load<pack_size>(dy_pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        y_buf[i * num_packs + pack_id] = y_pack[i];
        dy_buf[i * num_packs + pack_id] = dy_pack[i];
        if (algorithm == Algorithm::kSoftmax) {
          thread_sum += y_pack[i] * dy_pack[i];
        } else if (algorithm == Algorithm::kLogSoftmax) {
          thread_sum += dy_pack[i];
        } else {
          __trap();
        }
      }
    }
    const ComputeType row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        if (algorithm == Algorithm::kSoftmax) {
          pack[i] = (dy_buf[i * num_packs + pack_id] - row_sum) * y_buf[i * num_packs + pack_id];
        } else if (algorithm == Algorithm::kLogSoftmax) {
          pack[i] = dy_buf[i * num_packs + pack_id] - Exp(y_buf[i * num_packs + pack_id]) * row_sum;
        } else {
          __trap();
        }
      }
      store.template store<pack_size>(pack, row, pack_id * pack_size);
    }
  }
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         int block_size, Algorithm algorithm>
inline cudaError_t LaunchSoftmaxGradBlockSMemImpl(cudaStream_t stream, LOAD_Y load_y,
                                                  LOAD_DY load_dy, STORE store, int smem,
                                                  const int64_t rows, const int64_t cols) {
  constexpr int waves = 32;
  int grid_dim_x;
  {
    cudaError_t err = GetNumBlocks(block_size, rows, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  SoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size, algorithm>
      <<<grid_dim_x, block_size, smem, stream>>>(load_y, load_dy, store, rows, cols);
  return cudaPeekAtLastError();
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         Algorithm algorithm>
inline cudaError_t TryDispatchSoftmaxGradBlockSMemImplBlockSize(cudaStream_t stream, LOAD_Y load_y,
                                                                LOAD_DY load_dy, STORE store,
                                                                const int64_t rows,
                                                                const int64_t cols, bool* success) {
  constexpr int block_size_conf_1 = 128;
  constexpr int block_size_conf_2 = 256;
  constexpr int block_size_conf_3 = 512;
  constexpr int block_size_conf_4 = 1024;
  const size_t smem = cols * sizeof(ComputeType) * 2;
  int max_active_blocks_conf_1;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_1,
        SoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size_conf_1,
                                 algorithm>,
        block_size_conf_1, smem);
    if (err != cudaSuccess) { return err; }
  }
  if (max_active_blocks_conf_1 <= 0) {
    *success = false;
    return cudaSuccess;
  }
  int max_active_blocks_conf_4;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_4,
        SoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size_conf_4,
                                 algorithm>,
        block_size_conf_4, smem);
    if (err != cudaSuccess) { return err; }
  }
  if (max_active_blocks_conf_4 == max_active_blocks_conf_1) {
    *success = true;
    return LaunchSoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size,
                                          block_size_conf_4, algorithm>(stream, load_y, load_dy,
                                                                        store, smem, rows, cols);
  }
  int max_active_blocks_conf_3;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_3,
        SoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size_conf_3,
                                 algorithm>,
        block_size_conf_3, smem);
    if (err != cudaSuccess) { return err; }
  }
  if (max_active_blocks_conf_3 == max_active_blocks_conf_1) {
    *success = true;
    return LaunchSoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size,
                                          block_size_conf_3, algorithm>(stream, load_y, load_dy,
                                                                        store, smem, rows, cols);
  }
  int max_active_blocks_conf_2;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_2,
        SoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size_conf_2,
                                 algorithm>,
        block_size_conf_2, smem);
    if (err != cudaSuccess) { return err; }
  }
  if (max_active_blocks_conf_2 == max_active_blocks_conf_1) {
    *success = true;
    return LaunchSoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size,
                                          block_size_conf_2, algorithm>(stream, load_y, load_dy,
                                                                        store, smem, rows, cols);
  }
  *success = true;
  return LaunchSoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size,
                                        block_size_conf_1, algorithm>(stream, load_y, load_dy,
                                                                      store, smem, rows, cols);
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType,
         Algorithm algorithm>
struct TryDispatchSoftmaxGradBlockSMemImplPackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store,
                         const int64_t rows, const int64_t cols, bool* success) {
    if (cols % 2 == 0) {
      return TryDispatchSoftmaxGradBlockSMemImplBlockSize<LOAD_Y, LOAD_DY, STORE, ComputeType, 2,
                                                          algorithm>(stream, load_y, load_dy, store,
                                                                     rows, cols, success);
    } else {
      return TryDispatchSoftmaxGradBlockSMemImplBlockSize<LOAD_Y, LOAD_DY, STORE, ComputeType, 1,
                                                          algorithm>(stream, load_y, load_dy, store,
                                                                     rows, cols, success);
    }
  }
};

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType,
         Algorithm algorithm>
inline cudaError_t TryDispatchSoftmaxGradBlockSMemImpl(cudaStream_t stream, LOAD_Y load_y,
                                                       LOAD_DY load_dy, STORE store,
                                                       const int64_t rows, const int64_t cols,
                                                       bool* success) {
  return TryDispatchSoftmaxGradBlockSMemImplPackSize<LOAD_Y, LOAD_DY, STORE, ComputeType,
                                                     algorithm>()(stream, load_y, load_dy, store,
                                                                  rows, cols, success);
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         int block_size, Algorithm algorithm>
__global__ void SoftmaxGradBlockUncachedImpl(LOAD_Y load_y, LOAD_DY load_dy, STORE store,
                                             const int64_t rows, const int64_t cols) {
  const int tid = threadIdx.x;
  assert(cols % pack_size == 0);
  const int num_packs = cols / pack_size;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    ComputeType thread_sum = 0;
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType y_pack[pack_size];
      ComputeType dy_pack[pack_size];
      load_y.template load<pack_size>(y_pack, row, pack_id * pack_size);
      load_dy.template load<pack_size>(dy_pack, row, pack_id * pack_size);

#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        if (algorithm == Algorithm::kSoftmax) {
          thread_sum += y_pack[i] * dy_pack[i];
        } else if (algorithm == Algorithm::kLogSoftmax) {
          thread_sum += dy_pack[i];
        } else {
          __trap();
        }
      }
    }
    const ComputeType row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType y_pack[pack_size];
      ComputeType dy_pack[pack_size];
      load_y.template load<pack_size>(y_pack, row, pack_id * pack_size);
      load_dy.template load<pack_size>(dy_pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        if (algorithm == Algorithm::kSoftmax) {
          dy_pack[i] = (dy_pack[i] - row_sum) * y_pack[i];
        } else if (algorithm == Algorithm::kLogSoftmax) {
          dy_pack[i] -= Exp(y_pack[i]) * row_sum;
        } else {
          __trap();
        }
      }
      store.template store<pack_size>(dy_pack, row, pack_id * pack_size);
    }
  }
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         Algorithm algorithm>
inline cudaError_t LaunchSoftmaxGradBlockUncachedImpl(cudaStream_t stream, LOAD_Y load_y,
                                                      LOAD_DY load_dy, STORE store,
                                                      const int64_t rows, const int64_t cols) {
  constexpr int block_size = 1024;
  constexpr int waves = 32;
  int grid_dim_x;
  {
    cudaError_t err = GetNumBlocks(block_size, rows, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  SoftmaxGradBlockUncachedImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size,
                               algorithm>
      <<<grid_dim_x, block_size, 0, stream>>>(load_y, load_dy, store, rows, cols);
  return cudaPeekAtLastError();
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType,
         Algorithm algorithm>
struct DispatchSoftmaxGradBlockUncachedImplPackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store,
                         const int64_t rows, const int64_t cols) {
    if (cols % 2 == 0 && cols > kWarpSize) {
      return LaunchSoftmaxGradBlockUncachedImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, 2, algorithm>(
          stream, load_y, load_dy, store, rows, cols);
    } else {
      return LaunchSoftmaxGradBlockUncachedImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, 1, algorithm>(
          stream, load_y, load_dy, store, rows, cols);
    }
  }
};

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType,
         Algorithm algorithm>
inline cudaError_t DispatchSoftmaxGradBlockUncachedImpl(cudaStream_t stream, LOAD_Y load_y,
                                                        LOAD_DY load_dy, STORE store,
                                                        const int64_t rows, const int64_t cols) {
  return DispatchSoftmaxGradBlockUncachedImplPackSize<LOAD_Y, LOAD_DY, STORE, ComputeType,
                                                      algorithm>()(stream, load_y, load_dy, store,
                                                                   rows, cols);
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType>
inline typename std::enable_if<!std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchSoftmaxGrad(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store,
                    const int64_t rows, const int64_t cols) {
  if (cols <= 1024) {
    return DispatchSoftmaxGradWarpImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, Algorithm::kSoftmax>(
        stream, load_y, load_dy, store, rows, cols);
  } else {
    bool dispatch_smem_impl_success;
    {
      cudaError_t err = TryDispatchSoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType,
                                                            Algorithm::kSoftmax>(
          stream, load_y, load_dy, store, rows, cols, &dispatch_smem_impl_success);
      if (err != cudaSuccess) { return err; }
    }
    if (!dispatch_smem_impl_success) {
      return DispatchSoftmaxGradBlockUncachedImpl<LOAD_Y, LOAD_DY, STORE, ComputeType,
                                                  Algorithm::kSoftmax>(stream, load_y, load_dy,
                                                                       store, rows, cols);
    }
    return cudaSuccess;
  }
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType>
inline typename std::enable_if<std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchSoftmaxGrad(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store,
                    const int64_t rows, const int64_t cols) {
  return DispatchSoftmaxGradBlockUncachedImpl<LOAD_Y, LOAD_DY, STORE, ComputeType,
                                              Algorithm::kSoftmax>(stream, load_y, load_dy, store,
                                                                   rows, cols);
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType>
inline typename std::enable_if<!std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchLogSoftmaxGrad(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store,
                       const int64_t rows, const int64_t cols) {
  if (cols <= 1024) {
    return DispatchSoftmaxGradWarpImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, Algorithm::kLogSoftmax>(
        stream, load_y, load_dy, store, rows, cols);
  } else {
    bool dispatch_smem_impl_success;
    {
      cudaError_t err = TryDispatchSoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType,
                                                            Algorithm::kLogSoftmax>(
          stream, load_y, load_dy, store, rows, cols, &dispatch_smem_impl_success);
      if (err != cudaSuccess) { return err; }
    }
    if (!dispatch_smem_impl_success) {
      return DispatchSoftmaxGradBlockUncachedImpl<LOAD_Y, LOAD_DY, STORE, ComputeType,
                                                  Algorithm::kLogSoftmax>(stream, load_y, load_dy,
                                                                          store, rows, cols);
    }
    return cudaSuccess;
  }
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType>
inline typename std::enable_if<std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchLogSoftmaxGrad(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store,
                       const int64_t rows, const int64_t cols) {
  return DispatchSoftmaxGradBlockUncachedImpl<LOAD_Y, LOAD_DY, STORE, ComputeType,
                                              Algorithm::kLogSoftmax>(stream, load_y, load_dy,
                                                                      store, rows, cols);
}

// NdIndexHelper
#if defined(__CUDACC__)
#define OF_DEVICE_FUNC __device__ __host__ __forceinline__
#else
#define OF_DEVICE_FUNC inline
#endif

// 定义 NdIndexOffsetHelper 类来做多维张量的坐标映射
template<typename T, int N>
class NdIndexOffsetHelper {
 public:
  NdIndexOffsetHelper() = default;

  // 这段代码是一个构造函数模板，它的功能是根据给定的多维数组的维度初始化一个用于计算偏移量的辅助类。
  // 它的参数是一个可变参数模板，表示可以接受任意个数和类型的参数，但是第一个参数必须是一个T类型的值，
  // 后面的参数必须是一个参数包。它的函数体是一个调用另一个函数的语句，这个函数是用于初始化步长数组的。
  // 它使用了一个宏OF_DEVICE_FUNC，这个宏可能是用于指定函数在哪个设备上运行的，比如GPU或者CPU。
  // 这个构造函数模板可能是用于实现一些多维数组的操作，比如插值或者转置。
  template<class... Ts>
  OF_DEVICE_FUNC explicit NdIndexOffsetHelper(T d0, Ts... dims) {
    constexpr int n = 1 + sizeof...(dims);
    static_assert(n <= N, "");
    T dims_arr[n] = {d0, static_cast<T>(dims)...};
    // 初始化strides信息
    InitStrides(dims_arr, n);
  }

  // 从一个类型为T的数组进行构造，并初始化strides信息，注意这里的strides长度设置为N
  OF_DEVICE_FUNC explicit NdIndexOffsetHelper(const T* dims) { InitStrides(dims, N); }

  // 从一个类型为U的数组进行构造，并初始化strides信息，注意这里的strides长度设置为N
  template<typename U>
  OF_DEVICE_FUNC explicit NdIndexOffsetHelper(const U* dims) {
    T dims_arr[N];
    for (int i = 0; i < N; ++i) { dims_arr[i] = dims[i]; }
    InitStrides(dims_arr, N);
  }

  // 从一个类型为T的数组进行构造，并初始化strides信息，注意这里的strides长度自定义为n
  OF_DEVICE_FUNC explicit NdIndexOffsetHelper(const T* dims, int n) { InitStrides(dims, n); }

  // 从一个类型为U的数组进行构造，并初始化strides信息，注意这里的strides长度自定义为n
  template<typename U>
  OF_DEVICE_FUNC explicit NdIndexOffsetHelper(const U* dims, int n) {
    T dims_arr[N];
    for (int i = 0; i < N; ++i) {
      if (i < n) { dims_arr[i] = dims[i]; }
    }
    InitStrides(dims_arr, n);
  }

  // virtual 表示这是一个虚析构函数，用于在删除基类指针时调用派生类的析构函数，避免内存泄漏。
  // ~NdIndexOffsetHelper() 表示这是 NdIndexOffsetHelper 类的析构函数，用于释放类对象占用的资源。
  // = default; 表示这是一个默认的析构函数，没有自定义的操作，让编译器自动生成。
  virtual ~NdIndexOffsetHelper() = default;

  // 这段代码是一个模板函数，用于根据一个N维索引数组计算一个一维偏移量。函数的参数和返回值都是模板类型T，可以是任意数值类型。函数的主要步骤如下：
  OF_DEVICE_FUNC T NdIndexToOffset(const T* index) const {
    // 定义一个变量offset，初始值为0，用于存储最终的偏移量。
    T offset = 0;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    // 使用一个循环，从0到N-1，遍历索引数组的每一个元素。在循环中，使用一个数组stride_，
    // 用于存储每一个维度的步长，即每增加一个单位的索引，偏移量增加多少。
    // 将索引数组的第i个元素乘以步长数组的第i个元素，然后累加到offset上。
    for (int i = 0; i < N; ++i) { offset += index[i] * stride_[i]; }
    return offset;
  }
  
  // 类似上面，不过这里是从0到n进行循环
  OF_DEVICE_FUNC T NdIndexToOffset(const T* index, int n) const {
    assert(n <= N);
    T offset = 0;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      if (i < n) { offset += index[i] * stride_[i]; }
    }
    return offset;
  }

  // 这段代码是一个函数模板，它的功能是根据给定的多维索引计算一个一维偏移量。
  // 它的参数是一个可变参数模板，表示可以接受任意个数和类型的参数，但是第一个
  // 参数必须是一个T类型的值，后面的参数必须是一个参数包。它的返回值也是一个T类型的值。
  // 它的函数体是一个循环，用于累加每个维度的索引乘以对应的步长，得到最终的偏移量。
  // 它使用了一个宏OF_DEVICE_FUNC，这个宏可能是用于指定函数在哪个设备上运行的，
  // 比如GPU或者CPU。这个函数模板可能是用于实现一些多维数组的操作，比如插值或者转置。
  template<class... Ts>
  OF_DEVICE_FUNC T NdIndexToOffset(T d0, Ts... others) const {
    constexpr int n = 1 + sizeof...(others);
    static_assert(n <= N, "");
    T index[n] = {d0, others...};
    T offset = 0;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < n - 1; ++i) { offset += index[i] * stride_[i]; }
    if (n == N) {
      offset += index[n - 1];
    } else {
      offset += index[n - 1] * stride_[n - 1];
    }
    return offset;
  }

  // 这段代码是一个成员函数模板，它的功能是根据给定的一维偏移量计算一个多维索引。
  // 它的参数是一个T类型的值，表示偏移量，和一个T类型的指针，表示索引数组。
  // 它的函数体是一个循环，用于逐个维度地计算索引值，然后更新剩余的偏移量。
  // 它使用了一个宏OF_DEVICE_FUNC，这个宏可能是用于指定函数在哪个设备上运行的，比如GPU或者CPU 。
  // 这个成员函数模板可能是用于实现一些多维数组的操作，比如插值或者转置。
  OF_DEVICE_FUNC void OffsetToNdIndex(T offset, T* index) const {
    T remaining = offset;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < N - 1; ++i) {
      const T idx = remaining / stride_[i];
      index[i] = idx;
      remaining = remaining - idx * stride_[i];
    }
    index[N - 1] = remaining;
  }

  // 这段代码是用C++语言编写的，它定义了一个名为OffsetToNdIndex的函数，
  // 该函数的功能是将一维的偏移量转换为高维的索引1。这个函数是OneFlow的内部类，
  // OneFlow是一个深度学习框架，它支持分布式训练和推理。这个函数的参数有三个，分别是：
  // offset: 一个整数，表示一维的偏移量。
  // index: 一个整数数组，用于存储转换后的高维索引。
  // n: 一个整数，表示高维的维度数，不能超过N，N是一个常量。
  // 函数的主要逻辑是：
  // 首先，用一个变量remaining存储offset的值。
  // 然后，用一个循环遍历从0到N-1的整数i。
  // 在循环中，如果i小于n，那么就用remaining除以stride_[i]得到一个整数idx，这个stride_[i]是一个预定义的数组，表示每个维度的步长。
  // 然后，将idx赋值给index[i]，并用remaining减去idx乘以stride_[i]，更新remaining的值。
  // 最后，结束循环。
  // 这个函数的作用是将一维的偏移量映射到高维的索引，这在深度学习中有很多应用，比如Unfold和Fold算子，它们可以将图像的局部区域转换为一维的向量，或者反过来。
  OF_DEVICE_FUNC void OffsetToNdIndex(T offset, T* index, int n) const {
    assert(n <= N);
    T remaining = offset;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      if (i < n) {
        const T idx = remaining / stride_[i];
        index[i] = idx;
        remaining = remaining - idx * stride_[i];
      }
    }
  }

  // 它定义了一个名为OffsetToNdIndex的函数模板，该函数模板的功能和之前的函数类似，
  // 也是将一维的偏移量转换为高维的索引，但是它可以接受不同数量的参数。这个函数模板的参数有两个，分别是：
  // offset: 一个整数，表示一维的偏移量。
  // d0, others: 一系列的整数引用，用于存储转换后的高维索引。
  // 函数模板的主要逻辑是：
  // 首先，用一个常量n表示参数的个数，它等于1加上others的个数。
  // 然后，用一个静态断言检查n是否小于等于N，N是一个常量。
  // 然后，用一个指针数组index存储d0和others的地址。
  // 然后，用一个变量remaining存储offset的值。
  // 然后，用一个循环遍历从0到n-2的整数i。
  // 在循环中，如果i小于n-1，那么就用remaining除以stride_[i]得到一个整数idx，这个stride_[i]是一个预定义的数组，表示每个维度的步长。
  // 然后，将idx赋值给index[i]所指向的变量，并用remaining减去idx乘以stride_[i]，更新remaining的值。
  // 最后，根据n和N的关系，分两种情况处理最后一个参数：
  // 如果n等于N，那么就将remaining赋值给index[n-1]所指向的变量。
  // 如果n小于N，那么就用remaining除以stride_[n-1]得到一个整数，赋值给index[n-1]所指向的变量。
  // 这个函数模板的作用是将一维的偏移量映射到高维的索引，它可以根据不同的参数个数进行重载，这是C++的一种泛型编程的特性
  template<class... Ts>
  OF_DEVICE_FUNC void OffsetToNdIndex(T offset, T& d0, Ts&... others) const {
    constexpr int n = 1 + sizeof...(others);
    static_assert(n <= N, "");
    T* index[n] = {&d0, &others...};
    T remaining = offset;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < n - 1; ++i) {
      const T idx = remaining / stride_[i];
      *index[i] = idx;
      remaining = remaining - idx * stride_[i];
    }
    if (n == N) {
      *index[n - 1] = remaining;
    } else {
      *index[n - 1] = remaining / stride_[n - 1];
    }
  }

  OF_DEVICE_FUNC constexpr int Size() const { return N; }

 protected:
  // 这段代码也是用C++语言编写的，它定义了一个名为InitStrides的函数，
  // 该函数的功能是初始化stride_数组，该数组表示每个维度的步长。这个函数的参数有两个，分别是：
  // dims: 一个整数数组，表示高维的维度大小。
  // n: 一个整数，表示高维的维度数，不能超过N，N是一个常量。
  // 函数的主要逻辑是：
  // 首先，用一个循环遍历从n-1到N-1的整数i。
  // 在循环中，将stride_[i]赋值为1。
  // 然后，用一个循环遍历从n-2到0的整数i。
  // 在循环中，将stride_[i]赋值为dims[i+1]乘以stride_[i+1]。
  // 这个函数的作用是计算每个维度的步长，这在之前的OffsetToNdIndex函数中有用到，它可以根据不同的维度大小和维度数进行初始化。

  OF_DEVICE_FUNC void InitStrides(const T* dims, const int n) {
    for (int i = n - 1; i < N; ++i) { stride_[i] = 1; }
    for (int i = n - 2; i >= 0; --i) { stride_[i] = dims[i + 1] * stride_[i + 1]; }
  }

  T stride_[N];
};

// fused_softmax
// 这个函数的功能是简化两个张量的广播维度，使它们能够进行算术运算。
// 广播是一种在不同形状的张量之间进行运算的方法，它会将较小的张量扩展到较大的张量的形状上。
// 这段代码的输入参数有：
// num_a_dims: 张量A的维度数
// a_dims: 张量A的每个维度的大小
// num_b_dims: 张量B的维度数
// b_dims: 张量B的每个维度的大小
// simplified_num_dims: 简化后的广播维度数
// simplified_a_dims: 简化后的张量A的每个维度的大小
// simplified_b_dims: 简化后的张量B的每个维度的大小
// 这段代码的主要逻辑是：
// 首先，找到两个张量的最大维度数，然后用一个闭包函数MakeGetDim来获取每个张量在每个维度上的大小，如果维度数不足，就用1来填充。
// 然后，遍历每个维度，计算两个张量在该维度上的最大值，作为广播后的维度大小。
// 如果该维度大小为1，就跳过，否则就判断是否可以和上一个维度合并，如果可以，就乘以上一个维度的大小，
// 如果不可以，就添加到简化后的维度数组中，并记录是否是广播维度。
// 最后，返回简化后的广播维度数和两个张量的简化后的维度大小。
// 这段代码的目的是为了减少广播运算的开销，提高深度学习的性能。
inline void SimplifyBroadcastDims(size_t num_a_dims, const int64_t* a_dims, size_t num_b_dims,
                                  const int64_t* b_dims, size_t* simplified_num_dims,
                                  int64_t* simplified_a_dims, int64_t* simplified_b_dims) {
  const size_t num_max_dims = std::max(num_a_dims, num_b_dims);
  auto MakeGetDim = [num_max_dims](size_t num_dims, const int64_t* dims) {
    const int64_t num_padding_dims = num_max_dims - num_dims;
    return [num_padding_dims, dims](size_t index) {
      return index < num_padding_dims ? 1 : dims[index - num_padding_dims];
    };
  };
  auto GetADim = MakeGetDim(num_a_dims, a_dims);
  auto GetBDim = MakeGetDim(num_b_dims, b_dims);
  *simplified_num_dims = 0;
  bool prev_broadcast_a = false;
  bool prev_broadcast_b = false;
  for (int64_t i = 0; i < num_max_dims; ++i) {
    const int64_t a_dim = GetADim(i);
    const int64_t b_dim = GetBDim(i);
    const int64_t broadcast_dim = std::max(a_dim, b_dim);
    // CHECK_GT(broadcast_dim, 0);
    const bool broadcast_a = (a_dim == 1);
    const bool broadcast_b = (b_dim == 1);
    // CHECK((a_dim == broadcast_dim) || broadcast_a);
    // CHECK((b_dim == broadcast_dim) || broadcast_b);
    if (broadcast_dim == 1) {
      continue;
    } else if (*simplified_num_dims != 0
               && (prev_broadcast_a == broadcast_a && prev_broadcast_b == broadcast_b)) {
      simplified_a_dims[*simplified_num_dims - 1] *= a_dim;
      simplified_b_dims[*simplified_num_dims - 1] *= b_dim;
    } else {
      simplified_a_dims[*simplified_num_dims] = a_dim;
      simplified_b_dims[*simplified_num_dims] = b_dim;
      *simplified_num_dims += 1;
      prev_broadcast_a = broadcast_a;
      prev_broadcast_b = broadcast_b;
    }
  }
}

// 它的功能是定义一个结构体，用于存储广播掩码softmax的参数。
// 广播掩码softmax是一种在不同形状的张量之间进行softmax运算的方法，
// 它会将较小的张量扩展到较大的张量的形状上，并用一个掩码张量来指定哪些位置需要计算softmax，
// 哪些位置需要填充一个固定的值。这个结构体的成员变量有：
// src_index_helper: 一个用于计算源张量索引的辅助类
// mask_index_helper: 一个用于计算掩码张量索引的辅助类
// mask_dims: 掩码张量的每个维度的大小
// row_size: 每一行的元素个数
// fill: 填充的值
// scale: softmax的缩放因子
// 这个结构体的目的是为了方便在内核函数中使用广播掩码softmax的参数，提高深度学习的性能。
template<size_t num_dims, typename IndexType>
struct BroadcastMaskSoftmaxParams {
  NdIndexOffsetHelper<IndexType, num_dims> src_index_helper;
  NdIndexOffsetHelper<IndexType, num_dims> mask_index_helper;
  const int64_t* mask_dims{};
  int64_t row_size;
  float fill;
  float scale;
};

// 它的功能是定义一个结构体，用于存储逐元素掩码softmax的参数。
// 逐元素掩码softmax是一种在相同形状的张量之间进行softmax运算的方法，
// 它会用一个掩码张量来指定哪些位置需要计算softmax，哪些位置需要填充一个固定的值。
// 这个结构体的成员变量有：
// row_size: 每一行的元素个数
// fill: 填充的值
// scale: softmax的缩放因子
// 这个结构体的目的是为了方便在内核函数中使用逐元素掩码softmax的参数，提高深度学习的性能。
struct ElementwiseMaskSoftmaxParams {
  int64_t row_size;
  float fill;
  float scale;
};

// 它的功能是定义一个模板类，用于在内核函数中加载广播掩码softmax的输入。它的模板参数有：
// SRC: 源张量的数据类型
// DST: 目标张量的数据类型
// MASK: 掩码张量的数据类型
// num_dims: 张量的维度数
// IndexType: 索引的数据类型
// 它的构造函数接受以下参数：
// src: 源张量的指针
// mask: 掩码张量的指针
// params: 广播掩码softmax的参数，是一个BroadcastMaskSoftmaxParams结构体的实例
// 它的成员函数有：
// load: 一个模板函数，用于从源张量和掩码张量中加载N个元素，并将它们存储到目标张量中。它接受以下参数：
// dst: 目标张量的指针
// row: 当前的行号
// col: 当前的列号
// 它的内部逻辑是：
// 根据行号和列号计算源张量的偏移量
// 根据偏移量计算源张量的多维索引
// 根据多维索引计算掩码张量的偏移量
// 从源张量和掩码张量中读取N个元素，分别存储到pack和mask_pack中
// 遍历N个元素，如果掩码张量的元素为0，就将目标张量的元素设为填充值，否则就将源张量的元素乘以缩放因子，并转换为目标张量的数据类型
// 这个模板类的成员变量有：
// src: 源张量的指针
// mask: 掩码张量的指针
// mask_dims: 掩码张量的每个维度的大小，是一个数组
// params: 广播掩码softmax的参数，是一个BroadcastMaskSoftmaxParams结构体的实例
// 这个模板类的目的是为了方便在内核函数中使用广播掩码softmax的输入，提高深度学习的性能。
template<typename SRC, typename DST, typename MASK, size_t num_dims, typename IndexType>
struct BroadcastScaleMaskLoad {
  BroadcastScaleMaskLoad(const SRC* src, const MASK* mask,
                         BroadcastMaskSoftmaxParams<num_dims, IndexType> params)
      : src(src), mask(mask), params(params) {
    for (int i = 0; i < num_dims; i++) { mask_dims[i] = params.mask_dims[i]; }
  }
  template<int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) {
    Pack<SRC, N> pack;
    Pack<MASK, N> mask_pack;
    const IndexType offset = row * params.row_size + col;
    IndexType input_index[num_dims];
    IndexType mask_index[num_dims];
    params.src_index_helper.OffsetToNdIndex(offset, input_index);
    for (int dim = 0; dim < num_dims; ++dim) {
      if (mask_dims[dim] == 1) {
        mask_index[dim] = 0;
      } else {
        mask_index[dim] = input_index[dim];
      }
    }
    const IndexType mask_offset = params.mask_index_helper.NdIndexToOffset(mask_index);
    pack.storage = *(reinterpret_cast<const PackType<SRC, N>*>(src) + offset / N);
    mask_pack.storage =
        *(reinterpret_cast<const PackType<MASK, N>*>(mask) + mask_offset / N);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if (mask_pack.elem[i] == 0) {
        dst[i] = static_cast<DST>(params.fill);
      } else {
        dst[i] = static_cast<DST>(pack.elem[i]) * static_cast<DST>(params.scale);
      }
    }
  }
  const SRC* src;
  const MASK* mask;
  int64_t mask_dims[num_dims];
  BroadcastMaskSoftmaxParams<num_dims, IndexType> params;
};

// 它的功能是定义一个模板类，用于在内核函数中加载逐元素掩码softmax的输入。它的模板参数有：
// SRC: 源张量的数据类型
// DST: 目标张量的数据类型
// MASK: 掩码张量的数据类型
// 它的构造函数接受以下参数：
// src: 源张量的指针
// mask: 掩码张量的指针
// param: 逐元素掩码softmax的参数，是一个ElementwiseMaskSoftmaxParams结构体的实例
// load: 一个模板函数，用于从源张量和掩码张量中加载N个元素，并将它们存储到目标张量中。它接受以下参数：
// dst: 目标张量的指针
// row: 当前的行号
// col: 当前的列号
// 它的内部逻辑是：
// 根据行号和列号计算源张量和掩码张量的偏移量
// 从源张量和掩码张量中读取N个元素，分别存储到pack和mask_pack中
// 遍历N个元素，如果掩码张量的元素为0，就将目标张量的元素设为填充值，否则就将源张量的元素乘以缩放因子，并转换为目标张量的数据类型
// 这个模板类的成员变量有：
// src: 源张量的指针
// mask: 掩码张量的指针
// param: 逐元素掩码softmax的参数，是一个ElementwiseMaskSoftmaxParams结构体的实例
// 这个模板类的目的是为了方便在内核函数中使用逐元素掩码softmax的输入，提高深度学习的性能。
template<typename SRC, typename DST, typename MASK>
struct ElementwiseScaleMaskLoad {
  ElementwiseScaleMaskLoad(const SRC* src, const MASK* mask, ElementwiseMaskSoftmaxParams param)
      : src(src), mask(mask), param(param) {}
  template<int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) {
    Pack<SRC, N> pack;
    const int64_t offset = (row * param.row_size + col) / N;
    pack.storage = *(reinterpret_cast<const PackType<SRC, N>*>(src) + offset);
    Pack<int8_t, N> mask_pack;
    mask_pack.storage = *(reinterpret_cast<const PackType<MASK, N>*>(mask) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if (mask_pack.elem[i] == 0) {
        dst[i] = static_cast<DST>(param.fill);
      } else {
        dst[i] = static_cast<DST>(pack.elem[i]) * static_cast<DST>(param.scale);
      }
    }
  }
  const SRC* src;
  const MASK* mask;
  ElementwiseMaskSoftmaxParams param;
};

// 它的功能是定义一个模板类，用于在内核函数中存储广播掩码softmax的输出。它的模板参数有：
// SRC: 源张量的数据类型
// DST: 目标张量的数据类型
// MASK: 掩码张量的数据类型
// num_dims: 张量的维度数
// IndexType: 索引类型
// 它的构造函数接受以下参数：
// dst: 目标张量的指针
// mask: 掩码张量的指针
// params: 广播掩码softmax的参数，是一个BroadcastMaskSoftmaxParams结构体的实例
// 它的成员函数有：
// store: 用于从源张量加载N个元素，并将它们存储到目标张量中。它接受以下参数：
// src: 源张量的指针
// row: 当前的行号
// col: 当前的列号
// 它的内部逻辑是：
// 根据行号和列号计算源张量和目标张量的偏移量
// 根据偏移量计算源张量和掩码张量的索引
// 从掩码张量中读取N个元素，存储到mask_pack中
// 遍历N个元素，如果掩码张量的元素为0，就将目标张量的元素设为填充值，否则就将源张量的元素乘以缩放因子，并转换为目标张量的数据类型
// 将目标张量的元素存储到pack中，并写入到目标张量中
// 这个模板类的成员变量有：
// dst: 目标张量的指针
// mask: 掩码张量的指针
// mask_dims: 掩码张量的维度数组
// params: 广播掩码softmax的参数，是一个BroadcastMaskSoftmaxParams结构体的实例
// 这个模板类的目的是为了方便在内核函数中使用广播掩码softmax的输出，提高深度学习的性能。
template<typename SRC, typename DST, typename MASK, size_t num_dims, typename IndexType>
struct BroadcastScaleMaskStore {
  BroadcastScaleMaskStore(DST* dst, const MASK* mask,
                          BroadcastMaskSoftmaxParams<num_dims, IndexType> params)
      : dst(dst), mask(mask), params(params) {
    for (int i = 0; i < num_dims; ++i) { mask_dims[i] = params.mask_dims[i]; }
  }
  template<int N>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    Pack<DST, N> pack;
    Pack<MASK, N> mask_pack;
    const IndexType offset = row * params.row_size + col;
    IndexType input_index[num_dims];
    IndexType mask_index[num_dims];
    params.src_index_helper.OffsetToNdIndex(offset, input_index);
    for (int dim = 0; dim < num_dims; ++dim) {
      if (mask_dims[dim] == 1) {
        mask_index[dim] = 0;
      } else {
        mask_index[dim] = input_index[dim];
      }
    }
    const IndexType mask_offset = params.mask_index_helper.NdIndexToOffset(mask_index);
    mask_pack.storage =
        *(reinterpret_cast<const PackType<MASK, N>*>(mask) + mask_offset / N);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if (mask_pack.elem[i] == 0) {
        pack.elem[i] = static_cast<DST>(params.fill);
      } else {
        pack.elem[i] = static_cast<DST>(src[i]) * static_cast<DST>(params.scale);
      }
    }
    *(reinterpret_cast<PackType<DST, N>*>(dst) + offset / N) = pack.storage;
  }
  DST* dst;
  const MASK* mask;
  int64_t mask_dims[num_dims];
  BroadcastMaskSoftmaxParams<num_dims, IndexType> params;
};

// 它的功能是定义一个模板类，用于在内核函数中存储逐元素掩码softmax的输出。它的模板参数有：
// SRC: 源张量的数据类型
// DST: 目标张量的数据类型
// MASK: 掩码张量的数据类型
// 它的构造函数接受以下参数：
// dst: 目标张量的指针
// mask: 掩码张量的指针
// params: 逐元素掩码softmax的参数，是一个ElementwiseMaskSoftmaxParams结构体的实例
// 它的成员函数有：
// store: 一个模板函数，用于从源张量中读取N个元素，并将它们存储到目标张量中。它接受以下参数：
// src: 源张量的指针
// row: 当前的行号
// col: 当前的列号
// 它的内部逻辑是：
// 根据行号和列号计算源张量和目标张量的偏移量
// 根据偏移量从掩码张量中读取N个元素，存储到mask_pack中
// 遍历N个元素，如果掩码张量的元素为0，就将目标张量的元素设为填充值，否则就将源张量的元素乘以缩放因子，并转换为目标张量的数据类型
// 将目标张量的元素存储到pack中，并写入到目标张量中
template<typename SRC, typename DST, typename MASK>
struct ElementwiseScaleMaskStore {
  ElementwiseScaleMaskStore(DST* dst, const MASK* mask, ElementwiseMaskSoftmaxParams params)
      : dst(dst), mask(mask), params(params) {}
  template<int N>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    Pack<DST, N> pack;
    const int64_t offset = (row * params.row_size + col) / N;
    Pack<MASK, N> mask_pack;
    mask_pack.storage = *(reinterpret_cast<const PackType<MASK, N>*>(mask) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if (mask_pack.elem[i] == 0) {
        pack.elem[i] = params.fill;
      } else {
        pack.elem[i] = static_cast<DST>(src[i]) * static_cast<DST>(params.scale);
      }
    }
    *(reinterpret_cast<PackType<DST, N>*>(dst) + offset) = pack.storage;
  }
  DST* dst;
  const MASK* mask;
  ElementwiseMaskSoftmaxParams params;
};

// 这段代码定义了一个名为 MaskScaleLoad 的结构体，它有两个模板参数 SRC 和 DST，
// 分别表示源数据类型和目标数据类型。这个结构体有四个成员变量，分别是 src、mask、row_size 和 scale，
// 它们都是构造函数的参数。src 是一个指向源数据的指针，mask 是一个指向布尔值的指针，row_size 是一个表示每行数据的大小的整数，
// scale 是一个表示缩放因子的源数据类型的值。这个结构体还有一个模板成员函数 load，它有一个模板参数 N，表示每次加载的数据的数量（pack size）。
// 这个函数有三个参数，分别是 dst、row 和 col，表示要加载数据的目标地址、行号和列号。
// 这个函数的功能是从 src 和 mask 中读取 N 个数据，分别转换为目标数据类型，然后用 mask 的值和 scale 的值相乘，再和原始的输入相乘，
// 最后存储到 dst 中。这个函数使用了一些 C++ 的特性，比如 reinterpret_cast、Pack、PackType 和 static_cast，
// 它们都是用于类型转换和数据封装的工具。这个函数还使用了一个编译器指令 #pragma unroll，它是用于循环展开的优化技术。
// 这段代码可能是用于图像处理或者机器学习的领域，因为它涉及到了数据的加载、缩放和掩码的操作。
template<typename SRC, typename DST>
struct MaskScaleLoad {
  MaskScaleLoad(const SRC* src, const bool* mask, int64_t row_size, SRC scale)
      : src(src), mask(mask), row_size(row_size), scale(scale) {}
  template<int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) const {
    Pack<SRC, N> pack;
    const int64_t offset = (row * row_size + col) / N;
    pack.storage = *(reinterpret_cast<const PackType<SRC, N>*>(src) + offset);
    Pack<bool, N> mask_pack;
    mask_pack.storage = *(reinterpret_cast<const PackType<bool, N>*>(mask) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dst[i] = static_cast<DST>(pack.elem[i]) * static_cast<DST>(mask_pack.elem[i])
               * static_cast<DST>(scale);
    }
  }
  const SRC* src;
  const bool* mask;
  int64_t row_size;
  SRC scale;
};

// 这段代码是一个 C++ 的模板结构体，它定义了一个 DropoutStore 类，用于在神经网络中实现 dropout 的功能。
// Dropout 是一种正则化技术，它可以随机地将一些神经元的输出置为零，从而减少过拟合的风险。
// DropoutStore 类有以下几个成员变量和一个成员函数：
// dst: 一个指向 DST 类型的指针，用于存储 dropout 后的输出。
// softmax_y: 一个指向 DST 类型的指针，用于存储 softmax 的输出。
// mask: 一个指向 bool 类型的指针，用于存储 dropout 的掩码，表示哪些神经元被置为零。
// row_size: 一个 int64_t 类型的变量，用于表示每一行的元素个数。
// scale: 一个 DST 类型的变量，用于表示 dropout 的缩放因子，通常为 1 / (1 - dropout_rate)。
// store: 一个模板函数，用于将 SRC 类型的输入转换为 DST 类型，并根据 mask 和 scale 计算 dropout 后的输出，
// 同时将 softmax 的输出也保存下来。这个函数接受三个参数，分别是 src，row 和 col，表示输入的指针，行号和列号。
// 这个函数还使用了一个模板参数 N（pack size），表示每次处理的元素个数。这个函数的主要逻辑是：
// 定义三个 Pack 结构体，分别用于存储 softmax 的输出，dropout 的输出和 mask 的值。
// Pack 是一个模板结构体，它可以将 N 个元素打包成一个存储单元，以提高内存的访问效率。
// 计算当前处理的元素在内存中的偏移量，根据偏移量从 mask 中读取 N 个 bool 值，并将其转换为 Pack 结构体。
// 使用一个循环，对每个元素进行如下操作：
// 将 SRC 类型的输入转换为 DST 类型，并赋值给 softmax 的输出。
// 根据 mask 的值，将输入乘以 scale，并赋值给 dropout 的输出。
// 将 softmax 的输出和 dropout 的输出的 Pack 结构体转换为存储单元，并写入到 softmax_y 和 dst 中。
// 这段代码的优点是它可以高效地实现 dropout 的功能，同时保存 softmax 的输出，以便于后续的计算。
// 它也使用了模板和类型转换的技术，来保证数据的兼容性和安全性。它还使用了循环展开的优化技术，来减少循环的开销和提高性能。

template<typename SRC, typename DST>
struct DropoutStore {
  DropoutStore(DST* dst, DST* softmax_y, const bool* mask, int64_t row_size, DST scale)
      : dst(dst), softmax_y(softmax_y), mask(mask), row_size(row_size), scale(scale) {}
  template<int N>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    Pack<DST, N> softmax_y_pack;
    Pack<DST, N> dst_pack;
    const int64_t offset = (row * row_size + col) / N;
    Pack<bool, N> mask_pack;
    mask_pack.storage = *(reinterpret_cast<const PackType<bool, N>*>(mask) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      softmax_y_pack.elem[i] = static_cast<DST>(src[i]);
      dst_pack.elem[i] =
          static_cast<DST>(src[i]) * static_cast<DST>(mask_pack.elem[i]) * static_cast<DST>(scale);
    }
    *(reinterpret_cast<PackType<DST, N>*>(softmax_y) + offset) =
        softmax_y_pack.storage;
    *(reinterpret_cast<PackType<DST, N>*>(dst) + offset) = dst_pack.storage;
  }
  DST* dst;
  DST* softmax_y;
  const bool* mask;
  int64_t row_size;
  DST scale;
};

template<typename T, typename ComputeType, typename MASK, size_t num_dims>
void LaunchBroadcastForwardKernel(cudaStream_t stream, const T* x, T* y, const MASK* mask,
                                  const int64_t elem_cnt, const int64_t rows, const int64_t cols,
                                  const float fill, const float scale, const int64_t* input_dims,
                                  const int64_t* mask_dims) {
  NdIndexOffsetHelper<int32_t, num_dims> input_index_helper(input_dims);
  NdIndexOffsetHelper<int32_t, num_dims> mask_index_helper(mask_dims);
  BroadcastMaskSoftmaxParams<num_dims, int32_t> params;
  params.src_index_helper = input_index_helper;
  params.mask_index_helper = mask_index_helper;
  params.mask_dims = mask_dims;
  params.row_size = cols;
  params.fill = fill;
  params.scale = scale;
  BroadcastScaleMaskLoad<T, ComputeType, MASK, num_dims, int32_t> load(x, mask,
                                                                                            params);
  DirectStore<ComputeType, T> store(y, cols);
  (DispatchSoftmax<decltype(load), decltype(store), ComputeType>(
      stream, load, store, rows, cols));
   CUDA_CHECK();
}

template<typename T, typename ComputeType, typename MASK>
void LaunchElementwiseForwardKernel(cudaStream_t stream, const T* x, T* y, const MASK* mask,
                                    const int64_t rows, const int64_t cols, const float fill,
                                    const float scale) {
  ElementwiseMaskSoftmaxParams params;
  params.row_size = cols;
  params.fill = fill;
  params.scale = scale;
  ElementwiseScaleMaskLoad<T, ComputeType, MASK> load(x, mask, params);
  DirectStore<ComputeType, T> store(y, cols);
  (DispatchSoftmax<decltype(load), decltype(store), ComputeType>(
      stream, load, store, rows, cols));
   CUDA_CHECK();
}

int main(){
    const int batch_size = 4;
    const int num_heads = 8;
    const int seq_length = 64;
    const int N = batch_size * num_heads * seq_length * seq_length;

    float* x_host = (float*)malloc(N*sizeof(float));
    float *x_device;
    cudaMalloc((void **)&x_device, N*sizeof(float));
    for (int i = 0; i < N; i++) x_host[i] = 1.0;
    cudaMemcpy(x_device, x_host, N*sizeof(float), cudaMemcpyHostToDevice);

    float* y_host = (float*)malloc(N*sizeof(float));
    float *y_device;
    cudaMalloc((void **)&y_device, N*sizeof(float));

    bool *mask_host = (bool*)malloc(N*sizeof(bool));
    bool* mask_device;
    cudaMalloc((void **)&mask_device, N*sizeof(bool));
    for (int i = 0; i < N; i++) mask_host[i] = true;
    cudaMemcpy(mask_device, mask_host, N*sizeof(bool), cudaMemcpyHostToDevice);
    const float mask_fill_value = -10000.0;
    const float scale_value = 2.0;
    const int64_t cols = seq_length;
    const int64_t rows = N / seq_length;
    const size_t num_input_dims = 4;
    const int64_t input_dims[4] = {batch_size, num_heads, seq_length, seq_length};
    const size_t num_mask_dims = 4;
    const int64_t mask_dims[4] = {batch_size, num_heads, seq_length, seq_length};
    using ComputeType = typename DefaultComputeType<float>::type;
    size_t simplified_num_dims = 0;
    int64_t simplified_input_dims[4];
    int64_t simplified_mask_dims[4];

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    SimplifyBroadcastDims(num_input_dims, input_dims, num_mask_dims, mask_dims,
                                               &simplified_num_dims, simplified_input_dims,
                                               simplified_mask_dims);
    if (simplified_num_dims == 1) {
      LaunchElementwiseForwardKernel<float, ComputeType, bool>(stream, x_device, y_device,
          mask_device, rows, cols, mask_fill_value, scale_value);
    }
#define DEFINE_ONE_ELIF(dims)                                                               \
  else if (simplified_num_dims == dims) {                                                   \
    LaunchBroadcastForwardKernel<float, ComputeType, bool, dims>(                               \
        stream, x_device, y_device, \
        mask_device, N, rows, cols, mask_fill_value, scale_value,             \
        simplified_input_dims, simplified_mask_dims);                                       \
  }
    DEFINE_ONE_ELIF(2)
    DEFINE_ONE_ELIF(3)
    DEFINE_ONE_ELIF(4)
#undef DEFINE_ONE_ELIF
    else {
      exit(-1);
    }
    cudaMemcpy(y_host, y_device, N * sizeof(float), cudaMemcpyDeviceToHost);
    // 1 / 64 = 0.015625
    for (int i = 0; i < 32; i++){
      printf("%.6f\n", y_host[i]);
    }
    cudaFree(x_device);
    cudaFree(y_device);
    cudaFree(mask_device);
    free(x_host);
    free(y_host);
    free(mask_host);
    return 0;
}
