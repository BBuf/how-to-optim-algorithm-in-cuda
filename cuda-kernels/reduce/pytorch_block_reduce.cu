#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

#define C10_WARP_SIZE 32
#define N 32*1024*1024

constexpr int kCUDABlockReduceNumThreads = 512;
// Algorithmic limitation: BlockReduce does two WarpReduce calls, each
// of which reduces C10_WARP_SIZE elements. So, at most
// C10_WARP_SIZE**2 elements can be reduced at a time.
// NOTE: This is >= the max block size on current hardware anyway (1024).
constexpr int kCUDABlockReduceMaxThreads = C10_WARP_SIZE * C10_WARP_SIZE;

// Sums `val` accross all threads in a warp.
//
// Assumptions:
//   - The size of each block should be a multiple of `C10_WARP_SIZE`
// 实际上 WARP_SHFL_DOWN 的实现如下：
// template <typename T>
// __device__ __forceinline__ T WARP_SHFL_DOWN(T value, unsigned int delta, int width = warpSize, unsigned int mask = 0xffffffff)
// {
// #if !defined(USE_ROCM)
//     return __shfl_down_sync(mask, value, delta, width);
// #else
//     return __shfl_down(value, delta, width);
// #endif
// }


// 可以看到这里使用了warp原语__shfl_down_sync来对一个warp内的val进行规约求和。
// __shfl_down_sync的相关介绍：https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-shuffle-functions 
// && https://developer.nvidia.com/zh-cn/blog/using-cuda-warp-level-primitives/

// 示意图如：https://developer.nvidia.com/blog/wp-content/uploads/2018/01/reduce_shfl_down-625x275.png 所示
// 函数原型：T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
// mask表示一个warp中thread的激活表;
// var表示规约求和的变量；
// delta表示当前线程与另一个线程求和时跨越的线程偏移;
// width表示求和的宽度（个数）

// 根据循环体for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1)，
// 第一次循环时 delta 是 16，即 Lane0（Lane0表示一个warp中的lane_index）与 Lane16 求和，Lane1 与 Lane17 求和，***，
// 依此类推，一个 warp 内的 32 个 thread 的 val 规约成 16 个val。第二次循环时 delta 是 8 ，即 Lane0 与 Lane4 求和，Lane1 与 Lane5，
// 与上图的第一次行为相同。依次类推，最终不同 warp 中的 val 规约求和到了 Lane0 所持有的 val 中。
// 至此执行完 val = WarpReduceSum(val); 后，所有 warp 的和都规约到了每个 warp 中的 lane0 的线程中了，即 lid == 0 的线程，
// wid 则代表了不同的 lane（或不同的 warp ）。

template <typename T>
__inline__ __device__ T WarpReduceSum(T val) {
#pragma unroll
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    val += WARP_SHFL_DOWN(val, offset);
    // 或者：val += __shfl_down_sync(0xffffffff， val, offset, warpSize);//warpSize为内置数据，值为32
  }
  return val;
}

struct Block1D {
    static __forceinline__ __device__ int Tid() { return threadIdx.x; }

    static __forceinline__ __device__ int Warps() {
        return blockDim.x / C10_WARP_SIZE;
    }
};

struct Block2D {
    static __forceinline__ __device__ int Tid() {
        return threadIdx.x + threadIdx.y * blockDim.x;
    }

    static __forceinline__ __device__ int Warps() {
        return blockDim.x * blockDim.y / C10_WARP_SIZE;
    }
};

// Sums `val` across all threads in a block.
//
// Warning: the return value is only valid for thread 0.
// Assumptions:
//   - The size of each block should be a multiple of `C10_WARP_SIZE`
//   - `shared` should be a pointer to shared memory with size of, at least,
//     `sizeof(T) * number_of_warps`
// lane的中文翻译为车道（路宽），lane表示一个warp中的thread个数，
// 在 Block1D中在一个 lane 中的索引 lane_index 为 [0, warpSize - 1] 。
// 在一个 block 中会有多个 lane，lane_id = threadIdx.x / C10_WARP_SIZE ，最多有 1024 / C10_WARP_SIZE = 32 个lane。
template <typename T, typename B = Block1D>
__inline__ __device__ T BlockReduceSum(T val, T* shared) {
  const int tid = B::Tid(); // 获取 block 中的线程 id
  const int lid = tid % C10_WARP_SIZE; // lane id，表示当前 thread 在当前 lane 中的索引
  const int wid = tid / C10_WARP_SIZE; // wid表示当前 thread 在第多少个 lane
  // 每个 thread 对应寄存器中都有一个 val 值，WarpReduceSum 函数便是对 warp 中的所有 thread 所持有的 val 进行求和
  val = WarpReduceSum(val);
  // 下面要将各个 warp 规约求和的值进行一次规约，需要通过 shared-memory 将数据保存到同一个 warp 中的不同线程中，
  // 在数据保存前需要__syncthreads(); 同步一下
  __syncthreads(); // prevent races when BlockReduces are called in a row.
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  // 默认申请 32 大小的 shared-memory（32其实也是一个 Block 最大的 lane 个数），
  // 当 Block 内线程束较少时，无法刷新 shared-memory 上全部的 32 个值，需要对未使用到的内存进行初始化；
  val = (tid < B::Warps()) ? shared[lid] : T(0);
  // 再次使用 WarpReduceSum 对这 32 个 thread 的值进行求和，最终一个 Block 内的值便全部规约求和到了 threadIdx.x == 0 的线程所持有的 val 值了。
  // 这也就是说对于调用 BlockReduceSum 函数的代码来说，在使用规约求和后的值时需要通过 threadIdx.x == 0 的线程获取。

  if (wid == 0) {
    val = WarpReduceSum(val);
  }
  return val;
}

template <typename T, class ReduceOp>
__inline__ __device__ T WarpReduce(T val, const ReduceOp& op) {
#pragma unroll
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    val = op.combine(val, op.warp_shfl_down(val, offset));
  }
  return val;
}

template <typename T, class ReduceOp, typename B = Block1D>
__inline__ __device__ T
BlockReduce(T val, const ReduceOp& op, const T& identity_element, T* shared) {
  const int tid = B::Tid();
  const int lid = tid % C10_WARP_SIZE;
  const int wid = tid / C10_WARP_SIZE;
  val = WarpReduce(val, op);
  __syncthreads(); // prevent races when BlockReduces are called in a row.
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (tid < B::Warps()) ? shared[lid] : identity_element;
  if (wid == 0) {
    val = WarpReduce(val, op);
  }
  return val;
}

int main(){
    return 0;
}