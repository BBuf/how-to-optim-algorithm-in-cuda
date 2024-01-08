#include "ATen/ATen.h"
#include "ATen/AccumulateType.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/DeviceUtils.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#include "type_shim.h"
#include "static_switch.h"

// 这段代码定义了一个名为 cuWelfordOnlineSum 的 CUDA 设备函数，它实现了 Welford 算法用于在线计算均值（mu）和方差（sigma2）。
// Welford 算法是一种数值稳定的方法，用于逐步计算一系列数据的均值和方差。
// const U curr: 当前要处理的值。
// U& mu: 到目前为止所有值的均值。
// U& sigma2: 到目前为止所有值的方差。
// U& count: 到目前为止处理的元素数量。
template<typename U> __device__
void cuWelfordOnlineSum(
  const U curr,
  U& mu,
  U& sigma2,
  U& count)
{
  count = count + U(1); // 更新元素计数，每次调用函数时增加 1。
  U delta = curr - mu; // 计算当前值和当前均值的差值。
  U lmean = mu + delta / count; // 根据差值和元素数量计算新的均值。
  mu = lmean; // 将计算出的新均值赋值给 mu。
  U delta2 = curr - lmean; // 计算当前值和新均值的差值。
  sigma2 = sigma2 + delta * delta2; // 根据新旧均值之间的差值更新方差。
}

// 这段代码定义了一个名为 cuChanOnlineSum 的 CUDA 设备函数，它是另一种在线算法，
// 用于更新均值（mu）和方差（sigma2），考虑了两个独立样本的合并。
// const U muB, sigma2B, countB: 分别代表第二组数据的均值、方差和元素数量。
// U& mu, sigma2, count: 代表当前累积（第一组数据）的均值、方差和元素数量，这些将被更新以反映合并后的新值。
template<typename U> __device__
void cuChanOnlineSum(
  const U muB,
  const U sigma2B,
  const U countB,
  U& mu,
  U& sigma2,
  U& count)
{
  U delta = muB - mu; // 计算两组数据均值之间的差。
  U nA = count; // 保存当前组（A组）的元素数量。
  U nB = countB; // 获取第二组（B组）的元素数量。
  count = count + countB; // 更新元素总数。
  U nX = count; // 新的总元素数量。
  if (nX > U(0)) {
    nA = nA / nX; // 计算两组数据在新总数中的相对比例。
    nB = nB / nX; 
    mu = nA*mu + nB*muB; // 根据比例和各自的均值计算新的总均值。
    sigma2 = sigma2 + sigma2B + delta * delta * nA * nB * nX; // 更新方差，考虑两组数据方差和均值差的影响。
  } else {
    // 如果新的总数 nX 为 0，表明两组数据都是空的，因此将 mu 和 sigma2 设置为 0。
    mu = U(0);
    sigma2 = U(0);
  }
}

// 这段代码定义了一个名为 cuRMSOnlineSum 的 CUDA 设备函数，用于在线计算平方和，
// 从而可以用来计算均方根（RMS, Root Mean Square）值
template<typename U> __device__
void cuRMSOnlineSum(
  const U curr,
  U& sigma2)
{
  sigma2 = sigma2 + curr * curr;
}

// 这段代码定义了一个名为 cuChanRMSOnlineSum 的 CUDA 设备函数，用于在线计算两个数据集平方和的累加。
// 这个函数是用于合并两个独立数据集的均方根（RMS, Root Mean Square）计算的一部分。
template<typename U> __device__
void cuChanRMSOnlineSum(
  const U sigma2B,
  U& sigma2)
{
  sigma2 = sigma2 + sigma2B;
}

// cuWelfordMuSigma2 函数是一个用于 CUDA 设备的函数，专门设计来计算张量某一维度上的均值（mu）和方差（sigma2）。
// 它采用 Welford 方法进行计算以保证数值稳定性，并可选择只计算均方根（RMS）。
// template<typename T, typename U>: 模板参数，用于不同数据类型的张量值（T）和计算过程（U）。
// const T* __restrict__ vals: 指向张量值的指针。
// const int n1, n2: 张量的维度。n1 是进行计算的维度大小，n2 是被缩减的维度大小。
// const int i1: 在 n1 维度上当前处理的特定索引。
// U& mu, sigma2: 将被计算的均值和方差的引用。
// U* buf: 指向共享内存缓冲区的指针，用于线程间通信。
// bool rms_only: 标志，指示是否只计算 RMS（true）或均值和方差（false）。
template<typename T, typename U> __device__
void cuWelfordMuSigma2(
  const T* __restrict__ vals,
  const int n1,
  const int n2,
  const int i1,
  U& mu,
  U& sigma2,
  U* buf,
  bool rms_only)
{
  // 前提条件:
  // 1) blockDim.x == warpSize
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over n2
  // 初始化 count, mu, 和 sigma2 为零。
  U count = U(0);
  mu= U(0);
  sigma2 = U(0);
  // 这个条件判断确保当前线程处理的 i1 索引在张量的有效范围内。
  if (i1 < n1) {
    // one warp normalizes one n1 index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y; // 计算一个 CUDA 块中的线程总数。
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x; // 计算当前线程在块内的唯一线性索引
    // 将 lvals 指针设置为指向当前处理的 i1 索引处张量的开始位置。
    // vals 是整个张量数据的起始指针，i1*n2 计算出当前索引在张量中的线性位置。
    const T* lvals = vals + i1*n2;
    // 初始化一个局部变量 l，用于在接下来的循环中遍历张量的元素。这里每个线程会处理多个元素，起始位置是基于线程的索引的。
    int l = 4*thrx;
    // 这个循环以步长 4*numx 遍历张量的元素，每个线程处理四个元素（如果有足够的元素）。
    for (;  l+3 < n2;  l+=4*numx) {
      // 在每次外循环的迭代中，处理四个连续的元素。
      for (int k = 0;  k < 4;  ++k) {
        // 将当前处理的元素值转换为计算使用的数据类型（U）。
        U curr = static_cast<U>(lvals[l+k]);
        // 根据 rms_only 标志调用相应的函数来更新均值和方差或仅更新平方和（用于计算 RMS）。
        if (!rms_only) {
          cuWelfordOnlineSum<U>(curr,mu,sigma2,count);
        } else {
          cuRMSOnlineSum<U>(curr, sigma2);
        }
      }
    }
    // 这个循环处理了之前在步长为 4*numx 的循环中未处理的张量元素。每个线程独立处理它们剩余的部分。
    for (;  l < n2;  ++l) {
      U curr = static_cast<U>(lvals[l]);
      if (!rms_only) {
        cuWelfordOnlineSum<U>(curr,mu,sigma2,count);
      } else {
       cuRMSOnlineSum<U>(curr, sigma2);
      }
    }
    // intra-warp reductions
    // 这个循环是用于在同一个 warp 内部进行 reduce 的。
    for (int l = 0;  l <= 4;  ++l) {
      // 是在 CUDA 设备上进行 warp 内部数据交换的关键部分。
      // 这行代码用于确定在一个 warp（32个线程）内，每个线程应该从哪个“lane”（即其他线程）获取数据。
      // （1<<l）这个操作在这里用于逐步增加要从中获取数据的线程的距离。例如，当 l 为 0 时，
      // 线程将从它的“邻居”线程（即下一个线程）获取数据；当 l 为 1 时，它将从两个位置之外的线程获取数据，依此类推。
      // 这个表达式计算出当前线程应该从哪个线程获取数据。随着 l 的增加，每个线程从越来越远的线程获取数据。
      // &31是因为在一个 warp 内，线程索引是循环的。也就是说，如果一个线程的索引计算结果是 32，
      // 它实际上会从索引为 0 的线程获取数据，索引为 33 的线程实际上是索引为 1 的线程，依此类推。
      int srcLaneB = (threadIdx.x+(1<<l))&31;
      // 是一种 warp 内部的快速数据交换操作，用于从另一个线程（srcLaneB）获取数据。
      U sigma2B = WARP_SHFL(sigma2, srcLaneB);
      // 如果不是只计算 RMS（!rms_only），则使用 cuChanOnlineSum 合并两个线程的 mu、sigma2 和 count。
      // 如果只计算 RMS，则使用 cuChanRMSOnlineSum 合并 sigma2。
      if (!rms_only) {
        U muB = WARP_SHFL(mu, srcLaneB);
        U countB = WARP_SHFL(count, srcLaneB);
        cuChanOnlineSum<U>(muB,sigma2B,countB,mu,sigma2,count);
      } else {
        cuChanRMSOnlineSum<U>(sigma2B, sigma2);
      }
    }
    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    // 检查是否有多个 warp。如果 blockDim.y 大于 1，则表示块中有多个 warp 需要进行reduce操作。
    if (blockDim.y > 1) {
      // 为方差和均值的reduce操作分配共享内存。ubuf 用于存储方差和均值，ibuf 用于存储计数。
      U* ubuf = (U*)buf;
      U* ibuf = (U*)(ubuf + blockDim.y);
      // 这个循环是对 warp 间的reduce操作进行分层合并。
      for (int offset = blockDim.y/2;  offset > 0;  offset /= 2) {
        // upper half of warps write to shared
        // 确保只有部分线程（warp 的上半部分）将其计算的结果写入共享内存。
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int wrt_y = threadIdx.y - offset;
          if (!rms_only) {
            ubuf[2*wrt_y] = mu;
            ibuf[wrt_y] = count;
          }
          ubuf[2*wrt_y+1] = sigma2;
        }
        // 同步以等待共享内存存储完毕
        __syncthreads();
        // lower half merges
        // 此部分是对 warp 间数据的合并操作。
        // 确保只有部分线程（warp 的下半部分）从共享内存中读取数据并进行合并。
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          U sigma2B = ubuf[2*threadIdx.y+1];
          if (!rms_only) {
            U muB = ubuf[2*threadIdx.y];
            U countB = ibuf[threadIdx.y];
            cuChanOnlineSum<U>(muB,sigma2B,countB,mu,sigma2,count);
          } else {
            cuChanRMSOnlineSum<U>(sigma2B,sigma2);
          }
        }
        __syncthreads();
      }
      // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
      // 最终的结果由块内的第一个线程（threadIdx.x == 0 && threadIdx.y == 0）计算并写入共享内存。
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        if (!rms_only) {
          ubuf[0] = mu;
        }
        ubuf[1] = sigma2;
      }
      __syncthreads();
      // 如果不是只计算 RMS，则还需要更新均值 mu。
      if (!rms_only) {
        mu = ubuf[0];
      }
      // 计算最终的方差。
      sigma2 = ubuf[1]/U(n2);
      // don't care about final value of count, we know count == n2
    } 
    // 如果块中只有一个 warp（blockDim.y == 1），则通过 WARP_SHFL 直接在 warp 内进行数据交换和更新。
    else {
      if (!rms_only) {
        mu = WARP_SHFL(mu, 0);
      }
      sigma2 = WARP_SHFL(sigma2/U(n2), 0);
    }
  }
}

// 这个函数是上面的 cuWelfordMuSigma2 的Half特化，就不重复解析
template<> __device__
void cuWelfordMuSigma2(
  const at::Half* __restrict__ vals,
  const int n1,
  const int n2,
  const int i1,
  float& mu,
  float& sigma2,
  float* buf,
  bool rms_only)
{
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over n2
  float count = 0.0f;
  mu= float(0);
  sigma2 = float(0);
  if (i1 < n1) {
    // one warp normalizes one n1 index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const at::Half* lvals = vals + i1*n2;
    int l = 8*thrx;
    if ((((size_t)lvals)&3) != 0) {
      // 16 bit alignment
      // first thread consumes first point
      if (thrx == 0) {
        float curr = static_cast<float>(lvals[0]);
        if (!rms_only) {
          cuWelfordOnlineSum(curr,mu,sigma2,count);
        } else {
          cuRMSOnlineSum(curr, sigma2);
        }

      }
      ++l;
    }
    // at this point, lvals[l] are 32 bit aligned for all threads.
    for (;  l+7 < n2;  l+=8*numx) {
      for (int k = 0;  k < 8;  k+=2) {
        float2 curr = __half22float2(*((__half2*)(lvals+l+k)));
        if (!rms_only) {
          cuWelfordOnlineSum(curr.x,mu,sigma2,count);
          cuWelfordOnlineSum(curr.y,mu,sigma2,count);
        } else {
          cuRMSOnlineSum(curr.x, sigma2);
          cuRMSOnlineSum(curr.y, sigma2);
        }
      }
    }
    for (;  l < n2;  ++l) {
      float curr = static_cast<float>(lvals[l]);
      if (!rms_only) {
        cuWelfordOnlineSum(curr,mu,sigma2,count);
      } else {
        cuRMSOnlineSum(curr, sigma2);
      }
    }
    // intra-warp reductions
    for (int l = 0;  l <= 4;  ++l) {
      int srcLaneB = (threadIdx.x+(1<<l))&31;
      float sigma2B = WARP_SHFL(sigma2, srcLaneB);
      if (!rms_only) {
        float muB = WARP_SHFL(mu, srcLaneB);
        float countB = WARP_SHFL(count, srcLaneB);
        cuChanOnlineSum(muB,sigma2B,countB,mu,sigma2,count);
      } else {
        cuChanRMSOnlineSum(sigma2B, sigma2);
      }
    }
    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      float* ubuf = (float*)buf;
      float* ibuf = (float*)(ubuf + blockDim.y);
      for (int offset = blockDim.y/2;  offset > 0;  offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int wrt_y = threadIdx.y - offset;
          ubuf[2*wrt_y+1] = sigma2;
          if (!rms_only) {
            ubuf[2*wrt_y] = mu;
            ibuf[wrt_y] = count;
          }
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          float sigma2B = ubuf[2*threadIdx.y+1];
          if (!rms_only) {
            float muB = ubuf[2*threadIdx.y];
            float countB = ibuf[threadIdx.y];
            cuChanOnlineSum(muB,sigma2B,countB,mu,sigma2,count);
          } else {
            cuChanRMSOnlineSum(sigma2B, sigma2);
          }
        }
        __syncthreads();
      }
      // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        if (!rms_only) {
          ubuf[0] = mu;
        }
        ubuf[1] = sigma2;
      }
      __syncthreads();
      if (!rms_only) {
        mu = ubuf[0];
      }
      sigma2 = ubuf[1]/float(n2);
      // don't care about final value of count, we know count == n2
    } else {
      if (!rms_only) {
        mu = WARP_SHFL(mu, 0);
      }
      sigma2 = WARP_SHFL(sigma2/float(n2), 0);
    }
  }
}

template<typename U> U rsqrt(U v) {
  return U(1) / sqrt(v);
}
template<> float rsqrt(float v) {
  return rsqrtf(v);
}
template<> double rsqrt(double v) {
  return rsqrt(v);
}

namespace {
// This is the un-specialized struct.  Note that we prevent instantiation of this
// struct by putting an undefined symbol in the function body so it won't compile.
//  template <typename T>
//  struct SharedMemory
//  {
//      // Ensure that we won't compile any un-specialized types
//      __device__ T *getPointer()
//      {
//          extern __device__ void error(void);
//          error();
//          return NULL;
//      }
//  };
// https://github.com/NVIDIA/apex/issues/246
template <typename T>
struct SharedMemory;

template <>
struct SharedMemory <float>
{
    __device__ float *getPointer()
    {
        extern __shared__ float s_float[];
        return s_float;
    }
};

template <>
struct SharedMemory <double>
{
    __device__ double *getPointer()
    {
        extern __shared__ double s_double[];
        return s_double;
    }
};
}

template<typename T, typename U, typename V> __device__
void cuApplyLayerNorm_(
  V* __restrict__ output_vals,
  U* __restrict__ mean,
  U* __restrict__ invvar,
  const T* __restrict__ vals,
  const int n1,
  const int n2,
  const U epsilon,
  const V* __restrict__ gamma,
  const V* __restrict__ beta,
  bool rms_only
  )
{
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensors are contiguous
  //
  for (auto i1=blockIdx.y; i1 < n1; i1 += gridDim.y) {
    SharedMemory<U> shared;
    U* buf = shared.getPointer();
    U mu,sigma2;
    cuWelfordMuSigma2(vals,n1,n2,i1,mu,sigma2,buf,rms_only);

    const T* lvals = vals + i1*n2;
    V* ovals = output_vals + i1*n2;
    U c_invvar = rsqrt(sigma2 + epsilon);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (gamma != NULL && (beta != NULL || rms_only)) {
      for (int i = thrx;  i < n2;  i+=numx) {
        U curr = static_cast<U>(lvals[i]);
        if (!rms_only) {
          ovals[i] = gamma[i] * static_cast<V>(c_invvar * (curr - mu)) + beta[i];
        } else {
          ovals[i] = gamma[i] * static_cast<V>(c_invvar * curr);
        }

      }
    } else {
      for (int i = thrx;  i < n2;  i+=numx) {
        U curr = static_cast<U>(lvals[i]);
        if (!rms_only) {
          ovals[i] = static_cast<V>(c_invvar * (curr - mu));
        } else {
          ovals[i] = static_cast<V>(c_invvar * curr);
        }
      }
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      if (!rms_only) {
        mean[i1] = mu;
      }
      invvar[i1] = c_invvar;
    }
    __syncthreads();
  }
}

template<typename T, typename U, typename V=T> __global__
void cuApplyLayerNorm(
  V* __restrict__ output_vals,
  U* __restrict__ mean,
  U* __restrict__ invvar,
  const T* __restrict__ vals,
  const int n1,
  const int n2,
  const U epsilon,
  const V* __restrict__ gamma,
  const V* __restrict__ beta
  )
{
  cuApplyLayerNorm_<T, U, V>(output_vals, mean, invvar, vals, n1, n2, epsilon, gamma, beta, false);
}

template<typename V> __device__
V clamp_by_magnitude(V curr_gamma, double eps)
{
  const V kMinGamma = V(eps);
  if (curr_gamma >= 0) {
    if (curr_gamma < kMinGamma) {
      return kMinGamma;
    } else {
      return curr_gamma;
    }
  } else {
    if (curr_gamma > -kMinGamma) {
      return -kMinGamma;
    } else {
      return curr_gamma;
    }
  }
}


template<typename T, typename U, typename V, bool MemoryEfficient> __device__
void cuLoadWriteStridedInputs(
    const int i1_block,
    const int thr_load_row_off,
    const int thr_load_col_off,
    const int i2_off,
    const int row_stride,
    U* warp_buf1,
    U* warp_buf2,
    const T* input_or_output,
    const V* dout,
    const int i1_end,
    const int n2,
    const U* __restrict__ mean,
    const U* __restrict__ invvar,
    const V* __restrict__ gamma,
    const V* __restrict__ beta,
    const double eps,
    bool rms_only
    )
{
  int i1 = i1_block+thr_load_row_off;
  if (i1 < i1_end) {
    for (int k = 0;  k < blockDim.y;  ++k) {
      int i2 = i2_off + k;
      int load_idx = i1*n2+i2;
      int write_idx = thr_load_row_off*row_stride+thr_load_col_off+k;
      if (i2<n2) {
        U c_h = static_cast<U>(input_or_output[load_idx]);
        U curr_dout = static_cast<U>(dout[load_idx]);
        if (!rms_only) {
          warp_buf1[write_idx] = curr_dout;
          if (MemoryEfficient) {
            U curr_beta = static_cast<U>(beta[i2]);
            warp_buf2[write_idx] = curr_dout * (c_h - curr_beta) / static_cast<U>(clamp_by_magnitude(gamma[i2], eps));
          } else {
            warp_buf2[write_idx] = curr_dout * (c_h - mean[i1]) * invvar[i1];
          }
        } else {
          if (MemoryEfficient) {
            warp_buf2[write_idx] = curr_dout * (c_h) / static_cast<U>(clamp_by_magnitude(gamma[i2], eps));
          } else {
            warp_buf2[write_idx] = curr_dout * (c_h) * invvar[i1];
          }
        }
      } else {
        if (!rms_only) {
          warp_buf1[write_idx] = U(0);
        }
        warp_buf2[write_idx] = U(0);
      }
    }
  } else {
    for (int k = 0;  k < blockDim.y;  ++k) {
      int write_idx = thr_load_row_off*row_stride+thr_load_col_off+k;
      if (!rms_only) {
        warp_buf1[write_idx] = U(0);
      }
      warp_buf2[write_idx] = U(0);
    }
  }
}

template<typename T, typename U, typename V, bool MemoryEfficient> __device__
void cuLoadAddStridedInputs(
    const int i1_block,
    const int thr_load_row_off,
    const int thr_load_col_off,
    const int i2_off,
    const int row_stride,
    U* warp_buf1,
    U* warp_buf2,
    const T* input_or_output,
    const V* dout,
    const int i1_end,
    const int n2,
    const U* __restrict__ mean,
    const U* __restrict__ invvar,
    const V* __restrict__ gamma,
    const V* __restrict__ beta,
    const double eps,
    bool rms_only
    )
{
  int i1 = i1_block+thr_load_row_off;
  if (i1 < i1_end) {
    for (int k = 0;  k < blockDim.y;  ++k) {
      int i2 = i2_off + k;
      int load_idx = i1*n2+i2;
      int write_idx = thr_load_row_off*row_stride+thr_load_col_off+k;
      if (i2<n2) {
        U c_h = static_cast<U>(input_or_output[load_idx]);
        U curr_dout = static_cast<U>(dout[load_idx]);
        if (!rms_only) {
          U curr_beta = static_cast<U>(beta[i2]);
          warp_buf1[write_idx] += curr_dout;
          if (MemoryEfficient) {
            warp_buf2[write_idx] += curr_dout * (c_h - curr_beta) / static_cast<U>(clamp_by_magnitude(gamma[i2], eps));
          } else {
            warp_buf2[write_idx] += curr_dout * (c_h - mean[i1]) * invvar[i1];
          }
        } else {
          if (MemoryEfficient) {
            warp_buf2[write_idx] += curr_dout * (c_h) / static_cast<U>(clamp_by_magnitude(gamma[i2], eps));
          } else {
            warp_buf2[write_idx] += curr_dout * (c_h) * invvar[i1];
          }
        }
      }
    }
  }
}


template<typename T, typename U, typename V, bool MemoryEfficient> __global__
void cuComputePartGradGammaBeta(
    const V* __restrict__ dout,
    const T* __restrict__ input_or_output,
    const int n1,
    const int n2,
    const U* __restrict__ mean,
    const U* __restrict__ invvar,
    U epsilon,
    const V* __restrict__ gamma,
    const V* __restrict__ beta,
    U* part_grad_gamma,
    U* part_grad_beta,
    const double eps,
    bool rms_only)
{
    const int numsegs_n1 = (n1+blockDim.y*blockDim.y-1) / (blockDim.y*blockDim.y);
    const int segs_per_block = (numsegs_n1 + gridDim.y - 1) / gridDim.y;
    const int i1_beg = blockIdx.y * segs_per_block * blockDim.y*blockDim.y;
    const int i1_beg_plus_one = (blockIdx.y+1) * segs_per_block * blockDim.y*blockDim.y;
    const int i1_end = i1_beg_plus_one < n1 ? i1_beg_plus_one : n1;
    const int row_stride = blockDim.x+1;
    const int thr_load_col_off = (threadIdx.x*blockDim.y)&(blockDim.x-1);
    const int thr_load_row_off = (threadIdx.x*blockDim.y)/blockDim.x + threadIdx.y*blockDim.y;
    const int i2_off = blockIdx.x * blockDim.x + thr_load_col_off;
    SharedMemory<U> shared;
    U* buf = shared.getPointer(); // buf has at least blockDim.x * blockDim.y * blockDim.y + (blockDim.y - 1)*(blockDim.x/blockDim.y) elements
    U* warp_buf1 = (U*)buf;
    U* warp_buf2 = warp_buf1 + blockDim.y * blockDim.y * row_stride;
    // compute partial sums from strided inputs
    // do this to increase number of loads in flight
    cuLoadWriteStridedInputs<T, U, V, MemoryEfficient>(i1_beg,thr_load_row_off,thr_load_col_off,i2_off,row_stride,warp_buf1,warp_buf2,input_or_output,dout,i1_end,n2,mean,invvar,gamma,beta,eps, rms_only);
    for (int i1_block = i1_beg+blockDim.y*blockDim.y;  i1_block < i1_end;  i1_block+=blockDim.y*blockDim.y) {
      cuLoadAddStridedInputs<T, U, V, MemoryEfficient>(i1_block,thr_load_row_off,thr_load_col_off,i2_off,row_stride,warp_buf1,warp_buf2,input_or_output,dout,i1_end,n2,mean,invvar,gamma,beta,eps, rms_only);
    }
    __syncthreads();
    // inter-warp reductions
    // sum within each warp
    U acc1 = U(0);
    U acc2 = U(0);
    for (int k = 0;  k < blockDim.y;  ++k) {
      int row1 = threadIdx.y + k*blockDim.y;
      int idx1 = row1*row_stride + threadIdx.x;
      if (!rms_only) {
        acc1 += warp_buf1[idx1];
      }
      acc2 += warp_buf2[idx1];
    }
    if (!rms_only) {
      warp_buf1[threadIdx.y*row_stride+threadIdx.x] = acc1;
    }
    warp_buf2[threadIdx.y*row_stride+threadIdx.x] = acc2;
    __syncthreads();
    // sum all warps
    for (int offset = blockDim.y/2;  offset > 1;  offset /= 2) {
      if (threadIdx.y < offset) {
        int row1 = threadIdx.y;
        int row2 = threadIdx.y + offset;
        int idx1 = row1*row_stride + threadIdx.x;
        int idx2 = row2*row_stride + threadIdx.x;
        if (!rms_only) {
          warp_buf1[idx1] += warp_buf1[idx2];
        }
        warp_buf2[idx1] += warp_buf2[idx2];
      }
      __syncthreads();
    }
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.y == 0 && i2 < n2) {
      int row1 = threadIdx.y;
      int row2 = threadIdx.y + 1;
      int idx1 = row1*row_stride + threadIdx.x;
      int idx2 = row2*row_stride + threadIdx.x;
      if (!rms_only) {
        part_grad_beta[blockIdx.y*n2+i2] = warp_buf1[idx1] + warp_buf1[idx2];
      }
      part_grad_gamma[blockIdx.y*n2+i2] = warp_buf2[idx1] + warp_buf2[idx2];
    }
}

template<typename U, typename V> __global__
void cuComputeGradGammaBeta(
    const U* part_grad_gamma,
    const U* part_grad_beta,
    const int part_size,
    const int n1,
    const int n2,
    V* grad_gamma,
    V* grad_beta,
    bool rms_only)
{
    // sum partial gradients for gamma and beta
    SharedMemory<U> shared;
    U* buf = shared.getPointer();
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (i2 < n2) {
      // each warp does sequential reductions until reduced part_size is num_warps
      int num_warp_reductions = part_size / blockDim.y;
      U sum_gamma = U(0);
      U sum_beta = U(0);
      const U* part_grad_gamma_ptr = part_grad_gamma + threadIdx.y * num_warp_reductions * n2 + i2;
      const U* part_grad_beta_ptr = part_grad_beta + threadIdx.y * num_warp_reductions * n2 + i2;
      for (int warp_offset = 0;  warp_offset < num_warp_reductions;  ++warp_offset) {
        sum_gamma += part_grad_gamma_ptr[warp_offset*n2];
        if (!rms_only) {
          sum_beta += part_grad_beta_ptr[warp_offset*n2];
        }
      }
      // inter-warp reductions
      const int nbsize3 = blockDim.x * blockDim.y / 2;
      for (int offset = blockDim.y/2;  offset >= 1;  offset /= 2) {
        // top half write to shared memory
        if (threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int write_idx = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
          buf[write_idx] = sum_gamma;
          if (!rms_only) {
            buf[write_idx+nbsize3] = sum_beta;
          }
        }
        __syncthreads();
        // bottom half sums
        if (threadIdx.y < offset) {
          const int read_idx = threadIdx.y * blockDim.x + threadIdx.x;
          sum_gamma += buf[read_idx];
          if (!rms_only) {
            sum_beta += buf[read_idx+nbsize3];
          }
        }
        __syncthreads();
      }
      // write out fully summed gradients
      if (threadIdx.y == 0) {
        grad_gamma[i2] = sum_gamma;
        if (!rms_only) {
          grad_beta[i2] = sum_beta;
        }
      }
    }
}


template<typename T, typename U, typename V, bool MemoryEfficient> __global__
void cuComputeGradInput(
    const V* __restrict__ dout,
    const T* __restrict__ input_or_output,
    const int n1,
    const int n2,
    const U* __restrict__ mean,
    const U* __restrict__ invvar,
    U epsilon,
    const V* gamma,
    const V* beta,
    T* grad_input,
    const double eps,
    bool rms_only)
{
  for (auto i1=blockIdx.y; i1 < n1; i1 += gridDim.y) {
    U sum_loss1 = U(0);
    U sum_loss2 = U(0);
    const T* k_h = input_or_output + i1*n2;
    const V* k_dout = dout + i1*n2;
    const U c_invvar = invvar[i1];
    const U c_mean = !MemoryEfficient ? mean[i1] : 0.;
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (gamma != NULL) {
      int l = 4*thrx;
      for (;  l+3 < n2;  l+=4*numx) {
        for (int k = 0;  k < 4;  ++k) {
          const U c_h = static_cast<U>(k_h[l+k]);
          const U c_loss = static_cast<U>(k_dout[l+k]);
          if (!rms_only) {
            sum_loss1 += c_loss * gamma[l+k];
            if (MemoryEfficient) {
              sum_loss2 += c_loss * (c_h - beta[l+k]);
            } else {
              sum_loss2 += c_loss * gamma[l+k] * (c_h - c_mean) * c_invvar;
            }
          } else {
            if (MemoryEfficient) {
              sum_loss2 += c_loss * c_h;
            } else {
              sum_loss2 += c_loss * gamma[l+k] * (c_h) * c_invvar;
            }
          }
        }
      }
      for (;  l < n2;  ++l) {
        const U c_h = static_cast<U>(k_h[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        if (!rms_only) {
          sum_loss1 += c_loss * gamma[l];
          if (MemoryEfficient) {
            sum_loss2 += c_loss * (c_h - beta[l]);
          } else {
            sum_loss2 += c_loss * gamma[l] * (c_h - c_mean) * c_invvar;
          }
        } else {
          if (MemoryEfficient) {
            sum_loss2 += c_loss * c_h;
          } else {
            sum_loss2 += c_loss * gamma[l] * (c_h) * c_invvar;
          }
        }
      }
    } else {
      int l = 4*thrx;
      for (;  l+3 < n2;  l+=4*numx) {
        for (int k = 0;  k < 4;  ++k) {
          const U c_h = static_cast<U>(k_h[l+k]);
          const U c_loss = static_cast<U>(k_dout[l+k]);
          if (!rms_only) {
            sum_loss1 += c_loss;
            if (MemoryEfficient) {
              sum_loss2 += c_loss * c_h;
            } else {
              sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
            }
          } else {
            if (MemoryEfficient) {
              sum_loss2 += c_loss * c_h;
            } else {
              sum_loss2 += c_loss * (c_h) * c_invvar;
            }
          }
        }
      }
      for (;  l < n2;  ++l) {
        const U c_h = static_cast<U>(k_h[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        if (!rms_only) {
          sum_loss1 += c_loss;
          if (MemoryEfficient) {
            sum_loss2 += c_loss * c_h;
          } else {
            sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
          }
        } else {
          if (MemoryEfficient) {
            sum_loss2 += c_loss * c_h;
          } else {
            sum_loss2 += c_loss * (c_h) * c_invvar;
          }
        }
      }
    }
    // intra-warp reductions
    for (int mask = blockDim.x/2;  mask > 0;  mask /= 2) {
      if (!rms_only) {
        sum_loss1 += WARP_SHFL_XOR(sum_loss1, mask);
      }
      sum_loss2 += WARP_SHFL_XOR(sum_loss2, mask);
    }
    // inter-warp reductions
    if (blockDim.y > 1) {
      SharedMemory<U> shared;
      U* buf = shared.getPointer();
      for (int offset = blockDim.y/2;  offset > 0;  offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int wrt_i = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
          if (!rms_only) {
            buf[2*wrt_i] = sum_loss1;
          }
          buf[2*wrt_i+1] = sum_loss2;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.y < offset) {
          const int read_i = threadIdx.y * blockDim.x + threadIdx.x;
          if (!rms_only) {
            sum_loss1 += buf[2*read_i];
          }
          sum_loss2 += buf[2*read_i+1];
        }
        __syncthreads();
      }
      if (threadIdx.y == 0) {
        if (!rms_only) {
          buf[2*threadIdx.x] = sum_loss1;
        }
        buf[2*threadIdx.x+1] = sum_loss2;
      }
      __syncthreads();
      if (threadIdx.y !=0) {
        if (!rms_only) {
          sum_loss1 = buf[2*threadIdx.x];
        }
        sum_loss2 = buf[2*threadIdx.x+1];
      }
    }
    // all threads now have the two sums over l
    U fH = (U)n2;
    U term1 = (U(1) / fH) * c_invvar;
    T* k_grad_input = grad_input + i1*n2;
    if (gamma != NULL) {
      for (int l = thrx;  l < n2;  l+=numx) {
        const U c_h = static_cast<U>(k_h[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        const U k_gamma = static_cast<U>(clamp_by_magnitude(gamma[l], eps));
        U f_grad_input = fH * c_loss * k_gamma;
        if (!rms_only) {
          const U k_beta = beta[l];
          f_grad_input -= sum_loss1;
          if (MemoryEfficient) {
            f_grad_input -= (c_h - k_beta) / k_gamma * sum_loss2;
          } else {
            f_grad_input -= (c_h - c_mean) * c_invvar * sum_loss2;
          }
        } else {
          if (MemoryEfficient) {
            f_grad_input -= c_h / k_gamma * sum_loss2;
          } else {
            f_grad_input -= c_h * c_invvar * sum_loss2;
          }
        }
        f_grad_input *= term1;
        k_grad_input[l] = static_cast<T>(f_grad_input);
      }
    } else {
      for (int l = thrx;  l < n2;  l+=numx) {
        const U c_h = static_cast<U>(k_h[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        U f_grad_input = fH * c_loss;
        if (!rms_only) {
          f_grad_input -= sum_loss1;
          if (MemoryEfficient) {
            f_grad_input -= c_h * sum_loss2;
          } else {
            f_grad_input -= (c_h - c_mean) * c_invvar * sum_loss2;
          }
        } else {
          if (MemoryEfficient) {
            f_grad_input -= c_h * sum_loss2;
          } else {
            f_grad_input -= c_h * c_invvar * sum_loss2;
          }
        }
        f_grad_input *= term1;
        k_grad_input[l] = static_cast<T>(f_grad_input);
      }
    }
    // prevent race where buf is written again before reads are done
    __syncthreads();
  }
}


template<typename T, typename U, typename V=T>
void HostApplyLayerNorm(
    V* output,
    U* mean,
    U* invvar,
    const T* input,
    int n1,
    int n2,
    double epsilon,
    const V* gamma,
    const V* beta
    )
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const dim3 threads(32,4,1);
    const uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
    const dim3 blocks(1, std::min((uint64_t)n1, maxGridY), 1);
    int nshared =
        threads.y > 1 ?
            threads.y*sizeof(U)+(threads.y/2)*sizeof(U) :
            0;
    cuApplyLayerNorm<<<blocks, threads, nshared, stream>>>(
      output, mean, invvar, input, n1, n2, U(epsilon), gamma, beta);
}

void cuda_layer_norm(
    at::Tensor* output,
    at::Tensor* mean,
    at::Tensor* invvar,
    at::Tensor* input,
    int n1,
    int n2,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor* gamma,
    at::Tensor* beta,
    double epsilon)
{
    using namespace at;
    DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
        input->scalar_type(), output->scalar_type(), "layer_norm_cuda_kernel",
        using accscalar_t = at::acc_type<scalar_t_in, true>;
        HostApplyLayerNorm<scalar_t_in, accscalar_t, scalar_t_out>(
          output->DATA_PTR<scalar_t_out>(),
              mean->DATA_PTR<accscalar_t>(),
          invvar->DATA_PTR<accscalar_t>(),
          input->DATA_PTR<scalar_t_in>(),
          n1,n2,
          epsilon,
          gamma != NULL ? gamma->DATA_PTR<scalar_t_out>() : NULL,
          beta != NULL ? beta->DATA_PTR<scalar_t_out>() : NULL);
      )
}


template<typename T, typename U=float, typename V=T>
void HostLayerNormGradient(
    const V* dout,
    const U* mean,
    const U* invvar,
    at::Tensor* input_or_output,
    int n1,
    int n2,
    const V* gamma,
    const V* beta,
    double epsilon,
    T* grad_input,
    V* grad_gamma,
    V* grad_beta,
    bool memory_efficient
    )
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (gamma != NULL && beta != NULL) {
      // compute grad_gamma(j) and grad_beta(j)
      const int part_size = 16;
      const dim3 threads2(32,4,1);
      const dim3 blocks2((n2+threads2.x-1)/threads2.x,part_size,1);
      const int nshared2_a = 2 * sizeof(U) * threads2.y * threads2.y * (threads2.x + 1);
      const int nshared2_b = threads2.x * threads2.y * sizeof(U);
      const int nshared2 = nshared2_a > nshared2_b ? nshared2_a : nshared2_b;
      // note (mkozuki): I can hard code part_grad_gamma's dtype as float given that
      // the `cuda_layer_norm_gradient` doesn't support double.
      const auto part_grad_dtype =
        (input_or_output->scalar_type() == at::ScalarType::Half || input_or_output->scalar_type() == at::ScalarType::BFloat16) ?
        at::ScalarType::Float :
        input_or_output->scalar_type();
      at::Tensor part_grad_gamma = at::empty({part_size,n2}, input_or_output->options().dtype(part_grad_dtype));
      at::Tensor part_grad_beta = at::empty_like(part_grad_gamma);
      BOOL_SWITCH(memory_efficient, MemoryEfficient, [&]{
        auto kernel = &cuComputePartGradGammaBeta<T, U, V, MemoryEfficient>;
        kernel<<<blocks2, threads2, nshared2, stream>>>(
                        dout,
                        input_or_output->DATA_PTR<T>(),
                        n1,n2,
                        mean,
                        invvar,
                        U(epsilon),
                        gamma,
                        beta,
                        part_grad_gamma.DATA_PTR<U>(),
                        part_grad_beta.DATA_PTR<U>(),
                        epsilon,
                        false);
      });

      const dim3 threads3(32,8,1);
      const dim3 blocks3((n2+threads2.x-1)/threads2.x,1,1);
      const int nshared3 = threads3.x * threads3.y * sizeof(U);
      cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
                      part_grad_gamma.DATA_PTR<U>(),
                      part_grad_beta.DATA_PTR<U>(),
                      part_size,
                      n1,n2,
                      grad_gamma,
                      grad_beta,
                      false);
    }

    // compute grad_input
    const uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
    const dim3 blocks1(1, std::min((uint64_t)n1, maxGridY), 1);
    const dim3 threads1(32,4,1);
    int nshared =
            threads1.y > 1 ?
            threads1.y*threads1.x*sizeof(U) :
            0;
    BOOL_SWITCH(memory_efficient, MemoryEfficient, [&] {
      auto kernel = cuComputeGradInput<T, U, V, MemoryEfficient>;
      kernel<<<blocks1, threads1, nshared, stream>>>(
              dout,
              input_or_output->DATA_PTR<T>(),
              n1,n2,
              mean,
              invvar,
              U(epsilon),
              gamma,
              beta,
              grad_input,
              epsilon,
              false);
    });
}

void cuda_layer_norm_gradient(
    at::Tensor* dout,
    at::Tensor* mean,
    at::Tensor* invvar,
    at::Tensor* input_or_output,
    int n1,
    int n2,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor* gamma,
    at::Tensor* beta,
    double epsilon,
    at::Tensor* grad_input,
    at::Tensor* grad_gamma,
    at::Tensor* grad_beta,
    bool memory_efficient)
{
    using namespace at;
    // we can do away with `accscalar_t` as there're only three dtypes: fp32, fp16, bf16
    DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
      input_or_output->scalar_type(), gamma == NULL ? input_or_output->scalar_type() :  gamma->scalar_type(), "cuComputeGradInput",
      using accscalar_t = at::acc_type<scalar_t_in, true>;
      HostLayerNormGradient(
        dout->DATA_PTR<scalar_t_out>(),
        mean != NULL ? mean->DATA_PTR<accscalar_t>() : NULL,
        invvar->DATA_PTR<accscalar_t>(),
        input_or_output,
        n1,n2,
            // TMJ pass NULL argument for gamma, beta, grad_gamma and grad_beta
            // if gamma Tensor is NULL on input.
        gamma != NULL ? gamma->DATA_PTR<scalar_t_out>() : NULL,
        gamma != NULL ? beta->DATA_PTR<scalar_t_out>() : NULL,
        epsilon,
        grad_input->DATA_PTR<scalar_t_in>(),
        gamma != NULL ? grad_gamma->DATA_PTR<scalar_t_out>() : NULL,
        gamma != NULL ? grad_beta->DATA_PTR<scalar_t_out>() : NULL,
        memory_efficient);
    )
}
