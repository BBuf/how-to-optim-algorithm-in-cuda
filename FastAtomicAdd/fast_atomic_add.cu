#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <cuda_fp16.h>
using namespace std;

// FastAdd is referenced from
// https://github.com/pytorch/pytorch/blob/396c3b1d88d7624938a2bb0b287f2a19f1e89bb4/aten/src/ATen/native/cuda/KernelUtils.cuh#L29
template<typename T, typename std::enable_if<std::is_same<half, T>::value>::type* = nullptr>
__device__ __forceinline__ void FastSpecializedAtomicAdd(T* base, size_t offset,
                                                         const size_t length, T value) {
#if ((defined(CUDA_VERSION) && (CUDA_VERSION < 10000)) \
     || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))
  cuda::atomic::Add(reinterpret_cast<half*>(base) + offset, static_cast<half>(value));
#else
  // Accounts for the chance base falls on an odd 16 bit alignment (ie, not 32 bit aligned)
  __half* target_addr = reinterpret_cast<__half*>(base + offset);
  bool low_byte = (reinterpret_cast<std::uintptr_t>(target_addr) % sizeof(__half2) == 0);

  if (low_byte && offset < (length - 1)) {
    __half2 value2;
    value2.x = value;
    value2.y = __float2half_rz(0);
    cuda::atomic::Add(reinterpret_cast<__half2*>(target_addr), value2);

  } else if (!low_byte && offset > 0) {
    __half2 value2;
    value2.x = __float2half_rz(0);
    value2.y = value;
    cuda::atomic::Add(reinterpret_cast<__half2*>(target_addr - 1), value2);

  } else {
    cuda::atomic::Add(reinterpret_cast<__half*>(base) + offset, static_cast<__half>(value));
  }
#endif
}

template<typename T, typename std::enable_if<!std::is_same<half, T>::value>::type* = nullptr>
__device__ __forceinline__ void FastSpecializedAtomicAdd(T* base, size_t offset,
                                                         const size_t length, T value) {
  cuda::atomic::Add(base + offset, value);
}

template<class T>
__device__ __forceinline__ void FastAdd(T* base, size_t offset, const size_t length, T value) {
  FastSpecializedAtomicAdd(base, offset, length, value);
}

// vector inner product

template<typename T>
__global__ void dot(T* a, T* b, T* c, int n){
    const int nStep = gridDim.x * blockDim.x;
    double temp = 0.0;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    while (gid < n) {
        temp += a[gid] * b[gid];
        gid += nStep;
    }
    atomicAdd(c, temp);
}

int main(){
    
}

