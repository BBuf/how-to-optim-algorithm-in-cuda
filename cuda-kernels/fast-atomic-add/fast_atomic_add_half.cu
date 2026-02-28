#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <cuda_fp16.h>
using namespace std;
#define N 32*1024*1024
#define kBlockSize 256


// FastAdd is referenced from
// https://github.com/pytorch/pytorch/blob/396c3b1d88d7624938a2bb0b287f2a19f1e89bb4/aten/src/ATen/native/cuda/KernelUtils.cuh#L29
template<typename T, typename std::enable_if<std::is_same<half, T>::value>::type* = nullptr>
__device__ __forceinline__ void FastSpecializedAtomicAdd(T* base, size_t offset,
                                                         const size_t length, T value) {
#if ((defined(CUDA_VERSION) && (CUDA_VERSION < 10000)) \
     || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))
  atomicAdd(reinterpret_cast<half*>(base) + offset, static_cast<half>(value));
#else
  // Accounts for the chance base falls on an odd 16 bit alignment (ie, not 32 bit aligned)
  __half* target_addr = reinterpret_cast<__half*>(base + offset);
  bool low_byte = (reinterpret_cast<std::uintptr_t>(target_addr) % sizeof(__half2) == 0);

  if (low_byte && offset < (length - 1)) {
    __half2 value2;
    value2.x = value;
    value2.y = __float2half_rz(0);
    atomicAdd(reinterpret_cast<__half2*>(target_addr), value2);

  } else if (!low_byte && offset > 0) {
    __half2 value2;
    value2.x = __float2half_rz(0);
    value2.y = value;
    atomicAdd(reinterpret_cast<__half2*>(target_addr - 1), value2);

  } else {
    atomicAdd(reinterpret_cast<__half*>(base) + offset, static_cast<__half>(value));
  }
#endif
}

template<typename T, typename std::enable_if<!std::is_same<half, T>::value>::type* = nullptr>
__device__ __forceinline__ void FastSpecializedAtomicAdd(T* base, size_t offset,
                                                         const size_t length, T value) {
  atomicAdd(base + offset, value);
}

template<class T>
__device__ __forceinline__ void FastAdd(T* base, size_t offset, const size_t length, T value) {
  FastSpecializedAtomicAdd(base, offset, length, value);
}


// vector inner product

__global__ void dot(half* a, half* b, half* c, int n){
    const int nStep = gridDim.x * blockDim.x;
    half temp = 0.0;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    while (gid < n) {
        temp = temp + a[gid] * b[gid];
        gid += nStep;
    }
    // atomicAdd(c, temp);
    FastAdd(c, 0, 2, temp);
}

int main(){
    half *x_host = (half*)malloc(N*sizeof(half));
    half *x_device;
    cudaMalloc((void **)&x_device, N*sizeof(half));
    for (int i = 0; i < N; i++) x_host[i] = 0.1;
    cudaMemcpy(x_device, x_host, N*sizeof(half), cudaMemcpyHostToDevice);

    half *y_host = (half*)malloc(N*sizeof(half));
    half *y_device;
    cudaMalloc((void **)&y_device, N*sizeof(half));
    for (int i = 0; i < N; i++) y_host[i] = 0.1;
    cudaMemcpy(y_device, y_host, N*sizeof(half), cudaMemcpyHostToDevice);

    half *output_host = (half*)malloc(sizeof(half) * 2);
    half *output_device;
    cudaMalloc((void **)&output_device, sizeof(half) * 2);
    cudaMemset(output_device, 0, sizeof(half) * 2);

    int32_t block_num = (N + kBlockSize - 1) / kBlockSize;
    dim3 grid(block_num, 1);
    dim3 block(kBlockSize, 1);
    dot<<<grid, block>>>(x_device, y_device, output_device, N);
    cudaMemcpy(output_host, output_device, sizeof(half) * 2, cudaMemcpyDeviceToHost);
    printf("%.6f\n", static_cast<double>(output_host[0]));
    free(x_host);
    free(y_host);
    free(output_host);
    cudaFree(x_device);
    cudaFree(y_device);
    cudaFree(output_device);
}

