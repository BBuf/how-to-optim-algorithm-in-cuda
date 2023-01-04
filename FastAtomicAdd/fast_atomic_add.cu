#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <cuda_fp16.h>
using namespace std;
#define N 32*1024*2024
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

template<typename T>
struct DefaultComputeType {
  using type = T;
};

template<typename T, size_t pack_size>
struct alignas(sizeof(T) * pack_size) Pack {
  T elem[pack_size];
};

template<typename T, int32_t pack_size>
__device__ __inline__ void AtomicAdd(Pack<T, pack_size>* address,
                                     double val) {
#pragma unroll
  for (int i = 0; i < pack_size; ++i) {
    atomicAdd(reinterpret_cast<T*>(address) + i, static_cast<T>(val));
  }
}

template<>
__device__ __inline__ void AtomicAdd<half, 2>(Pack<half, 2>* address, double val) {
  half2 h2_val;
  h2_val.x = static_cast<half>(val);
  h2_val.y = static_cast<half>(val);
  atomicAdd(reinterpret_cast<half2*>(address), h2_val);
}

template<typename T, int32_t pack_size>
__global__ void dot(Pack<T, pack_size>* a, Pack<T, pack_size>* b, Pack<T, pack_size>* c, int n){
    const int nStep = gridDim.x * blockDim.x;
    double temp = 0.0;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    while (gid < n) {
        for (int i = 0; i < pack_size; i++) {
            temp += static_cast<double>(a[gid].elem[i]) * static_cast<double>(b[gid].elem[i]);
        }
        gid += nStep;
    }
    AtomicAdd<T, pack_size>(c, temp);
}

int main(){
    half *x_host = (half*)malloc(N*sizeof(half));
    half *x_device;
    cudaMalloc((void **)&x_device, N*sizeof(half));
    for (int i = 0; i < N; i++) x_host[i] = 0.001;
    cudaMemcpy(x_device, x_host, N*sizeof(half), cudaMemcpyHostToDevice);
    Pack<half, 2>* x_pack = reinterpret_cast<Pack<half, 2>*>(x_device);

    half *y_host = (half*)malloc(N*sizeof(half));
    half *y_device;
    cudaMalloc((void **)&y_device, N*sizeof(half));
    for (int i = 0; i < N; i++) y_host[i] = 0.001;
    cudaMemcpy(y_device, y_host, N*sizeof(half), cudaMemcpyHostToDevice);
    Pack<half, 2>* y_pack = reinterpret_cast<Pack<half, 2>*>(y_device);

    half *output_host = (half*)malloc(2 * sizeof(half));
    half *output_device;
    cudaMalloc((void **)&output_device, 2 * sizeof(half));
    Pack<half, 2>* output_pack = reinterpret_cast<Pack<half, 2>*>(output_device);

    int32_t block_num = (N + kBlockSize - 1) / kBlockSize;
    dim3 grid(block_num, 1);
    dim3 block(kBlockSize, 1);
    dot<half, 2><<<grid, block>>>(x_pack, y_pack, output_pack, N);
    cudaMemcpy(output_device, output_host, 2 * sizeof(half), cudaMemcpyDeviceToHost);
    printf("%.6f\n", static_cast<float>(output_host[0])); // the right answer should be 0.001*0.001*N=0.000001*32*1024*1024=33.554432

    free(x_host);
    free(y_host);
    free(output_host);
    cudaFree(x_device);
    cudaFree(y_device);
    cudaFree(output_device);
}

