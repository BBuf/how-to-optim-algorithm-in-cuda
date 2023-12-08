#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <cuda_fp16.h>
using namespace std;
#define N 32*1024*1024
#define kBlockSize 256

// vector inner product

template<typename T, size_t pack_size>
struct alignas(sizeof(T) * pack_size) Pack {
  T elem[pack_size];
};

template<typename T, int32_t pack_size>
__device__ __inline__ void AtomicAdd(Pack<T, pack_size>* address,
                                     T val) {
#pragma unroll
  for (int i = 0; i < pack_size; ++i) {
    atomicAdd(reinterpret_cast<T*>(address) + i, static_cast<T>(val));
  }
}

template<>
__device__ __inline__ void AtomicAdd<half, 2>(Pack<half, 2>* address, half val) {
  half2 h2_val;
  h2_val.x = static_cast<half>(val);
  h2_val.y = static_cast<half>(val);
  atomicAdd(reinterpret_cast<half2*>(address), h2_val);
}

template<typename T, int32_t pack_size>
__global__ void dot(Pack<T, pack_size>* a, Pack<T, pack_size>* b, Pack<T, pack_size>* c, int n){
    const int nStep = gridDim.x * blockDim.x;
    T temp = 0.0;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    while (gid < n / pack_size) {
        for (int i = 0; i < pack_size; i++) {
            temp = temp + a[gid].elem[i] * b[gid].elem[i];
        }
        gid += nStep;
    }
    AtomicAdd<T, pack_size>(c, temp);
}

int main(){
    half *x_host = (half*)malloc(N*sizeof(half));
    half *x_device;
    cudaMalloc((void **)&x_device, N*sizeof(half));
    for (int i = 0; i < N; i++) x_host[i] = 0.1;
    cudaMemcpy(x_device, x_host, N*sizeof(half), cudaMemcpyHostToDevice);
    Pack<half, 2>* x_pack = reinterpret_cast<Pack<half, 2>*>(x_device);

    half *y_host = (half*)malloc(N*sizeof(half));
    half *y_device;
    cudaMalloc((void **)&y_device, N*sizeof(half));
    for (int i = 0; i < N; i++) y_host[i] = 0.1;
    cudaMemcpy(y_device, y_host, N*sizeof(half), cudaMemcpyHostToDevice);
    Pack<half, 2>* y_pack = reinterpret_cast<Pack<half, 2>*>(y_device);

    half *output_host = (half*)malloc(2 * sizeof(half));
    half *output_device;
    cudaMalloc((void **)&output_device, 2 * sizeof(half));
    cudaMemset(output_device, 0, sizeof(half) * 2);
    Pack<half, 2>* output_pack = reinterpret_cast<Pack<half, 2>*>(output_device);

    int32_t block_num = (N + kBlockSize - 1) / kBlockSize;
    dim3 grid(block_num, 1);
    dim3 block(kBlockSize, 1);
    dot<half, 2><<<grid, block>>>(x_pack, y_pack, output_pack, N);
    cudaMemcpy(output_host, output_device, 2 * sizeof(half), cudaMemcpyDeviceToHost);
    printf("%.6f\n", static_cast<double>(output_host[0]));
    free(x_host);
    free(y_host);
    free(output_host);
    cudaFree(x_device);
    cudaFree(y_device);
    cudaFree(output_device);
}

