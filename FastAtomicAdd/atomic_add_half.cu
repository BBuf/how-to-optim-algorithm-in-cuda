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

__global__ void dot(half* a, half* b, half* c, int n){
    const int nStep = gridDim.x * blockDim.x;
    half temp = 0.0;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    while (gid < n) {
        temp = temp + a[gid] * b[gid];
        gid += nStep;
    }
    atomicAdd(c, temp);
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

    half *output_host = (half*)malloc(sizeof(half));
    half *output_device;
    cudaMalloc((void **)&output_device, sizeof(half));
    cudaMemset(output_device, 0, sizeof(half));

    int32_t block_num = (N + kBlockSize - 1) / kBlockSize;
    dim3 grid(block_num, 1);
    dim3 block(kBlockSize, 1);
    dot<<<grid, block>>>(x_device, y_device, output_device, N);
    cudaMemcpy(output_host, output_device, sizeof(half), cudaMemcpyDeviceToHost);
    printf("%.6f\n", static_cast<double>(output_host[0]));
    free(x_host);
    free(y_host);
    free(output_host);
    cudaFree(x_device);
    cudaFree(y_device);
    cudaFree(output_device);
}

