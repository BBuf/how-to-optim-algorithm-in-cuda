#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 32*1024*1024
#define BLOCK_SIZE 256

template <unsigned int blockSize>
__device__ void warpReduce(volatile float* cache,int tid){
    if(blockSize >= 64)cache[tid]+=cache[tid+32];
    if(blockSize >= 32)cache[tid]+=cache[tid+16];
    if(blockSize >= 16)cache[tid]+=cache[tid+8];
    if(blockSize >= 8)cache[tid]+=cache[tid+4];
    if(blockSize >= 4)cache[tid]+=cache[tid+2];
    if(blockSize >= 2)cache[tid]+=cache[tid+1];
}

template <unsigned int blockSize>
__global__ void reduce_v5(float *g_idata,float *g_odata){
    __shared__ float sdata[BLOCK_SIZE];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();

    // do reduction in shared mem
    if(blockSize>=512){
        if(tid<256){
            sdata[tid]+=sdata[tid+256];
        }
        __syncthreads();
    }
    if(blockSize>=256){
        if(tid<128){
            sdata[tid]+=sdata[tid+128];
        }
        __syncthreads();
    }
    if(blockSize>=128){
        if(tid<64){
            sdata[tid]+=sdata[tid+64];
        }
        __syncthreads();
    }
    
    // write result for this block to global mem
    if(tid<32)warpReduce<blockSize>(sdata,tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int main() {
    float *input_host = (float*)malloc(N*sizeof(float));
    float *input_device;
    cudaMalloc((void **)&input_device, N*sizeof(float));
    for (int i = 0; i < N; i++) input_host[i] = 2.0;
    cudaMemcpy(input_device, input_host, N*sizeof(float), cudaMemcpyHostToDevice);

    int32_t block_num = (N + BLOCK_SIZE - 1) / BLOCK_SIZE / 2;
    float *output_host = (float*)malloc((block_num) * sizeof(float));
    float *output_device;
    cudaMalloc((void **)&output_device, (block_num) * sizeof(float));
    
    dim3 grid(block_num, 1);
    dim3 block(BLOCK_SIZE, 1);
    reduce_v5<BLOCK_SIZE><<<grid, block>>>(input_device, output_device);
    cudaMemcpy(output_host, output_device, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    return 0;
}


