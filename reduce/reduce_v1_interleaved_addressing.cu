#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 32*1024*1024
#define BLOCK_SIZE 256

__global__ void reduce_v1(float *g_idata,float *g_odata){
    __shared__ float sdata[BLOCK_SIZE];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        // if (tid % (2*s) == 0) {
        //     sdata[tid] += sdata[tid + s];
        // }
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int main() {
    float *input_host = (float*)malloc(N*sizeof(float));
    float *input_device;
    cudaMalloc((void **)&input_device, N*sizeof(float));
    for (int i = 0; i < N; i++) input_host[i] = 2.0;
    cudaMemcpy(input_device, input_host, N*sizeof(float), cudaMemcpyHostToDevice);

    int32_t block_num = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float *output_host = (float*)malloc((N / BLOCK_SIZE) * sizeof(float));
    float *output_device;
    cudaMalloc((void **)&output_device, (N / BLOCK_SIZE) * sizeof(float));
    
    dim3 grid(N / BLOCK_SIZE, 1);
    dim3 block(BLOCK_SIZE, 1);
    reduce_v1<<<grid, block>>>(input_device, output_device);
    cudaMemcpy(output_host, output_device, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    return 0;
}
