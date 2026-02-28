// copy from https://github.com/Liu-xiandong/How_to_optimize_in_GPU/blob/master/sgemv/Sgemv_v0.cu
#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h> 

// cal offset from row col and ld , in row-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// transfer float4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}



template <unsigned int WarpSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if (WarpSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (WarpSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
    if (WarpSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    if (WarpSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    if (WarpSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
    return sum;
}

// if N == 32
__global__ void Sgemv_v0( 
    float * __restrict__ A,
    float * __restrict__ x,
    float * __restrict__ y, 
    const int M,
    const int N) {
    // Block index
    int bx = blockIdx.x;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const int warp_size=32;
    int laneId= tx % warp_size;
    int current_row = blockDim.y * bx + ty;

    if(current_row < M){
        float res=0;
        int kIteration = N/warp_size;
        if(kIteration==0) kIteration=1;
        #pragma unroll
        for(int i=0; i< kIteration; i++){
            int current_col = i*warp_size + laneId;
            res += A[current_row*N + current_col] * x[current_col];
        }
        res = warpReduceSum<warp_size>(res);
        if(laneId==0) y[current_row]=res;
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("usage: ./main [M] [N]\n");
        exit(0);
    }
    size_t M = atoi(argv[1]);
    size_t N = atoi(argv[2]);

    size_t bytes_A = sizeof(float) * M * N;
    size_t bytes_x = sizeof(float) * N;
    size_t bytes_y = sizeof(float) * M;
    float* h_A = (float*)malloc(bytes_A);
    float* h_x = (float*)malloc(bytes_x);
    float* h_y = (float*)malloc(bytes_y);
    float* h_y1 = (float*)malloc(bytes_y);

    float* d_A;
    float* d_x;
    float* d_y;

    checkCudaErrors(cudaMalloc(&d_A, bytes_A));
    checkCudaErrors(cudaMalloc(&d_x, bytes_x));
    checkCudaErrors(cudaMalloc(&d_y, bytes_y));

    // 生成A的数据
    for( int i = 0; i < M * N; i++ ) {
        h_A[i] = (float)i/N;
    }

    // 生成x的数据
    for( int i = 0; i < N; i++ ) {
        h_x[i] = 1;
    }
    memset(h_y,0,M*sizeof(float));
    memset(h_y1,0,M*sizeof(float));

    int nIter = 1000;
    checkCudaErrors(cudaMemcpy( d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_x, h_x, bytes_x, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_y, h_y, bytes_y, cudaMemcpyHostToDevice));
    for (int run = 0 ; run < nIter; run ++ ) {
        dim3 dimGrid(M/4);
        dim3 dimBlock(32,4);
        Sgemv_v0<<< dimGrid, dimBlock >>>(d_A, d_x, d_y, M, N);
    }
    checkCudaErrors(cudaMemcpy( h_y, d_y, bytes_y, cudaMemcpyDeviceToHost));

    // cublas
    cublasHandle_t blas_handle;  
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0;
    checkCudaErrors(cudaMemcpy( d_y, h_y1, bytes_y, cudaMemcpyHostToDevice));
    for (int run = 0 ; run < nIter; run ++ ) {
        cublasSgemv (blas_handle, CUBLAS_OP_T, 
            N, M, &alpha, 
            d_A, N, d_x, 1, &beta, d_y, 1
        );
    }
    checkCudaErrors(cudaMemcpy( h_y1, d_y, bytes_y, cudaMemcpyDeviceToHost));
    cublasDestroy(blas_handle); 
    
    double eps = 1.e-6;  // machine zero
    bool correct = true;
    for (int i = 0; i < M; i++) {
        double abs_err = fabs(h_y[i] - h_y1[i]);
        double dot_length = M;
        double abs_val = fabs(h_y[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    i, h_y[i], h_y1[i], eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    
    // Free Memory
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    
    free(h_A);
    free(h_x);
    free(h_y);
    free(h_y1);
}