// Copy from https://github.com/Liu-xiandong/How_to_optimize_in_GPU/blob/master/sgemv/ComplexHalfGemv.cu
#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h> 
#include <cuComplex.h>
#include <thrust/complex.h>
#include "cuHalfComplex.cuh"

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

template <unsigned int WarpSize>
__device__ __forceinline__ half warpReduceSum(half sum) {
    if (WarpSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (WarpSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
    if (WarpSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    if (WarpSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    if (WarpSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
    return sum;
}

__global__ void ComplexHalfGemv( 
    cuHalfComplex * __restrict__ A,
    cuHalfComplex * __restrict__ x,
    cuHalfComplex * __restrict__ y, 
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
        cuHalfComplex res = cuHalfComplex(0,0);
        int kIteration = (N/warp_size)/4;
        if(kIteration==0) kIteration=1;
        A = &A[current_row*N];

        #pragma unroll
        for(int i=0; i< kIteration; i++){
            int current_col_vec = (i*warp_size + laneId)/4;
            float4 current_val= reinterpret_cast<float4 *>(A)[current_col_vec]; //FETCH_FLOAT4(A[current_col_vec]); 
            float4 current_x = reinterpret_cast<float4 *>(x)[current_col_vec];
            cuHalfComplex val0 =  reinterpret_cast<cuHalfComplex *>(&current_val)[0];
            cuHalfComplex val1 =  reinterpret_cast<cuHalfComplex *>(&current_val)[1];
            cuHalfComplex val2 =  reinterpret_cast<cuHalfComplex *>(&current_val)[2];
            cuHalfComplex val3 =  reinterpret_cast<cuHalfComplex *>(&current_val)[3];
            cuHalfComplex x0 =  reinterpret_cast<cuHalfComplex *>(&current_x)[0];
            cuHalfComplex x1 =  reinterpret_cast<cuHalfComplex *>(&current_x)[1];
            cuHalfComplex x2 =  reinterpret_cast<cuHalfComplex *>(&current_x)[2];
            cuHalfComplex x3 =  reinterpret_cast<cuHalfComplex *>(&current_x)[3];

            res = res + val0 * x0;
            res = res + val1 * x1;
            res = res + val2 * x2;
            res = res + val3 * x3;
        }
        half res_r = res.r;
        half res_i = res.i;
        res_r = warpReduceSum<warp_size>(res_r);
        res_i = warpReduceSum<warp_size>(res_i);
        if(laneId == 0) y[current_row]=cuHalfComplex(res_r,res_i);
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("usage: ./main [M] [N]\n");
        exit(0);
    }
    size_t M = atoi(argv[1]);
    size_t N = atoi(argv[2]);

    size_t bytes_A = sizeof(cuHalfComplex) * M * N;
    size_t bytes_x = sizeof(cuHalfComplex) * N;
    size_t bytes_y = sizeof(cuHalfComplex) * M;
    size_t bytes_y1 = sizeof(float2) * M;
    cuHalfComplex* h_A = (cuHalfComplex*)malloc(bytes_A);
    cuHalfComplex* h_x = (cuHalfComplex*)malloc(bytes_x);
    cuHalfComplex* h_y = (cuHalfComplex*)malloc(bytes_y);
    float2* h_y1 = (float2*)malloc(bytes_y1);

    cuHalfComplex* d_A;
    cuHalfComplex* d_x;
    cuHalfComplex* d_y;

    checkCudaErrors(cudaMalloc((void**)&d_A, bytes_A));
    checkCudaErrors(cudaMalloc((void**)&d_x, bytes_x));
    checkCudaErrors(cudaMalloc((void**)&d_y, bytes_y));

    // 生成A的数据
    for( int i = 0; i < M * N; i++ ) {
        half x = 1;
        half y = 1;
        h_A[i] = cuHalfComplex(x,y);
    }

    // 生成x的数据
    for( int i = 0; i < N; i++ ) {
        half x = 1;
        half y = 1;
        h_x[i] = cuHalfComplex(x,y);
    }
    memset(h_y, 0, M * sizeof(cuHalfComplex));
    memset(h_y1, 0, M * sizeof(float2));

    int nIter = 1000;
    checkCudaErrors(cudaMemcpy( d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_x, h_x, bytes_x, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_y, h_y, bytes_y, cudaMemcpyHostToDevice));
    for (int run = 0 ; run < nIter; run ++ ) {
        dim3 dimGrid(M/4);
        dim3 dimBlock(32,4);
        ComplexHalfGemv<<< dimGrid, dimBlock >>>(d_A, d_x, d_y, M, N);
    }
    checkCudaErrors(cudaMemcpy( h_y, d_y, bytes_y, cudaMemcpyDeviceToHost));

    // compute the result in cpu
    // fp16 is not support in CPU, so use float
    for(int i=0; i<M; i++){
        float result_r = 0;
        float result_i = 0;
        for(int j=0; j<N; j++){
            float a_r = h_A[i*N+j].r;
            float a_i = h_A[i*N+j].i;
            float b_r = h_x[j].r;
            float b_i = h_x[j].i;
            float res_r = a_r*b_r - a_i*b_i;
            float res_i = a_i*b_r + a_r*b_i;
            result_r += res_r;
            result_i += res_i;
        }
        float2 result;
        result.x = result_r;
        result.y = result_i;
        h_y1[i] = result;
    }

    // simple check, not reasonable
    double eps = 1.e-3;
    bool correct = true;
    for (int i = 0; i < M; i++) {
        double abs_err = fabs((float)(h_y[i].r) - h_y1[i].x)+fabs((float)(h_y[i].i) - h_y1[i].y);
        if (abs_err > eps) {
            printf("Error! Matrix[%05d]=(%.8f,%.8f), ref=(%.8f,%.8f) error term is > %E\n",
                    i, (float)(h_y[i].r), (float)(h_y[i].i), (h_y1[i].x), (h_y1[i].y), eps);
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
