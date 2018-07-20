#include <cuda_runtime.h>
#include <curand_mtgp32_kernel.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cmath>

#define MY_THREAD_NUM 512
#define BLOCK_NUM 32
#define MY_RAND_MAX 32767
#define MATRIX_SIZE  1000
const int blocks_num = (MATRIX_SIZE + MY_THREAD_NUM - 1) / MY_THREAD_NUM;

//n * n  Matrix Multiply A * B = C
__global__ static void MatrixMulCUDA(const float *a, const float *b, float *c, int n)
{
    const int tid = threadIdx.x; //目前的thread是第几个thread
    const int bid = blockIdx.x; //目前的thread是属于哪个block
    const int idx = bid * THREAD_NUM + tid; //从bid和tid推出目前的thread应该计算的A矩阵的行数和B矩阵列数
    const int row = idx / n;
    const int col = idx % n;
    if(row < n && col < n){
        float sum = 0;
        for(int i = 0; i < n; i++){
            sum += a[row * n + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

//生成随机矩阵

void RandomMatrix(float *A, int n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            A[i * n + j] = (float)rand() / MY_RAND_MAX + (float)rand() / (MY_RAND_MAX * MY_RAND_MAX);
        }
    }
}

void MatrixMul(){
    //定义矩阵
    float *a, *b, *c, *d;
    int n = MATRIX_SIZE;
    a = (float*)malloc(sizeof(float)*n*n);
    b = (float*)malloc(sizeof(float)*n*n);
    c = (float*)malloc(sizeof(float)*n*n);
    d = (float*)malloc(sizeof(float)*n*n);
    srand(time(NULL));
    RandomMatrix(a, n);
    RandomMatrix(b, n);
    //把数据复制到显卡内存中
    float *cuda_a, *cuda_b, *cuda_c;
    //cudaMalloc 取得一块显存内存
    cudaMalloc((void**)&cuda_a, sizeof(float) * n * n);
    cudaMalloc((void**)&cuda_b, sizeof(float) * n * n);
    cudaMalloc((void**)&cuda_c, sizeof(float) * n * n);
    //cudaMemcpy 将产生的矩阵复制到显卡内存中
    //cudaMemcpyHostToDevice - 从内存复制到显卡内存
    //cudaMemcpyDeviceToHost - 从显卡内存复制到内存
    cudaMemcpy(cuda_a, a, sizeof(float)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, b, sizeof(float)*n*n, cudaMemcpyHostToDevice);
    float time_elapsed=0;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);    //创建Event
    cudaEventCreate(&stop);
    cudaEventRecord( start,0);    //记录当前时间
    MatrixMulCUDA<<<blocks_num, THREAD_NUM, 0>>>(cuda_a, cuda_b, cuda_c, n);
    cudaEventRecord( stop,0);    //记录当前时间
    cudaEventSynchronize(start);    //Waits for an event to complete.
    cudaEventSynchronize(stop);    //Waits for an event to complete.Record之前的任务
    //把结果从显示芯片复制回主内存
    //cudaMemcpy 将结果从显存中复制回内存
    cudaMemcpy(c, cuda_c, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&time_elapsed,start,stop);    //计算时间差
    cudaEventDestroy(start);    //destory the event
    cudaEventDestroy(stop);
    //Free
    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_c);
    printf("Matrix multiply GPU time: %.10f\n", time_elapsed);
    clock_t start_time = clock();
    //CPU计算矩阵乘法
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            double temp = 0;
            for(int k = 0; k < n; k++){
                temp += a[i * n + k]* b[k * n + j];
            }
            d[i * n + j] = temp;
        }
    }
    clock_t end_time = clock();
    //验证正确性和准确性
    float max_error = 0.0, average_error = 0;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            if(d[i * n + j] != 0){
                float err = fabs((c[i * n + j] - d[i * n + j]) / d[i * n + j]);
                if(max_error < err) max_error = err;
                average_error += err;
            }
        }
    }
    double cpu_time = (double)(end_time - start_time) / CLOCKS_PER_SEC * 1000.0;
    printf("Matrix multiply CPU time: %.10f\n", cpu_time);
    printf("Max error: %.10f Average error: %.10f\n", max_error, average_error / (n * n));
    printf("%.10f\n", cpu_time/time_elapsed);
}

