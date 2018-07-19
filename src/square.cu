#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand_mtgp32_kernel.h>

#define DATA_SIZE 10485760
#define THREAD_NUM 128
int data[DATA_SIZE];

void GenerateNumbers(int *number, int size)
{
    for (int i = 0; i < size; i++) {
        number[i] = i % 10;
    }
}
__global__ static void sumOfSquares(int *num, int* result, clock_t* time)
{
    //表示目前的thread 是第几个 thread（由0开始计算）
    const int tid = threadIdx.x;
    //计算每个线程需要完成的量
    const int size = DATA_SIZE / THREAD_NUM;
    int sum = 0;
    int i;
    clock_t  start;
    //只在 thread 0（即 threadIdx.x = 0 的时候）进行记录
    if(tid == 0) start = clock();
    for (i = tid; i < DATA_SIZE; i+=THREAD_NUM) {
        sum += num[i] * num[i] * num[i];
    }
//    for (i = tid * size; i < (tid+1)*size; i++) {
//        sum += num[i] * num[i] * num[i];
//    }
    result[tid] = sum;
    //计算时间的动作，只在 thread 0（即 threadIdx.x = 0 的时候）进行
    if(tid == 0) *time = clock() - start;
}

int Cal_Squares_Sum(){
    //生成随机数
    GenerateNumbers(data, DATA_SIZE);
    //把数据复制到显卡内存中
    int* gpudata, *result;
    clock_t* time;
    //cudaMalloc 取得一块显卡内存 ( 其中result用来存储计算结果 )
    cudaMalloc((void**)&gpudata, sizeof(int)* DATA_SIZE);
    cudaMalloc((void**)&result, sizeof(int)*THREAD_NUM);
    cudaMalloc((void**)&time, sizeof(clock_t));
    //cudaMemcpy 将产生的随机数复制到显卡内存中
    //cudaMemcpyHostToDevice - 从内存复制到显卡内存
    //cudaMemcpyDeviceToHost - 从显卡内存复制到内存
    cudaMemcpy(gpudata, data, sizeof(int)* DATA_SIZE, cudaMemcpyHostToDevice);
    // 在CUDA 中执行函数 语法：函数名称<<<block 数目, thread 数目, shared memory 大小>>>(参数...);
    sumOfSquares << <1, THREAD_NUM, 0 >> >(gpudata, result, time);
    //把结果从显示芯片复制回主内存
    int sum[THREAD_NUM];
    clock_t used_time;
    //cudaMemcpy 将结果从显存中复制回内存
    cudaMemcpy(&sum, result, sizeof(int) * THREAD_NUM, cudaMemcpyDeviceToHost);
    cudaMemcpy(&used_time, time, sizeof(clock_t), cudaMemcpyDeviceToHost);
    int sum2 = 0;
    for(int i=0; i<DATA_SIZE; i++){
        sum2 += data[i] * data[i] * data[i];
    }
    //used_time是GPU的时钟周期（timestamp），需要除以GPU的运行频率才能得到以秒为单位的时间
    printf("Time: %d\n", used_time);
    //Free
    cudaFree(gpudata);
    cudaFree(result);
    cudaFree(time);
    int final_sum = 0;
    for(int i = 0; i < THREAD_NUM; i++){
        final_sum += sum[i];
    }
    printf("GPU sum: %d gputime: %.10f\n", final_sum, 1.0*used_time/1076000000.0);
    final_sum = 0;
    clock_t cpu_start_time = clock();
    for(int i=0; i<DATA_SIZE; i++){
        final_sum += data[i] * data[i] * data[i];
    }
    clock_t cpu_used_time = clock()-cpu_start_time;
    printf("CPU sum: %d: cputime: %d\n", final_sum, cpu_used_time);
    printf("Speed Ratio %.5f\n", (used_time*1.0)/(cpu_used_time*1.0));
    return 1;
}