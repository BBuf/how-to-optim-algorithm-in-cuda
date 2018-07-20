// 	Multithreads with syncthreads block optimizer and Update way of cacaulate time
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand_mtgp32_kernel.h>
#include <device_launch_parameters.h>
#include <iostream>

using namespace std;

#define DATA_SIZE 1048576
#define MY_THREAD_NUM 256
#define BLOCK_NUM 128
//32 * 256 = 81902 threads
int data[DATA_SIZE];

void GenerateNumbers(int *number, int size)
{
    for (int i = 0; i < size; i++) {
        number[i] = i % 10;
    }
}
__global__ static void sumOfSquares(int *num, int* result)
{
    //声明一块共享内存
    extern __shared__ int shared[];
    //表示目前的thread是第几个thread(由0开始计算)
    const int tid = threadIdx.x;
    //表示目前的thread是第几个block(从0开始计算)
    const int bid = blockIdx.x;
    //计算每个线程需要完成的量
//    const int size = DATA_SIZE / MY_THREAD_NUM;
    shared[tid] = 0;
    int sum = 0;
    int i;
    //多线程使用运行内存连续优化技巧
//    for (i = tid; i < DATA_SIZE; i+=MY_THREAD_NUM) {
//        sum += num[i] * num[i] * num[i];
//    }
    //普通多线程
//    for (i = tid * size; i < (tid+1)*size; i++) {
//        sum += num[i] * num[i] * num[i];
//    }
    //多线程使用block和内存连续优化
//    for(i = bid * MY_THREAD_NUM + tid; i < DATA_SIZE; i+=BLOCK_NUM*MY_THREAD_NUM){
//        sum += num[i] * num[i] * num[i];
//    }
    //thread需要同时通过tid和bid来确定，同时不要忘记保证内存连续性
    for (i = bid * MY_THREAD_NUM + tid; i < DATA_SIZE; i += BLOCK_NUM * MY_THREAD_NUM) {
        shared[tid] += num[i] * num[i] * num[i];
    }
    //同步 保证每个 thread 都已经把结果写到 shared[tid] 里面
    __syncthreads();
    //使用线程0完成加和
    //树装加法
    int offset = 1, mask = 1;
    while(offset < MY_THREAD_NUM){
        if((tid&mask) == 0){
            shared[tid] += shared[tid+offset];
        }
        offset += offset;
        mask += offset;
        __syncthreads();
    }
    if(tid == 0){
        result[bid] = shared[0];
    }
}

int Cal_Squares_Sum(){
    //生成随机数
    GenerateNumbers(data, DATA_SIZE);
    //把数据复制到显卡内存中
    int* gpudata, *result;
    //cudaMalloc 取得一块显卡内存 ( 其中result用来存储计算结果 )
    cudaMalloc((void**)&gpudata, sizeof(int)* DATA_SIZE);
    cudaMalloc((void**)&result, sizeof(int)*BLOCK_NUM);
    //cudaMemcpy 将产生的随机数复制到显卡内存中
    //cudaMemcpyHostToDevice - 从内存复制到显卡内存
    //cudaMemcpyDeviceToHost - 从显卡内存复制到内存
    cudaMemcpy(gpudata, data, sizeof(int)* DATA_SIZE, cudaMemcpyHostToDevice);
    // 在CUDA 中执行函数 语法：函数名称<<<block 数目, thread 数目, shared memory 大小>>>(参数...);
    //使用event计算时间
    float time_elapsed=0;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);    //创建Event
    cudaEventCreate(&stop);
    cudaEventRecord( start,0);    //记录当前时间
    sumOfSquares <<<BLOCK_NUM, MY_THREAD_NUM, MY_THREAD_NUM * sizeof(int) >>>(gpudata, result);
    cudaEventRecord( stop,0);    //记录当前时间
    cudaEventSynchronize(start);    //Waits for an event to complete.
    cudaEventSynchronize(stop);    //Waits for an event to complete.Record之前的任务
    //把结果从显示芯片复制回主内存
    int sum[BLOCK_NUM];
    //cudaMemcpy 将结果从显存中复制回内存
    cudaMemcpy(&sum, result, sizeof(int) *  BLOCK_NUM, cudaMemcpyDeviceToHost);
    //used_time是GPU的时钟周期（timestamp），需要除以GPU的运行频率才能得到以秒为单位的时间
    //Free
    int final_sum = 0;
    for(int i = 0; i < BLOCK_NUM; i++){
        final_sum += sum[i];
    }
    cudaEventElapsedTime(&time_elapsed,start,stop);    //计算时间差
    cudaFree(gpudata);
    cudaFree(result);
    //采取新的计时策略 把每个 block 最早的开始时间，和最晚的结束时间相减，取得总运行时间
    cudaEventDestroy(start);    //destory the event
    cudaEventDestroy(stop);
    printf("GPU sum: %d GPU time: %.10f ms\n", final_sum, time_elapsed);

    final_sum = 0;
    clock_t cpu_start_time = clock();
    for(int i=0; i<DATA_SIZE; i++){
        final_sum += data[i] * data[i] * data[i];
    }
    clock_t cpu_used_time = clock()-cpu_start_time;
    double cpu_time = (double)(cpu_used_time)/CLOCKS_PER_SEC*1000.0;
    printf("CPU sum: %d CPU time: %.10f ms\n", final_sum, (double)(cpu_used_time)/CLOCKS_PER_SEC*1000.0);
    printf("Speed Ratio %.10f\n", 2.555/time_elapsed);
    return 1;
}