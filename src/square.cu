#include <cuda_runtime.h>
#include <stdlib.h>
#define DATA_SIZE 1048576
int data[DATA_SIZE];
void GenerateNumbers(int *number, int size)
{
    for (int i = 0; i < size; i++) {
        number[i] = rand() % 10;
    }
}
__global__ static void sumOfSquares(int *num, int* result)
{
    int sum = 0;
    int i;
    for (i = 0; i < DATA_SIZE; i++) {
        sum += num[i] * num[i] * num[i];
    }
    *result = sum;
}

int Cal_Squares_Sum(){
    //生成随机数
    GenerateNumbers(data, DATA_SIZE);
    //把数据复制到显卡内存中
    int* gpudata, *result;
    //cudaMalloc 取得一块显卡内存 ( 其中result用来存储计算结果 )
    cudaMalloc((void**)&gpudata, sizeof(int)* DATA_SIZE);
    cudaMalloc((void**)&result, sizeof(int));
    //cudaMemcpy 将产生的随机数复制到显卡内存中
    //cudaMemcpyHostToDevice - 从内存复制到显卡内存
    //cudaMemcpyDeviceToHost - 从显卡内存复制到内存
    cudaMemcpy(gpudata, data, sizeof(int)* DATA_SIZE, cudaMemcpyHostToDevice);
    // 在CUDA 中执行函数 语法：函数名称<<<block 数目, thread 数目, shared memory 大小>>>(参数...);
    sumOfSquares << <1, 1, 0 >> >(gpudata, result);
    //把结果从显示芯片复制回主内存
    int sum;
    //cudaMemcpy 将结果从显存中复制回内存
    cudaMemcpy(&sum, result, sizeof(int), cudaMemcpyDeviceToHost);
    int sum2 = 0;
    for(int i=0; i<DATA_SIZE; i++){
        sum2 += data[i] * data[i] * data[i];
    }
    //Free
    cudaFree(gpudata);
    cudaFree(result);
    return sum == sum2;
}