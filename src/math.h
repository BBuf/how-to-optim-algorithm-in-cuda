//
// Created by zxy on 18-7-18.
//

#ifndef CUDA_LEARN_SAMPLES_MATH_H
#define CUDA_LEARN_SAMPLES_MATH_H

#include <cuda_runtime.h>

int zxy_add(int a, int b); //使用CUDA实现了两个数加法
int zxy_cal_squares(); //计算立方和
void zxy_matrix_mul(); //计算两个矩阵相乘
bool InitCUDA(); //对CUDA进行初始化
void printDeviceProp(const cudaDeviceProp &prop); //打印设备信息的函数

#endif //CUDA_LEARN_SAMPLES_MATH_H