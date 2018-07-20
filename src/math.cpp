//
// Created by zxy on 18-7-18.
//

#include "math.h"
#include "add.cuh"
#include "square.cuh"
#include "matrix.cuh"
#include <iostream>
using namespace std;

int zxy_add(int a, int b){
    return add(a, b);
}

int zxy_cal_squares(){
    return Cal_Squares_Sum();
}

void zxy_matrix_mul(){
    MatrixMul();
}

void printDeviceProp(const cudaDeviceProp &prop)
{
    printf("Device Name : %s.\n", prop.name);
    printf("totalGlobalMem : %d.\n", prop.totalGlobalMem);
    printf("sharedMemPerBlock : %d.\n", prop.sharedMemPerBlock);
    printf("regsPerBlock : %d.\n", prop.regsPerBlock);
    printf("warpSize : %d.\n", prop.warpSize);
    printf("memPitch : %d.\n", prop.memPitch);
    printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("totalConstMem : %d.\n", prop.totalConstMem);
    printf("major.minor : %d.%d.\n", prop.major, prop.minor);
    printf("clockRate : %d.\n", prop.clockRate);
    printf("textureAlignment : %d.\n", prop.textureAlignment);
    printf("deviceOverlap : %d.\n", prop.deviceOverlap);
    printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
}

bool InitCUDA(){
    int count;
    cudaGetDeviceCount(&count);
    if (count == 0) {
        cout << "There is no device" << endl;
        return false;
    }
    int i;
    for (i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            printDeviceProp(prop);
            if (prop.major >= 1) {
                break;
            }
        }
    }
    if (i == count) {
        cout << "There is no device supporting CUDA 1.x." << endl;
        return false;
    }
    cudaSetDevice(i);
    return true;
}