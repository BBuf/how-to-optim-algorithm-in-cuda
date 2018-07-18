//
// Created by zxy on 18-7-18.
//

#include "math.h"
#include "add.cuh"
#include "square.cuh"
#include <cuda_runtime.h>

int zxy_add(int a, int b){
    return add(a, b);
}

int zxy_cal_squares(){
    return Cal_Squares_Sum();
}