#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int add(int a,int b);

struct TEST
{
    int a;
    int b;
    int ADD();
};


