#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


#include "repeat.h"

__global__ void kpath_test1 (unsigned int *ts, unsigned int* out, int p1, int p2, int its)
{
	unsigned int t1 = p1;
	unsigned int t2 = p2;
	unsigned int start_time = 0 , stop_time = 0;
	unsigned int tid = (blockIdx.x*blockDim.x + threadIdx.x);

	__syncthreads();

	if (tid < 16)
	{
		start_time = clock();
		repeat64(t1+=t2;t2+=t1;)
		stop_time = clock();
	}
	else 
	{
		start_time = clock();
		repeat64(t1-=t2;t2-=t1;)
		stop_time = clock();
	}

	out[0] = (t1 + t2);		
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2] = start_time; 
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2+1] = stop_time; 
}


void measure_path()
{
	unsigned int ts[1024];			// ts, output from kernel. Two elements used per thread.
	unsigned int *d_ts;
	unsigned int *d_out;			// Unused memory for storing output

	//run two warps
    	dim3 Db = dim3(32 * 2);
    	dim3 Dg = dim3(1,1,1);


	cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);

	int warpsize = deviceProp.warpSize;

	
	// Allocate device array.
	cudaError_t errcode;
	if (cudaSuccess != (errcode = cudaMalloc((void**)&d_ts, sizeof(ts))))
	{
		printf ("cudaMalloc failed %s:%d\n", __FILE__, __LINE__);
		printf ("   %s\n", cudaGetErrorString(errcode));
		return;
	}
	if (cudaSuccess != cudaMalloc((void**)&d_out, 4))
	{
		printf ("cudaMalloc failed %s:%d\n", __FILE__, __LINE__);
		return;
	}

	cudaGetLastError();
    	printf ("Running __syncthreads() tests...\n");

	Db.x = warpsize * 1;
	printf("\nExecution order of two-way diverged warp: ");
	kpath_test1 <<<Dg, Db>>>(d_ts, d_out, 4, 6, 2);
	cudaThreadSynchronize();
	cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);

	if (cudaSuccess != cudaGetLastError())
		printf("failed.\n");

	printf("\n");
	printf("  warp:%2d, thread:%2d: start=%u, stop=%u\n", 0, 0, ts[0 * 2], ts[0*2+1]);
	printf("  warp:%2d, thread:%2d: start=%u, stop=%u\n", 0, 16, ts[16 * 2], ts[16*2+1]);


	printf("\n");

	cudaFree(d_ts);
	cudaFree(d_out);
}

