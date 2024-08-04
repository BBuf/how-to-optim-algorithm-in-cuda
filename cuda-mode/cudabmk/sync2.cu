#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


#include "repeat.h"



__global__ void ksync2_test4 (unsigned int *ts, unsigned int* out, int p1, int p2, int its)
{
	unsigned int t1 = p1;
	unsigned int t2 = p2;
	unsigned int start_time = 0 , stop_time = 0, start_time2 = 0, stop_time2 = 0;
	unsigned int tid = (blockIdx.x*blockDim.x + threadIdx.x);
	unsigned int tid2 = tid%32;
	__shared__ volatile unsigned int count[32];
	__shared__ volatile unsigned int count2[32];

	count[tid2] = 0;
	count2[tid2] = 0;
	__syncthreads();


	if (tid/32 == 0)
	{
		if (tid < 16) 
		{
			start_time = clock();
			__syncthreads(); 
			stop_time = clock();
			repeat16(t1&=t2;t2&=t1;)
			count2[tid2] = count[tid];
		}
		else 
		{
			repeat16(t1|=t2;t2|=t1;)
			start_time = clock();
			__syncthreads();	
			stop_time = clock();	
			count2[tid2] = count[tid];
		}
		
	}
	else
	{
		repeat32(t1+=t2;t2+=t1;)
		count[tid2] = tid;
		start_time = clock();
		__syncthreads();
		stop_time = clock();
		
		repeat32(t1+=t2;t2+=t1;)
		count[tid2] = tid + 100;
		start_time2 = clock();
		__syncthreads();
		stop_time2 = clock();
	}

	out[0] = (t1 + t2);		
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*3] =  count2[tid2];
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*3 + 1] = (threadIdx.x&1) ? start_time2 : start_time;
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*3 + 2] = (threadIdx.x&1) ? stop_time2 : stop_time;
}


void measure_sync2()
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



	Db.x = warpsize * 2;
	printf("\nksync2_test4: Shows syncthreads in two warps, one of which is diverged.");
	ksync2_test4 <<<Dg, Db>>>(d_ts, d_out, 4, 6, 2);
	cudaThreadSynchronize();
	cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);

	printf("\n  count[0..15]:  {");
	for (int i = 0; i < warpsize/2 - 1; i++)
	{
		printf("%3d, ", ts[i*3]);
	}
	printf("%3d}\n", ts[15 * 3]);
	printf("  count[16..31]: {");
	for (int i = 16; i < warpsize - 1; i++)
	{
		printf("%3d, ", ts[i*3]);
	}
	printf("%3d}\n", ts[31 * 3]);

	printf ("  Warp 0 thread  0 sync time: %5d - %5d\n", ts[0*3+1], ts[0*3+2]);
	printf ("  Warp 0 thread 16 sync time: %5d - %5d\n", ts[16*3+1], ts[16*3+2]);
	printf ("  Warp 1 thread  0 first sync time: %5d - %5d\n", ts[32*3+1], ts[32*3+2]);	
	printf ("  Warp 1 thread  1 second sync time: %5d - %5d\n", ts[33*3+1], ts[33*3+2]);	

	if (cudaSuccess != cudaGetLastError())
		printf("failed.\n");


	cudaFree(d_ts);
	cudaFree(d_out);
}

