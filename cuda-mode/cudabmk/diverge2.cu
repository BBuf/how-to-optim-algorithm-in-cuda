#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "repeat.h"

/* THIS KERNEL HANGS */
__global__ void kdiverge2_test1 (unsigned int *ts, unsigned int* out, int p1, int p2, int its)
{
	int t1 = p1;
	int t2 = p2;
	unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
	__shared__ volatile unsigned int sharedvar;


	sharedvar = 0;
	__syncthreads();


	while (sharedvar != tid);
	sharedvar++;


	out[0] = (t1 + t2);		
}

/* THIS KERNEL HANGS */
__global__ void kdiverge2_test2 (unsigned int *ts, unsigned int* out, int p1, int p2, int its)
{
	int t1 = p1;
	int t2 = p2;
	unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
	__shared__ volatile unsigned int sharedvar;


	sharedvar = 0;
	__syncthreads();


	label0: if (sharedvar != tid) goto label0;
	else sharedvar++;


	out[0] = (t1 + t2);		
}

__global__ void kdiverge2_test3 (unsigned int *ts, unsigned int* out, int p1, int p2, int its)
{
	int t1 = p1;
	int t2 = p2;
	unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
	__shared__ volatile unsigned int sharedvar;


	sharedvar = 0;
	__syncthreads();


	label0: if (sharedvar != tid) 
	{
		if (t1 > 0) goto label0;
	}
	else 
		sharedvar++;

	out[0] = (t1 + t2);		
}

/* THIS KERNEL HANGS */
__global__ void kdiverge2_test4 (unsigned int *ts, unsigned int* out, int p1, int p2, int its)
{
	int t1 = p1;
	int t2 = p2;
	unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
	__shared__ volatile unsigned int sharedvar;


	sharedvar = 0;
	__syncthreads();


	while (sharedvar != tid);
	atomicAdd((unsigned int *)&sharedvar, 1);


	out[0] = (t1 + t2);		
}

/* THIS KERNEL HANGS */
__global__ void kdiverge2_test5 (unsigned int *ts, unsigned int* out, int p1, int p2, int its)
{
	int t1 = p1;
	int t2 = p2;
	unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
	__shared__  unsigned int sharedvar;


	sharedvar = 0;
	__syncthreads();

	while (sharedvar != tid);
	atomicAdd((unsigned int *)&sharedvar, 1);


	out[0] = (t1 + t2);		
}


void measure_diverge2()
{

	unsigned int ts[1024];			// ts, output from kernel. Two elements used per thread.
	unsigned int *d_ts;
	unsigned int *d_out;			// Unused memory for storing output

    	dim3 Db = dim3(32 * 2);
    	dim3 Dg = dim3(1,1,1);
	
	// Allocate device array.
	cudaError_t errorcode;
	if (cudaSuccess != (errorcode = cudaMalloc((void**)&d_ts, sizeof(ts))))
	{
		printf ("cudaMalloc failed %s:%d\n", __FILE__, __LINE__);
		printf ("   %s\n", cudaGetErrorString(errorcode));
		return;
	}
	if (cudaSuccess != (errorcode = cudaMalloc((void**)&d_out, 4)))
	{
		printf ("cudaMalloc failed %s:%d\n", __FILE__, __LINE__);
		return;
	}

	errorcode = cudaGetLastError();
    	
	printf ("Running divergence tests ...\n");

	Db.x = 32;

	// Not runnning kdiverge2_test1, 2, 4, and 5 because they hang the system.
	printf("kdiverge2_test3: ");
	kdiverge2_test3 <<<Dg, Db>>>(d_ts, d_out, 4, 6, 100);
	cudaThreadSynchronize();
	if (cudaSuccess != (errorcode = cudaGetLastError()))
		printf("failed. %s\n", cudaGetErrorString(errorcode));

	cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);
	if (cudaSuccess != (errorcode = cudaGetLastError()))
		printf("failed. %s\n", cudaGetErrorString(errorcode));
	printf(" PASS.\n");
	


	printf("\n");

	cudaFree(d_ts);
	cudaFree(d_out);
}

