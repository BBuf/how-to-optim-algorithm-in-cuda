
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "repeat.h"


__global__ void kclock(unsigned int *ts)
{
	unsigned int start_time = 0, stop_time = 0;

	start_time = clock();

	// Measure something here
	
	stop_time = clock();
	
	
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2] = start_time;
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2+1] = stop_time;
}

__global__ void kclock_test2 (unsigned int *ts, unsigned int *out, int p1, int p2, unsigned int its)
{
	unsigned int t1 = p1;
	unsigned int t2 = p2;
	unsigned int start_time = 0, stop_time = 0;

	for (int i = 0; i < its; i++)
	{
		start_time = clock();
		repeat64(t1+=t2;t2+=t1;)
		stop_time = clock();
	}

	out[0] = t1+t2;
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2] = start_time;
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2+1] = stop_time;
}

void measure_clock()
{ 
	unsigned int ts[1024];		// Timestamps, output from kernel.
	unsigned int *d_ts;
	unsigned int *d_ts2;
	unsigned int *d_out;

	dim3 Db = dim3(1);
    	dim3 Dg = dim3(1,1,1);
	
	// Allocate device array.
	cudaError_t errcode;
	if (cudaSuccess != (errcode = cudaMalloc((void**)&d_ts, sizeof(ts))))
	{
		printf ("cudaMalloc failed %s:%d\n", __FILE__, __LINE__);
		printf ("   %s\n", cudaGetErrorString(errcode));
		return;
	}
	if (cudaSuccess != (errcode = cudaMalloc((void**)&d_ts2, sizeof(ts))))
	{
		printf ("cudaMalloc failed %s:%d\n", __FILE__, __LINE__);
		printf ("   %s\n", cudaGetErrorString(errcode));
		return;
	}
		if (cudaSuccess != (errcode = cudaMalloc((void**)&d_out, 4)))
	{
		printf ("cudaMalloc failed %s:%d\n", __FILE__, __LINE__);
		printf ("   %s\n", cudaGetErrorString(errcode));
		return;
	}


	cudaGetLastError();
    	printf ("Running clock() test...");
	
	
	printf ("\nkclock: ");
	kclock <<<Dg, Db>>> (d_ts);
	cudaThreadSynchronize();
	cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);

	printf("\n");
	for (int i=0; i< Dg.x*Db.x; i++)
		printf ("   (%8u, %8u): %u\n", ts[i*2], ts[i*2+1], ts[i*2+1]-ts[i*2]);
    	printf ("\n");	



	Dg.x = 10; 
	Db.x = 1;  //1 thread per block
		
	printf ("\nkclock_test2: [%d blocks, %d thread(s)/block]", Dg.x, Db.x);
	kclock_test2 <<<Dg, Db>>>(d_ts2, d_out, 4, 6, 2);
				

	Dg.x = 30; //1 block per SM
	Db.x = 1;  //1 thread per block
	
	printf ("\nkclock_test2: [%d blocks, %d thread(s)/block]", Dg.x, Db.x);
	kclock_test2 <<<Dg, Db>>>(d_ts, d_out, 4, 6, 2);
	cudaThreadSynchronize();

	printf("\n");
	cudaMemcpy(ts, d_ts2, sizeof(ts), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 10; i++)
	{
		printf ("  Block %02d: start: %08u, stop: %08u\n", i, ts[i*2], ts[(i)*2+1]);

	}
	printf ("\n");
	
	cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);
	int m = 0;
	for (int i=0; i < 10; i++)
	{	
		for (int j = 0; j < Dg.x/10; j++, m++) 
		{
			printf ("  Block %02d: start: %08u, stop: %08u\n", 10*j+i, ts[(10*j+i)*2], ts[(10*j+i)*2+1]);
	
		}
	}
	printf ("\n");	
	
	printf("\n");
	cudaFree (d_ts);
	cudaFree (d_ts2);
}

