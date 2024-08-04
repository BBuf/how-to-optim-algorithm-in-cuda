#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "repeat.h"

__global__ void kdiverge3_test1 (unsigned int *ts, unsigned int* out, int p1, int p2, int its)
{
	unsigned int t1 = p1;
	unsigned int t2 = p2;
	unsigned int start_time = 0 , stop_time = 0;
	unsigned int tid = (blockIdx.x*blockDim.x + threadIdx.x);
	
	int c0 = ts[0];	
	int c1 = ts[1];	
	int c2 = ts[2];	
	int c3 = ts[3];	
	int c4 = ts[4];	
	int c5 = ts[5];	
	int c6 = ts[6];	
	int c7 = ts[7];	
	int c8 = ts[8];	
	int c9 = ts[9];	
	int c10 = ts[10];	
	int c11 = ts[11];	
	int c12 = ts[12];	
	int c13 = ts[13];	
	int c14 = ts[14];	
	int c15 = ts[15];	
	int c16 = ts[16];	
	int c17 = ts[17];	
	int c18 = ts[18];	
	int c19 = ts[19];	
	int c20 = ts[20];	
	int c21 = ts[21];	
	int c22 = ts[22];	
	int c23 = ts[23];	
	int c24 = ts[24];	
	int c25 = ts[25];	
	int c26 = ts[26];	
	int c27 = ts[27];	
	int c28 = ts[28];	
	int c29 = ts[29];	
	int c30 = ts[30];	


	__syncthreads();

	if (tid == c0)
	{
		start_time = clock();								
		repeat16(t1+=t2;t2+=t1;)							
		stop_time = clock();								
	}
	else if (tid == c1)
	{
		start_time = clock();								
		repeat16(t1+=t2;t2+=t1;)							
		stop_time = clock();								
	}
	else if (tid == c2)
	{
		start_time = clock();								
		repeat16(t1+=t2;t2+=t1;)							
		stop_time = clock();								
	}
  	else if (tid == c3)
	{
		start_time = clock();								
		repeat16(t1+=t2;t2+=t1;)							
		stop_time = clock();								
	}
  	else if (tid == c4)
	{
		start_time = clock();								
		repeat16(t1+=t2;t2+=t1;)							
		stop_time = clock();								
	}
  	else if (tid == c5)
	{
		start_time = clock();								
		repeat16(t1+=t2;t2+=t1;)							
		stop_time = clock();								
	}
	else if (tid == c6)
	{
		start_time = clock();								
		repeat16(t1+=t2;t2+=t1;)							
		stop_time = clock();								
	}
	else if (tid == c7)
	{
		start_time = clock();								
		repeat16(t1+=t2;t2+=t1;)							
		stop_time = clock();								
	}
	else if (tid == c8)
	{
		start_time = clock();								
		repeat16(t1+=t2;t2+=t1;)							
		stop_time = clock();								
	}
	else if (tid == c9)
	{
		start_time = clock();								
		repeat16(t1+=t2;t2+=t1;)							
		stop_time = clock();								
	}
	else if (tid == c10)
	{
		start_time = clock();								
		repeat16(t1+=t2;t2+=t1;)							
		stop_time = clock();								
	}
	else if (tid == c11)
	{
		start_time = clock();								
		repeat16(t1+=t2;t2+=t1;)							
		stop_time = clock();								
	}
	else if (tid == c12)
	{
		start_time = clock();								
		repeat16(t1+=t2;t2+=t1;)							
		stop_time = clock();								
	}
	else if (tid == c13)
	{
		start_time = clock();								
		repeat16(t1+=t2;t2+=t1;)							
		stop_time = clock();								
	}
	else if (tid == c14)
	{
		start_time = clock();								
		repeat16(t1+=t2;t2+=t1;)							
		stop_time = clock();								
	}
	else if (tid == c15)
	{
		start_time = clock();								
		repeat16(t1+=t2;t2+=t1;)							
		stop_time = clock();								
	}
	else if (tid == c16)
	{
		start_time = clock();								
		repeat16(t1+=t2;t2+=t1;)							
		stop_time = clock();								
	}
	else if (tid == c17)
	{
		start_time = clock();								
		repeat16(t1+=t2;t2+=t1;)							
		stop_time = clock();								
	}
	else if (tid == c18)
	{
		start_time = clock();								
		repeat16(t1+=t2;t2+=t1;)							
		stop_time = clock();								
	}
  	else if (tid == c19)
	{
		start_time = clock();								
		repeat16(t1+=t2;t2+=t1;)							
		stop_time = clock();								
	}
  	else if (tid == c20)
	{
		start_time = clock();								
		repeat16(t1+=t2;t2+=t1;)							
		stop_time = clock();								
	}
  	else if (tid == c21)
	{
		start_time = clock();								
		repeat16(t1+=t2;t2+=t1;)							
		stop_time = clock();								
	}
	else if (tid == c22)
	{
		start_time = clock();								
		repeat16(t1+=t2;t2+=t1;)							
		stop_time = clock();								
	}
	else if (tid == c23) 
	{
		start_time = clock();								
		repeat16(t1+=t2;t2+=t1;)							
		stop_time = clock();								
	}
	else if (tid == c24)
	{
		start_time = clock();								
		repeat16(t1+=t2;t2+=t1;)							
		stop_time = clock();								
	}
	else if (tid == c25)
	{
		start_time = clock();								
		repeat16(t1+=t2;t2+=t1;)							
		stop_time = clock();								
	}
	else if (tid == c26)
	{
		start_time = clock();								
		repeat16(t1+=t2;t2+=t1;)							
		stop_time = clock();								
	}
	else if (tid == c27)
	{
		start_time = clock();								
		repeat16(t1+=t2;t2+=t1;)							
		stop_time = clock();								
	}
	else if (tid == c28)
	{
		start_time = clock();								
		repeat16(t1+=t2;t2+=t1;)							
		stop_time = clock();								
	}
	else if (tid == c29)
	{
		start_time = clock();								
		repeat16(t1+=t2;t2+=t1;)							
		stop_time = clock();								
	}
	else if (tid == c30)
	{
		start_time = clock();								
		repeat16(t1+=t2;t2+=t1;)							
		stop_time = clock();								
	}
	else 
	{
		start_time = clock();
		repeat16(t1+=t2;t2+=t1;)							
		stop_time = clock();
	}
			
	out[0] = (t1 + t2);		

	ts[(blockIdx.x*blockDim.x + threadIdx.x)*3 + 1] = start_time;
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*3 + 2] = stop_time;
}

// Similar to kiverge3_test1, but create different group sizes to rule
// out execution ordering due to number of threads that branched coherently..
__global__ void kdiverge3_test2 (unsigned int *ts, unsigned int* out, int p1, int p2, int its)
{
	unsigned int t1 = p1;
	unsigned int t2 = p2;
	unsigned int start_time = 0 , stop_time = 0;
	unsigned int tid = (blockIdx.x*blockDim.x + threadIdx.x);
	

	if (tid < 15)
	{
		start_time = clock();			
		repeat16(t1+=t2;t2+=t1;)	
		stop_time = clock();								
	}
	else if (tid < 24)
	{
		start_time = clock();			
		repeat16(t1+=t2;t2+=t1;)
		stop_time = clock();								
	}
	else if (tid < 26)
	{
		start_time = clock();		
		repeat16(t1+=t2;t2+=t1;)	
		stop_time = clock();								
	}
  	else if (tid < 30)
	{
		start_time = clock();				
		repeat16(t1+=t2;t2+=t1;)			
		stop_time = clock();					
	}
  	else 
	{
		start_time = clock();
		repeat16(t1+=t2;t2+=t1;)		
		stop_time = clock();
	}
			
	out[0] = (t1 + t2);		
	//ts[(blockIdx.x*blockDim.x + threadIdx.x)*3] =  count2[tid2];
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*3 + 1] = start_time;
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*3 + 2] = stop_time;
}

// Create some branch divergence, then see that all threads reconverge
// before reaching stop_time.
__global__ void kdiverge3_test3 (unsigned int *ts, unsigned int* out, int p1, int p2, int its)
{
	unsigned int t1 = p1 &3;
	unsigned int t2 = p2 &7;
	unsigned int start_time = 0 , stop_time = 0;
	unsigned int tid = (blockIdx.x*blockDim.x + threadIdx.x);
	
	unsigned int t3 = p1;
	int i = 0;

	for (i = 0; i < its; i++)
	{
		if (tid < t3)
		{
			start_time = clock();	
			repeat8(t1+=t2;t2+=t1;)
			t3--;
		}
		else
		{ 
			repeat8(t1^=t2;t2^=t1;)
			continue;
		}
	}

	repeat32(t1+=t2;t2+=t1;)
	stop_time = clock();
		
	out[0] = (t1 + t2 + t3);		

	ts[(blockIdx.x*blockDim.x + threadIdx.x)*3 + 1] = start_time;
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*3 + 2] = stop_time;
}

// Create some branch divergence. The 'break' creates branch divergence.
// Because the number of iterations is independent of threadIdx, the
// code generated does not generate a reconvergence point after the loop
// so the warps don't reconverge after break.
__global__ void kdiverge3_test4 (unsigned int *ts, unsigned int* out, int p1, int p2, int its)
{
	unsigned int t1 = p1 &3;
	unsigned int t2 = p2 &7;
	unsigned int start_time = 0 , stop_time = 0;
	unsigned int tid = (blockIdx.x*blockDim.x + threadIdx.x);
	
	unsigned int t3 = p1;
	int i = 0;

	for (i = 0; i < its; i++)
	{
		if (tid < t3)
		{
			start_time = clock();	
			repeat8(t1+=t2;t2+=t1;)
			t3--;
		}
		else
		{ 
			repeat8(t1^=t2;t2^=t1;)
			break;
		}
	}

	repeat32(t1+=t2;t2+=t1;)
	stop_time = clock();
		
	out[0] = (t1 + t2 + t3);		

	ts[(blockIdx.x*blockDim.x + threadIdx.x)*3 + 1] = start_time;
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*3 + 2] = stop_time;
}


// This is the same as kdiverge_test4 except that the number of iterations
// depends on the thread index. The 'break' creates branch divergence.
// The code generated does generate a reconvergence point after the loop
// so the warps reconverge after break.
__global__ void kdiverge3_test5 (unsigned int *ts, unsigned int* out, int p1, int p2, int its)
{
	unsigned int t1 = p1 &3;
	unsigned int t2 = p2 &7;
	unsigned int start_time = 0 , stop_time = 0;
	unsigned int tid = (blockIdx.x*blockDim.x + threadIdx.x);
	
	unsigned int t3 = p1;
	unsigned int t4 = threadIdx.x*threadIdx.y+its;
	int i = 0;

	for (i = 0; i < t4; i++)
	{
		if (tid < t3)
		{
			start_time = clock();	
			repeat8(t1+=t2;t2+=t1;)
			t3--;
		}
		else
		{ 
			repeat8(t1^=t2;t2^=t1;)
			break;
		}
	}

	repeat32(t1+=t2;t2+=t1;)
	stop_time = clock();
		
	out[0] = (t1 + t2 + t3);		

	ts[(blockIdx.x*blockDim.x + threadIdx.x)*3 + 1] = start_time;
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*3 + 2] = stop_time;
}


void measure_diverge3()
{
	
	unsigned int ts[2048];		// ts, output from kernel. Two elements used per thread.
	unsigned int *d_ts;
	unsigned int *d_out;		// Unused memory for storing output
	unsigned int min;

    	dim3 Db = dim3(1,1,1);
    	dim3 Dg = dim3(1,1);

	cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);

	int warpsize = deviceProp.warpSize;

	for (int i = 0; i < 2048; i++)
		ts[i] = (i*7)%32;
	
	cudaError_t errcode;
	// Allocate device array.
	if (cudaSuccess != (errcode = cudaMalloc((void**)&d_ts, sizeof(ts))))
	{
		printf ("cudaMalloc failed %s:%d\n", __FILE__, __LINE__);
		return;
	}
	if (cudaSuccess != cudaMalloc((void**)&d_out, 4))
	{
		printf ("cudaMalloc failed %s:%d\n", __FILE__, __LINE__);
		return;
	}
	
    	errcode = cudaGetLastError();
	if (errcode != cudaSuccess)
		printf("  failed. %s\n", cudaGetErrorString(errcode));    	

	fprintf (stderr, "Running divergence3 tests to probe execution ordering when diverged...\n");

	
	// Set up threadID 
	for (int i = 0; i < 2048; i++)
		ts[i] = i%32;
	cudaMemcpy(d_ts, ts, sizeof(ts), cudaMemcpyHostToDevice);

	Db.x = warpsize * 1;
	printf("\nkdiverge3_test1: [order:forward]");
	kdiverge3_test1 <<<Dg, Db>>>(d_ts, d_out, 4, 6, 2);
	cudaThreadSynchronize();
	cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);

	if (cudaSuccess != cudaGetLastError())
		printf("failed.\n");

	min = ts[0*3 + 1];
	for (int i = 0; i < Db.x; i++)
	{
		min = (ts[i*3 + 1] < min)? ts[i*3+1]: min;
	}
	
	printf("\n");	
	for (int i = 0; i < Db.x; i++)
	{
		printf("  warp:%2d, thread:%2d: start=%5u, \tstop=%5u\n", i/warpsize, i%warpsize, ts[i * 3 + 1]-min, ts[i * 3 + 2]-min);
	}



	for (int i = 0; i < 2048; i++)
		ts[i] = (31-i)%32;
	cudaMemcpy(d_ts, ts, sizeof(ts), cudaMemcpyHostToDevice);

	Db.x = warpsize * 1;
	printf("\nkdiverge3_test1: [order:backward]");
	kdiverge3_test1 <<<Dg, Db>>>(d_ts, d_out, 4, 6, 2);
	cudaThreadSynchronize();
	cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);

	if (cudaSuccess != cudaGetLastError())
		printf("failed.\n");

	min = ts[0*3 + 1];
	for (int i = 0; i < Db.x; i++)
	{
		min = (ts[i*3 + 1] < min)? ts[i*3+1]: min;
	}

	printf("\n");	
	for (int i = 0; i < Db.x; i++)
	{
		printf("  warp:%2d, thread:%2d: start=%5u, \tstop=%5u\n", i/warpsize, i%warpsize, ts[i * 3 + 1]-min, ts[i * 3 + 2]-min);
	}
	


	for (int i = 0; i < 2048; i++)
		ts[i] = (i*7)%32;
	cudaMemcpy(d_ts, ts, sizeof(ts), cudaMemcpyHostToDevice);

	Db.x = warpsize * 1;
	printf("\nkdiverge3_test1: [order:random]");
	kdiverge3_test1 <<<Dg, Db>>>(d_ts, d_out, 4, 6, 2);
	cudaThreadSynchronize();
	cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);

	if (cudaSuccess != cudaGetLastError())
		printf("failed.\n");

	min = ts[0*3 + 1];
	for (int i = 0; i < Db.x; i++)
	{
		min = (ts[i*3 + 1] < min)? ts[i*3+1]: min;
	}

	printf("\n");	
	for (int i = 0; i < Db.x; i++)
	{
		printf("  warp:%2d, thread:%2d: start=%5u, \tstop=%5u\n", i/warpsize, i%warpsize, ts[i * 3 + 1]-min, ts[i * 3 + 2]-min);
	}


	Db.x = warpsize * 1;
	printf("\nkdiverge3_test2: Testing if size of group that branched coherently affects execution order.");
	kdiverge3_test2 <<<Dg, Db>>>(d_ts, d_out, 4, 6, 2);
	cudaThreadSynchronize();
	cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);

	if (cudaSuccess != cudaGetLastError())
		printf("failed.\n");

	min = ts[0*3 + 1];
	for (int i = 0; i < Db.x; i++)
	{
		min = (ts[i*3 + 1] < min)? ts[i*3+1]: min;
	}

	printf("\n");	
	for (int i = 0; i < Db.x; i++)
	{
		printf("  warp:%2d, thread:%2d: start=%5u, \tstop=%5u\n", i/warpsize, i%warpsize, ts[i * 3 + 1]-min, ts[i * 3 + 2]-min);
	}
	



	Db.x = warpsize * 1;
	printf("\nkdiverge3_test3: Showing reconvergence after a loop.");
	kdiverge3_test3 <<<Dg, Db>>>(d_ts, d_out, 32, 6, 100);
	cudaThreadSynchronize();
	cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);

	if (cudaSuccess != cudaGetLastError())
		printf("failed.\n");

	printf("\n");	
	for (int i = 0; i < Db.x; i++)
	{
		printf("  warp:%2d, thread:%2d: start=%5u, \tstop=%5u\n", i/warpsize, i%warpsize, ts[i * 3 + 1], ts[i * 3 + 2]);
	}
	


	Db.x = warpsize * 1;
	printf("\nkdiverge3_test4: Showing weird code generation causing no reconvergence.");
	kdiverge3_test4 <<<Dg, Db>>>(d_ts, d_out, 32, 6, 100);
	cudaThreadSynchronize();
	cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);

	if (cudaSuccess != cudaGetLastError())
		printf("failed.\n");

	printf("\n");	
	for (int i = 0; i < Db.x; i++)
	{
		printf("  warp:%2d, thread:%2d: start=%5u, \tstop=%5u\n", i/warpsize, i%warpsize, ts[i * 3 + 1], ts[i * 3 + 2]);
	}
	printf("\n");


	Db.x = warpsize * 1;
	printf("\nkdiverge3_test5: Showing weird code generation that does reconverge.");
	kdiverge3_test5 <<<Dg, Db>>>(d_ts, d_out, 32, 6, 100);
	cudaThreadSynchronize();
	cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);

	if (cudaSuccess != cudaGetLastError())
		printf("failed.\n");

	printf("\n");	
	for (int i = 0; i < Db.x; i++)
	{
		printf("  warp:%2d, thread:%2d: start=%5u, \tstop=%5u\n", i/warpsize, i%warpsize, ts[i * 3 + 1], ts[i * 3 + 2]);
	}
	printf("\n");




	cudaFree(d_ts);
	cudaFree(d_out);
}

