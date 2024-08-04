#include <stdio.h>

#include "repeat.h"

#define CARRAY_SIZE 16384

__constant__ unsigned int d_carray[CARRAY_SIZE];
unsigned int h_test[CARRAY_SIZE];
__global__ void kclat (unsigned int *ts, unsigned int *out, int p1, int p2, int its2)
{																					
	int t1 = p1; int t2 = p1*p1; int t3 = p1*p1+p1; int t4 = p1*p1+p2;			
	int t5 = p1*p2; int t6 = p1*p2+p1; int t7 = p1*p2+p2; int t8 = p2*p1*p2;			

	int start_time, end_time;

	unsigned int p;
	int p_start = (blockIdx.x == 0) ? 0 : 8256;
	if (((1<<blockIdx.x) & p1) == 0) return;
	
	
	
	for (int j=0;j<2;j++)
	{
		p = p_start;
		int its = (j==0)? 2 : its2;
		start_time = clock();
		for (int i=0;i<its;i++)
		{
			repeat256(p = d_carray[p];)
		}
		
		end_time = clock();
	}
	
	t1 = p;
	out[0] = t1+t2+t3+t4+t5+t6+t7+t8;		
	
	if ((threadIdx.x & 31) == 0)
	{
		ts[(((blockIdx.x)*((blockDim.x+31)/32))+(threadIdx.x/32))*2] = start_time;
		ts[(((blockIdx.x)*((blockDim.x+31)/32))+(threadIdx.x/32))*2 + 1] = end_time;
	}
}

__global__ void kcicache_interfere (unsigned int *ts, unsigned int *out, int p1, int p2, int its2)
{																					
	int t1 = p1; int t2 = p1*p1; int t3 = p1*p1+p1; int t4 = p1*p1+p2;			
	int t5 = p1*p2; int t6 = p1*p2+p1; int t7 = p1*p2+p2; int t8 = p2*p1*p2;			

	int start_time, end_time;

	unsigned int p;
	if (((1<<blockIdx.x) & p1) == 0) return;
	
	
	if (blockIdx.x == 0)
	{
		for (int j=0;j<2;j++)
		{
			p = 0;
			int its = (j==0)? 2 : its2;
			start_time = clock();
			for (int i=0;i<its;i++)
			{
				repeat256(p = d_carray[p];)
			}
			
			end_time = clock();
		}
	}
	else
	{
		int its = its2 * 4;
		for (int i=0;i<its;i++) {
			repeat159(t1 = abs(t2); t2 = abs(t3); t3=abs(t4); t4=abs(t5); t5=abs(t6); t6=abs(t7); t7=abs(t8);t8=abs(t1);)
		}	
	}
	
	t1 = p;
	out[0] = t1+t2+t3+t4+t5+t6+t7+t8;		
	
	if ((threadIdx.x & 31) == 0)
	{
		ts[(((blockIdx.x)*((blockDim.x+31)/32))+(threadIdx.x/32))*2] = start_time;
		ts[(((blockIdx.x)*((blockDim.x+31)/32))+(threadIdx.x/32))*2 + 1] = end_time;
	}
}
		
__global__ void kcbw (unsigned int *ts, unsigned int *out, int p1, int p2, int its2)
{																					
	int t1 = p1; int t2 = p1*p1; int t3 = p1*p1+p1; int t4 = p1*p1+p2;			
	int t5 = p1*p2; int t6 = p1*p2+p1; int t7 = p1*p2+p2; int t8 = p2*p1*p2;			

	int start_time, end_time;

	volatile int p;
	if (blockIdx.x != 0) its2 *= 1.5f;
		
	for (int j=0;j<2;j++)
	{
		p = ((blockIdx.x)&4095)*64;
		int its = (j==0)? 2 : its2;
		start_time = clock();
		for (int i=0;i<its;i++)
		{
			repeat32(t1 += d_carray[p]; t2+=d_carray[p+512]; t3+=d_carray[p+1024]; t4+=d_carray[p+1536]; 
					t5+=d_carray[p+2048]; t6+=d_carray[p+2560]; t7+=d_carray[p+3072]; t8+=d_carray[p+3584];
					)
		}
		
		end_time = clock();
	}
	
	t1 += p;
	out[0] = t1+t2+t3+t4+t5+t6+t7+t8;		
	
	if ((threadIdx.x & 31) == 0)
	{
		ts[(((blockIdx.x)*((blockDim.x+31)/32))+(threadIdx.x/32))*2] = start_time;
		ts[(((blockIdx.x)*((blockDim.x+31)/32))+(threadIdx.x/32))*2 + 1] = end_time;
	}
}
__global__ void kcbw_8t (unsigned int *ts, unsigned int *out, int p1, int p2, int its2)
{																					
	int t1 = p1; int t2 = p1*p1; int t3 = p1*p1+p1; int t4 = p1*p1+p2;			
	int t5 = p1*p2; int t6 = p1*p2+p1; int t7 = p1*p2+p2; int t8 = p2*p1*p2;			

	int start_time, end_time;

	volatile int p;
	if (blockIdx.x != 0) its2 *= 1.5f;
		
	for (int j=0;j<2;j++)
	{
		p = threadIdx.x*64+((blockIdx.x/10)&1)*4096;
		int its = (j==0)? 2 : its2;
		start_time = clock();
		for (int i=0;i<its;i++)
		{
			repeat32(t1 += d_carray[p]; t2+=d_carray[p+512]; t3+=d_carray[p+1024]; t4+=d_carray[p+1536]; 
					t5+=d_carray[p+2048]; t6+=d_carray[p+2560]; t7+=d_carray[p+3072]; t8+=d_carray[p+3584];
					)
		}
		
		end_time = clock();
	}
	
	t1 += p;
	out[0] = t1+t2+t3+t4+t5+t6+t7+t8;		
	
	if ((threadIdx.x & 31) == 0)
	{
		ts[(((blockIdx.x)*((blockDim.x+31)/32))+(threadIdx.x/32))*2] = start_time;
		ts[(((blockIdx.x)*((blockDim.x+31)/32))+(threadIdx.x/32))*2 + 1] = end_time;
	}
}


		
		
void cmem_stride(unsigned int *h_carray, unsigned int *d_ts, unsigned int *d_out, unsigned int *ts, int stride, int min_size, int max_size, int step_size)
{
   	dim3 Db = dim3(1);
   	dim3 Dg = dim3(1,1,1);
	cudaError_t errcode;

	printf ("Constant memory, %d-byte stride\n", stride*4);
	printf ("  [array size]: [clocks per read], [max], [min]\n");
	for (int size = min_size; size <= max_size; size+=step_size)
	{
		// Set up array contents
		for (int i=0;i<size;i++)
		{
			h_carray[i] = i+stride;
			if (h_carray[i] >= size) 
				h_carray[i] %= stride;
		}
		
		
		
		cudaMemcpyToSymbol(d_carray, h_carray, CARRAY_SIZE*4);
		
		unsigned long long sum_time = {0};
		unsigned int max_time=0, min_time=(unsigned)-1;
		int kits = 20;
		int its = 30;
		for (int k = 0; k < kits; k++)
		{
			// Launch kernel
			kclat<<<Dg, Db>>> (d_ts, d_out, 1,3, its);
			
			
			errcode = cudaGetLastError();
			if (errcode != cudaSuccess)
			{
				printf ("Failed: %s\n", cudaGetErrorString(errcode));
			}
			cudaThreadSynchronize();			
			cudaMemcpy(ts, d_ts, 16, cudaMemcpyDeviceToHost);	
							
			sum_time += ts[1]-ts[0];
			if (ts[1]-ts[0] > max_time) max_time = ts[1]-ts[0];
			if (ts[1]-ts[0] < min_time) min_time = ts[1]-ts[0];
		}

		printf ("  %d: %.3f, %.3f, %.3f clk\n", size*4, 
			sum_time/(kits*its*256.0),
			min_time/(its*256.0),
			max_time/(its*256.0));
	}
	printf ("\n");
}

void cmem_stride_2 (unsigned int *h_carray, unsigned int *d_ts, unsigned int *d_out, unsigned int *ts, int stride, int min_size, int max_size, int step_size, unsigned int exec_mask)
{
   	dim3 Db = dim3(1);
   	dim3 Dg = dim3(31,1,1);
	cudaError_t errcode;
	if (max_size > 8128)
	{
		printf ("Size %d too big. Must be <= 8128 elements\n", max_size);
		return;
	}

	printf ("Constant memory, %d-byte stride, blocks [", stride);
	for (int i=0;i<31;i++)
		if ((1<<i)&exec_mask) printf (" %d", i);
	printf (" ]\n");
	printf ("  [array size]: [clocks per read], [max], [min]\n");
	for (int size = min_size; size <= max_size; size+=step_size)
	{
		// Set up array contents
		for (int i=0;i<size;i++)
		{
			h_carray[i] = i+stride;
			if (h_carray[i] >= size) 
				h_carray[i] %= stride;
			h_carray[8256+i] = 8256+i+stride;
			if (h_carray[8256+i] >= 8256+size) 
				h_carray[8256+i] = 8256 + (h_carray[8256+i]%stride);
		}
		
		
		
		cudaMemcpyToSymbol(d_carray, h_carray, CARRAY_SIZE*4);
		
		unsigned long long sum_time = {0};
		unsigned int max_time=0, min_time=(unsigned)-1;
		int kits = 20;
		int its = 30;
		for (int k = 0; k < kits; k++)
		{
			// Launch kernel
			kclat<<<Dg, Db>>> (d_ts, d_out, exec_mask, 3, its);
			
			
			errcode = cudaGetLastError();
			if (errcode != cudaSuccess)
			{
				printf ("Failed: %s\n", cudaGetErrorString(errcode));
			}
			cudaThreadSynchronize();			
			cudaMemcpy(ts, d_ts, 16, cudaMemcpyDeviceToHost);	
							
			sum_time += ts[1]-ts[0];
			if (ts[1]-ts[0] > max_time) max_time = ts[1]-ts[0];
			if (ts[1]-ts[0] < min_time) min_time = ts[1]-ts[0];
		}

		printf ("  %d: %.3f, %.3f, %.3f clk\n", size*4, 
			sum_time/(kits*its*256.0),
			min_time/(its*256.0),
			max_time/(its*256.0));
	}
	printf ("\n");
}


void cmem_icache_sharing (unsigned int *h_carray, unsigned int *d_ts, unsigned int *d_out, unsigned int *ts, int stride, int min_size, int max_size, int step_size, unsigned int exec_mask)
{
   	dim3 Db = dim3(1);
   	dim3 Dg = dim3(31,1,1);
	cudaError_t errcode;
	if (max_size > 16384)
	{
		printf ("Size %d too big. Must be <= 16384 elements\n", max_size);
		return;
	}

	printf ("Constant memory and Icache sharing, %d-byte stride, blocks [", stride);
	for (int i=0;i<31;i++)
		if ((1<<i)&exec_mask) printf (" %d", i);
	printf (" ]\n");
	printf ("  [array size]: [clocks per read], [max], [min]\n");
	for (int size = min_size; size <= max_size; size+=step_size)
	{
		// Set up array contents
		for (int i=0;i<size;i++)
		{
			h_carray[i] = i+stride;
			if (h_carray[i] >= size) 
				h_carray[i] %= stride;
		}
		
		
		
		cudaMemcpyToSymbol(d_carray, h_carray, CARRAY_SIZE*4);
		
		unsigned long long sum_time = {0};
		unsigned int max_time=0, min_time=(unsigned)-1;
		int kits = 20;
		int its = 30;
		for (int k = 0; k < kits; k++)
		{
			// Launch kernel
			kcicache_interfere<<<Dg, Db>>> (d_ts, d_out, exec_mask, 3, its);
			
			
			errcode = cudaGetLastError();
			if (errcode != cudaSuccess)
			{
				printf ("Failed: %s\n", cudaGetErrorString(errcode));
			}
			cudaThreadSynchronize();			
			cudaMemcpy(ts, d_ts, 16, cudaMemcpyDeviceToHost);	
							
			sum_time += ts[1]-ts[0];
			if (ts[1]-ts[0] > max_time) max_time = ts[1]-ts[0];
			if (ts[1]-ts[0] < min_time) min_time = ts[1]-ts[0];
		}

		printf ("  %d: %.3f, %.3f, %.3f clk\n", size*4, 
			sum_time/(kits*its*256.0),
			min_time/(its*256.0),
			max_time/(its*256.0));
	}
	printf ("\n");
}


void cmem_bandwidth (unsigned int *d_ts, unsigned int *d_out, unsigned int *ts, unsigned int nblocks, int nthreads)
{
   	dim3 Db = dim3(nthreads);
   	dim3 Dg = dim3(nblocks,1,1);
	dim3 Dgpad = dim3(0,1,1);
	while ((Dg.x+Dgpad.x)%2 == 0 || (Dg.x+Dgpad.x)%5 == 0) Dgpad.x++;
	cudaError_t errcode;

		
	unsigned long long sum_time = {0};
	unsigned int max_time=0, min_time=(unsigned)-1;
	int kits = 20;
	int its = 30;
	for (int k = 0; k < kits; k++)
	{
		// Launch kernel
		if (nthreads == 1)
			kcbw<<<Dg, Db>>> (d_ts, d_out, 0, 0, its);
		else
			kcbw_8t<<<Dg, Db>>> (d_ts, d_out, 0, 0, its);
		
		
		errcode = cudaGetLastError();
		if (errcode != cudaSuccess)
		{
			printf ("Failed: %s\n", cudaGetErrorString(errcode));
		}
		cudaThreadSynchronize();			
		cudaMemcpy(ts, d_ts, 16, cudaMemcpyDeviceToHost);	
		
		sum_time += ts[1]-ts[0];
		if (ts[1]-ts[0] > max_time) max_time = ts[1]-ts[0];
		if (ts[1]-ts[0] < min_time) min_time = ts[1]-ts[0];
		
		if (Dgpad.x > 0)
		{
			kcbw<<<Dgpad, Db>>> (d_ts, d_out, 0, 0, 1);
			cudaThreadSynchronize();	
		}
	}

	printf ("  %d: %.3f, %.3f, %.3f bytes/clk\n", nblocks, 
		(nblocks*kits*its*256.0*256.0*Db.x)/sum_time,
		(nblocks*its*256.0*256.0*Db.x)/max_time,
		(nblocks*its*256.0*256.0*Db.x)/min_time);
	
}



int main()
{
	
	unsigned int ts[4096];			// ts, output from kernel. Two elements used per thread.
	unsigned int *d_ts;
	unsigned int *d_out;			// Unused memory for storing output
	unsigned int *h_carray;
	

	
	// Allocate device array.
	cudaError_t errcode;
	if (cudaSuccess != (errcode = cudaMalloc((void**)&d_ts, sizeof(ts))))
	{
		printf ("cudaMalloc failed %s:%d\n", __FILE__, __LINE__);
		printf ("   %s\n", cudaGetErrorString(errcode));
		return -1;
	}
	if (cudaSuccess != cudaMalloc((void**)&d_out, 4))
	{
		printf ("cudaMalloc failed %s:%d\n", __FILE__, __LINE__);
		return -1;
	}
	
	h_carray = (unsigned int*)malloc(CARRAY_SIZE*4);


	// Stride 256 overview
	cmem_stride(h_carray, d_ts, d_out, ts, 256/4, 16, 16384, 256/4);
	
	// Stride 64 L2 and L3
	cmem_stride(h_carray, d_ts, d_out, ts, 64/4, 6144/4, 40960/4, 64/4);

	// Stride 16 L1
	cmem_stride(h_carray, d_ts, d_out, ts, 16/4, 512-64, 512+192, 16/4);

	// Different-TPC Sharing
	printf ("Different TPC Testing ");
	cmem_stride_2(h_carray, d_ts, d_out, ts, 256/4, 16, 8128, 256/4, 0x81);

	// Same-TPC Sharing
	printf ("Shared TPC Testing ");
	cmem_stride_2(h_carray, d_ts, d_out, ts, 256/4, 16, 8128, 256/4, 0x401);
	
	// Same-SM sharing, L1
	printf ("Shared SM Testing ");
	cmem_stride_2(h_carray, d_ts, d_out, ts, 256/4, 16, 8128, 256/4, 0x40000001);
	
	// Instruction cache sharing
	printf ("Different TPC ");
	cmem_icache_sharing(h_carray, d_ts, d_out, ts, 256/4, 16, 16384, 256/4, 0x81);
	printf ("Shared TPC ");
	cmem_icache_sharing(h_carray, d_ts, d_out, ts, 256/4, 16, 16384, 256/4, 0x401);
	printf ("Shared SM ");
	cmem_icache_sharing(h_carray, d_ts, d_out, ts, 256/4, 16, 16384, 256/4, 0x40000001);
	
	
	// Constant cache bandwidth
	printf ("Constant cache L3 bandwidth, blocks touching addresses 2KB apart\n");
	for (int nblocks = 1; nblocks <= 60; nblocks++)
		cmem_bandwidth (d_ts, d_out, ts, nblocks, 1);
	printf ("\n");
	printf ("Constant cache L3 bandwidth, 8 threads/warp\n");
	for (int nblocks = 1; nblocks <= 60; nblocks++)
		cmem_bandwidth (d_ts, d_out, ts, nblocks, 8);
	printf ("\n");


	
	cudaFree(d_ts);
	cudaFree(d_out);
	free(h_carray);

	return 0;
}

