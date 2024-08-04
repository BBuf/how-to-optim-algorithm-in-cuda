#include <stdio.h>


__global__ void kicache_test4_2 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_4 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_6 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_8 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_10 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_12 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_14 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_16 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_18 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_20 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_22 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_24 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_26 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_28 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_30 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_32 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_34 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_36 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_38 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_40 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_42 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_44 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_46 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_48 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_50 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_52 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_54 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_56 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_58 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_60 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_62 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_64 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_72 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_80 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_88 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_96 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_104 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_112 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_120 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_128 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_136 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_144 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_152 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_160 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_168 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_176 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_184 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_192 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_200 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_208 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_216 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_224 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_232 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_240 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_248 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_test4_256 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);




void measure_icache()
{
	
	unsigned int ts[1024];			// ts, output from kernel. Two elements used per thread.
	unsigned int *d_ts;
	unsigned int *d_out;			// Unused memory for storing output
	

    	dim3 Db = dim3(1);
    	dim3 Dg = dim3(1,1,1);
	
	// Allocate device array.
	if (cudaSuccess != cudaMalloc((void**)&d_ts, sizeof(ts)))
	{
		printf ("cudaMalloc failed %s:%d\n", __FILE__, __LINE__);
		return;
	}
	if (cudaSuccess != cudaMalloc((void**)&d_out, 4))
	{
		printf ("cudaMalloc failed %s:%d\n", __FILE__, __LINE__);
		return;
	}
	
    fprintf (stderr, "Running icache test...\n");
	
	

	// Measure instruction cache size
	printf ("Instruction cache size:\n");
	



	unsigned int sum_times[32];
	
	
	printf ("  0.5 KB steps: ");
	Db.x = 1;
	for (int p2 = 1; p2 <= 32; p2++)
	{
		unsigned int sum_time = 0;
		bool failed = false;
		for (int i=0;i<100 && !failed ;i++)
		{
			cudaGetLastError();		// Clear previous error code, if any
			switch (p2) {
				case 1: kicache_test4_2 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 2: kicache_test4_4 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 3: kicache_test4_6 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 4: kicache_test4_8 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 5: kicache_test4_10 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 6: kicache_test4_12 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 7: kicache_test4_14 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 8: kicache_test4_16 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 9: kicache_test4_18 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 10: kicache_test4_20 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 11: kicache_test4_22 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 12: kicache_test4_24 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 13: kicache_test4_26 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 14: kicache_test4_28 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 15: kicache_test4_30 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 16: kicache_test4_32 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 17: kicache_test4_34 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 18: kicache_test4_36 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 19: kicache_test4_38 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 20: kicache_test4_40 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 21: kicache_test4_42 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 22: kicache_test4_44 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 23: kicache_test4_46 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 24: kicache_test4_48 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 25: kicache_test4_50 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 26: kicache_test4_52 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 27: kicache_test4_54 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 28: kicache_test4_56 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 29: kicache_test4_58 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 30: kicache_test4_60 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 31: kicache_test4_62 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 32: kicache_test4_64 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
			}
				
			if (cudaGetLastError() != cudaSuccess)
			{
				failed = true;
				break;
			}
				
			cudaThreadSynchronize();
			cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);
			
			sum_time += (ts[1] - ts[0]);
		}
		
		if (failed)
		{
			printf ("xxxx ");
		}
		else
		{
			// Compute average latency over the lifetime of each warp (sum_time), and average throughput of the kernel (sum_max_time).
			printf ("%.1f ", sum_time/100.0/(p2*64)); fflush(stdout);
		}
		
		sum_times[p2-1] = sum_time;
	}
	printf (" (icache = ");
	for (int last_i=1, i=1;i<32;i++)
	{
		if (sum_times[i]/(i+1) > sum_times[last_i]/(last_i+1) *1.33)
		{
			printf ("%.1fKB ", i*0.5);
			last_i = i;
		}	
	}
	printf (")\n");
	
	
	printf ("  2 KB steps: ");
	Db.x = 1;
	for (int p2 = 1; p2 <= 32; p2++)
	{
		unsigned int sum_time = 0;
		bool failed = false;
		for (int i=0;i<100 && !failed ;i++)
		{
			cudaGetLastError();		// Clear previous error code, if any
			switch (p2) {
				case 1: kicache_test4_8 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 2: kicache_test4_16 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 3: kicache_test4_24 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 4: kicache_test4_32 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 5: kicache_test4_40 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 6: kicache_test4_48 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 7: kicache_test4_56 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 8: kicache_test4_64 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 9: kicache_test4_72 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 10: kicache_test4_80 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 11: kicache_test4_88 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 12: kicache_test4_96 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 13: kicache_test4_104 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 14: kicache_test4_112 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 15: kicache_test4_120 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 16: kicache_test4_128 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 17: kicache_test4_136 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 18: kicache_test4_144 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 19: kicache_test4_152 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 20: kicache_test4_160 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 21: kicache_test4_168 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 22: kicache_test4_176 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 23: kicache_test4_184 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 24: kicache_test4_192 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 25: kicache_test4_200 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 26: kicache_test4_208 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 27: kicache_test4_216 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 28: kicache_test4_224 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 29: kicache_test4_232 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 30: kicache_test4_240 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 31: kicache_test4_248 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;
				case 32: kicache_test4_256 <<<Dg, Db>>>(d_ts, d_out, 1, p2, 2); break;

			}			

				
			if (cudaGetLastError() != cudaSuccess)
			{
				failed = true;
				break;
			}
				
			cudaThreadSynchronize();
			cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);
			
			sum_time += (ts[1] - ts[0]);
		}
		
		if (failed)
		{
			printf ("xxxx ");
		}
		else
		{
			// Compute average latency over the lifetime of each warp (sum_time), and average throughput of the kernel (sum_max_time).
			printf ("%.1f ", sum_time/100.0/(p2*256)); fflush(stdout);
		}
		sum_times[p2-1] = sum_time;
	}
	printf (" (icache = ");
	for (int last_i=1, i=1;i<32;i++)
	{
		if (sum_times[i]/(i+1) > sum_times[last_i]/(last_i+1) *1.33)
		{
			printf ("%.1fKB ", i*2.0);
			last_i = i;
		}	
	}
	printf (")\n");
	
	
	printf ("\n Test instruction cache sharing by running two thread blocks concurently.\n");
	
	Db.x = 1;
	dim3 Dg2 = dim3(31,1,1);
	
	for (int blk2 = 1; blk2 <31; blk2+= (blk2 == 1? 9 : 10))
	{
		printf ("  TPC 0,%d (2 KB steps): ", blk2);
		int mask = (1<<blk2) | 1;	// Enable two blocks for execution
		for (int p2 = 1; p2 <= 32; p2++)
		{
			unsigned int sum_time = 0;
			bool failed = false;
			for (int i=0;i<50 && !failed ;i++)
			{
				cudaGetLastError();		// Clear previous error code, if any
				switch (p2) {
					case 1: kicache_test4_8 <<<Dg2, Db>>>(d_ts, d_out, mask, p2, 2); break;
					case 2: kicache_test4_16 <<<Dg2, Db>>>(d_ts, d_out, mask, p2, 2); break;
					case 3: kicache_test4_24 <<<Dg2, Db>>>(d_ts, d_out, mask, p2, 2); break;
					case 4: kicache_test4_32 <<<Dg2, Db>>>(d_ts, d_out, mask, p2, 2); break;
					case 5: kicache_test4_40 <<<Dg2, Db>>>(d_ts, d_out, mask, p2, 2); break;
					case 6: kicache_test4_48 <<<Dg2, Db>>>(d_ts, d_out, mask, p2, 2); break;
					case 7: kicache_test4_56 <<<Dg2, Db>>>(d_ts, d_out, mask, p2, 2); break;
					case 8: kicache_test4_64 <<<Dg2, Db>>>(d_ts, d_out, mask, p2, 2); break;
					case 9: kicache_test4_72 <<<Dg2, Db>>>(d_ts, d_out, mask, p2, 2); break;
					case 10: kicache_test4_80 <<<Dg2, Db>>>(d_ts, d_out, mask, p2, 2); break;
					case 11: kicache_test4_88 <<<Dg2, Db>>>(d_ts, d_out, mask, p2, 2); break;
					case 12: kicache_test4_96 <<<Dg2, Db>>>(d_ts, d_out, mask, p2, 2); break;
					case 13: kicache_test4_104 <<<Dg2, Db>>>(d_ts, d_out, mask, p2, 2); break;
					case 14: kicache_test4_112 <<<Dg2, Db>>>(d_ts, d_out, mask, p2, 2); break;
					case 15: kicache_test4_120 <<<Dg2, Db>>>(d_ts, d_out, mask, p2, 2); break;
					case 16: kicache_test4_128 <<<Dg2, Db>>>(d_ts, d_out, mask, p2, 2); break;
					case 17: kicache_test4_136 <<<Dg2, Db>>>(d_ts, d_out, mask, p2, 2); break;
					case 18: kicache_test4_144 <<<Dg2, Db>>>(d_ts, d_out, mask, p2, 2); break;
					case 19: kicache_test4_152 <<<Dg2, Db>>>(d_ts, d_out, mask, p2, 2); break;
					case 20: kicache_test4_160 <<<Dg2, Db>>>(d_ts, d_out, mask, p2, 2); break;
					case 21: kicache_test4_168 <<<Dg2, Db>>>(d_ts, d_out, mask, p2, 2); break;
					case 22: kicache_test4_176 <<<Dg2, Db>>>(d_ts, d_out, mask, p2, 2); break;
					case 23: kicache_test4_184 <<<Dg2, Db>>>(d_ts, d_out, mask, p2, 2); break;
					case 24: kicache_test4_192 <<<Dg2, Db>>>(d_ts, d_out, mask, p2, 2); break;
					case 25: kicache_test4_200 <<<Dg2, Db>>>(d_ts, d_out, mask, p2, 2); break;
					case 26: kicache_test4_208 <<<Dg2, Db>>>(d_ts, d_out, mask, p2, 2); break;
					case 27: kicache_test4_216 <<<Dg2, Db>>>(d_ts, d_out, mask, p2, 2); break;
					case 28: kicache_test4_224 <<<Dg2, Db>>>(d_ts, d_out, mask, p2, 2); break;
					case 29: kicache_test4_232 <<<Dg2, Db>>>(d_ts, d_out, mask, p2, 2); break;
					case 30: kicache_test4_240 <<<Dg2, Db>>>(d_ts, d_out, mask, p2, 2); break;
					case 31: kicache_test4_248 <<<Dg2, Db>>>(d_ts, d_out, mask, p2, 2); break;
					case 32: kicache_test4_256 <<<Dg2, Db>>>(d_ts, d_out, mask, p2, 2); break;

				}

				if (cudaGetLastError() != cudaSuccess)
				{
					failed = true;
					break;
				}
					
				cudaThreadSynchronize();
				cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);
				
				sum_time += (ts[1] - ts[0]);
			}
			
			if (failed)
			{
				printf ("xxxx ");
			}
			else
			{
				printf ("%.1f ", sum_time/50.0/(p2*256)); fflush(stdout);
			}
			sum_times[p2-1] = sum_time;
		}
		printf (" (apparent icache = ");
		for (int last_i=1, i=1;i<32;i++)
		{
			if (sum_times[i]/(i+1) > sum_times[last_i]/(last_i+1) *1.25)
			{
				printf ("%.1fKB ", i*2.0);
				last_i = i;
			}	
		}
		printf (")\n");	
	}
	
	
	printf ("\n");
	
	cudaFree(d_ts);
	cudaFree(d_out);

	
	
}

int main()
{
	measure_icache();
}



