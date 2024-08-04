#include <stdio.h>

#include "repeat.h"

#define DECLS(N)	\
__global__ void kicache_line_##N##_0 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);	\
__global__ void kicache_line_##N##_1 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);	\
__global__ void kicache_line_##N##_2 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);	\
__global__ void kicache_line_##N##_3 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);	\
__global__ void kicache_line_##N##_4 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);	\
__global__ void kicache_line_##N##_5 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);	\
__global__ void kicache_line_##N##_6 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);	\
__global__ void kicache_line_##N##_7 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);


DECLS(117) DECLS(118) DECLS(119) DECLS(120) DECLS(121)
DECLS(122) DECLS(123) DECLS(124) DECLS(125) DECLS(126)
DECLS(127) DECLS(128) DECLS(129) DECLS(130) DECLS(131)
DECLS(132) DECLS(133) DECLS(134) DECLS(135) DECLS(136)
DECLS(137) DECLS(138) DECLS(139) DECLS(140) DECLS(141)
DECLS(142) DECLS(143) DECLS(144) DECLS(145) DECLS(146)
DECLS(147) DECLS(148) DECLS(149) DECLS(150) DECLS(151)
DECLS(152) DECLS(153) DECLS(154) DECLS(155) DECLS(156)
DECLS(157) DECLS(158) DECLS(159) DECLS(160) DECLS(161)
DECLS(162) DECLS(163) DECLS(164) DECLS(165) DECLS(166)
DECLS(167) DECLS(168)


__global__ void kibuffer (unsigned int *ts, unsigned int *out, int p1, int p2, int its);

void measure_icache2();

int main()
{
	measure_icache2();
}

void measure_icache2()
{
	
	unsigned int ts[4096];			// ts, output from kernel. Two elements used per thread.
	unsigned int *d_ts;
	unsigned int *d_out;			// Unused memory for storing output
	

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
	if (cudaSuccess != cudaMalloc((void**)&d_out, 4))
	{
		printf ("cudaMalloc failed %s:%d\n", __FILE__, __LINE__);
		return;
	}
	
	printf ("L2 icache parameters (7-10KB of independent ops):\n");
	printf ("    [Codesize: avg, min, max runtime]\n");
	
#define DO_LTEST(N,M)	\
	{int stime=0, max_stime=0, min_stime=0x80000000;	\
	for (int k=0;k<20;k++) {							\
		kicache_line_##N##_##M <<<Dg, Db>>>(d_ts, d_out, 4, 3, 100); cudaThreadSynchronize();	\
		cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);			\
		stime += ts[1]-ts[0];				\
		if (max_stime < ts[1]-ts[0]) max_stime = ts[1]-ts[0];			\
		if (min_stime > ts[1]-ts[0]) min_stime = ts[1]-ts[0];			\
	}										\
	printf ("   %d: %.4f, %.4f, %.4f\n", N*64+M*8, (stime)/(2000.0*(N*8+M)), (min_stime)/(100.0*(N*8+M)), (max_stime)/(100.0*(N*8+M))); }

	
#define DO_LTESTS(N) DO_LTEST(N,0) DO_LTEST(N,1) DO_LTEST(N,2) DO_LTEST(N,3) DO_LTEST(N,4) DO_LTEST(N,5) DO_LTEST(N,6) DO_LTEST(N,7)
	
	DO_LTESTS(118);
	DO_LTESTS(119);
	DO_LTESTS(120);
	DO_LTESTS(121);
	
	DO_LTESTS(122);
	DO_LTESTS(123);
	DO_LTESTS(124);
	DO_LTESTS(125);
	DO_LTESTS(126);
	DO_LTESTS(127);
	DO_LTESTS(128);
	DO_LTESTS(129);
	DO_LTESTS(130);
	DO_LTESTS(131);
	DO_LTESTS(132);
	DO_LTESTS(133);
	DO_LTESTS(134);
	DO_LTESTS(135);
	DO_LTESTS(136);
	DO_LTESTS(137);
	DO_LTESTS(138);
	DO_LTESTS(139);
	DO_LTESTS(140);
	
	
	DO_LTESTS(141);
	DO_LTESTS(142);
	DO_LTESTS(143);
	DO_LTESTS(144);
	DO_LTESTS(145);
	DO_LTESTS(146);
	DO_LTESTS(147);
	DO_LTESTS(148);
	DO_LTESTS(149);
	DO_LTESTS(150);
	DO_LTESTS(151);
	DO_LTESTS(152);
	DO_LTESTS(153);
	DO_LTESTS(154);
	DO_LTESTS(155);
	DO_LTESTS(156);
	DO_LTESTS(157);
	DO_LTESTS(158);
	DO_LTESTS(159);
	DO_LTESTS(160);	
	DO_LTESTS(161);
	DO_LTESTS(162);
	DO_LTESTS(163);
	DO_LTESTS(164);
	
	//DO_LTESTS(165);
	//DO_LTESTS(166);
	//DO_LTESTS(167);
	//DO_LTESTS(168);

	
	
	if (1)
	{
		printf ("\nMeasuring instruction buffer size:\n");
		unsigned times[36] = {0};
		Db.x = 256;
		for (int k=0;k<10000;k++) {							
			kibuffer <<<Dg, Db>>>(d_ts, d_out, k%200+4, 3, 300); cudaThreadSynchronize();	
			cudaError_t errcode = cudaGetLastError();
			if (errcode != cudaSuccess)
			{
				printf ("Failed: %s\n", cudaGetErrorString(errcode));
				break;
			}
			cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);			
			for (int i=0;i<36;i++)
				times[i] += (unsigned)ts[i]-(unsigned)ts[0];		
		}
		for (int i=0;i<36;i++)
			printf ("  %d: %.3f\n", i*16, times[i]/(10000.0));
		printf ("\n");
	}
	
	
	
	cudaFree(d_ts);
	cudaFree(d_out);
}
