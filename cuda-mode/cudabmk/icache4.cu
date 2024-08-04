#include <stdio.h>

#include "repeat.h"


#define DECLS_L1(N)	\
__global__ void kicache_L1_##N##_0 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);	\
__global__ void kicache_L1_##N##_1 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);	\
__global__ void kicache_L1_##N##_2 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);	\
__global__ void kicache_L1_##N##_3 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);	\
__global__ void kicache_L1_##N##_4 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);	\
__global__ void kicache_L1_##N##_5 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);	\
__global__ void kicache_L1_##N##_6 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);	\
__global__ void kicache_L1_##N##_7 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);

DECLS_L1(48)
DECLS_L1(49)
DECLS_L1(50)
DECLS_L1(51)
DECLS_L1(52)
DECLS_L1(53)
DECLS_L1(54)
DECLS_L1(55)
DECLS_L1(56)
DECLS_L1(57)
DECLS_L1(58)
DECLS_L1(59)
DECLS_L1(60)
DECLS_L1(61)
DECLS_L1(62)
DECLS_L1(63)
DECLS_L1(64)
DECLS_L1(65)
DECLS_L1(66)
DECLS_L1(67)
DECLS_L1(68)
DECLS_L1(69)
DECLS_L1(70)
DECLS_L1(71)
DECLS_L1(72)
DECLS_L1(73)
DECLS_L1(74)
DECLS_L1(75)
DECLS_L1(76)
DECLS_L1(77)
DECLS_L1(78)
DECLS_L1(79)
DECLS_L1(80)
DECLS_L1(81)
DECLS_L1(82)
DECLS_L1(83)
DECLS_L1(84)
DECLS_L1(85)
DECLS_L1(86)
DECLS_L1(87)
DECLS_L1(88)
DECLS_L1(89)
DECLS_L1(90)
DECLS_L1(91)
DECLS_L1(92)



void measure_icache4_L1()
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

#define DO_LTEST_L1(N,M)	\
	{long long stime=0; int max_stime=0, min_stime=0x80000000;	\
	for (int k=0;k<50;k++) {							\
		kicache_L1_##N##_##M <<<Dg, Db>>>(d_ts, d_out, 4, 3, 200); cudaThreadSynchronize();	\
		cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);			\
		stime += ts[1]-ts[0];				\
		if (max_stime < ts[1]-ts[0]) max_stime = ts[1]-ts[0];			\
		if (min_stime > ts[1]-ts[0]) min_stime = ts[1]-ts[0];			\
	}										\
	printf ("   %d: %.4f, %.4f, %.4f\n", N*64+M*8, (stime)/(10000.0*(N*8+M+3)), (min_stime)/(200.*(N*8.+M+3)), (max_stime)/(200.*(N*8.+M+3))); }
	
	
#define DO_LTESTS_L1(N) DO_LTEST_L1(N,0) DO_LTEST_L1(N,4)
	
	printf ("L1 icache parameters (without and with contention in TPC):\n");
	printf ("    [Codesize: avg, min, max runtime]\n");
	
	for (Dg.x = 1; Dg.x <= 51; Dg.x += 50)
	{
		printf ("    %d blocks in TPC\n", Dg.x == 1 ? 1 : 5);
		/*
		DO_LTESTS_L1(48);
		DO_LTESTS_L1(49);
		DO_LTESTS_L1(50);
		DO_LTESTS_L1(51);
		DO_LTESTS_L1(52);
		DO_LTESTS_L1(53);
		DO_LTESTS_L1(54);
		DO_LTESTS_L1(55); */
		DO_LTESTS_L1(56);
		DO_LTESTS_L1(57);
		DO_LTESTS_L1(58);
		DO_LTESTS_L1(59);
		DO_LTESTS_L1(60);
		DO_LTESTS_L1(61);
		DO_LTESTS_L1(62);
		DO_LTESTS_L1(63);
		DO_LTESTS_L1(64);
		DO_LTESTS_L1(65);
		DO_LTESTS_L1(66);
		DO_LTESTS_L1(67);
		DO_LTESTS_L1(68);
		DO_LTESTS_L1(69);
		DO_LTESTS_L1(70);
		DO_LTESTS_L1(71);
		DO_LTESTS_L1(72);
		DO_LTESTS_L1(73);
		DO_LTESTS_L1(74);
		DO_LTESTS_L1(75);
		DO_LTESTS_L1(76);
		DO_LTESTS_L1(77);
		DO_LTESTS_L1(78);
		DO_LTESTS_L1(79);
		DO_LTESTS_L1(80);
		DO_LTESTS_L1(81);
		DO_LTESTS_L1(82);
		DO_LTESTS_L1(83);
		DO_LTESTS_L1(84);
		DO_LTESTS_L1(85);
		DO_LTESTS_L1(86);
		DO_LTESTS_L1(87);
		DO_LTESTS_L1(88);
		DO_LTESTS_L1(89);
		DO_LTESTS_L1(90);
		DO_LTESTS_L1(91);
		DO_LTESTS_L1(92);

	}
	
	
	
	cudaFree(d_ts);
	cudaFree(d_out);
}
int main()
{
	measure_icache4_L1();
}
