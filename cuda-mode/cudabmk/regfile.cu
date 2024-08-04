#include <stdio.h>

#include "repeat.h"


/* These kernels are empty. They will be replaced by compiled cubin versions
   from regfile.real_cubin */
extern "C" {
__global__ void kempty_4 (unsigned int* out) {}
__global__ void kempty_8 (unsigned int* out) {}
__global__ void kempty_12 (unsigned int* out) {}
__global__ void kempty_16 (unsigned int* out) {}
__global__ void kempty_20 (unsigned int* out) {}
__global__ void kempty_24 (unsigned int* out) {}
__global__ void kempty_28 (unsigned int* out) {}
__global__ void kempty_32 (unsigned int* out) {}
__global__ void kempty_36 (unsigned int* out) {}
__global__ void kempty_40 (unsigned int* out) {}
__global__ void kempty_44 (unsigned int* out) {}
__global__ void kempty_48 (unsigned int* out) {}
__global__ void kempty_52 (unsigned int* out) {}
__global__ void kempty_56 (unsigned int* out) {}
__global__ void kempty_60 (unsigned int* out) {}
__global__ void kempty_64 (unsigned int* out) {}
__global__ void kempty_68 (unsigned int* out) {}
__global__ void kempty_72 (unsigned int* out) {}
__global__ void kempty_76 (unsigned int* out) {}
__global__ void kempty_80 (unsigned int* out) {}
__global__ void kempty_84 (unsigned int* out) {}
__global__ void kempty_88 (unsigned int* out) {}
__global__ void kempty_92 (unsigned int* out) {}
__global__ void kempty_96 (unsigned int* out) {}
__global__ void kempty_100 (unsigned int* out) {}
__global__ void kempty_104 (unsigned int* out) {}
__global__ void kempty_108 (unsigned int* out) {}
__global__ void kempty_112 (unsigned int* out) {}
__global__ void kempty_116 (unsigned int* out) {}
__global__ void kempty_120 (unsigned int* out) {}
__global__ void kempty_124 (unsigned int* out) {}
__global__ void kempty_128 (unsigned int* out) {}
__global__ void kempty_132 (unsigned int* out) {}
}


#define REGTEST1(RPT)	\
	for (Db.x = 0, max_success = -1; Db.x <= 516; Db.x+=1)	\
	{														\
		try{ kempty_##RPT <<<Dg, Db>>>(d_out);} catch (...)	{continue;}		\
		if (cudaGetLastError() == cudaSuccess) { max_success = Db.x;}			\
	}	\
	printf ("  [%3d x %3d = %5d]\n", max_success, RPT, max_success*RPT);	\

void measure_regfile()
{
	unsigned int *d_out;			// Unused memory for storing output
	
   	dim3 Db = dim3(1);
   	dim3 Dg = dim3(1,1,1);
	
	if (cudaSuccess != cudaMalloc((void**)&d_out, 4))
	{
		printf ("cudaMalloc failed %s:%d\n", __FILE__, __LINE__);
		return;
	}
	
	printf ("\n");
    	printf ("Running register file capacity test...\n");
	cudaGetLastError();		// Clear previous error code, if any
	
	printf ("  [Max threads x regs/thread = registers used] before kernel spawn failure.\n");
	int max_success;
	REGTEST1(4);
	REGTEST1(8);
	REGTEST1(12);
	REGTEST1(16);
	REGTEST1(20);
	REGTEST1(24);
	REGTEST1(28);
	REGTEST1(32);
	REGTEST1(36);
	REGTEST1(40);
	REGTEST1(44);
	REGTEST1(48);
	REGTEST1(52);
	REGTEST1(56);
	REGTEST1(60);
	REGTEST1(64);
	REGTEST1(68);
	REGTEST1(72);
	REGTEST1(76);
	REGTEST1(80);
	REGTEST1(84);
	REGTEST1(88);
	REGTEST1(92);
	REGTEST1(96);
	REGTEST1(100);
	REGTEST1(104);
	REGTEST1(108);
	REGTEST1(112);
	REGTEST1(116);
	REGTEST1(120);
	REGTEST1(124);
	REGTEST1(128);
			
	printf ("\n");
	
	cudaFree(d_out);
}

