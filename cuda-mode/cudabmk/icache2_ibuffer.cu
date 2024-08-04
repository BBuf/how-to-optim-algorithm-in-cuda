#include "repeat.h"

__global__ void kibuffer (unsigned int *ts, unsigned int *out, int p1, int p2, int its)
{																					
	volatile int t1 = p1, t2 = p1*p1, t3 = p1*p1+p1, t4 = p1*p1+p2;			
	volatile int t5 = p1*p2, t6 = p1*p2+p1, t7 = p1*p2+p2, t8 = p2*p1*p2;			
	volatile int t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21;
	volatile int t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t35, t36, t37;
	
	if (threadIdx.x < 32)
	{
		__syncthreads();
		for (int i=0;i<its;i++) {
			repeat3(t1 = abs(t2); t2 = abs(t3); t3=abs(t4); t4=abs(t5); t5=abs(t6); t6=abs(t7); t7=abs(t8); t8=abs(t1);)
		}
		repeat28(t1 = abs(t2); t2 = abs(t3); t3=abs(t4); t4=abs(t5); t5=abs(t6); t6=abs(t7); t7=abs(t8); t8=abs(t1);)
	}
	else if (threadIdx.x < 64)
	{
		__syncthreads();
		for (int i=0;i<its;i++) {
			repeat3(t1 = abs(t2); t2 = abs(t3); t3=abs(t4); t4=abs(t5); t5=abs(t6); t6=abs(t7); t7=abs(t8); t8=abs(t1);)
		}
		repeat28(t1 = abs(t2); t2 = abs(t3); t3=abs(t4); t4=abs(t5); t5=abs(t6); t6=abs(t7); t7=abs(t8); t8=abs(t1);)
	}
	else if (threadIdx.x < 96)
	{
		__syncthreads();
		for (int i=0;i<its;i++) {
			repeat3(t1 = abs(t2); t2 = abs(t3); t3=abs(t4); t4=abs(t5); t5=abs(t6); t6=abs(t7); t7=abs(t8); t8=abs(t1);)
		}
		repeat28(t1 = abs(t2); t2 = abs(t3); t3=abs(t4); t4=abs(t5); t5=abs(t6); t6=abs(t7); t7=abs(t8); t8=abs(t1);)
	}
	else if (threadIdx.x < 128)
	{
		__syncthreads();
		for (int i=0;i<its;i++) {
			repeat3(t1 = abs(t2); t2 = abs(t3); t3=abs(t4); t4=abs(t5); t5=abs(t6); t6=abs(t7); t7=abs(t8); t8=abs(t1);)
		}
		repeat28(t1 = abs(t2); t2 = abs(t3); t3=abs(t4); t4=abs(t5); t5=abs(t6); t6=abs(t7); t7=abs(t8); t8=abs(t1);)
	}
	else if (threadIdx.x < 160)
	{
		__syncthreads();
		for (int i=0;i<its;i++) {
			repeat3(t1 = abs(t2); t2 = abs(t3); t3=abs(t4); t4=abs(t5); t5=abs(t6); t6=abs(t7); t7=abs(t8); t8=abs(t1);)
		}
		repeat28(t1 = abs(t2); t2 = abs(t3); t3=abs(t4); t4=abs(t5); t5=abs(t6); t6=abs(t7); t7=abs(t8); t8=abs(t1);)
	}
	else if (threadIdx.x < 192)
	{
		__syncthreads();
		for (int i=0;i<its;i++) {
			repeat3(t1 = abs(t2); t2 = abs(t3); t3=abs(t4); t4=abs(t5); t5=abs(t6); t6=abs(t7); t7=abs(t8); t8=abs(t1);)
		}
		repeat28(t1 = abs(t2); t2 = abs(t3); t3=abs(t4); t4=abs(t5); t5=abs(t6); t6=abs(t7); t7=abs(t8); t8=abs(t1);)
	}
	else if (threadIdx.x < 224)
	{
		__syncthreads();
		for (int i=0;i<its;i++) {
			repeat3(t1 = abs(t2); t2 = abs(t3); t3=abs(t4); t4=abs(t5); t5=abs(t6); t6=abs(t7); t7=abs(t8); t8=abs(t1);)
		}
		// Start measurement code 256 bytes before alignment.
		repeat24(t1 = abs(t2); t2 = abs(t3); t3=abs(t4); t4=abs(t5); t5=abs(t6); t6=abs(t7); t7=abs(t8); t8=abs(t1);)
	}
	else if (threadIdx.x < 256)		// This warp measures.
	{
		__syncthreads();
		for (int i=0;i<p1; i++) {
			t2 = clock(); t3 = clock();	t4 = clock(); t5 = clock();
			t6 = clock(); t7 = clock(); t8 = clock(); t9 = clock();
			t10 = clock(); t11 = clock(); t12 = clock(); t13 = clock();
			t14 = clock(); t15 = clock(); t16 = clock(); t17 = clock();
			t18 = clock(); t19 = clock(); t20 = clock(); t21 = clock();
			t22 = clock(); t23 = clock(); t24 = clock(); t25 = clock();
			t26 = clock(); t27 = clock(); t28 = clock(); t29 = clock();
			t30 = clock(); t31 = clock(); t32 = clock(); t33 = clock();
			t34 = clock(); t35 = clock(); t36 = clock(); t37 = clock();
		}
	}
	
	out[0] = t1+t2+t3+t4+t5+t6+t7+t8;		
	if (threadIdx.x == 224)
	{
		ts[0] = t2; ts[1] = t3; ts[2] = t4; ts[3] = t5;		
		ts[4] = t6; ts[5] = t7; ts[6] = t8; ts[7] = t9;		
		ts[8] = t10; ts[9] = t11; ts[10] = t12; ts[11] = t13;		
		ts[12] = t14; ts[13] = t15; ts[14] = t16; ts[15] = t17;		
		ts[16] = t18; ts[17] = t19; ts[18] = t20; ts[19] = t21;		
		ts[20] = t22; ts[21] = t23; ts[22] = t24; ts[23] = t25;		
		ts[24] = t26; ts[25] = t27; ts[26] = t28; ts[27] = t29;		
		ts[28] = t30; ts[29] = t31; ts[30] = t32; ts[31] = t33;		
		ts[32] = t34; ts[33] = t35; ts[34] = t36; ts[35] = t37;		
	}
}

