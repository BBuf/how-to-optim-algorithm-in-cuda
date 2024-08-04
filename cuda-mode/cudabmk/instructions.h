typedef short		SHORT;
typedef	unsigned int 	UINT;
typedef int		INT;
typedef float		FLOAT;
typedef double 		DOUBLE;


/* WARP INTRINSICS */
#define ALL(a, b)	a=__all(a==b)
#define ANY(a, b)	a=__any(a==b)
/* #define SYNC(a, b)	__syncthreads() */

/* ARITHMETIC INSTRUCTIONS */
#define ADD(a, b) 	a+=b		
#define SUB(a, b)	a-=b	
#define MUL(a, b)	a*=b		
#define MAD(a, b)	a=a*b+a
#define DIV(a, b)	a/=b				
#define REM(a, b)	a%=b			
#define ABS(a, b)	a+=abs(b)		
#define NEG(a, b)	a^=-b		
#define MIN(a, b)	a=min(a+b,b)	
#define MAX(a, b)	a=max(a+b,b)	

/* LOGIC INSTRUCTIONS */
#define AND(a, b)	a&=b				
#define OR(a, b)	a|=b				
#define XOR(a, b)	a^=b				
#define SHL(a, b)	a<<=b			
#define SHR(a, b)	a>>=b			

/* TO BE TESTED LATER */
#define NOT(a, b)	a=~b				
#define NOT2(a, b)	if (a>=b) a=~b				
#define CNOT(a, b)	a^=(b==0)?1:0

/* TO BE TESTED LATER */
#define ANDNOT(a, b)	a&=~b
#define ORNOT(a, b)	a|=~b
#define XORNOT(a, b)	a^=~b
#define ADDNOT(a, b) 	a+=~b
#define ANDNOTNOT(a, b)	a=~a&~b

/* ARITHMETIC INSTRINSICS: INTEGER */
#define MUL24(a, b)	a=__mul24(a, b)
#define UMUL24(a, b)	a=__umul24(a, b)
#define MULHI(a, b)	a=__mulhi(a, b)
#define UMULHI(a, b)	a=__umulhi(a, b)
#define SAD(a, b)	a=__sad(a, b, a)	
#define USAD(a, b)	a=__usad(a, b, a)	

/* ARITHMETIC INTRINSICS: FLOAT */
#define FADD_RN(a, b)	a=__fadd_rn(a, b)
#define FADD_RZ(a, b)	a=__fadd_rz(a, b)
#define FMUL_RN(a, b)	a=__fmul_rn(a, b)
#define FMUL_RZ(a, b)	a=__fmul_rz(a, b)
#define FDIVIDEF(a, b)	a=__fdividef(a, b)

/* ARITHMETIC INTRINSICS: DOUBLE. Requires SM1.3 */
#define DADD_RN(a, b)	a=__dadd_rn(a, b)

/* MATH INSTRUCTIONS: FLOAT */
#define RCP(a, b)	a+=1/b
#define SQRT(a, b)	a=sqrt(b)
#define RSQRT(a, b)	a=rsqrt(b)

#define SIN(a, b)	a=sinf(b)
#define COS(a, b)	a=cosf(b)
#define TAN(a, b)	a=tanf(b)
#define EXP(a, b)	a=expf(b)
#define EXP10(a, b)	a=exp10f(b)
#define LOG(a, b)	a=logf(b)
#define LOG2(a, b)	a=log2f(b)
#define LOG10(a, b)	a=log10f(b)
#define POW(a, b)	a=powf(a, b)


/* MATH INTRINSICS: FLOAT */
#define SINF(a, b)	a=__sinf(b)
#define COSF(a, b)	a=__cosf(b)
//#define SINCOSF
#define TANF(a, b)	a=__tanf(b)
#define EXPF(a, b)	a=__expf(b)
#define EXP2F(a, b)	a=exp2f(b)
#define EXP10F(a, b)	a=__exp10f(b)
#define LOGF(a, b)	a=__logf(b)
#define LOG2F(a, b)	a=__log2f(b)
#define LOG10F(a, b)	a=__log10f(b)
#define POWF(a, b)	a=__powf(a, b)

/* CONVERSION INTRINSICS */
#define INTASFLOAT(a, b)		a=__int_as_float(b)
#define FLOATASINT(a, b)		a=__float_as_int(b)

/* MISC INTRINSICS */
#define POPC(a, b)	a=__popc(b)
#define SATURATE(a, b)	a=saturate(b)
#define CLZ(a, b)	a=__clz(b)  //count leading zeros	
#define CLZLL(a, b)	a=__clzll(b)  //count leading zeros	
#define FFS(a, b)	a=__ffs(b)
#define FFSLL(a, b)	a=__ffsll(b)


/* COMPARISON AND SELECTION INSTRUCTIONS */
//#define SET
//#define SETP
//#define SELP
//#define SLCT

/* DATA MOVEMENT AND CONVERSION INSTRUCTIONS */
#define MOV(a, b)	a+=b; b=a 
#define MOV4(a, b, c)	a=b^c; b=a
//#define LD
//#define ST
//#define CVT

/* TEXTURE INSTRUCTIONS */
//#define TEX

/* CONTROL FLOW INSTRUCTIONS */

/* PARALLEL SYNCHRONIZATION AND COMMUNICATION INSTRUCTIONS */

/* MISCELLANEOUS INSTRUCTIONS */
//#define TRAP
//#define BRKPT

/* CONVERSION FUNCTIONS: FLOAT */
//#define RINTF()
//#define ROUNDF()

/* this are all supposed to map to a single machine instr according to the guide */
//#define TRUNCF()
//#define CEILF()
//#define FLOORF()




#define IF(a, b)	if(a == b) a^=b

/* ATOMIC INSTRUCTIONS */
#define ATOMICADD(a,b)	atomicAdd(a, b)

#define K_OP_DEP(OP, DEP, TYPE)\
extern "C"\
__global__ void K_##OP##_##TYPE##_DEP##DEP (unsigned int *ts, unsigned int* out, TYPE p1, TYPE p2, int its) 	\
{														\
	TYPE t1 = p1;												\
	TYPE t2 = p2;												\
	unsigned int start_time=0, stop_time=1;									\
														\
	do													\
	{													\
		for (int i=0;i<its;i++)										\
		{												\
			__syncthreads();									\
			start_time = clock();									\
			repeat##DEP(OP(t1, t2); OP(t2, t1);)							\
			stop_time = clock();									\
		}												\
	} while(stop_time < start_time);									\
														\
	out[0] = (unsigned int )(t1 + t2);									\
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2] = start_time;						\
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2 +1] = stop_time;						\
}

#define K_OP_DEP_DOUBLE_ISSUE(OP, DEP, TYPE)\
__global__ void K_##OP##_##TYPE##_DEP##DEP##_DI (unsigned int *ts, unsigned int* out, TYPE p1, TYPE p2, int its) 	\
{														\
	TYPE t1 = p1;												\
	TYPE t2 = p2;												\
	unsigned int start_time=0, stop_time=1;									\
														\
	do													\
	{													\
		for (int i=0;i<its;i++)										\
		{												\
			__syncthreads();									\
			start_time = clock();									\
			repeat##DEP(OP(t1, t2); OP(t2, t1);)							\
			stop_time = clock();									\
		}												\
	} while(stop_time < start_time);									\
														\
	out[0] = (unsigned int )(t1 + t2);									\
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2] = start_time;						\
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2 +1] = stop_time;						\
}

#define K_OP_DEP3(OP, DEP, TYPE)\
__global__ void K_##OP##_##TYPE##_DEP##DEP (unsigned int *ts, unsigned int* out, TYPE p1, TYPE p2, TYPE p3, int its) 	\
{														\
	TYPE t1 = p1;												\
	TYPE t2 = p2;												\
	TYPE t3 = p3;												\
	unsigned int start_time, stop_time;									\
														\
	for (int i=0;i<its;i++)											\
	{													\
		__syncthreads();										\
		start_time = clock();										\
		repeat##DEP(OP(t1, t2, t3); OP(t2, t1, t3); OP(t3, t1, t2);)					\
		stop_time = clock();										\
	}													\
														\
	out[0] = (unsigned int )(t1 + t2);									\
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2] = start_time;						\
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2 +1] = stop_time;						\
}

#define K_OP_DEP4(OP, DEP, TYPE)\
__global__ void K_##OP##_##TYPE##_DEP##DEP (unsigned int *ts, unsigned int* out, TYPE p1, TYPE p2, TYPE p3, TYPE p4, int its) 	\
{														\
	TYPE t1 = p1;												\
	TYPE t2 = p2;												\
	TYPE t3 = p3;												\
	TYPE t4 = p4;												\
	unsigned int start_time, stop_time;									\
														\
	for (int i=0;i<its;i++)											\
	{													\
		__syncthreads();										\
		start_time = clock();										\
		repeat##DEP(OP(t1, t2, t3); OP(t3, t4, t2);) 	 						\
		stop_time = clock();										\
	}													\
														\
	out[0] = (unsigned int )(t1 + t2 + t3 + t4);								\
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2] = start_time;						\
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2 +1] = stop_time;						\
}

//This gives me a move from a register to local memory with ADD, keeping it for now
#define K_OP_DEP_LOCAL(OP, DEP, TYPE)\
__global__ void K_##OP##_##TYPE##_DEP##DEP (unsigned int *ts, unsigned int* out, TYPE p1, TYPE p2, int its) 	\
{														\
	TYPE t[128];												\
	t[0] = p1;												\
	t[1] = p2;												\
	unsigned int start_time, stop_time;									\
														\
	for (int i=0;i<its;i++)											\
	{													\
		__syncthreads();										\
		start_time = clock();										\
		repeat##DEP(OP(t[0], t[1]); OP(t[1],t[0]);) 	 						\
		stop_time = clock();										\
	}													\
														\
	for (int i = 0; i < 128; i++)										\
		out[0] += (unsigned int )(t[i]);								\
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2] = start_time;						\
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2 +1] = stop_time;						\
}

#define K_OP_DEPX(OP, DEP, TYPE)\
__global__ void K_##OP##_##TYPE##_DEP##DEP (unsigned int *ts, unsigned int* out, TYPE p1, TYPE p2, int its) 	\
{														\
	TYPE t0 = p1;												\
	TYPE t1 = p2;												\
	TYPE t2 = p1^p2;											\
	TYPE t3 = p1+p2;											\
	unsigned int start_time, stop_time;									\
														\
	for (int i=0;i<its;i++)											\
	{													\
		__syncthreads();										\
		start_time = clock();										\
		repeat32(											\
		if (t0 == t1) t1=t0;										\
		if (t1 == t2) t2=t1;										\
		if (t2 == t3) t3=t2;										\
		if (t3 == t0) t0=t3;										\
		)												\
		stop_time = clock();										\
	}													\
														\
	out[0] += (unsigned int )(t0 + t1 + t2 + t3);								\
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2] = start_time;						\
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2 +1] = stop_time;						\
}

#define K_OP_DEP_ATOMIC(OP, DEP, TYPE)\
__global__ void K_##OP##_##TYPE##_DEP##DEP (unsigned int *ts, unsigned int* out, TYPE p1, TYPE p2, int its) 	\
{														\
	TYPE t1 = p1;												\
	TYPE t2 = p2;												\
	unsigned int start_time, stop_time;									\
														\
	for (int i=0;i<its;i++)											\
	{													\
		__syncthreads();										\
		start_time = clock();										\
		repeat##DEP(OP(&out[0], t1); OP(&out[0], t2);) 							\
		/*atomicAdd(out, 1);*/\
		stop_time = clock();										\
	}													\
														\
	/* out[0] = (unsigned int )(ts[0]); */									\
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2] = start_time;						\
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2 +1] = stop_time;						\
}


/* WARP VOTE INTRINSICS -- NEEDS SM1.2 */
K_OP_DEP(ALL, 128, UINT)
K_OP_DEP(ANY, 128, UINT)

/* K_OP_DEP(SYNC, 128, UINT) 
  Use the cubin in ksync_uint_dep128.real_cubin Recent versions of the CUDA compiler will
  optimize away repeated syncthreads() calls. Seems like the optimization will break
  program correctness, but that's what's done...
*/

/* ARITHMETIC INSTRUCTIONS: UINT*/
K_OP_DEP(ADD, 128, UINT)
K_OP_DEP(SUB, 128, UINT)
K_OP_DEP(MUL, 128, UINT)
K_OP_DEP(DIV, 128, UINT)
K_OP_DEP(REM, 128, UINT)
K_OP_DEP(MAD, 128, UINT)
K_OP_DEP(MIN, 128, UINT)
K_OP_DEP(MAX, 128, UINT)

/* ARITHMETIC INSTRUCTIONS: INT */
K_OP_DEP(ADD, 128, INT)
K_OP_DEP(SUB, 128, INT)
K_OP_DEP(MUL, 128, INT)
K_OP_DEP(DIV, 128, INT)
K_OP_DEP(REM, 128, INT)
K_OP_DEP(MAD, 128, INT)
K_OP_DEP(ABS, 128, INT)
K_OP_DEP(NEG, 128, INT)
K_OP_DEP(MIN, 128, INT)
K_OP_DEP(MAX, 128, INT)

/* ARITHMETIC INSTRUCTIONS: FLOAT */
K_OP_DEP(ADD, 128, FLOAT)
K_OP_DEP(SUB, 128, FLOAT)
K_OP_DEP(MUL, 128, FLOAT)
K_OP_DEP(DIV, 128, FLOAT)
K_OP_DEP(MAD, 128, FLOAT)
K_OP_DEP(ABS, 128, FLOAT)
K_OP_DEP(MIN, 128, FLOAT)
K_OP_DEP(MAX, 128, FLOAT)

/* ARITHMETIC INSTRUCTIONS: DOUBLE */
K_OP_DEP(ADD, 128, DOUBLE)
K_OP_DEP(SUB, 128, DOUBLE)
K_OP_DEP(MUL, 128, DOUBLE)
K_OP_DEP(DIV, 128, DOUBLE)
K_OP_DEP(MAD, 128, DOUBLE)
K_OP_DEP(ABS, 128, DOUBLE)
K_OP_DEP(MIN, 128, DOUBLE)
K_OP_DEP(MAX, 128, DOUBLE)

/* LOGIC INSTRUCTIONS */
K_OP_DEP(AND,  128, UINT) 
K_OP_DEP(OR,   128, UINT) 
K_OP_DEP(XOR,  128, UINT) 
K_OP_DEP(SHL,  128, UINT) 
K_OP_DEP(SHR,  128, UINT) 

K_OP_DEP(NOT,  128, UINT) 
K_OP_DEP(NOT2,  128, INT) 
K_OP_DEP(CNOT, 128, UINT) 

K_OP_DEP(ANDNOT,  128, UINT) 
K_OP_DEP(ORNOT,   128, UINT) 
K_OP_DEP(XORNOT,  128, UINT) 
K_OP_DEP(ADDNOT,  128, UINT) 
K_OP_DEP(ANDNOTNOT,  128, UINT) 

/* ARITHMETIC INSTRINSICS: UINT/INT */
K_OP_DEP(UMUL24, 128, UINT)
K_OP_DEP(MUL24, 128, INT)
K_OP_DEP(UMULHI, 128, UINT)
K_OP_DEP(MULHI, 128, INT)
K_OP_DEP(USAD, 128, UINT)
K_OP_DEP(SAD, 128, INT)

/* ARITHMETIC INSTRINSICS: FLOAT */
K_OP_DEP(FADD_RN, 128, FLOAT)
K_OP_DEP(FADD_RZ, 128, FLOAT)
K_OP_DEP(FMUL_RN, 128, FLOAT)
K_OP_DEP(FMUL_RZ, 128, FLOAT)
K_OP_DEP(FDIVIDEF, 128, FLOAT)

/* INSTRINSICS: DOUBLE */
K_OP_DEP(DADD_RN, 128, DOUBLE)

/* MATH INSTRUCTIONS: FLOAT */
K_OP_DEP(RCP, 128, FLOAT)
K_OP_DEP(SQRT, 128, FLOAT)
K_OP_DEP(RSQRT, 128, FLOAT)
/*
K_OP_DEP(SIN, 128, FLOAT)
K_OP_DEP(COS, 128, FLOAT)
K_OP_DEP(TAN, 128, FLOAT)
K_OP_DEP(EXP, 128, FLOAT)
K_OP_DEP(EXP10, 128, FLOAT)
K_OP_DEP(LOG, 128, FLOAT)
K_OP_DEP(LOG2, 128, FLOAT)
K_OP_DEP(LOG10, 128, FLOAT)
K_OP_DEP(POW, 128, FLOAT)
*/

/* MATH INTRINSICS: FLOAT */
K_OP_DEP(SINF, 128, FLOAT)
K_OP_DEP(COSF, 128, FLOAT)
K_OP_DEP(TANF, 128, FLOAT)
K_OP_DEP(EXPF, 128, FLOAT)
K_OP_DEP(EXP2F, 128, FLOAT)
K_OP_DEP(EXP10F, 128, FLOAT)
K_OP_DEP(LOGF, 128, FLOAT)
K_OP_DEP(LOG2F, 128, FLOAT)
K_OP_DEP(LOG10F, 128, FLOAT)
K_OP_DEP(POWF, 128, FLOAT)

/* CONVERSION */
K_OP_DEP(INTASFLOAT, 128, UINT)
K_OP_DEP(FLOATASINT, 128, FLOAT)

/* MISC */
K_OP_DEP(POPC, 128, UINT)
K_OP_DEP(CLZ, 128, UINT)
K_OP_DEP(CLZLL, 128, UINT)
K_OP_DEP(FFS, 128, UINT)
K_OP_DEP(FFSLL, 128, UINT)
K_OP_DEP(SATURATE, 128, FLOAT)


/* ATOMIC INSTRUCTIONS NEEDS SM1.1 and SM1.2 for SHARED*/
//K_OP_DEP_ATOMIC(ATOMICADD, 128, UINT)
