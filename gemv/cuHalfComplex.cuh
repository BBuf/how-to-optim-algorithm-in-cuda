#if !defined(CU_HALFCOMPLEX_H_)
#define CU_HALFCOMPLEX_H_

#if !defined(__CUDACC__)
#include <math.h>       /* import fabsf, sqrt */
#endif /* !defined(__CUDACC__) */
#include <cuda_fp16.h>

struct cuHalfComplex {
    half r;
    half i;
    __host__ __device__ __forceinline__ cuHalfComplex( half a, half b ) : r(a), i(b) {}

    __device__ __forceinline__ cuHalfComplex operator*(const cuHalfComplex& a) {
        return cuHalfComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }

    __device__ __forceinline__ cuHalfComplex operator+(const cuHalfComplex& a) {
        return cuHalfComplex(r+a.r, i+a.i);
    }
};

#endif /* !defined(CU_HALFCOMPLEX_H_) */
