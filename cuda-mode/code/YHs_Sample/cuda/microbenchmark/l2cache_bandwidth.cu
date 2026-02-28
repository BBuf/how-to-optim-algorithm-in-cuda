#include <cstdio>

// accessed data size in byte, should be smaller than l2 cache size
const int DATA_SIZE_IN_BYTE = (1lu << 20) * 2;
// number of LDG instructions
const int N_LDG = (1lu << 20) * 512;

const int WARMUP_ITER = 200;
const int BENCH_ITER = 200;

__device__ __forceinline__
int ldg_cg(const void *ptr) {
    int ret;
    asm volatile (
        "ld.global.cg.b32 %0, [%1];"
        : "=r"(ret)
        : "l"(ptr)
    );

    return ret;
}

template <int BLOCK, int UNROLL, int N_DATA>
__global__ void kernel(const int *x, int *y) {
    int offset = (BLOCK * UNROLL * blockIdx.x + threadIdx.x) % N_DATA;
    const int *ldg_ptr = x + offset;
    int reg[UNROLL];

    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        reg[i] = ldg_cg(ldg_ptr + BLOCK * i);
    }

    int sum = 0;
    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        sum += reg[i];
    }

    if (sum != 0) {
        *y = sum;
    }
}

int main() {
    const int N_DATA = DATA_SIZE_IN_BYTE / sizeof(int);

    const int UNROLL = 16;
    const int BLOCK = 128;

    static_assert(N_DATA >= UNROLL * BLOCK && N_DATA % (UNROLL * BLOCK) == 0,
                  "UNROLL or BLOCK is invalid");

    int *x, *y;
    cudaMalloc(&x, N_DATA * sizeof(int));
    cudaMalloc(&y, N_DATA * sizeof(int));
    cudaMemset(x, 0, N_DATA * sizeof(int));

    int grid = N_LDG / UNROLL / BLOCK;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warm up to cache data into L2
    for (int i = 0; i < WARMUP_ITER; ++i) {
        kernel<BLOCK, UNROLL, N_DATA><<<grid, BLOCK>>>(x, y);
    }

    cudaEventRecord(start);
    for (int i = 0; i < BENCH_ITER ; ++i) {
        kernel<BLOCK, UNROLL, N_DATA><<<grid, BLOCK>>>(x, y);
    }
    cudaEventRecord(stop);

    float time_ms = 0.f;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    double gbps = ((double)(N_LDG * sizeof(int)) / 1e9) /
                  ((double)time_ms / BENCH_ITER / 1e3);
    printf("L2 cache bandwidth: %fGB/s\n", gbps);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(x);
    cudaFree(y);

    return 0;
}


