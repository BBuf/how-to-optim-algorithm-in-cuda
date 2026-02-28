#include <cstdio>
#include <cstdint>

const int MEMORY_OFFSET = (1u << 20) * 16;
const int BENCH_ITER = 100;

const int BLOCK = 128;
const int LDG_UNROLL = 1;

__device__ __forceinline__
uint4 ldg_cs(const void *ptr) {
    uint4 ret;
    asm volatile (
        "ld.global.cs.v4.b32 {%0, %1, %2, %3}, [%4];"
        : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
        : "l"(ptr)
    );

    return ret;
}

__device__ __forceinline__
void stg_cs(const uint4 &reg, void *ptr) {
    asm volatile (
        "st.global.cs.v4.b32 [%4], {%0, %1, %2, %3};"
        : : "r"(reg.x), "r"(reg.y), "r"(reg.z), "r"(reg.w), "l"(ptr)
    );
}

template <int BLOCK, int VEC_UNROLL>
__global__ void read_kernel(const void *x, void *y) {
    uint32_t idx = blockIdx.x * BLOCK * VEC_UNROLL + threadIdx.x;

    const uint4 *ldg_ptr = (const uint4 *)x + idx;
    uint4 reg[VEC_UNROLL];

    #pragma unroll
    for (int i = 0; i < VEC_UNROLL; ++i) {
        reg[i] = ldg_cs(ldg_ptr + i * BLOCK);
    }

    // dummy STG
    #pragma unroll
    for (int i = 0; i < VEC_UNROLL; ++i) {
        if (reg[i].x != 0) {
            stg_cs(reg[i], (uint4 *)y + i);
        }
    }
}

template <int BLOCK, int VEC_UNROLL>
__global__ void write_kernel(void *y) {
    uint32_t idx = blockIdx.x * BLOCK * VEC_UNROLL + threadIdx.x;

    uint4 *stg_ptr = (uint4 *)y + idx;

    #pragma unroll
    for (int i = 0; i < VEC_UNROLL; ++i) {
        uint4 reg = make_uint4(0, 0, 0, 0);
        stg_cs(reg, stg_ptr + i * BLOCK);
    }
}

template <int BLOCK, int VEC_UNROLL>
__global__ void copy_kernel(const void *x, void *y) {
    uint32_t idx = blockIdx.x * BLOCK * VEC_UNROLL + threadIdx.x;

    const uint4 *ldg_ptr = (const uint4 *)x + idx;
    uint4 *stg_ptr = (uint4 *)y + idx;
    uint4 reg[VEC_UNROLL];

    #pragma unroll
    for (int i = 0; i < VEC_UNROLL; ++i) {
        reg[i] = ldg_cs(ldg_ptr + i * BLOCK);
    }

    #pragma unroll
    for (int i = 0; i < VEC_UNROLL; ++i) {
        stg_cs(reg[i], stg_ptr + i * BLOCK);
    }
}

void benchmark(size_t size_in_byte) {
    printf("%luMB (r+w)\n", size_in_byte / (1 << 20));

    double size_gb = (double)size_in_byte / (1 << 30);

    size_t n = size_in_byte / sizeof(uint4);
    size_t grid = n / (BLOCK * LDG_UNROLL);

    static_assert(MEMORY_OFFSET % sizeof(uint4) == 0,
                  "invalid MEMORY_OFFSET");

    char *ws;
    cudaMalloc(&ws, size_in_byte + MEMORY_OFFSET * BENCH_ITER);

    // set all zero for read-only kernel
    cudaMemset(ws, 0, size_in_byte + MEMORY_OFFSET * BENCH_ITER);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time_ms = 0.f;

    // warmup
    read_kernel<BLOCK, LDG_UNROLL><<<grid, BLOCK>>>(ws, nullptr);
    write_kernel<BLOCK, LDG_UNROLL><<<grid, BLOCK>>>(ws);
    copy_kernel<BLOCK, LDG_UNROLL><<<grid / 2, BLOCK>>>(ws, ws + size_in_byte / 2);

    // read
    cudaEventRecord(start);
    for (int i = BENCH_ITER - 1; i >= 0; --i) {
        read_kernel<BLOCK, LDG_UNROLL><<<grid, BLOCK>>>(ws + i * MEMORY_OFFSET, nullptr);
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("read %fGB/s\n", size_gb * BENCH_ITER / ((double)time_ms / 1000));

    // write
    cudaEventRecord(start);
    for (int i = BENCH_ITER - 1; i >= 0; --i) {
        write_kernel<BLOCK, LDG_UNROLL><<<grid, BLOCK>>>(ws + i * MEMORY_OFFSET);
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("write %fGB/s\n", size_gb * BENCH_ITER / ((double)time_ms / 1000));


    // copy
    cudaEventRecord(start);
    for (int i = BENCH_ITER - 1; i >= 0; --i) {
        copy_kernel<BLOCK, LDG_UNROLL><<<grid / 2, BLOCK>>>(
            ws + i * MEMORY_OFFSET,
            ws + i * MEMORY_OFFSET + size_in_byte / 2);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time_ms, start, stop);
    printf("copy %fGB/s\n", size_gb * BENCH_ITER / ((double)time_ms / 1000));

    printf("---------------------------\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(ws);
}

int main() {
    size_t size = (1lu << 20) * 4;

    // 4MB~1GB
    while (size <= (1lu << 30)) {
        benchmark(size);
        size *= 2;
    }

    return 0;
}


