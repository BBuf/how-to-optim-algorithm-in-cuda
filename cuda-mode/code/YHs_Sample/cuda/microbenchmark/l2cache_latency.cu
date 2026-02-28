#include <cstdio>
#include <cstdint>

const int WARMUP = 100;
// number of LDG instructions to be timed
const int ROUND = 10;
// stride in byte between LDG instructions
const int STRIDE = 128;

template <int ROUND>
__global__ __launch_bounds__(32, 1)
void l2_latency_kernel(const uint32_t *stride,
                       uint32_t *ret,
                       uint32_t *clk) {
    const char *ldg_ptr = reinterpret_cast<const char *>(stride + threadIdx.x);
    uint32_t val;

    // populate TLB
    asm volatile (
        "ld.global.cg.b32 %0, [%1];\n"
        : "=r"(val)
        : "l"(ldg_ptr)
        : "memory"
    );

    ldg_ptr += val;

    uint32_t start;
    uint32_t stop;

    asm volatile (
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(start) : : "memory"
    );

    #pragma unroll
    for (int i = 0; i < ROUND; ++i) {
        asm volatile (
            "ld.global.cg.b32 %0, [%1];\n"
            : "=r"(val)
            : "l"(ldg_ptr)
            : "memory"
        );

        /*
         * dependent LDG instructions to make sure that
         * LDG latency can not be hidden by parallel LDG.
         *
         * IADD/IMAD/XMAD's latency is much lower than
         * l2 cache and can be ignored.
         */
        ldg_ptr += val;
    }

    asm volatile (
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(stop) : : "memory"
    );

    clk[threadIdx.x] = stop - start;

    // dummy write back
    if (val == 0) {
        *ret = val;
    }
}

int main() {
    static_assert(STRIDE >= 32 * sizeof(uint32_t) &&
                  STRIDE % sizeof(uint32_t) == 0,
                  "invalid 'STRIDE'");

    const uint32_t STRIDE_MEM_SIZE = (ROUND + 1) * STRIDE;

    uint32_t *h_stride;
    cudaMallocHost(&h_stride, STRIDE_MEM_SIZE);

    for (int i = 0; i < STRIDE_MEM_SIZE / sizeof(uint32_t); ++i) {
        h_stride[i] = STRIDE;
    }

    uint32_t *d_stride, *d_ret;
    cudaMalloc(&d_stride, STRIDE_MEM_SIZE);
    cudaMalloc(&d_ret, sizeof(uint32_t));
    cudaMemcpy(d_stride, h_stride, STRIDE_MEM_SIZE, cudaMemcpyHostToDevice);

    uint32_t *d_clk;
    cudaMalloc(&d_clk, 32 * sizeof(uint32_t));

    // pupulate l0/l1 i-cache and l2 cache
    for (int i = 0; i < WARMUP; ++i) {
        l2_latency_kernel<ROUND><<<1, 32>>>(d_stride, d_ret, d_clk);
    }

    // l2 cache latency benchmark
    l2_latency_kernel<ROUND><<<1, 32>>>(d_stride, d_ret, d_clk);

    uint32_t h_clk[32];
    cudaMemcpy(h_clk, d_clk, 32 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    printf("l2 cache latency %u cycles\n", h_clk[0] / ROUND);

    cudaFree(d_stride);
    cudaFree(d_ret);
    cudaFree(d_clk);
    cudaFreeHost(h_stride);

    return 0;
}


