#include <cstdio>
#include <cstdint>

// number of LDG instructions to be timed
const int ROUND = 10;
// stride in byte between LDG instructions
// should be greater than L2 cache line size to avoid L2 cache hit
const int STRIDE = 1024;

// workspace size in byte to flush L2 cache
const int L2_FLUSH_SIZE = (1 << 20) * 128;

template <int BLOCK>
__global__ void flush_l2_kernel(const int *x, int *y) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    const int *x_ptr = x + blockIdx.x * BLOCK + warp_id * 32;
    int sum = 0;

    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        const int *ldg_ptr = x_ptr + (lane_id ^ i);

        asm volatile (
            "{.reg .s32 val;\n"
            " ld.global.cg.b32 val, [%1];\n"
            " add.s32 %0, val, %0;}\n"
            : "+r"(sum) : "l"(ldg_ptr)
        );
    }

    if (sum != 0) {
        *y = sum;
    }
}

void flush_l2() {
    int *x;
    int *y;
    cudaMalloc(&x, L2_FLUSH_SIZE);
    cudaMalloc(&y, sizeof(int));
    cudaMemset(x, 0, L2_FLUSH_SIZE);

    int n = L2_FLUSH_SIZE / sizeof(int);
    flush_l2_kernel<128><<<n / 128, 128>>>(x, y);

    cudaFree(x);
    cudaFree(y);
}

template <int ROUND>
__global__ __launch_bounds__(32, 1)
void dram_latency_kernel(const uint32_t *stride,
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
         * IADD/IMAD/XMAD's latency is much lower than dram and can be ignored.
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

    // pupulate l0/l1 i-cache
    dram_latency_kernel<ROUND><<<1, 32>>>(d_stride, d_ret, d_clk);

    // flush L2 cache
    flush_l2();

    // DRAM latency benchmark
    dram_latency_kernel<ROUND><<<1, 32>>>(d_stride, d_ret, d_clk);

    uint32_t h_clk[32];
    cudaMemcpy(h_clk, d_clk, 32 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    printf("DRAM latency %u cycles\n", h_clk[0] / ROUND);

    cudaFree(d_stride);
    cudaFree(d_ret);
    cudaFree(d_clk);
    cudaFreeHost(h_stride);

    return 0;
}


