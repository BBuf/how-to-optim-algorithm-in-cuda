#include <cstdio>
#include <cstdint>

const int WARMUP = 100;
// number of LDS instructions to be timed
const int ROUND = 50;

__global__ __launch_bounds__(16, 1)
void smem_latency_kernel(const uint32_t *addr, uint32_t *ret, uint32_t *clk) {
    __shared__ uint32_t smem[16];

    smem[threadIdx.x] = addr[threadIdx.x];

    uint32_t start;
    uint32_t stop;
    uint32_t smem_addr;

    asm volatile (
        "{.reg .u64 u64addr;\n"
        " cvta.to.shared.u64 u64addr, %1;\n"
        " cvt.u32.u64 %0, u64addr;}\n"
        : "=r"(smem_addr)
        : "l"(smem + threadIdx.x)
    );

    asm volatile (
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(start) : : "memory"
    );

    #pragma unroll
    for (int i = 0; i < ROUND; ++i) {
        /*
         * dependent LDS instructions to make sure that
         * LDS latency can not be hidden by parallel LDS.
         */
        asm volatile (
            "ld.shared.b32 %0, [%0];\n"
            : "+r"(smem_addr) : : "memory"
        );
    }

    asm volatile (
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(stop) : : "memory"
    );

    clk[threadIdx.x] = stop - start;

    // dummy write back
    if (smem_addr == ~0x0) {
        *ret = smem_addr;
    }
}

int main() {
    uint32_t *h_addr;
    cudaMallocHost(&h_addr, 16 * sizeof(uint32_t));

    for (int i = 0; i < 16; ++i) {
        h_addr[i] = i * sizeof(uint32_t);
    }

    uint32_t *d_addr, *d_ret;
    cudaMalloc(&d_addr, 16 * sizeof(uint32_t));
    cudaMalloc(&d_ret, sizeof(uint32_t));
    cudaMemcpy(d_addr, h_addr, 16 * sizeof(uint32_t), cudaMemcpyHostToDevice);

    uint32_t *d_clk;
    cudaMalloc(&d_clk, 16 * sizeof(uint32_t));

    // pupulate l0/l1 i-cache
    for (int i = 0; i < WARMUP; ++i) {
        smem_latency_kernel<<<1, 16>>>(d_addr, d_ret, d_clk);
    }

    // shared memory latency benchmark
    smem_latency_kernel<<<1, 16>>>(d_addr, d_ret, d_clk);

    uint32_t h_clk[16];
    cudaMemcpy(h_clk, d_clk, 16 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    printf("shared memory latency %u cycles\n", h_clk[0] / ROUND);

    cudaFree(d_addr);
    cudaFree(d_ret);
    cudaFree(d_clk);
    cudaFreeHost(h_addr);

    return 0;
}


