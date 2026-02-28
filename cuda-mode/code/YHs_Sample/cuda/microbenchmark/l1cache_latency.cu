#include <cstdio>
#include <cstdint>

const int WARMUP = 100;
const int THREAD = 4;
// number of LDG instructions to be timed
const int ROUND = 50;

template <int ROUND>
__global__ __launch_bounds__(THREAD, 1)
void l1_latency_kernel(void **ptr, void **ret, uint32_t *clk) {
    void **ldg_ptr = ptr + threadIdx.x;

    // warm up to populate l1 cache
    for (int i = 0; i < ROUND; ++i) {
        asm volatile (
            "ld.global.nc.b64 %0, [%0];\n"
            : "+l"(ldg_ptr) : : "memory"
        );
    }

    uint32_t start;
    uint32_t stop;

    asm volatile (
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(start) : : "memory"
    );

    for (int i = 0; i < ROUND; ++i) {
        /*
         * dependent LDG instructions to make sure that
         * LDG latency can not be hidden by parallel LDG.
         */
        asm volatile (
            "ld.global.nc.b64 %0, [%0];\n"
            : "+l"(ldg_ptr) : : "memory"
        );
    }

    asm volatile (
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(stop) : : "memory"
    );

    clk[threadIdx.x] = stop - start;

    // dummy write back
    if (ldg_ptr == nullptr) {
        *ret = ldg_ptr;
    }
}

int main() {

    void **d_ptr;
    void **d_ret;
    uint32_t *d_clk;
    cudaMalloc(&d_ptr, THREAD * sizeof(void *));
    cudaMalloc(&d_ret, sizeof(void *));
    cudaMalloc(&d_clk, THREAD * sizeof(uint32_t));

    void **h_ptr;
    cudaMallocHost(&h_ptr, THREAD * sizeof(void *));

    for (int i = 0; i < THREAD; ++i) {
        h_ptr[i] = d_ptr + i;
    }

    cudaMemcpy(d_ptr, h_ptr, THREAD * sizeof(void *), cudaMemcpyHostToDevice);

    // populate instruction cache
    for (int i = 0; i < WARMUP; ++i) {
        l1_latency_kernel<ROUND><<<1, THREAD>>>(d_ptr, d_ret, d_clk);
    }

    // l1 cache latency benchmark
    l1_latency_kernel<ROUND><<<1, THREAD>>>(d_ptr, d_ret, d_clk);

    uint32_t h_clk[THREAD];
    cudaMemcpy(h_clk, d_clk, THREAD * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    printf("l1 cache latency %u cycles\n", h_clk[0] / ROUND);

    cudaFree(d_ptr);
    cudaFree(d_ret);
    cudaFree(d_clk);
    cudaFreeHost(h_ptr);

    return 0;
}


