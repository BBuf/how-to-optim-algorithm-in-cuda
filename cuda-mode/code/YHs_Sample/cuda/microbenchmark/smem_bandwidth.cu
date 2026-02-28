#include <cstdio>
#include <cstdint>

const int WARMUP = 100;
const int BLOCK = 256;

// number of LDS instructions to be timed
const int ROUND = 512;

__global__ void smem_bandwidth_kernel(
        int *ret, uint32_t *clk_start, uint32_t *clk_stop) {
    __shared__ int4 smem[BLOCK + ROUND];

    uint32_t tid = threadIdx.x;

    uint32_t start;
    uint32_t stop;
    uint32_t smem_addr;
    int4 reg = make_int4(tid, tid + 1, tid + 2, tid + 3);

    asm volatile (
        "{.reg .u64 u64addr;\n"
        " cvta.to.shared.u64 u64addr, %1;\n"
        " cvt.u32.u64 %0, u64addr;}\n"
        : "=r"(smem_addr)
        : "l"(smem + tid)
    );

    asm volatile (
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(start) : : "memory"
    );

    #pragma unroll
    for (int i = 0; i < ROUND; ++i) {
        asm volatile (
            "st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n"
            :
            : "r"(smem_addr + i * (uint32_t)sizeof(int4))
              "r"(reg.x), "r"(reg.y), "r"(reg.z), "r"(reg.w)
            : "memory"
        );
    }

    asm volatile (
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(stop) : : "memory"
    );

    if (threadIdx.x % 32 == 0) {
        clk_start[threadIdx.x / 32] = start;
        clk_stop[threadIdx.x / 32] = stop;
    }

    // dummy write back
    int tmp = ((int *)smem)[tid];
    if (tmp < 0) {
        *ret = tmp;
    }
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    if (prop.major == 3) {
        // enable 64-bit bank for Kepler GPU
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
        if (BLOCK < 256) {
            // 2 warps each schedular for Kepler GPU
            printf("thread block size is not enough to utilze all LSU.\n");
        }
    } else {
        if (BLOCK < 128) {
            // 1 warp each schedular for Maxwell+ GPU
            printf("thread block size is not enough to utilze all LSU.\n");
        }
    }

    int *d_ret;
    uint32_t *d_clk_start;
    uint32_t *d_clk_stop;
    cudaMalloc(&d_ret, BLOCK * sizeof(int));
    cudaMalloc(&d_clk_start, BLOCK / 32 * sizeof(uint32_t));
    cudaMalloc(&d_clk_stop, BLOCK / 32 * sizeof(uint32_t));

    // pupulate l0/l1 i-cache
    for (int i = 0; i < WARMUP; ++i) {
        smem_bandwidth_kernel<<<1, BLOCK>>>(d_ret, d_clk_start, d_clk_stop);
    }

    // shared memory bandwidth benchmark
    smem_bandwidth_kernel<<<1, BLOCK>>>(d_ret, d_clk_start, d_clk_stop);

    uint32_t h_clk_start[BLOCK];
    uint32_t h_clk_stop[BLOCK];
    cudaMemcpy(h_clk_start, d_clk_start, BLOCK / 32 * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_clk_stop, d_clk_stop, BLOCK / 32 * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);

    uint32_t start_min = ~0;
    uint32_t stop_max = 0;

    for (int i = 0; i < BLOCK / 32; ++i) {
        if (h_clk_start[i] < start_min) {
            start_min = h_clk_start[i];
        }
        if (h_clk_stop[i] > stop_max) {
            stop_max = h_clk_stop[i];
        }
    }

    uint32_t smem_size = BLOCK * ROUND * sizeof(int4);
    uint32_t duration = stop_max - start_min;
    float bw_measured = float(smem_size) / duration;
    // round up by 32
    uint32_t bw_theoretical = ((uint32_t)bw_measured + 31) / 32 * 32;

    printf("shared memory accessed: %u byte\n", smem_size);
    printf("duration: %u cycles\n", duration);
    printf("shared memory bandwidth per SM (measured): %f byte/cycle\n", bw_measured);
    printf("shared memory bandwidth per SM (theoretical): %u byte/cycle\n", bw_theoretical);

    uint32_t clk = prop.clockRate / 1000;
    uint32_t sm = prop.multiProcessorCount;
    float chip_bandwidth = float(sm) * bw_theoretical * clk / 1000;
    printf("standard clock frequency: %u MHz\n", clk);
    printf("SM: %u\n", sm);
    printf("whole chip shared memory bandwidth (theoretical): %f GB/s\n", chip_bandwidth);

    cudaFree(d_ret);
    cudaFree(d_clk_start);
    cudaFree(d_clk_stop);

    return 0;
}


