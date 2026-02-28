#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>

void random_init(float *data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = float(rand()) / RAND_MAX;
    }
}

bool check(const float *A,
           const float *B,
           const float *C,
           int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.f;
            for (int p = 0; p < k; ++p) {
                sum += A[i * k + p] * B[j + p * n];
            }

            if (std::fabs(sum - C[i * n + j]) / std::fabs(sum) > 1e-5f) {
                printf("C[%d][%d] not match, %f vs %f\n", i, j, sum, C[i * n + j]);
                return false;
            }
        }
    }

    return true;
}

__device__ __forceinline__
uint32_t smem_u32addr(const void *smem_ptr) {
    uint32_t addr;
    asm ("{.reg .u64 u64addr;\n"
         " cvta.to.shared.u64 u64addr, %1;\n"
         " cvt.u32.u64 %0, u64addr;}\n"
         : "=r"(addr)
         : "l"(smem_ptr)
    );

    return addr;
}

__device__ __forceinline__
void ldgsts32(const uint32_t &smem_addr,
              const void *gmem_ptr,
              bool guard) {
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4
        " @p cp.async.ca.shared.global.L2::128B [%0], [%1], 4;}\n"
#else
        " @p cp.async.ca.shared.global [%0], [%1], 4;}\n"
#endif
        : : "r"(smem_addr), "l"(gmem_ptr), "r"((int)guard)
    );
}

__device__ __forceinline__
void ldgsts32(const uint32_t &smem_addr,
              const void *gmem_ptr,
              const uint32_t &src_size,
              bool guard) {
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %3, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4
        " @p cp.async.ca.shared.global.L2::128B [%0], [%1], 4, %2;}\n"
#else
        " @p cp.async.ca.shared.global [%0], [%1], 4, %2;}\n"
#endif
        : : "r"(smem_addr), "l"(gmem_ptr), "r"(src_size), "r"((int)guard)
    );
}

__device__ __forceinline__
void ldgsts_commit() {
    asm volatile ("cp.async.wait_all;\n"::);
}

__device__ __forceinline__
void stg32(const float &reg, void *ptr, bool guard) {
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @p st.global.f32 [%0], %1;}\n"
        : : "l"(ptr), "f"(reg), "r"((int)guard)
    );
}

__device__ __forceinline__
void lds128(float &reg0, float &reg1,
            float &reg2, float &reg3,
            const uint32_t &addr) {
    asm volatile (
        "ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
        : "=f"(reg0), "=f"(reg1), "=f"(reg2), "=f"(reg3)
        : "r"(addr)
    );
}

__device__ __forceinline__
void sts32(const float &reg, const uint32_t &addr) {
    asm volatile (
        "st.shared.f32 [%0], %1;\n"
        : : "r"(addr), "f"(reg)
    );
}

__device__ __forceinline__
void sts128(const float &reg0, const float &reg1,
            const float &reg2, const float &reg3,
            const uint32_t &addr) {
    asm volatile (
        "st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n"
        : : "r"(addr), "f"(reg0), "f"(reg1), "f"(reg2), "f"(reg3)
    );
}

struct StgFrag {
    float data[4][4];

    __device__ __forceinline__
    StgFrag(const float (&C_frag)[16][8], int tile_x, int tile_y) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                data[i][j] = C_frag[tile_y * 4 + i][tile_x * 4 + j];
            }
        }
    }
};

__device__ __noinline__
void C_tile_wb(StgFrag C_frag,
               float *C_stg_ptr,
               const float *C_lds_ptr,
               uint32_t C_sts_addr,
               uint32_t m,
               uint32_t n,
               uint32_t m_idx,
               uint32_t n_idx) {
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        sts128(C_frag.data[i][0],
               C_frag.data[i][1],
               C_frag.data[i][2],
               C_frag.data[i][3],
               C_sts_addr + i * 9 * sizeof(float4));
    }

    __syncthreads();

    uint32_t m_guard = m < m_idx ? 0 : m - m_idx;

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        stg32(C_lds_ptr[i * 36],
              C_stg_ptr + i * n,
              i < m_guard && n_idx < n);
    }
}

/*
 * matrix A, B and C: row-major
 *
 * mma block:
 * thread block tile: m128n256k8
 * warp tile: m64n64k8
 * thread tile: m16n8k8
 * thread fragment:
 *     matrixA: 16x1 FP32
 *     matrixB: 1x8 FP32
 *
 * ----------------------------------------------------------------
 * thread block tile map:
 *
 *                                         256
 *                    --|---------------------------------------|
 *             B_tile  8|                                       |
 *                    --|---------------------------------------|
 *
 *  A_tile   | 8 |      |    64   |
 *         --|---|    --|---------|---------|---------|---------|
 *           |   |      |         |         |         |         |
 *           |   |    64|  warp0  |  warp1  |  warp2  |  warp3  |
 *           |   |      |         |         |         |         |
 *        128|   |    --|---------|---------|---------|---------|
 *           |   |      |         |         |         |         |
 *           |   |      |  warp4  |  warp5  |  warp6  |  warp7  |
 *           |   |      |         |         |         |         |
 *         --|---|      |---------|---------|---------|---------|
 *
 * ----------------------------------------------------------------
 * warp tile map:
 *
 * 'z' thread map to avoid LDS.128 shared memory broadcast limitation.
 *
 *              |              32               ||
 *     B_frag --|---|---|---|---|---|---|---|---||---|---|---|---|---|---|---|---|
 *             1|///|   |   |   |   |   |   |   ||///|   |   |   |   |   |   |   |
 *            --|---|---|---|---|---|---|---|---||---|---|---|---|---|---|---|---|
 * A_frag       | 4 |                           ||
 *    | 1 |                                     ||
 *  --|---|--   |---|---|---|---|---|---|---|---||---|---------------------------|
 *    |///|4    |t0 |t2 |t4 |t6 |t8 |t10|t12|t14||t0 |                           |
 *    |---|--   |---|---|---|---|---|---|---|---||---|                           |
 *    |   |     |t1 |t3 |t5 |t7 |t9 |t11|t13|t15||                               |
 *  16|---|     |---|---|---|---|---|---|---|---||                               |
 *    |   |     |t16|t18|t20|t22|t24|t26|t28|t30||                               |
 *    |---|     |---|---|---|---|---|---|---|---||                               |
 *    |   |     |t17|t19|t21|t23|t25|t27|t29|t31||                               |
 *  ==|===|=====|===|===|===|===|===|===|===|===||===|============================
 *    |///|     |t0 |                           ||t0 |                           |
 *    |---|     |---|                           ||---|                           |
 *    |   |     |                               ||                               |
 *    |---|     |                               ||                               |
 *    |   |     |                               ||                               |
 *    |---|     |                               ||                               |
 *    |   |     |                               ||                               |
 *    |---|     |-------------------------------||-------------------------------|
 *    |///|     |t0 |                           ||t0 |                           |
 *    |---|     |---|                           ||---|                           |
 *    |   |     |                               ||                               |
 *    |---|     |                               ||                               |
 *    |   |     |                               ||                               |
 *    |---|     |                               ||                               |
 *    |   |     |                               ||                               |
 *    |---|     |-------------------------------||-------------------------------|
 *    |///|     |t0 |                           ||t0 |                           |
 *    |---|     |---|                           ||---|                           |
 *    |   |     |                               ||                               |
 *    |---|     |                               ||                               |
 *    |   |     |                               ||                               |
 *    |---|     |                               ||                               |
 *    |   |     |                               ||                               |
 *    |---|     |-------------------------------||-------------------------------|
 *
 */
__global__ __launch_bounds__(256)
void ampere_sgemm_128x256x8_kernel(
        const float *A,
        const float *B,
        float *C,
        uint32_t m,
        uint32_t n,
        uint32_t k,
        uint32_t B_ldg_step) {  // n * sizeof(float) * 8
    /*
     * matrix A & B thread block tile shared memory (double buffer)
     * matrix A: 132 * 8 * 4Byte/item * double buffer = 4.125KB * 2
     * matrix B: 256 * 8 * 4Byte/item * double buffer = 16KB
     *
     * for double buffer faster switch, A_smem requires 8KB * 2 shared memory
     * and 16KB aligned, B_smem should be 16KB aligned, then the double buffer
     * can be switched by only 1 xor instruction:
     *     (uint32_t &)A_smem ^= 0x2000;
     *     (uint32_t &)B_smem ^= 0x2000;
     */
    __shared__ __align__(16 * 1024) char smem[32 * 1024];
    float *A_smem = reinterpret_cast<float *>(smem);
    float *B_smem = reinterpret_cast<float *>(smem + 16 * 1024);

    // A, B and C register fragment
    float A_frag[2][16];
    float B_frag[2][8];
    float C_frag[16][8];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            C_frag[i][j] = 0;
        }
    }

    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t warp_id = threadIdx.x / 32;

    // 4x8 threads each warp for FFMA
    const uint32_t mma_tid_x = (lane_id / 2) % 8;
    const uint32_t mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);

    // A_tile & B_tile ldg pointer
    const char *A_ldg_ptr = (const char *)(
        A + (blockIdx.y * 128 + threadIdx.x / 8) * k + threadIdx.x % 8);
    const char *B_ldg_ptr = (const char *)(
        B + (threadIdx.x / 128) * n + blockIdx.x * 256 + threadIdx.x % 128);

    // A_ldg_offset
    uint32_t A_ldg_offset[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        A_ldg_offset[i] = i * 32 * k * sizeof(float);
    }

    // B_ldg_offset
    uint32_t B_ldg_offset[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        B_ldg_offset[i] = i * 2 * n * sizeof(float);
    }

    // A_tile & B_tile sts/lds pointer
    // using uint32_t pointer for faster double buffer switch
    uint32_t A_sts_addr = smem_u32addr(
        A_smem + (threadIdx.x % 8) * 132 + (threadIdx.x / 8));
    uint32_t B_sts_addr = smem_u32addr(
        B_smem + (threadIdx.x / 128) * 256 + (threadIdx.x % 128));

    uint32_t A_lds_addr = smem_u32addr(
        A_smem + (warp_id / 4) * 64 + mma_tid_y * 4);
    uint32_t B_lds_addr = smem_u32addr(
        B_smem + (warp_id % 4) * 64 + mma_tid_x * 4);

    // ldg_guard to avoid LDG out of bound
    uint32_t A_ldg_guard = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int m_idx = blockIdx.y * 128 + threadIdx.x / 8 + i * 32;
        if (m_idx < m) {
            A_ldg_guard |= (1u << i);
        }
    }

    uint32_t B_ldg_guard = 0;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n_idx = blockIdx.x * 256 + threadIdx.x % 128 + i * 128;
        if (n_idx < n) {
            B_ldg_guard |= (1u << i);
        }
    }

    // 1'st A&B tile loaded before the k_tile loop
    uint32_t k_tiles = (k + 7) / 8 - 1;

    // load 1'st tile to shared memory
    {
        uint32_t first_k_tile = k - k_tiles * 8;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint32_t src_size = threadIdx.x % 8 < first_k_tile ? 4 : 0;

            ldgsts32(A_sts_addr + i * 32 * sizeof(float),
                     A_ldg_ptr + A_ldg_offset[i],
                     src_size,
                     (A_ldg_guard & (1u << i)) != 0);
        }

        A_ldg_ptr += first_k_tile * sizeof(float);

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint32_t src_size = i * 2 + threadIdx.x / 128 < first_k_tile ? 4 : 0;

            ldgsts32(B_sts_addr + i * 2 * 256 * sizeof(float),
                     B_ldg_ptr + B_ldg_offset[i],
                     src_size,
                     (B_ldg_guard & (1u << 0)) != 0);
            ldgsts32(B_sts_addr + (i * 2 * 256 + 128) * sizeof(float),
                     B_ldg_ptr + B_ldg_offset[i] + 128 * sizeof(float),
                     src_size,
                     (B_ldg_guard & (1u << 1)) != 0);
        }

        B_ldg_ptr += n * first_k_tile * sizeof(float);

        ldgsts_commit();
        __syncthreads();

        // switch double buffer
        A_sts_addr ^= 0x2000;
        B_sts_addr ^= 0x2000;
    }

    // load 1'st fragment
    lds128(A_frag[0][0], A_frag[0][1], A_frag[0][2], A_frag[0][3],
           A_lds_addr);
    lds128(A_frag[0][4], A_frag[0][5], A_frag[0][6], A_frag[0][7],
           A_lds_addr + 16 * sizeof(float));
    lds128(A_frag[0][8], A_frag[0][9], A_frag[0][10], A_frag[0][11],
           A_lds_addr + 32 * sizeof(float));
    lds128(A_frag[0][12], A_frag[0][13], A_frag[0][14], A_frag[0][15],
           A_lds_addr + 48 * sizeof(float));
    lds128(B_frag[0][0], B_frag[0][1], B_frag[0][2], B_frag[0][3],
           B_lds_addr);
    lds128(B_frag[0][4], B_frag[0][5], B_frag[0][6], B_frag[0][7],
           B_lds_addr + 32 * sizeof(float));

    // k_tiles loop
    for (; k_tiles > 0; --k_tiles) {
        #pragma unroll
        for (int k_frag = 0; k_frag < 8; ++k_frag) {
            // store next A&B tile to shared memory
            if (k_frag == 7) {
                ldgsts_commit();
                __syncthreads();

                // switch double buffer
                A_lds_addr ^= 0x2000;
                B_lds_addr ^= 0x2000;
                A_sts_addr ^= 0x2000;
                B_sts_addr ^= 0x2000;

                // ldg pointer for next tile
                A_ldg_ptr += 8 * sizeof(float);
                B_ldg_ptr += B_ldg_step;
            }

            // load next A&B fragment from shared memory to register
            lds128(A_frag[(k_frag + 1) % 2][0],
                   A_frag[(k_frag + 1) % 2][1],
                   A_frag[(k_frag + 1) % 2][2],
                   A_frag[(k_frag + 1) % 2][3],
                   A_lds_addr + (k_frag + 1) % 8 * 132 * sizeof(float));
            lds128(A_frag[(k_frag + 1) % 2][4],
                   A_frag[(k_frag + 1) % 2][5],
                   A_frag[(k_frag + 1) % 2][6],
                   A_frag[(k_frag + 1) % 2][7],
                   A_lds_addr + ((k_frag + 1) % 8 * 132 + 16) * sizeof(float));
            lds128(A_frag[(k_frag + 1) % 2][8],
                   A_frag[(k_frag + 1) % 2][9],
                   A_frag[(k_frag + 1) % 2][10],
                   A_frag[(k_frag + 1) % 2][11],
                   A_lds_addr + ((k_frag + 1) % 8 * 132 + 32) * sizeof(float));
            lds128(A_frag[(k_frag + 1) % 2][12],
                   A_frag[(k_frag + 1) % 2][13],
                   A_frag[(k_frag + 1) % 2][14],
                   A_frag[(k_frag + 1) % 2][15],
                   A_lds_addr + ((k_frag + 1) % 8 * 132 + 48) * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][0],
                   B_frag[(k_frag + 1) % 2][1],
                   B_frag[(k_frag + 1) % 2][2],
                   B_frag[(k_frag + 1) % 2][3],
                   B_lds_addr + (k_frag + 1) % 8 * 256 * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][4],
                   B_frag[(k_frag + 1) % 2][5],
                   B_frag[(k_frag + 1) % 2][6],
                   B_frag[(k_frag + 1) % 2][7],
                   B_lds_addr + ((k_frag + 1) % 8 * 256 + 32) * sizeof(float));

            // load next A&B tile
            if (k_frag < 4) {
                ldgsts32(A_sts_addr + k_frag * 32 * sizeof(float),
                         A_ldg_ptr + A_ldg_offset[k_frag],
                         (A_ldg_guard & (1u << k_frag)) != 0);

                ldgsts32(B_sts_addr + k_frag * 2 * 256 * sizeof(float),
                         B_ldg_ptr + B_ldg_offset[k_frag],
                         (B_ldg_guard & (1u << 0)) != 0);
                ldgsts32(B_sts_addr + (k_frag * 2 * 256 + 128) * sizeof(float),
                         B_ldg_ptr + B_ldg_offset[k_frag] + 128 * sizeof(float),
                         (B_ldg_guard & (1u << 1)) != 0);
            }

            // FFMA loop
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    C_frag[i][j] += A_frag[k_frag % 2][i] *
                                    B_frag[k_frag % 2][j];
                }
            }
        }
    }

    // FFMA for the last tile
    #pragma unroll
    for (int k_frag = 0; k_frag < 8; ++k_frag) {
        if (k_frag < 7) {
            // load next A&B fragment from shared memory to register
            lds128(A_frag[(k_frag + 1) % 2][0],
                   A_frag[(k_frag + 1) % 2][1],
                   A_frag[(k_frag + 1) % 2][2],
                   A_frag[(k_frag + 1) % 2][3],
                   A_lds_addr + (k_frag + 1) % 8 * 132 * sizeof(float));
            lds128(A_frag[(k_frag + 1) % 2][4],
                   A_frag[(k_frag + 1) % 2][5],
                   A_frag[(k_frag + 1) % 2][6],
                   A_frag[(k_frag + 1) % 2][7],
                   A_lds_addr + ((k_frag + 1) % 8 * 132 + 16) * sizeof(float));
            lds128(A_frag[(k_frag + 1) % 2][8],
                   A_frag[(k_frag + 1) % 2][9],
                   A_frag[(k_frag + 1) % 2][10],
                   A_frag[(k_frag + 1) % 2][11],
                   A_lds_addr + ((k_frag + 1) % 8 * 132 + 32) * sizeof(float));
            lds128(A_frag[(k_frag + 1) % 2][12],
                   A_frag[(k_frag + 1) % 2][13],
                   A_frag[(k_frag + 1) % 2][14],
                   A_frag[(k_frag + 1) % 2][15],
                   A_lds_addr + ((k_frag + 1) % 8 * 132 + 48) * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][0],
                   B_frag[(k_frag + 1) % 2][1],
                   B_frag[(k_frag + 1) % 2][2],
                   B_frag[(k_frag + 1) % 2][3],
                   B_lds_addr + (k_frag + 1) % 8 * 256 * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][4],
                   B_frag[(k_frag + 1) % 2][5],
                   B_frag[(k_frag + 1) % 2][6],
                   B_frag[(k_frag + 1) % 2][7],
                   B_lds_addr + ((k_frag + 1) % 8 * 256 + 32) * sizeof(float));
        }

        // FFMA loop
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                C_frag[i][j] += A_frag[k_frag % 2][i] *
                                B_frag[k_frag % 2][j];
            }
        }
    }

    // C_tile write back, reuse A&B tile shared memory buffer
    uint32_t C_sts_addr = smem_u32addr((float4 *)(smem + warp_id * 4096) +
                                       mma_tid_y * 4 * 9 + mma_tid_x);
    const float *C_lds_ptr = (float *)(smem + warp_id * 4096) + lane_id;

    uint32_t m_idx = blockIdx.y * 128 + warp_id / 4 * 64;
    uint32_t n_idx = blockIdx.x * 256 + warp_id % 4 * 64 + lane_id;

    float *C_stg_ptr = C + m_idx * n + n_idx;

    if (m_idx >= m) {
        return;
    } else if (m_idx + 64 <= m) {
        uint32_t n_guard = n < n_idx ? 0 : n - n_idx;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                __syncthreads();

                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    sts128(C_frag[i * 4 + p][j * 4],
                           C_frag[i * 4 + p][j * 4 + 1],
                           C_frag[i * 4 + p][j * 4 + 2],
                           C_frag[i * 4 + p][j * 4 + 3],
                           C_sts_addr + p * 9 * sizeof(float4));
                }

                __syncthreads();

                #pragma unroll
                for (int p = 0; p < 16; ++p) {
                    stg32(C_lds_ptr[p * 36],
                          C_stg_ptr + (i * 16 + p) * n + j * 32,
                          j * 32 < n_guard);
                }
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                StgFrag stg_frag(C_frag, j, i);

                C_tile_wb(stg_frag,
                          C_stg_ptr + i * 16 * n + j * 32,
                          C_lds_ptr,
                          C_sts_addr,
                          m,
                          n,
                          m_idx + i * 16,
                          n_idx + j * 32);
            }
        }
    }
}

int main() {
    int m = 5120;
    int n = 4096;
    int k = 4096;
    int n_iter = 10;

    float *h_A, *h_B, *h_C;
    cudaMallocHost(&h_A, m * k * sizeof(float));
    cudaMallocHost(&h_B, k * n * sizeof(float));
    cudaMallocHost(&h_C, m * n * sizeof(float));
    random_init(h_A, m * k);
    random_init(h_B, k * n);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));

    cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyDefault);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    dim3 grid((n + 255) / 256, (m + 127) / 128);

    // warmup
    ampere_sgemm_128x256x8_kernel<<<grid, 256>>>(
        d_A, d_B, d_C, m, n, k,
        n * sizeof(float) * 8);

    cudaEventRecord(start);
    for (int i = 0; i < n_iter; ++i) {
        ampere_sgemm_128x256x8_kernel<<<grid, 256>>>(
            d_A, d_B, d_C, m, n, k,
            n * sizeof(float) * 8);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float ms;
    cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    long workload = n_iter * long(m) * n * k * 2;
    double gflops = (double(workload) / 1e9) / (double(ms) / 1e3);
    printf("Performance: %fGFLOPS\n", gflops);

    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDefault);

    bool chk = check(h_A, h_B, h_C, m, n, k);
    printf("Matrix_C check: %s\n", chk ? "OK" : "Failed");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
}


