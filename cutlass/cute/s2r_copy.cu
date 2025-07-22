#include <cuda.h>
#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_sm75.hpp>


template<typename T>
__device__ void initShm(T* shm, int size) {
    int idx = threadIdx.x;
    int offset = 0;
    while (offset * warpSize + idx < size) {
        shm[offset * warpSize + idx] = static_cast<T>(offset * warpSize + idx);
        offset+= 1;
    }
}


template<typename T, typename Config, typename ALayout_, typename BLayout_>
__global__ void S2RCopyKernel(T* ptr) {
    using namespace cute;
    // using T = typename Config::T;
    using S2RCopyA = typename Config::s2r_copy_atom_A;
    using S2RCopyB = typename Config::s2r_copy_atom_B;
    using BasicMMA = typename Config::mma;
    using ShmLayoutA = ALayout_;
    using ShmLayoutB = BLayout_;

    // Thread ID
    int idx = threadIdx.x;

    // Make Share Memory Tensor with shape ShmMatrixLayout
    extern __shared__ T shm_data[];
    T* shm = shm_data;
    initShm<T>(shm, cosize(ShmLayoutA{}));
    initShm<T>(shm + cosize(ShmLayoutA{}), cosize(ShmLayoutB{}));
    __syncthreads();
    auto shmMatrixA = make_tensor(make_smem_ptr(shm), ShmLayoutA{});    // (16, 8):(8, 1)
    auto shmMatrixB = make_tensor(make_smem_ptr(shm + cosize(ShmLayoutA{})), ShmLayoutB{});    // (8, 8):(1, 8)

    BasicMMA basic_mma; // MNK 16x8x8 --> A: 16x8, B: 8x8
    auto thr_mma = basic_mma.get_slice(idx);
    auto tCrA = thr_mma.partition_fragment_A(shmMatrixA);    // Fragment(Registers) for Matrix A
    auto tCrB = thr_mma.partition_fragment_B(shmMatrixB);    // Fragment(Registers) for Matrix B
    /*
    * A: (16, 8):(8, 1)
    * MMA can deal with 16x8x8 for on time,
    * so matrix A will be tiled into (MMA, 1, 1),
    * tCrA is ((_2,_2),_1,_1):((_1,_2),_0,_0), _1,_1 -> (MMA, [1, 1]),
    * (_2,_2) --> 4 elements per thread
    * 
    * B: (8, 8):(1, 8)
    * tCrB is (_2,_1,_1):(_1,_0,_0)
    */

#ifdef DEBUG
    if (thread0()) {
        print(tCrA);
        print("\n");
        print(tCrB);
        print("\n");
    }
#endif

    auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyA{}, basic_mma);
    auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
    auto tAsA = s2r_thr_copy_a.partition_S(shmMatrixA);
    auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA); // ((_4,_1),_1,_1):((_1,_0),_0,_0)

    auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyB{}, basic_mma);
    auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
    auto tBsB = s2r_thr_copy_b.partition_S(shmMatrixB);
    auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);

    cute::copy(s2r_tiled_copy_a, tAsA, tCrA_view);
    cute::copy(s2r_tiled_copy_b, tBsB, tCrB_view);

    T* vals_A = static_cast<T*>(tCrA_view.data());
    T* vals_B = static_cast<T*>(tCrB_view.data());
    ptr[idx] = (*vals_A) + (*vals_B);

#ifdef DEBUG
    printf("Thread ID: %d hold A value %.1f, %.1f, %.1f, %.1f\n",
        idx, float(tCrA_view(0)), float(tCrA_view(1)), float(tCrA_view(2)), float(tCrA_view(3)));
    printf("Thread ID: %d hold B value %.1f, %.1f\n",
        idx, float(tCrB_view(0)), float(tCrB_view(1)));
#endif
}


namespace config {
    using namespace cute;

    template<typename T_, typename S2RCopyOpA_, typename S2RCopyOpB_, typename TiledMMA_>
    struct CopyConfig {
        // Define 16-bits datatype
        using T = T_;

        // Define Shm -> Reg CopyAtom
        using s2r_copy_op_A = S2RCopyOpA_;
        using s2r_copy_traits_A = Copy_Traits<s2r_copy_op_A>;
        using s2r_copy_atom_A = Copy_Atom<s2r_copy_traits_A, T>;

        using s2r_copy_op_B = S2RCopyOpB_;
        using s2r_copy_traits_B = Copy_Traits<s2r_copy_op_B>;
        using s2r_copy_atom_B = Copy_Atom<s2r_copy_traits_B, T>;

        // Define Basic MMA
        using mma = TiledMMA_;
    };

    template<typename TiledMMA_, int permuteM_, int permuteN_, int permuteK_>
    struct InputConfig {
        using mnk = typename TiledMMA_::Shape_MNK;
        
        // Define A/B Shared memory layout
        static constexpr int permute_m = permuteM_;
        static constexpr int permute_n = permuteN_;
        static constexpr int permute_k = permuteK_;
        static constexpr int _M = permute_m * get<0>(mnk{});
        static constexpr int _N = permute_n * get<1>(mnk{});
        static constexpr int _K = permute_k * get<2>(mnk{});

        using ShmLayoutA = Layout<Shape<Int<_M>, Int<_K>>,
                                Stride<Int<_K>, Int<1>>
                                >;  // Row Major
        using ShmLayoutTransA = Layout<Shape<Int<_M>, Int<_K>>,
                                    Stride<Int<1>, Int<_M>>
                                    >;  // Column Major
        using ShmLayoutB = Layout<Shape<Int<_K>, Int<_N>>,
                                Stride<Int<1>, Int<_K>>
                                >;  // Column Major
        using ShmLayoutTransB = Layout<Shape<Int<_K>, Int<_N>>,
                                Stride<Int<_N>, Int<1>>
                                >;  // Row Major
    };
}   // namespace config


template<typename MMA_Op_, typename CopyA_, typename CopyB_,
    int permuteM_ = 1, int permuteN_ = 1, int permuteK_ = 1>
int S2RCopyKernelStub() {
    using namespace cute;
    static constexpr int permute_m = permuteM_;
    static constexpr int permute_n = permuteN_;
    static constexpr int permute_k = permuteK_;

    using mma_op = MMA_Op_;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    using mma_atom_shape = typename mma_traits::Shape_MNK;
    static constexpr int kMmaPM = permute_m * get<0>(mma_atom_shape{});
    static constexpr int kMmaPN = permute_n * get<1>(mma_atom_shape{});
    static constexpr int kMmaPK = permute_k * get<2>(mma_atom_shape{});
    using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;

    using tiled_mma = decltype(make_tiled_mma(mma_atom{},
                        make_layout(make_shape(_1{}, _1{}, _1{})),  // Just on warp
                        MMA_P_T{}));

    using copyConfig = config::CopyConfig<half_t, CopyA_, CopyB_, tiled_mma>;
    using inputConfig = config::InputConfig<tiled_mma, permute_m, permute_n, permute_k>;

    half_t* dev_out;
    cudaMalloc((void**)&dev_out, sizeof(half_t) * 32);

    dim3 block(size(tiled_mma{}), 1, 1);   // One warp
    dim3 grid(1, 1, 1); // One ThreadBlock

    // using ShmLayoutA = typename inputConfig::ShmLayoutA;    // Row-Major
    using ShmLayoutA = typename inputConfig::ShmLayoutTransA;   // Column-Major
    // using ShmLayoutB = typename inputConfig::ShmLayoutB;    // Column-Major
    using ShmLayoutB = typename inputConfig::ShmLayoutTransB;   // Row-Major
#ifdef DEBUG
    print_layout(ShmLayoutA{});
    print_layout(ShmLayoutB{});
#endif
    constexpr int shm_size = cosize(ShmLayoutA{}) + cosize(ShmLayoutB{});
    cudaFuncSetAttribute(S2RCopyKernel<half_t, copyConfig, ShmLayoutA, ShmLayoutB>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    S2RCopyKernel<half_t, copyConfig, ShmLayoutA, ShmLayoutB>
            <<<grid, block, shm_size>>>(dev_out);

    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    printf("Error Code: %d, State: %s\n", err, cudaGetErrorString(err));
    cudaFree(dev_out);
    return 0;
}


int main(int argc, char** argv) {
    using namespace cute;

    // S2RCopyKernelStub<SM80_16x8x8_F16F16F16F16_TN, SM75_U32x2_LDSM_N, SM75_U32x2_LDSM_N, 1, 2, 1>();    // OK
    // S2RCopyKernelStub<SM80_16x8x8_F16F16F16F16_TN, SM75_U32x1_LDSM_N, SM75_U32x1_LDSM_N, 1, 2, 1>();    // OK
    // S2RCopyKernelStub<SM80_16x8x8_F16F16F16F16_TN, SM75_U32x4_LDSM_N, SM75_U32x4_LDSM_N, 1, 2, 1>();    // ERROR

    S2RCopyKernelStub<SM80_16x8x8_F16F16F16F16_TN, SM75_U16x8_LDSM_T, SM75_U32x1_LDSM_N, 2, 1, 1>();
}
