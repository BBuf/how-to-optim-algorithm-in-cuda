#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/types.h>

#define USE_CUTLASS_LAUNCHER 1

namespace cute {

// Invalidate barrier present in shared memory
CUTE_HOST_DEVICE
void invalidate_barrier(uint64_t &smem_barrier) // 64 bits user-managed barrier in smem
{
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_barrier);
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" ::"r"(smem_int_ptr));
#elif defined(__CUDA_ARCH__)
  asm volatile("brkpt;\n" ::);
#endif
}

} // namespace cute

template <class ElementA,
          class ElementB,
          class SmemLayoutA, // (M,K)
          class SmemLayoutB> // (N,K)
struct SharedStorage_AB {
  alignas(128) cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
  alignas(128) cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;

  uint64_t tma_barrier[1];
};

template <class ElementC,
          class SmemLayoutC> // (M,N)
struct SharedStorage_C {
  alignas(128) cute::ArrayEngine<ElementC, cute::cosize_v<SmemLayoutC>> C;

  uint64_t tma_barrier[1];
};

template <typename Spec,
          bool IsGemm,
          bool use_tma_store_reduce_add,
          typename TMA_A,
          typename TMA_B,
          typename TMA_C,
          typename TMA_D>
__global__ __launch_bounds__(Spec::kThreadNum) void
#if !(USE_CUTLASS_LAUNCHER)
    __cluster_dims__(Spec::kClusterDimX, Spec::kClusterDimY, Spec::kClusterDimZ)
#endif
        tma_multicast_reduce(__grid_constant__ TMA_A const tma_A,
                             __grid_constant__ TMA_B const tma_B,
                             __grid_constant__ TMA_C const tma_C,
                             __grid_constant__ TMA_D const tma_D,
                             int M,
                             int N,
                             int K) {
  using namespace cute;

  using X = Underscore;
  using OutType = typename Spec::OutType;
  using ComputeTypeA = typename Spec::ComputeTypeA;
  using ComputeTypeB = typename Spec::ComputeTypeB;
  using ComputeTypeC = typename Spec::ComputeTypeC;
  using SmemLayoutA = typename Spec::SmemLayoutA;
  using SmemLayoutB = typename Spec::SmemLayoutB;
  using SmemLayoutC = typename Spec::SmemLayoutC;
  using SmemLayoutD = typename Spec::SmemLayoutD;

  constexpr int kTileM = Spec::kTileM;
  constexpr int kTileN = Spec::kTileN;
  constexpr int kTileK = Spec::kTileK;

  extern __shared__ __align__(1024) uint8_t smem_raw[];

  using SharedStorage = SharedStorage_AB<ComputeTypeA, ComputeTypeB, SmemLayoutA, SmemLayoutB>;
  SharedStorage &smem = *reinterpret_cast<SharedStorage *>(smem_raw);
  // >=8-bit SMEM pointer
  auto *Aptr_smem = smem.A.begin();
  auto *Bptr_smem = smem.B.begin();

  // 8-bit SMEM pointer
  // uint8_t *Aptr_smem = reinterpret_cast<uint8_t*>(smem.A.begin());
  // uint8_t *Bptr_smem = reinterpret_cast<uint8_t*>(smem.B.begin());

  // 8-bit SMEM pointer
  // uint8_t *Aptr_smem = smem_raw;
  // uint8_t *Bptr_smem = smem_raw + Spec::kShmSizeA;
  uint8_t *Cptr_smem = smem_raw;
  uint8_t *Dptr_smem = smem_raw;

  int tid = threadIdx.x;
  int bidx = blockIdx.x;
  int bidy = blockIdx.y;

  Tensor mA = tma_A.get_tma_tensor(make_shape(M, K));
  Tensor mB = tma_B.get_tma_tensor(make_shape(N, K));
  Tensor mC = tma_C.get_tma_tensor(make_shape(M, N));
  Tensor mD = tma_D.get_tma_tensor(make_shape(M, N));

  auto tiler = make_tile(Int<kTileM>{}, Int<kTileN>{}, Int<kTileK>{});
  auto coord = make_coord(bidy, bidx, _);

  Tensor gA = local_tile(mA, tiler, coord, Step<_1, X, _1>{}); // (BLK_M, BLK_K, K_TILES)
  Tensor gB = local_tile(mB, tiler, coord, Step<X, _1, _1>{}); // (BLK_N, BLK_K, K_TILES)
  Tensor gC = local_tile(mC, tiler, coord, Step<_1, _1, X>{}); // (BLK_M, BLK_N)
  Tensor gD = local_tile(mD, tiler, coord, Step<_1, _1, X>{}); // (BLK_M, BLK_N)

  Tensor sA = make_tensor(make_smem_ptr((ComputeTypeA *)Aptr_smem), SmemLayoutA{}); // (BLK_M, BLK_K)
  Tensor sB = make_tensor(make_smem_ptr((ComputeTypeB *)Bptr_smem), SmemLayoutB{}); // (BLK_N, BLK_K)
  Tensor sC = make_tensor(make_smem_ptr((ComputeTypeC *)Cptr_smem), SmemLayoutC{}); // (BLK_M, BLK_N)
  Tensor sD = make_tensor(make_smem_ptr((OutType *)Dptr_smem), SmemLayoutD{});      // (BLK_M, BLK_N)

  // Define the CTA-in-cluster Layout and Coord
  uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
  auto cta_layout_mnk = make_layout(typename Spec::ClusterShape{});
  auto cta_coord_mnk = cta_layout_mnk.get_flat_coord(block_rank_in_cluster);

  // TMA Multicast Masks
  uint16_t mcast_mask_a = create_tma_multicast_mask<0>(cta_layout_mnk, cta_coord_mnk);
  uint16_t mcast_mask_b = create_tma_multicast_mask<1>(cta_layout_mnk, cta_coord_mnk);

  auto [tAgA, tAsA] = tma_partition(tma_A,
                                    get<0>(cta_coord_mnk), // The CTA coordinate along N mode of the cluster
                                    make_layout(size<0>(cta_layout_mnk)), // The CTA layout along N mode of the cluster
                                    group_modes<0, 2>(sA), group_modes<0, 2>(gA)); // (TMA,K_TILES) and (TMA,kStages)
  auto [tBgB, tBsB] = tma_partition(tma_B,
                                    get<1>(cta_coord_mnk), // The CTA coordinate along M mode of the cluster
                                    make_layout(size<1>(cta_layout_mnk)), // The CTA layout along M mode of the cluster
                                    group_modes<0, 2>(sB), group_modes<0, 2>(gB)); // (TMA,K_TILES) and (TMA,kStages)
  auto [tCgC, tCsC] = tma_partition(tma_C, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sC),
                                    group_modes<0, 2>(gC)); // (TMA,K_TILES) and (TMA)
  auto [tDgD, tDsD] = tma_partition(tma_D, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sD),
                                    group_modes<0, 2>(gD)); // (TMA,K_TILES) and (TMA)

  constexpr int tma_transaction_bytes =
      kTileM * kTileK * sizeof(ComputeTypeA) + kTileN * kTileK * sizeof(ComputeTypeB);

  typename Spec::TiledMMA tiled_mma;
  ThrMMA thr_mma = tiled_mma.get_slice(tid);

  // Note: Here we cannot use gA(_, _, 0) to subsitute sA
  Tensor tCrA = thr_mma.partition_fragment_A(sA); // (MMA, MMA_M, MMA_K)
  Tensor tCrB = thr_mma.partition_fragment_B(sB); // (MMA, MMA_N, MMA_K)
  Tensor tCrC = thr_mma.partition_fragment_C(sC); // (MMA, MMA_M, MMA_N)
  clear(tCrC);                                    // Set the accumulators to zero

  typename Spec::TiledCopyA_S2R s2r_tiled_copy_a;
  ThrCopy s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(tid);
  Tensor tAsA_s2r = s2r_thr_copy_a.partition_S(sA); // (CPY, CPY_M, CPY_K)
  Tensor tArA_s2r = s2r_thr_copy_a.retile_D(tCrA);  // (CPY, CPY_M, CPY_K)

  typename Spec::TiledCopyB_S2R s2r_tiled_copy_b;
  ThrCopy s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(tid);
  Tensor tBsB_s2r = s2r_thr_copy_b.partition_S(sB); // (CPY, CPY_M, CPY_K)
  Tensor tBrB_s2r = s2r_thr_copy_b.retile_D(tCrB);  // (CPY, CPY_M, CPY_K)

  //
  // MAINLOOP
  //

  int NTilesK = ceil_div(K, kTileK);

  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();
  uint64_t &tma_load_mbarrier = smem.tma_barrier[0];
  // uint64_t &tma_load_mbarrier = reinterpret_cast<uint64_t*>(smem_raw + Spec::kShmSizeA + Spec::kShmSizeB)[0];

  // Init TMA load barrier
  if ((warp_idx == 0) && lane_predicate) {
    initialize_barrier(tma_load_mbarrier, /* arrival thread count */ 1);
    // Make initialized barrier visible in async proxy.
    cutlass::arch::fence_view_async_shared();
  }

  // Ensure barrier init is complete on all CTAs
  if constexpr (Spec::kClusterSize > 1) {
    cluster_sync();
  } else {
    __syncthreads();
  }

  for (int ik = 0; ik < NTilesK; ++ik) {
    if ((warp_idx == 0) && lane_predicate) {
      copy(tma_A.with(tma_load_mbarrier, mcast_mask_a), tAgA(_, ik), tAsA);
      copy(tma_B.with(tma_load_mbarrier, mcast_mask_b), tBgB(_, ik), tBsB);

      // Arrive on the barrier and tell how many bytes are expected to come in.
      set_barrier_transaction_bytes(tma_load_mbarrier, tma_transaction_bytes);
    }

    wait_barrier(tma_load_mbarrier, /* phase */ ik & 1);

    copy(s2r_tiled_copy_a, tAsA_s2r, tArA_s2r);
    copy(s2r_tiled_copy_b, tBsB_s2r, tBrB_s2r);

    gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);

    if constexpr (Spec::kClusterSize > 1) {
      cluster_sync();
    } else {
      __syncthreads();
    }
  }

  // Important!
  if ((warp_idx == 0) && lane_predicate) {
    invalidate_barrier(tma_load_mbarrier);
  }
  __syncthreads();

  //
  // EPILOGUE
  //

  if constexpr (use_tma_store_reduce_add) {
    typename Spec::TiledCopyD_R2S r2s_tiled_copy_d;
    ThrCopy r2s_thr_copy_d = r2s_tiled_copy_d.get_slice(tid);
    Tensor tCrC_r2s = r2s_thr_copy_d.retile_S(tCrC);  // (CPY, CPY_M, CPY_N)
    Tensor tCsC_r2s = r2s_thr_copy_d.partition_D(sC); // (CPY, CPY_M, CPY_N)
    copy(r2s_tiled_copy_d, tCrC_r2s, tCsC_r2s);

    tma_store_fence();
    __syncthreads();

    if ((warp_idx == 0) && lane_predicate) {
      copy(tma_C, tCsC, tCgC);
      tma_store_arrive();
      tma_store_wait<0>();
    }

    return;

  } else {

    if constexpr (!IsGemm) {
      // Load C matrix with TMA
      constexpr int tma_transaction_load_c_bytes = kTileM * kTileN * sizeof(ComputeTypeC);
      using SharedStorage_C = SharedStorage_C<ComputeTypeC, SmemLayoutC>;
      SharedStorage_C &smem_c = *reinterpret_cast<SharedStorage_C *>(smem_raw);
      uint64_t &tma_load_c_mbarrier = smem_c.tma_barrier[0];

      auto tCrC_load = make_tensor_like<ComputeTypeC>(tCrC);

      typename Spec::TiledCopyC_S2R s2r_tiled_copy_c;
      ThrCopy s2r_thr_copy_c = s2r_tiled_copy_c.get_slice(tid);
      Tensor tCsC_s2r = s2r_thr_copy_c.partition_S(sC);     // (CPY, CPY_M, CPY_K)
      Tensor tCrC_s2r = s2r_thr_copy_c.retile_D(tCrC_load); // (CPY, CPY_M, CPY_K)

      if ((warp_idx == 0) && lane_predicate) {
        initialize_barrier(tma_load_c_mbarrier, /* arrival thread count */ 1);
        cutlass::arch::fence_view_async_shared();

        copy(tma_C.with(tma_load_c_mbarrier), tCgC, tCsC);
        set_barrier_transaction_bytes(tma_load_c_mbarrier, tma_transaction_load_c_bytes);
      }

      __syncthreads();
      wait_barrier(tma_load_c_mbarrier, /* phase */ 0);

      copy(s2r_tiled_copy_c, tCsC_s2r, tCrC_s2r);

// compute with CUDA core
#pragma unroll
      for (int i = 0; i < size(tCrC); ++i) {
        tCrC(i) = tCrC(i) + tCrC_load(i);
      }

      // Important!
      if ((warp_idx == 0) && lane_predicate) {
        invalidate_barrier(tma_load_c_mbarrier);
      }
      __syncthreads();
    }

    auto tCrD = make_tensor_like<OutType>(tCrC);
    copy(tCrC, tCrD); // Convert precision

    typename Spec::TiledCopyD_R2S r2s_tiled_copy_d;
    ThrCopy r2s_thr_copy_d = r2s_tiled_copy_d.get_slice(tid);
    Tensor tDrD_r2s = r2s_thr_copy_d.retile_S(tCrD);  // (CPY, CPY_M, CPY_N)
    Tensor tDsD_r2s = r2s_thr_copy_d.partition_D(sD); // (CPY, CPY_M, CPY_N)
    copy(r2s_tiled_copy_d, tDrD_r2s, tDsD_r2s);

    tma_store_fence();
    __syncthreads();

    if ((warp_idx == 0) && lane_predicate) {
      copy(tma_D, tDsD, tDgD);
      tma_store_arrive();
      tma_store_wait<0>();
    }
  }
}

namespace spec {

using namespace cute;

template <typename OutType_,
          typename ComputeTypeA_,
          typename ComputeTypeB_,
          typename ComputeTypeC_,
          int kTileM_ = 128,
          int kTileN_ = 128,
          int kTileK_ = 64>
struct KernelSpec {
  using OutType = OutType_;
  using ComputeTypeA = ComputeTypeA_;
  using ComputeTypeB = ComputeTypeB_;
  using ComputeTypeC = ComputeTypeC_;

  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;

  using MMA_op = std::conditional_t<
      std::is_same_v<ComputeTypeA, cute::bfloat16_t> && std::is_same_v<ComputeTypeB, cute::bfloat16_t> &&
          std::is_same_v<ComputeTypeC, float>,
      SM80_16x8x16_F32BF16BF16F32_TN,
      std::conditional_t<
          std::is_same_v<ComputeTypeA, cute::half_t> && std::is_same_v<ComputeTypeB, cute::half_t> &&
              std::is_same_v<ComputeTypeC, cute::half_t>,
          SM80_16x8x16_F16F16F16F16_TN,
          std::conditional_t<std::is_same_v<ComputeTypeA, cute::half_t> &&
                                 std::is_same_v<ComputeTypeB, cute::half_t> && std::is_same_v<ComputeTypeC, float>,
                             SM80_16x8x16_F32F16F16F32_TN,
                             void>>>;

  static_assert(!std::is_same_v<MMA_op, void>, "Unsupported MMA op!");

  using MMA_traits = MMA_Traits<MMA_op>;
  using MMA_atom = MMA_Atom<MMA_traits>;
  using MMA_shape = typename MMA_traits::Shape_MNK;

  static constexpr int kMmaThrExpandM = 2;
  static constexpr int kMmaThrExpandN = 4;
  static constexpr int kMmaThrExpandK = 1;

  static constexpr int kMmaValExpandM = 1;
  static constexpr int kMmaValExpandN = 2;
  static constexpr int kMmaValExpandK = 2;

  static constexpr int kMmaTileM = kMmaThrExpandM * kMmaValExpandM * get<0>(MMA_shape{});
  static constexpr int kMmaTileN = kMmaThrExpandN * kMmaValExpandN * get<1>(MMA_shape{});
  static constexpr int kMmaTileK = kMmaThrExpandK * kMmaValExpandK * get<2>(MMA_shape{});

  using MMAThrLayout =
      decltype(make_layout(make_shape(Int<kMmaThrExpandM>{}, Int<kMmaThrExpandN>{}, Int<kMmaThrExpandK>{})));
  using MMATileLayout = Tile<Int<kMmaTileM>, Int<kMmaTileN>, Int<kMmaTileK>>;

  using TiledMMA = decltype(make_tiled_mma(MMA_op{}, MMAThrLayout{}, MMATileLayout{}));
  static constexpr int kThreadNum = size(TiledMMA{});

  using ClusterShape = Shape<_2, _1, _1>;
  static constexpr int kClusterDimX = get<0>(ClusterShape{});
  static constexpr int kClusterDimY = get<1>(ClusterShape{});
  static constexpr int kClusterDimZ = get<2>(ClusterShape{});
  static constexpr int kClusterSize = size(ClusterShape{});

  using Copy_S2R_op_A = std::conditional_t<sizeof(ComputeTypeA) == 2, SM75_U32x4_LDSM_N, AutoVectorizingCopy>;
  using Copy_S2R_op_B = std::conditional_t<sizeof(ComputeTypeB) == 2, SM75_U32x4_LDSM_N, AutoVectorizingCopy>;
  using Copy_S2R_op_C = std::conditional_t<sizeof(ComputeTypeC) == 2, SM75_U32x4_LDSM_N, AutoVectorizingCopy>;

  using CopyA_S2R_atom = Copy_Atom<Copy_S2R_op_A, ComputeTypeA>;
  using CopyB_S2R_atom = Copy_Atom<Copy_S2R_op_B, ComputeTypeB>;
  using CopyC_S2R_atom = Copy_Atom<Copy_S2R_op_C, ComputeTypeC>;

  using Copy_R2S_op = SM90_U32x4_STSM_N;
  using CopyD_R2S_atom = Copy_Atom<Copy_R2S_op, OutType>;

  using TiledCopyA_S2R = decltype(make_tiled_copy_A(CopyA_S2R_atom{}, TiledMMA{}));
  using TiledCopyB_S2R = decltype(make_tiled_copy_B(CopyB_S2R_atom{}, TiledMMA{}));
  using TiledCopyC_S2R = decltype(make_tiled_copy_C(CopyC_S2R_atom{}, TiledMMA{}));
  using TiledCopyD_R2S = decltype(make_tiled_copy_C(CopyD_R2S_atom{}, TiledMMA{}));

  using SmemLayoutA = decltype(tile_to_shape(
      GMMA::Layout_K_SW128_Atom<ComputeTypeA>{}, make_shape(Int<kTileM>{}, Int<kTileK>{}), Step<_1, _2>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      GMMA::Layout_K_SW128_Atom<ComputeTypeB>{}, make_shape(Int<kTileN>{}, Int<kTileK>{}), Step<_1, _2>{}));
  using SmemLayoutC = decltype(tile_to_shape(
      GMMA::Layout_K_SW128_Atom<ComputeTypeC>{}, make_shape(Int<kTileM>{}, Int<kTileN>{}), Step<_1, _2>{}));
  using SmemLayoutD = decltype(tile_to_shape(
      GMMA::Layout_K_SW128_Atom<OutType>{}, make_shape(Int<kTileM>{}, Int<kTileN>{}), Step<_1, _2>{}));

  static constexpr int kShmSizeA = cosize_v<SmemLayoutA> * sizeof(ComputeTypeA);
  static constexpr int kShmSizeB = cosize_v<SmemLayoutB> * sizeof(ComputeTypeB);
  static constexpr int kShmSizeC = cosize_v<SmemLayoutC> * sizeof(ComputeTypeC);
  static constexpr int kShmSizeD = cosize_v<SmemLayoutD> * sizeof(OutType);

  static constexpr int kShmSize =
      cute::max(sizeof(SharedStorage_AB<ComputeTypeA, ComputeTypeB, SmemLayoutA, SmemLayoutB>),
                cute::max(sizeof(SharedStorage_C<ComputeTypeC, SmemLayoutC>), kShmSizeD));
};

} // namespace spec

#define CHECK_TORCH_TENSOR_DTYPE(T, DTYPE)                                                                            \
  do {                                                                                                                \
    if ((T).options().dtype() != (DTYPE)) {                                                                           \
      std::cerr << "Tensor dtype mismatch! Expected: " << (DTYPE) << ", but got: " << (T).options().dtype() << " at " \
                << __FILE__ << ":" << __LINE__ << std::endl;                                                          \
      std::exit(EXIT_FAILURE);                                                                                        \
    }                                                                                                                 \
  } while (0);

#define CHECK_TORCH_TENSOR_SHAPE(T, M, N)                                                                             \
  do {                                                                                                                \
    auto actual_shape = (T).sizes();                                                                                  \
    if (actual_shape != torch::IntArrayRef({M, N})) {                                                                 \
      std::cerr << "Tensor shape mismatch! Expected: " << torch::IntArrayRef({M, N}) << ", but got: " << actual_shape \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;                                                \
      std::exit(EXIT_FAILURE);                                                                                        \
    }                                                                                                                 \
  } while (0);

#define BOOL_SWITCH(COND, CONST_NAME, ...)                                                                            \
  [&] {                                                                                                               \
    if (COND) {                                                                                                       \
      constexpr static bool CONST_NAME = true;                                                                        \
      return __VA_ARGS__();                                                                                           \
    } else {                                                                                                          \
      constexpr static bool CONST_NAME = false;                                                                       \
      return __VA_ARGS__();                                                                                           \
    }                                                                                                                 \
  }()

template <typename T> constexpr torch::ScalarType to_torch_scalar_type() {
  if constexpr (std::is_same_v<T, cute::half_t>)
    return torch::kHalf;
  else if constexpr (std::is_same_v<T, cute::bfloat16_t>)
    return torch::kBFloat16;
  else if constexpr (std::is_same_v<T, float>)
    return torch::kFloat32;
  else if constexpr (std::is_same_v<T, cute::float_e4m3_t>)
    return torch::kFloat8_e4m3fn;
  else if constexpr (std::is_same_v<T, cute::float_e5m2_t>)
    return torch::kFloat8_e5m2;
  else
    throw std::runtime_error("Unsupported type!");
}

template <typename ComputeTypeC, typename OutType> constexpr bool if_use_tma_store_reduce_add() {
  if constexpr (std::is_same_v<ComputeTypeC, OutType>)
    return true;
  else
    return false;
}

template <int kTileM,
          int kTileN,
          int kTileK,
          typename OutType,
          typename ComputeTypeA,
          typename ComputeTypeB,
          typename ComputeTypeC = OutType>
torch::Tensor run_tma_multicast_reduce(const torch::Tensor a, const torch::Tensor b, std::optional<torch::Tensor> _c) {

  at::cuda::CUDAGuard device_guard{a.get_device()};
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  auto M = a.size(0);
  auto N = b.size(0);
  auto K = a.size(1);

  auto torch_compute_type_a = to_torch_scalar_type<ComputeTypeA>();
  auto torch_compute_type_b = to_torch_scalar_type<ComputeTypeB>();
  auto torch_compute_type_c = to_torch_scalar_type<ComputeTypeC>();
  auto torch_compute_type_d = to_torch_scalar_type<OutType>();

  torch::Tensor c, d;
  bool is_gemm;

  auto options = torch::TensorOptions().dtype(torch_compute_type_d).device(torch::kCUDA);
  d = torch::zeros({M, N}, options);

  if (!_c.has_value()) {
    c = d;
    is_gemm = true;
  } else {
    c = _c.value();
    is_gemm = false;
  }

  CHECK_TORCH_TENSOR_DTYPE(a, torch_compute_type_a)
  CHECK_TORCH_TENSOR_DTYPE(b, torch_compute_type_b)
  if (!is_gemm) CHECK_TORCH_TENSOR_DTYPE(c, torch_compute_type_c)
  CHECK_TORCH_TENSOR_DTYPE(d, torch_compute_type_d)

  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, N, K)
  if (!is_gemm) CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  CHECK_TORCH_TENSOR_SHAPE(d, M, N)

  using Spec = spec::KernelSpec<OutType, ComputeTypeA, ComputeTypeB, ComputeTypeC, kTileM, kTileN, kTileK>;
  auto cluster_shape = typename Spec::ClusterShape{};

  auto [cls_x, cls_y, cls_z] = cluster_shape;

  dim3 block = Spec::kThreadNum;
  dim3 cluster(cls_x, cls_y, cls_z);
  dim3 grid(cute::ceil_div(N, Spec::kTileN), cute::ceil_div(M, Spec::kTileM));
  int shm_size = Spec::kShmSize;

  printf("Block Size: (%d, %d, %d) | Cluster Size: (%d, %d, %d) | Grid Size: (%d, %d, %d) | Shared Memory Size: %d "
         "Bytes\n",
         block.x, block.y, block.z, cluster.x, cluster.y, cluster.z, grid.x, grid.y, grid.z, shm_size);

  ////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////// TMA Specific Code ///////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////

  using namespace cute;

  auto mA = make_tensor(make_gmem_ptr((ComputeTypeA *)a.data_ptr()), make_layout(make_shape(M, K), LayoutRight{}));
  // auto mA = make_tensor(make_gmem_ptr((ComputeTypeA *)a.data_ptr()), make_shape(M, K), GenRowMajor{});
  auto mB = make_tensor(make_gmem_ptr((ComputeTypeB *)b.data_ptr()), make_layout(make_shape(N, K), LayoutRight{}));
  auto mC = make_tensor(make_gmem_ptr((ComputeTypeC *)c.data_ptr()), make_layout(make_shape(M, N), LayoutRight{}));
  auto mD = make_tensor(make_gmem_ptr((OutType *)d.data_ptr()), make_layout(make_shape(M, N), LayoutRight{}));

  auto smem_layout_A = typename Spec::SmemLayoutA{};
  auto smem_layout_B = typename Spec::SmemLayoutB{};
  auto smem_layout_C = typename Spec::SmemLayoutC{};
  auto smem_layout_D = typename Spec::SmemLayoutD{};

  // A loads can be optimized with multicast if cluster-x > 1
  using TMA_Load_CopyA = std::conditional_t<get<0>(cluster_shape) == 1, SM90_TMA_LOAD, SM90_TMA_LOAD_MULTICAST>;

  // B loads can be optimized with multicast if cluster-y > 1
  using TMA_Load_CopyB = std::conditional_t<get<1>(cluster_shape) == 1, SM90_TMA_LOAD, SM90_TMA_LOAD_MULTICAST>;

  auto tma_A = make_tma_copy(TMA_Load_CopyA{}, mA, smem_layout_A, Int<get<0>(cluster_shape)>{});
  // auto tma_A = make_tma_atom(TMA_Load_CopyA{}, mA, smem_layout_A, make_shape(kTileM, kTileK),
  // Int<get<0>(cluster_shape)>{});
  auto tma_B = make_tma_copy(TMA_Load_CopyB{}, mB, smem_layout_B, Int<get<1>(cluster_shape)>{});

  constexpr bool use_tma_store_reduce_add = if_use_tma_store_reduce_add<ComputeTypeC, OutType>();
  using TMA_CopyC = std::conditional_t<use_tma_store_reduce_add, SM90_TMA_REDUCE_ADD, SM90_TMA_LOAD>;
  auto tma_C = make_tma_copy(TMA_CopyC{}, mC, smem_layout_C);
  auto tma_D = make_tma_copy(SM90_TMA_STORE{}, mD, smem_layout_D);

  ////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////// End TMA Code //////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaDeviceSynchronize();

  // Kernel launch
  BOOL_SWITCH(is_gemm, IsGemm, [&] {
    cudaEventRecord(start, stream);

#if USE_CUTLASS_LAUNCHER
    void *kernel_ptr = (void *)&tma_multicast_reduce<Spec, IsGemm, use_tma_store_reduce_add, decltype(tma_A),
                                                     decltype(tma_B), decltype(tma_C), decltype(tma_D)>;
    cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cutlass::launch_kernel_on_cluster({grid, block, cluster, shm_size, stream}, kernel_ptr, tma_A, tma_B, tma_C, tma_D,
                                      M, N, K);
#else
    cudaFuncSetAttribute(tma_multicast_reduce<Spec, IsGemm, use_tma_store_reduce_add, decltype(tma_A), decltype(tma_B), decltype(tma_C), decltype(tma_D)>,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    tma_multicast_reduce<Spec, IsGemm, use_tma_store_reduce_add><<<grid, block, shm_size, stream>>>(
      tma_A, tma_B, tma_C, tma_D, M, N, K
    );
#endif

    cudaEventRecord(stop, stream);
  });

  cudaDeviceSynchronize();

  auto error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error) +
                             " (error code: " + std::to_string(error) + ")");
  }

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel execution time: %.3f ms\n", milliseconds);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  if (use_tma_store_reduce_add)
    return c;
  else
    return d;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tma_multicast_reduce_fp16_fp16_fp16_fp16",
        &(run_tma_multicast_reduce<128, 128, 64, cute::half_t, cute::half_t, cute::half_t>),
        "Run a mixed-precision half 16x8x8 MMA operation.");
  m.def("tma_multicast_reduce_fp16_fp16_fp16_fp32",
        &(run_tma_multicast_reduce<128, 128, 64, cute::half_t, cute::half_t, cute::half_t, float>),
        "Run a mixed-precision half 16x8x8 MMA operation.");
  m.def("tma_multicast_reduce_bf16_bf16_bf16_fp32",
        &(run_tma_multicast_reduce<128, 128, 64, cute::bfloat16_t, cute::bfloat16_t, cute::bfloat16_t, float>),
        "Run a mixed-precision half 16x8x8 MMA operation.");
}
