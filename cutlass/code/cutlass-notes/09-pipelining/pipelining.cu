#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <torch/extension.h>
#include <torch/types.h>

template <typename Spec, bool IsGemm, bool IsCvtPrecision>
__global__ __launch_bounds__(Spec::kThreadNum) void pipelining(void *__restrict__ Cptr,
                                                               const void *__restrict__ Aptr,
                                                               const void *__restrict__ Bptr,
                                                               int M,
                                                               int N,
                                                               int K,
                                                               void *__restrict__ Outptr) {
  using namespace cute;

  using X = Underscore;
  using MMA_shape = typename Spec::MMA_shape;
  using OutType = typename Spec::OutType;
  using ComputeTypeA = typename Spec::ComputeTypeA;
  using ComputeTypeB = typename Spec::ComputeTypeB;
  using ComputeTypeC = typename Spec::ComputeTypeC;
  using SmemLayoutA = typename Spec::SmemLayoutA;
  using SmemLayoutB = typename Spec::SmemLayoutB;
  using SmemLayoutC = typename Spec::SmemLayoutC;
  using SmemLayoutO = typename Spec::SmemLayoutO;

  constexpr int kTileM = Spec::kTileM;
  constexpr int kTileN = Spec::kTileN;
  constexpr int kTileK = Spec::kTileK;
  constexpr int kShmSizeA = Spec::kShmSizeA;
  constexpr int kShmSizeB = Spec::kShmSizeB;
  constexpr int G2S_Stages = Spec::G2S_Stages;
  constexpr int kMmaValExpandK = Spec::kMmaValExpandK;

  extern __shared__ __align__(1024) uint8_t smem[];

  uint8_t *Aptr_smem = smem;
  uint8_t *Bptr_smem = smem + kShmSizeA;
  uint8_t *Cptr_smem;
  if constexpr (!IsGemm)
    Cptr_smem = smem + kShmSizeA + kShmSizeB;
  else
    Cptr_smem = smem;
  uint8_t *Optr_smem = smem;

  int tid = threadIdx.x;
  int bidx = blockIdx.x;
  int bidy = blockIdx.y;

  Tensor mA = make_tensor(make_gmem_ptr((ComputeTypeA *)Aptr), make_shape(M, K), make_stride(K, Int<1>{})); // (M, K)
  Tensor mB = make_tensor(make_gmem_ptr((ComputeTypeB *)Bptr), make_shape(N, K), make_stride(K, Int<1>{})); // (N, K)
  Tensor mC = make_tensor(make_gmem_ptr((ComputeTypeC *)Cptr), make_shape(M, N), make_stride(N, Int<1>{})); // (M, N)
  Tensor mO = make_tensor(make_gmem_ptr((OutType *)Outptr), make_shape(M, N), make_stride(N, Int<1>{}));    // (M, N)

  auto tiler = make_tile(Int<kTileM>{}, Int<kTileN>{}, Int<kTileK>{});
  auto coord = make_coord(bidy, bidx, _);

  Tensor gA = local_tile(mA, tiler, coord, Step<_1, X, _1>{}); // (BLK_M, BLK_K, K_TILES)
  Tensor gB = local_tile(mB, tiler, coord, Step<X, _1, _1>{}); // (BLK_N, BLK_K, K_TILES)
  Tensor gC = local_tile(mC, tiler, coord, Step<_1, _1, X>{}); // (BLK_M, BLK_N)
  Tensor gO = local_tile(mO, tiler, coord, Step<_1, _1, X>{}); // (BLK_M, BLK_N)

  auto m_max_coord = M - size<0>(gA) * bidy;      // M - BLK_M * m_coord
  auto n_max_coord = N - size<0>(gB) * bidx;      // N - BLK_N * n_coord
  auto k_residue = K - size<1>(gA) * size<2>(gA); // K - BLK_K * k_coord_max

  gA = domain_offset(make_coord(0, k_residue, 0), gA);
  gB = domain_offset(make_coord(0, k_residue, 0), gB);

  Tensor sA = make_tensor(make_smem_ptr((ComputeTypeA *)Aptr_smem), SmemLayoutA{}); // (BLK_M, BLK_K, G2S_PIPE)
  Tensor sB = make_tensor(make_smem_ptr((ComputeTypeB *)Bptr_smem), SmemLayoutB{}); // (BLK_N, BLK_K, G2S_PIPE)
  Tensor sC = make_tensor(make_smem_ptr((ComputeTypeC *)Cptr_smem), SmemLayoutC{}); // (BLK_M, BLK_N)
  Tensor sO = make_tensor(make_smem_ptr((OutType *)Optr_smem), SmemLayoutO{});      // (BLK_M, BLK_N)

  typename Spec::TiledMMA tiled_mma;
  ThrMMA thr_mma = tiled_mma.get_slice(tid);

  Tensor tCrA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
  Tensor tCrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K)
  Tensor tCrC = thr_mma.partition_fragment_C(gC);          // (MMA, MMA_M, MMA_N)

  typename Spec::TiledCopyA_G2S g2s_tiled_copy_a;
  ThrCopy g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(tid);
  Tensor tAgA_g2s = g2s_thr_copy_a.partition_S(gA); // (ACPY, ACPY_M, ACPY_K, K_TILES)
  Tensor tAsA_g2s = g2s_thr_copy_a.partition_D(sA); // (ACPY, ACPY_M, ACPY_K, G2S_PIPE)

  typename Spec::TiledCopyB_G2S g2s_tiled_copy_b;
  ThrCopy g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(tid);
  Tensor tBgB_g2s = g2s_thr_copy_b.partition_S(gB); // (BCPY, BCPY_N, BCPY_K, K_TILES)
  Tensor tBsB_g2s = g2s_thr_copy_b.partition_D(sB); // (BCPY, BCPY_N, BCPY_K, G2S_PIPE)

  typename Spec::TiledCopyC_G2S g2s_tiled_copy_c;
  ThrCopy g2s_thr_copy_c = g2s_tiled_copy_c.get_slice(tid);
  Tensor tCgC_g2s = g2s_thr_copy_c.partition_S(gC); // (CCPY, CCPY_M, CCPY_N)
  Tensor tCsC_g2s = g2s_thr_copy_c.partition_D(sC); // (CCPY, CCPY_M, CCPY_N)

  //
  // PREDICATES
  //

  Tensor tApA_g2s =
      make_tensor<bool>(make_shape(size<1>(tAsA_g2s), size<2>(tAsA_g2s)), Stride<_1, _0>{}); // (ACPY_M, ACPY_K)
  Tensor tBpB_g2s =
      make_tensor<bool>(make_shape(size<1>(tBsB_g2s), size<2>(tBsB_g2s)), Stride<_1, _0>{}); // (BCPY_N, BCPY_K)
  Tensor tCpC_g2s = make_tensor<bool>(make_shape(size<1>(tCsC_g2s), size<2>(tCsC_g2s)),
                                      Stride<_1, Int<size<1>(tCsC_g2s)>>{}); // (CCPY_M, CCPY_N)

  Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA))); // (BLK_M,BLK_K) -> (blk_m,blk_k)
  Tensor cB = make_identity_tensor(make_shape(size<0>(sB), size<1>(sB))); // (BLK_N,BLK_K) -> (blk_n,blk_k)
  Tensor cC = make_identity_tensor(make_shape(size<0>(sC), size<1>(sC))); // (BLK_M,BLK_N) -> (blk_m,blk_n)

  Tensor tAcA_g2s = g2s_thr_copy_a.partition_S(cA); // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
  Tensor tBcB_g2s = g2s_thr_copy_b.partition_S(cB); // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
  Tensor tCcC_g2s = g2s_thr_copy_c.partition_S(cC); // (CCPY,CCPY_M,CCPY_N) -> (blk_m,blk_n)

#pragma unroll
  for (int m = 0; m < size<0>(tApA_g2s); ++m) {
    tApA_g2s(m, 0) = get<0>(tAcA_g2s(0, m, 0)) < m_max_coord;
  }
#pragma unroll
  for (int n = 0; n < size<0>(tBpB_g2s); ++n) {
    tBpB_g2s(n, 0) = get<0>(tBcB_g2s(0, n, 0)) < n_max_coord;
  }
#pragma unroll
  for (int m = 0; m < size<0>(tCpC_g2s); ++m) {
#pragma unroll
    for (int n = 0; n < size<1>(tCpC_g2s); ++n) {
      tCpC_g2s(m, n) = elem_less(tCcC_g2s(0, m, n), make_coord(m_max_coord, n_max_coord));
    }
  }

  //
  // END PREDICATES
  //

  typename Spec::TiledCopyA_S2R s2r_tiled_copy_a;
  ThrCopy s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(tid);
  Tensor tAsA_s2r = s2r_thr_copy_a.partition_S(sA); // (CPY, CPY_M, CPY_K, PIPE)
  Tensor tArA_s2r = s2r_thr_copy_a.retile_D(tCrA);  // (CPY, CPY_M, CPY_K)

  typename Spec::TiledCopyB_S2R s2r_tiled_copy_b;
  ThrCopy s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(tid);
  Tensor tBsB_s2r = s2r_thr_copy_b.partition_S(sB); // (CPY, CPY_M, CPY_K, PIPE)
  Tensor tBrB_s2r = s2r_thr_copy_b.retile_D(tCrB);  // (CPY, CPY_M, CPY_K)

  typename Spec::TiledCopyC_S2R s2r_tiled_copy_c;
  ThrCopy s2r_thr_copy_c = s2r_tiled_copy_c.get_slice(tid);
  Tensor tCsC_s2r = s2r_thr_copy_c.partition_S(sC); // (CPY, CPY_M, CPY_K)
  Tensor tCrC_s2r = s2r_thr_copy_c.retile_D(tCrC);  // (CPY, CPY_M, CPY_K)

  //
  // Prefetch
  //

  if constexpr (!IsGemm) {
    clear(tCsC_g2s);
    copy_if(g2s_tiled_copy_c, tCpC_g2s, tCgC_g2s, tCsC_g2s);
  }

  int NTilesK = ceil_div(K, kTileK);

  clear(tAsA_g2s);
  clear(tBsB_g2s);

#pragma unroll
  for (int k = 0; k < size<2>(tAsA_g2s); ++k) {
    if (get<1>(tAcA_g2s(0, 0, k)) >= -k_residue) { // blk_k coord < residue_k (gA shifted)
      copy_if(g2s_tiled_copy_a, tApA_g2s(_, k), tAgA_g2s(_, _, k, 0), tAsA_g2s(_, _, k, 0));
    }
  }

#pragma unroll
  for (int k = 0; k < size<2>(tBsB_g2s); ++k) {
    if (get<1>(tBcB_g2s(0, 0, k)) >= -k_residue) { // blk_k coord < residue_k (gB shifted)
      copy_if(g2s_tiled_copy_b, tBpB_g2s(_, k), tBgB_g2s(_, _, k, 0), tBsB_g2s(_, _, k, 0));
    }
  }

  cp_async_fence();

#pragma unroll
  for (int ik = 1; ik < G2S_Stages - 1; ++ik) {
    // Set all predicates to false if we are going to overshoot bounds
    if (ik == NTilesK) {
      clear(tApA_g2s);
      clear(tBpB_g2s);
    }

    copy_if(g2s_tiled_copy_a, tApA_g2s, tAgA_g2s(_, _, _, ik), tAsA_g2s(_, _, _, ik));
    copy_if(g2s_tiled_copy_b, tBpB_g2s, tBgB_g2s(_, _, _, ik), tBsB_g2s(_, _, _, ik));

    cp_async_fence();
  }

  // Prefetch register stage

  constexpr int k_tiles = size<2>(tArA_s2r);
  constexpr int prefetch_s2r_tiles = 1;

  cp_async_wait<G2S_Stages - 2>();
  __syncthreads();

#pragma unroll
  for (int k = 0; k < prefetch_s2r_tiles; ++k) {
    copy(s2r_tiled_copy_a, tAsA_s2r(_, _, k, 0), tArA_s2r(_, _, k));
    copy(s2r_tiled_copy_b, tBsB_s2r(_, _, k, 0), tBrB_s2r(_, _, k));
  }

  if constexpr (IsGemm) {
    clear(tCrC); // Set the accumulators to zero
  } else {
    copy(s2r_tiled_copy_c, tCsC_s2r, tCrC_s2r);
  }

  //
  // MAINLOOP
  //

  int g2s_gmem_pipe = G2S_Stages - 1;
  int g2s_smem_pipe = G2S_Stages - 1;
  int s2r_smem_pipe = 0;

  for (int ik = 0; ik < NTilesK; ++ik) {
    // Note, the for_each() function is required here to ensure `k` is of type Int<N>.
    for_each(make_int_sequence<k_tiles>{}, [&](auto k) {
      if (k == k_tiles - prefetch_s2r_tiles) {
        cp_async_wait<G2S_Stages - 2>();
        __syncthreads();
        ++s2r_smem_pipe;
        s2r_smem_pipe = (s2r_smem_pipe == G2S_Stages) ? 0 : s2r_smem_pipe;
      }

      // copy A and B
      auto k_next = (k + Int<prefetch_s2r_tiles>{}) % k_tiles;
      copy(s2r_tiled_copy_a, tAsA_s2r(_, _, k_next, s2r_smem_pipe), tArA_s2r(_, _, k_next));
      copy(s2r_tiled_copy_b, tBsB_s2r(_, _, k_next, s2r_smem_pipe), tBrB_s2r(_, _, k_next));

      if (k == 0) {
        // Set all predicates to false if we are going to overshoot bounds
        if (g2s_gmem_pipe == NTilesK) {
          clear(tApA_g2s);
          clear(tBpB_g2s);
        }

        copy_if(g2s_tiled_copy_a, tApA_g2s, tAgA_g2s(_, _, _, g2s_gmem_pipe), tAsA_g2s(_, _, _, g2s_smem_pipe));
        copy_if(g2s_tiled_copy_b, tBpB_g2s, tBgB_g2s(_, _, _, g2s_gmem_pipe), tBsB_g2s(_, _, _, g2s_smem_pipe));

        cp_async_fence();
        ++g2s_gmem_pipe;
        ++g2s_smem_pipe;
        g2s_smem_pipe = (g2s_smem_pipe == G2S_Stages) ? 0 : g2s_smem_pipe;
      }

#pragma unroll
      for (int ik = k * kMmaValExpandK; ik < (k + 1) * kMmaValExpandK; ++ik) {
        gemm(tiled_mma, tCrC, tCrA(_, _, ik), tCrB(_, _, ik), tCrC);
      }
    });
  }

  cp_async_wait<0>();
  __syncthreads();

  if constexpr (!IsCvtPrecision) {
    typename Spec::TiledCopyC_R2S r2s_tiled_copy_c;
    ThrCopy r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(tid);
    Tensor tCrC_r2s = r2s_thr_copy_c.retile_S(tCrC);  // (CPY, CPY_M, CPY_N)
    Tensor tCsC_r2s = r2s_thr_copy_c.partition_D(sC); // (CPY, CPY_M, CPY_N)
    copy(r2s_tiled_copy_c, tCrC_r2s, tCsC_r2s);

    __syncthreads();

    typename Spec::TiledCopyC_S2G s2g_tiled_copy_c;
    ThrCopy s2g_thr_copy_c = s2g_tiled_copy_c.get_slice(tid);
    Tensor tCsC_s2g = s2g_thr_copy_c.partition_S(sC); // (CPY, CPY_M, CPY_N)
    Tensor tCgC_s2g = s2g_thr_copy_c.partition_D(gC); // (CPY, CPY_M, CPY_N)

    //
    // PREDICATES
    //

    Tensor tCpC_s2g = make_tensor<bool>(make_shape(size<1>(tCgC_s2g), size<2>(tCgC_s2g)),
                                        Stride<_1, Int<size<1>(tCgC_s2g)>>{}); // (CCPY_M, CCPY_N)
    Tensor tCcC_s2g = s2g_thr_copy_c.partition_S(cC);                          // (CCPY,CCPY_M,CCPY_N) -> (blk_m,blk_n)

#pragma unroll
    for (int m = 0; m < size<0>(tCpC_s2g); ++m) {
#pragma unroll
      for (int n = 0; n < size<1>(tCpC_s2g); ++n) {
        tCpC_s2g(m, n) = elem_less(tCcC_s2g(0, m, n), make_coord(m_max_coord, n_max_coord));
      }
    }

    //
    // END PREDICATES
    //

    copy_if(s2g_tiled_copy_c, tCpC_s2g, tCsC_s2g, tCgC_s2g);

  } else {

    auto t = make_tensor_like<OutType>(tCrC);
    copy(tCrC, t); // Convert precision

    typename Spec::TiledCopyO_R2S r2s_tiled_copy_o;
    ThrCopy r2s_thr_copy_o = r2s_tiled_copy_o.get_slice(tid);
    Tensor tOrC_r2s = r2s_thr_copy_o.retile_S(t);     // (CPY, CPY_M, CPY_N)
    Tensor tOsO_r2s = r2s_thr_copy_o.partition_D(sO); // (CPY, CPY_M, CPY_N)
    copy(r2s_tiled_copy_o, tOrC_r2s, tOsO_r2s);

    __syncthreads();

    typename Spec::TiledCopyO_S2G s2g_tiled_copy_o;
    ThrCopy s2g_thr_copy_o = s2g_tiled_copy_o.get_slice(tid);
    Tensor tOsO_s2g = s2g_thr_copy_o.partition_S(sO); // (CPY, CPY_M, CPY_N)
    Tensor tOgO_s2g = s2g_thr_copy_o.partition_D(gO); // (CPY, CPY_M, CPY_N)

    //
    // PREDICATES
    //

    Tensor tOpO_s2g = make_tensor<bool>(make_shape(size<1>(tOgO_s2g), size<2>(tOgO_s2g)),
                                        Stride<_1, Int<size<1>(tOgO_s2g)>>{}); // (OCPY_M, OCPY_N)
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tOcO_s2g = s2g_thr_copy_o.partition_S(cO);                          // (OCPY,OCPY_M,OCPY_N) -> (blk_m,blk_n)

#pragma unroll
    for (int m = 0; m < size<0>(tOpO_s2g); ++m) {
#pragma unroll
      for (int n = 0; n < size<1>(tOpO_s2g); ++n) {
        tOpO_s2g(m, n) = elem_less(tOcO_s2g(0, m, n), make_coord(m_max_coord, n_max_coord));
      }
    }

    //
    // END PREDICATES
    //

    copy_if(s2g_tiled_copy_o, tOpO_s2g, tOsO_s2g, tOgO_s2g);
  }
}

namespace spec {

using namespace cute;

template <typename OutType_,
          typename ComputeTypeA_,
          typename ComputeTypeB_,
          typename ComputeTypeC_,
          int kTileM_,
          int kTileN_,
          int kTileK_,
          int G2S_Stages_ = 3>
struct KernelSpec {
  using OutType = OutType_;
  using ComputeTypeA = ComputeTypeA_;
  using ComputeTypeB = ComputeTypeB_;
  using ComputeTypeC = ComputeTypeC_;

  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;

  static constexpr int G2S_Stages = G2S_Stages_;
  static_assert(G2S_Stages >= 2, "G2S_Stages should not be less than 2.");

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

  using Copy_G2S_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;

  using Copy_S2R_op_A = std::conditional_t<sizeof(ComputeTypeA) == 2, SM75_U32x4_LDSM_N, AutoVectorizingCopy>;
  using Copy_S2R_op_B = std::conditional_t<sizeof(ComputeTypeB) == 2, SM75_U32x4_LDSM_N, AutoVectorizingCopy>;
  using Copy_S2R_op_C = std::conditional_t<sizeof(ComputeTypeC) == 2, SM75_U32x4_LDSM_N, AutoVectorizingCopy>;

  using CopyA_G2S_atom = Copy_Atom<Copy_G2S_op, ComputeTypeA>;
  using CopyB_G2S_atom = Copy_Atom<Copy_G2S_op, ComputeTypeB>;
  using CopyC_G2S_atom = Copy_Atom<Copy_G2S_op, ComputeTypeC>;

  using CopyA_S2R_atom = Copy_Atom<Copy_S2R_op_A, ComputeTypeA>;
  using CopyB_S2R_atom = Copy_Atom<Copy_S2R_op_B, ComputeTypeB>;
  using CopyC_S2R_atom = Copy_Atom<Copy_S2R_op_C, ComputeTypeC>;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  using Copy_R2S_op = SM90_U32x4_STSM_N;
#else
  using Copy_R2S_op = AutoVectorizingCopy;
#endif

  using Copy_S2G_op = UniversalCopy<cute::uint128_t>;

  using CopyC_R2S_atom = Copy_Atom<Copy_R2S_op, ComputeTypeC>;
  using CopyO_R2S_atom = Copy_Atom<Copy_R2S_op, OutType>;

  using CopyC_S2G_atom = Copy_Atom<Copy_S2G_op, ComputeTypeC>;
  using CopyO_S2G_atom = Copy_Atom<Copy_S2G_op, OutType>;

  static constexpr int kThreadNum = size(TiledMMA{});
  static constexpr int kThreadsPerWarp = 32;
  static constexpr int kTileM_Copy = cute::min(kThreadsPerWarp, kTileM);
  static constexpr int kTileN_Copy = cute::min(kThreadsPerWarp, kTileN);

  // Here we omit cases that `kAlignedCopyItems < 1`
  static constexpr int kAlignedCopyItemsA =
      cute::min(128 / 8 / sizeof(ComputeTypeA), kTileK *kTileM_Copy / kThreadNum);
  static constexpr int kAlignedCopyItemsB =
      cute::min(128 / 8 / sizeof(ComputeTypeB), kTileK *kTileN_Copy / kThreadNum);
  static constexpr int kAlignedCopyItemsC =
      cute::min(128 / 8 / sizeof(ComputeTypeC), kTileN *kTileM_Copy / kThreadNum);
  static constexpr int kAlignedCopyItemsO = cute::min(128 / 8 / sizeof(OutType), kTileN *kTileM_Copy / kThreadNum);

  using TiledCopyA_G2S =
      decltype(make_tiled_copy(CopyA_G2S_atom{},
                               make_layout(make_shape(Int<kTileM_Copy>{}, Int<kThreadNum / kTileM_Copy>{}),
                                           make_stride(Int<kThreadNum / kTileM_Copy>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<kAlignedCopyItemsA>{}))));
  using TiledCopyB_G2S =
      decltype(make_tiled_copy(CopyB_G2S_atom{},
                               make_layout(make_shape(Int<kTileN_Copy>{}, Int<kThreadNum / kTileN_Copy>{}),
                                           make_stride(Int<kThreadNum / kTileN_Copy>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<kAlignedCopyItemsB>{}))));
  using TiledCopyC_G2S =
      decltype(make_tiled_copy(CopyC_G2S_atom{},
                               make_layout(make_shape(Int<kTileM_Copy>{}, Int<kThreadNum / kTileM_Copy>{}),
                                           make_stride(Int<kThreadNum / kTileM_Copy>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<kAlignedCopyItemsC>{}))));

  using TiledCopyA_S2R = decltype(make_tiled_copy_A(CopyA_S2R_atom{}, TiledMMA{}));
  using TiledCopyB_S2R = decltype(make_tiled_copy_B(CopyB_S2R_atom{}, TiledMMA{}));
  using TiledCopyC_S2R = decltype(make_tiled_copy_C(CopyC_S2R_atom{}, TiledMMA{}));

  using TiledCopyC_R2S = decltype(make_tiled_copy_C(CopyC_R2S_atom{}, TiledMMA{}));
  using TiledCopyO_R2S = decltype(make_tiled_copy_C(CopyO_R2S_atom{}, TiledMMA{}));

  using TiledCopyC_S2G =
      decltype(make_tiled_copy(CopyC_S2G_atom{},
                               make_layout(make_shape(Int<kTileM_Copy>{}, Int<kThreadNum / kTileM_Copy>{}),
                                           make_stride(Int<kThreadNum / kTileM_Copy>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<kAlignedCopyItemsC>{}))));
  using TiledCopyO_S2G =
      decltype(make_tiled_copy(CopyO_S2G_atom{},
                               make_layout(make_shape(Int<kTileM_Copy>{}, Int<kThreadNum / kTileM_Copy>{}),
                                           make_stride(Int<kThreadNum / kTileM_Copy>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<kAlignedCopyItemsO>{}))));

  using SmemLayoutAtomAB = decltype(composition(Swizzle<3, 3, 3>{},
                                                make_layout(make_shape(Int<8>{}, Int<cute::min(64, kTileK)>{}),
                                                            make_stride(Int<cute::min(64, kTileK)>{}, Int<1>{}))));
  using SmemLayoutAtomC = decltype(composition(Swizzle<3, 3, 3>{},
                                               make_layout(make_shape(Int<8>{}, Int<cute::min(64, kTileN)>{}),
                                                           make_stride(Int<cute::min(64, kTileN)>{}, Int<1>{}))));

  using SmemLayoutA =
      decltype(tile_to_shape(SmemLayoutAtomAB{}, make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<G2S_Stages>{})));
  using SmemLayoutB =
      decltype(tile_to_shape(SmemLayoutAtomAB{}, make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<G2S_Stages>{})));
  using SmemLayoutC = decltype(tile_to_shape(SmemLayoutAtomC{}, make_shape(Int<kTileM>{}, Int<kTileN>{})));
  using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomC{}, make_shape(Int<kTileM>{}, Int<kTileN>{})));

  static constexpr int kShmSizeA = cosize_v<SmemLayoutA> * sizeof(ComputeTypeA);
  static constexpr int kShmSizeB = cosize_v<SmemLayoutB> * sizeof(ComputeTypeB);
  static constexpr int kShmSizeC = cosize_v<SmemLayoutC> * sizeof(ComputeTypeC);
  static constexpr int kShmSizeO = cosize_v<SmemLayoutO> * sizeof(OutType);
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

template <typename ComputeTypeC, typename OutType> constexpr bool needs_precision_conversion() {
  return !std::is_same_v<ComputeTypeC, OutType>;
}

template <int kTileM,
          int kTileN,
          int kTileK,
          int G2S_Stages,
          typename OutType,
          typename ComputeTypeA,
          typename ComputeTypeB,
          typename ComputeTypeC = OutType>
torch::Tensor run_pipelining(const torch::Tensor a, const torch::Tensor b, std::optional<torch::Tensor> _c) {

  at::cuda::CUDAGuard device_guard{a.device()};
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  auto M = a.size(0);
  auto N = b.size(0);
  auto K = a.size(1);

  auto torch_compute_type_a = to_torch_scalar_type<ComputeTypeA>();
  auto torch_compute_type_b = to_torch_scalar_type<ComputeTypeB>();
  auto torch_compute_type_c = to_torch_scalar_type<ComputeTypeC>();

  torch::Tensor c, out;
  bool is_gemm;

  if (!_c.has_value()) {
    auto options = torch::TensorOptions().dtype(torch_compute_type_c).device(torch::kCUDA);
    c = torch::empty({M, N}, options);
    is_gemm = true;
  } else {
    c = _c.value();
    is_gemm = false;
  }

  CHECK_TORCH_TENSOR_DTYPE(a, torch_compute_type_a)
  CHECK_TORCH_TENSOR_DTYPE(b, torch_compute_type_b)
  CHECK_TORCH_TENSOR_DTYPE(c, torch_compute_type_c)

  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, N, K)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

  constexpr bool IsCvtPrecision = needs_precision_conversion<ComputeTypeC, OutType>();

  if constexpr (IsCvtPrecision) {
    auto torch_compute_type_out = to_torch_scalar_type<OutType>();
    auto options = torch::TensorOptions().dtype(torch_compute_type_out).device(torch::kCUDA);
    out = torch::empty({M, N}, options);

    CHECK_TORCH_TENSOR_DTYPE(out, torch_compute_type_out)
    CHECK_TORCH_TENSOR_SHAPE(out, M, N)
  }

  using Spec = spec::KernelSpec<OutType, ComputeTypeA, ComputeTypeB, ComputeTypeC, kTileM, kTileN, kTileK, G2S_Stages>;

  dim3 block = Spec::kThreadNum;
  dim3 grid(cute::ceil_div(N, Spec::kTileN), cute::ceil_div(M, Spec::kTileM));

  constexpr int kShmSizeA = Spec::kShmSizeA;
  constexpr int kShmSizeB = Spec::kShmSizeB;
  constexpr int kShmSizeC = Spec::kShmSizeC;
  constexpr int kShmSizeO = Spec::kShmSizeO;

  int shm_size = (is_gemm) ? ((IsCvtPrecision) ? (cute::max(kShmSizeA + kShmSizeB, kShmSizeO))
                                               : (cute::max(kShmSizeA + kShmSizeB, kShmSizeC)))
                           : ((IsCvtPrecision) ? (cute::max(kShmSizeA + kShmSizeB + kShmSizeC, kShmSizeO))
                                               : (kShmSizeA + kShmSizeB + kShmSizeC));

  printf("Block Size: (%d, %d, %d) | Grid Size: (%d, %d, %d) | Shared Memory Size: %d Bytes\n", block.x, block.y,
         block.z, grid.x, grid.y, grid.z, shm_size);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaDeviceSynchronize();

  auto get_data_ptr = [](const torch::Tensor &tensor) -> void * {
    return tensor.defined() ? tensor.data_ptr() : nullptr;
  };
  void *out_ptr = get_data_ptr(out);

  // Kernel launch
  BOOL_SWITCH(is_gemm, IsGemm, [&] {
    cudaEventRecord(start, stream);
    if (shm_size >= 48 * 1024) {
      cudaFuncSetAttribute(pipelining<Spec, IsGemm, IsCvtPrecision>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           shm_size);
    }
    pipelining<Spec, IsGemm, IsCvtPrecision>
        <<<grid, block, shm_size, stream>>>(c.data_ptr(), a.data_ptr(), b.data_ptr(), M, N, K, out_ptr);
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

  if constexpr (IsCvtPrecision)
    return out;
  else
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pipelining_fp16_fp16_fp16_fp16", &(run_pipelining<128, 128, 64, 3, cute::half_t, cute::half_t, cute::half_t>),
        "Run a mixed-precision half 16x8x8 MMA operation.");
  m.def("pipelining_fp16_fp16_fp16_fp32",
        &(run_pipelining<128, 128, 64, 3, cute::half_t, cute::half_t, cute::half_t, float>),
        "Run a mixed-precision half 16x8x8 MMA operation.");
  m.def("pipelining_bf16_bf16_bf16_fp32",
        &(run_pipelining<128, 128, 64, 3, cute::bfloat16_t, cute::bfloat16_t, cute::bfloat16_t, float>),
        "Run a mixed-precision half 16x8x8 MMA operation.");
}
