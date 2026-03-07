#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <torch/extension.h>
#include <torch/types.h>

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
#define CP_ASYNC_ENABLED
#endif

template <typename Spec, bool IsGemm, bool IsCvtPrecision>
__global__ void block_copy(void *__restrict__ Cptr,
                           const void *__restrict__ Aptr,
                           const void *__restrict__ Bptr,
                           int m,
                           int n,
                           int k,
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

  constexpr int kBlockM = Spec::kBlockM;
  constexpr int kBlockN = Spec::kBlockN;
  constexpr int kBlockK = Spec::kBlockK;
  constexpr int kShmSizeA = Spec::kShmSizeA;
  constexpr int kShmSizeB = Spec::kShmSizeB;

  extern __shared__ __align__(1024) uint8_t smem[];

  uint8_t *Aptr_smem = smem;
  uint8_t *Bptr_smem = smem + kShmSizeA;
  uint8_t *Cptr_smem = smem + kShmSizeA + kShmSizeB;
  uint8_t *Optr_smem = smem;

  int tid = threadIdx.x;

  Tensor mA = make_tensor(make_gmem_ptr((ComputeTypeA *)Aptr), make_shape(m, k), make_stride(k, Int<1>{})); // (M, K)
  Tensor mB = make_tensor(make_gmem_ptr((ComputeTypeB *)Bptr), make_shape(n, k), make_stride(k, Int<1>{})); // (N, K)
  Tensor mC = make_tensor(make_gmem_ptr((ComputeTypeC *)Cptr), make_shape(m, n), make_stride(n, Int<1>{})); // (M, N)
  Tensor mO = make_tensor(make_gmem_ptr((OutType *)Outptr), make_shape(m, n), make_stride(n, Int<1>{}));    // (M, N)

  auto tiler = make_tile(Int<kBlockM>{}, Int<kBlockN>{}, Int<kBlockK>{});
  auto coord = make_coord(0, 0, 0);

  Tensor gA = local_tile(mA, tiler, coord, Step<_1, X, _1>{}); // (kBlockM, kBlockK)
  Tensor gB = local_tile(mB, tiler, coord, Step<X, _1, _1>{}); // (kBlockN, kBlockK)
  Tensor gC = local_tile(mC, tiler, coord, Step<_1, _1, X>{}); // (kBlockM, kBlockN)
  Tensor gO = local_tile(mO, tiler, coord, Step<_1, _1, X>{}); // (kBlockM, kBlockN)

  Tensor sA = make_tensor(make_smem_ptr((ComputeTypeA *)Aptr_smem), SmemLayoutA{}); // (kBlockM, kBlockK)
  Tensor sB = make_tensor(make_smem_ptr((ComputeTypeB *)Bptr_smem), SmemLayoutB{}); // (kBlockN, kBlockK)
  Tensor sC = make_tensor(make_smem_ptr((ComputeTypeC *)Cptr_smem), SmemLayoutC{}); // (kBlockM, kBlockN)
  Tensor sO = make_tensor(make_smem_ptr((OutType *)Optr_smem), SmemLayoutO{});      // (kBlockM, kBlockN)

  typename Spec::TiledMMA tiled_mma;
  ThrMMA thr_mma = tiled_mma.get_slice(tid);

  Tensor tCgA = thr_mma.partition_A(gA); // (MMA, MMA_M, MMA_K)
  Tensor tCgB = thr_mma.partition_B(gB); // (MMA, MMA_N, MMA_K)
  Tensor tCgC = thr_mma.partition_C(gC); // (MMA, MMA_M, MMA_N)

  Tensor tCrA = thr_mma.partition_fragment_A(gA); // (MMA, MMA_M, MMA_K)
  Tensor tCrB = thr_mma.partition_fragment_B(gB); // (MMA, MMA_N, MMA_K)
  Tensor tCrC = thr_mma.partition_fragment_C(gC); // (MMA, MMA_M, MMA_N)

  typename Spec::TiledCopyA_G2S g2s_tiled_copy_a;
  ThrCopy g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(tid);
  Tensor tAgA_g2s = g2s_thr_copy_a.partition_S(gA); // (CPY, CPY_M, CPY_K)
  Tensor tAsA_g2s = g2s_thr_copy_a.partition_D(sA); // (CPY, CPY_M, CPY_K)

  typename Spec::TiledCopyB_G2S g2s_tiled_copy_b;
  ThrCopy g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(tid);
  Tensor tBgB_g2s = g2s_thr_copy_b.partition_S(gB); // (CPY, CPY_N, CPY_K)
  Tensor tBsB_g2s = g2s_thr_copy_b.partition_D(sB); // (CPY, CPY_N, CPY_K)

  typename Spec::TiledCopyC_G2S g2s_tiled_copy_c;
  ThrCopy g2s_thr_copy_c = g2s_tiled_copy_c.get_slice(tid);
  Tensor tCgC_g2s = g2s_thr_copy_c.partition_S(gC); // (CPY, CPY_M, CPY_N)
  Tensor tCsC_g2s = g2s_thr_copy_c.partition_D(sC); // (CPY, CPY_M, CPY_N)

  // ----- Copy all global matrix Tile A/B/C to SMEM -----

  copy(g2s_tiled_copy_a, tAgA_g2s, tAsA_g2s);
  copy(g2s_tiled_copy_b, tBgB_g2s, tBsB_g2s);
  if constexpr (!IsGemm) {
    copy(g2s_tiled_copy_c, tCgC_g2s, tCsC_g2s);
  }

#if defined(CP_ASYNC_ENABLED)
  cp_async_fence();
  cp_async_wait<0>();
#endif
  __syncthreads();

  // ----- Complete copy from GMEM to SMEM -----

  typename Spec::TiledCopyA_S2R s2r_tiled_copy_a;
  ThrCopy s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(tid);
  Tensor tAsA_s2r = s2r_thr_copy_a.partition_S(sA); // (CPY, CPY_M, CPY_K)
  Tensor tArA_s2r = s2r_thr_copy_a.retile_D(tCrA);  // (CPY, CPY_M, CPY_K)

  typename Spec::TiledCopyB_S2R s2r_tiled_copy_b;
  ThrCopy s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(tid);
  Tensor tBsB_s2r = s2r_thr_copy_b.partition_S(sB); // (CPY, CPY_M, CPY_K)
  Tensor tBrB_s2r = s2r_thr_copy_b.retile_D(tCrB);  // (CPY, CPY_M, CPY_K)

  typename Spec::TiledCopyC_S2R s2r_tiled_copy_c;
  ThrCopy s2r_thr_copy_c = s2r_tiled_copy_c.get_slice(tid);
  Tensor tCsC_s2r = s2r_thr_copy_c.partition_S(sC); // (CPY, CPY_M, CPY_K)
  Tensor tCrC_s2r = s2r_thr_copy_c.retile_D(tCrC);  // (CPY, CPY_M, CPY_K)

  if constexpr (IsGemm) {
    clear(tCrC); // Set the accumulators to zero
  } else {
    copy(s2r_tiled_copy_c, tCsC_s2r, tCrC_s2r);
  }

#if 1
  copy(s2r_tiled_copy_a, tAsA_s2r, tArA_s2r);
  copy(s2r_tiled_copy_b, tBsB_s2r, tBrB_s2r);

  gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);

#else
  constexpr int kMmaValExpandM = Spec::kMmaValExpandM;
  constexpr int kMmaValExpandN = Spec::kMmaValExpandN;
  constexpr int kMmaValExpandK = Spec::kMmaValExpandK;

  constexpr int kMmaTileM = Spec::kMmaTileM;
  constexpr int kMmaTileN = Spec::kMmaTileN;
  constexpr int kMmaTileK = Spec::kMmaTileK;

  constexpr int NTilesM = kBlockM / kMmaTileM;
  constexpr int NTilesN = kBlockN / kMmaTileN;
  constexpr int NTilesK = kBlockK / kMmaTileK;

#pragma unroll
  for (int m_tile = 0; m_tile < NTilesM; ++m_tile) {
#pragma unroll
    for (int n_tile = 0; n_tile < NTilesN; ++n_tile) {
#pragma unroll
      for (int k_tile = 0; k_tile < NTilesK; ++k_tile) {
        copy(s2r_tiled_copy_a, tAsA_s2r(_, m_tile, k_tile), tArA_s2r(_, m_tile, k_tile));
        copy(s2r_tiled_copy_b, tBsB_s2r(_, n_tile, k_tile), tBrB_s2r(_, n_tile, k_tile));
#pragma unroll
        for (int im = m_tile * kMmaValExpandM; im < (m_tile + 1) * kMmaValExpandM; ++im) {
#pragma unroll
          for (int in = n_tile * kMmaValExpandN; in < (n_tile + 1) * kMmaValExpandN; ++in) {
#pragma unroll
            for (int ik = k_tile * kMmaValExpandK; ik < (k_tile + 1) * kMmaValExpandK; ++ik) {
              gemm(tiled_mma, tCrC(_, im, in), tCrA(_, im, ik), tCrB(_, in, ik), tCrC(_, im, in));
            }
          }
        }
      }
    }
  }
#endif

  // Question: Why should we sync here?
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
    copy(s2g_tiled_copy_c, tCsC_s2g, tCgC_s2g);

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
    copy(s2g_tiled_copy_o, tOsO_s2g, tOgO_s2g);
  }
}

namespace spec {

using namespace cute;

template <typename OutType_,
          typename ComputeTypeA_,
          typename ComputeTypeB_,
          typename ComputeTypeC_,
          int kBlockM_ = 128,
          int kBlockN_ = 128,
          int kBlockK_ = 64>
struct KernelSpec {
  using OutType = OutType_;
  using ComputeTypeA = ComputeTypeA_;
  using ComputeTypeB = ComputeTypeB_;
  using ComputeTypeC = ComputeTypeC_;

  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockN = kBlockN_;
  static constexpr int kBlockK = kBlockK_;

  using MMA_op = SM80_16x8x16_F32BF16BF16F32_TN;
  using MMA_traits = MMA_Traits<MMA_op>;
  using MMA_atom = MMA_Atom<MMA_traits>;
  using MMA_shape = MMA_traits::Shape_MNK;

  static constexpr int kMmaThrExpandM = 2;
  static constexpr int kMmaThrExpandN = 4;
  static constexpr int kMmaThrExpandK = 1;

  static constexpr int kMmaValExpandM = 1;
  static constexpr int kMmaValExpandN = 1;
  static constexpr int kMmaValExpandK = 2;

  static constexpr int kMmaTileM = kMmaThrExpandM * kMmaValExpandM * get<0>(MMA_shape{});
  static constexpr int kMmaTileN = kMmaThrExpandN * kMmaValExpandN * get<1>(MMA_shape{});
  static constexpr int kMmaTileK = kMmaThrExpandK * kMmaValExpandK * get<2>(MMA_shape{});

  using MMAThrLayout =
      decltype(make_layout(make_shape(Int<kMmaThrExpandM>{}, Int<kMmaThrExpandN>{}, Int<kMmaThrExpandK>{})));
  using MMATileLayout = Tile<Int<kMmaTileM>, Int<kMmaTileN>, Int<kMmaTileK>>;

  using TiledMMA = decltype(make_tiled_mma(MMA_op{}, MMAThrLayout{}, MMATileLayout{}));

  // We use CACHEGLOBAL instead of CACHEALWAYS, since we won't be reading
  // from the same address by the same threadblock. This is slightly faster.
  using Copy_G2S_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using Copy_S2R_op = AutoVectorizingCopy;

  using CopyA_G2S_atom = Copy_Atom<Copy_G2S_op, ComputeTypeA>;
  using CopyB_G2S_atom = Copy_Atom<Copy_G2S_op, ComputeTypeB>;
  using CopyC_G2S_atom = Copy_Atom<Copy_G2S_op, ComputeTypeC>;

  using CopyA_S2R_atom = Copy_Atom<Copy_S2R_op, ComputeTypeA>;
  using CopyB_S2R_atom = Copy_Atom<Copy_S2R_op, ComputeTypeB>;
  using CopyC_S2R_atom = Copy_Atom<Copy_S2R_op, ComputeTypeC>;

  using Copy_R2S_op = AutoVectorizingCopy;
  using Copy_S2G_op = AutoVectorizingCopy;

  using CopyC_R2S_atom = Copy_Atom<Copy_R2S_op, ComputeTypeC>;
  using CopyO_R2S_atom = Copy_Atom<Copy_R2S_op, OutType>;

  using CopyC_S2G_atom = Copy_Atom<Copy_S2G_op, ComputeTypeC>;
  using CopyO_S2G_atom = Copy_Atom<Copy_S2G_op, OutType>;

  static constexpr int kThreadNum = size(TiledMMA{});
  static constexpr int kBlockK_Copy = cute::min(64, kBlockK) / 8;
  static constexpr int kBlockN_Copy = cute::min(64, kBlockN) / 8;

  using TiledCopyA_G2S =
      decltype(make_tiled_copy(CopyA_G2S_atom{},
                               make_layout(make_shape(Int<kThreadNum / kBlockK_Copy>{}, Int<kBlockK_Copy>{}),
                                           make_stride(Int<kBlockK_Copy>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));
  using TiledCopyB_G2S =
      decltype(make_tiled_copy(CopyB_G2S_atom{},
                               make_layout(make_shape(Int<kThreadNum / kBlockK_Copy>{}, Int<kBlockK_Copy>{}),
                                           make_stride(Int<kBlockK_Copy>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));
  using TiledCopyC_G2S =
      decltype(make_tiled_copy(CopyC_G2S_atom{},
                               make_layout(make_shape(Int<kThreadNum / kBlockN_Copy>{}, Int<kBlockN_Copy>{}),
                                           make_stride(Int<kBlockN_Copy>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));

  using TiledCopyA_S2R = decltype(make_tiled_copy_A(CopyA_S2R_atom{}, TiledMMA{}));
  using TiledCopyB_S2R = decltype(make_tiled_copy_B(CopyB_S2R_atom{}, TiledMMA{}));
  using TiledCopyC_S2R = decltype(make_tiled_copy_C(CopyC_S2R_atom{}, TiledMMA{}));

  using TiledCopyC_R2S = decltype(make_tiled_copy_C(CopyC_R2S_atom{}, TiledMMA{}));
  using TiledCopyO_R2S = decltype(make_tiled_copy_C(CopyO_R2S_atom{}, TiledMMA{}));

  using TiledCopyC_S2G =
      decltype(make_tiled_copy(CopyC_S2G_atom{},
                               make_layout(make_shape(Int<kThreadNum / kBlockN_Copy>{}, Int<kBlockN_Copy>{}),
                                           make_stride(Int<kBlockN_Copy>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));
  using TiledCopyO_S2G =
      decltype(make_tiled_copy(CopyO_S2G_atom{},
                               make_layout(make_shape(Int<kThreadNum / kBlockN_Copy>{}, Int<kBlockN_Copy>{}),
                                           make_stride(Int<kBlockN_Copy>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));

  using SmemLayoutA =
      decltype(make_layout(make_shape(Int<kBlockM>{}, Int<kBlockK>{}), make_stride(Int<kBlockK>{}, Int<1>{})));
  using SmemLayoutB =
      decltype(make_layout(make_shape(Int<kBlockN>{}, Int<kBlockK>{}), make_stride(Int<kBlockK>{}, Int<1>{})));
  using SmemLayoutC =
      decltype(make_layout(make_shape(Int<kBlockM>{}, Int<kBlockN>{}), make_stride(Int<kBlockN>{}, Int<1>{})));
  using SmemLayoutO =
      decltype(make_layout(make_shape(Int<kBlockM>{}, Int<kBlockN>{}), make_stride(Int<kBlockN>{}, Int<1>{})));

  static constexpr int kShmSizeA = cosize(SmemLayoutA{}) * sizeof(ComputeTypeA);
  static constexpr int kShmSizeB = cosize(SmemLayoutB{}) * sizeof(ComputeTypeB);
  static constexpr int kShmSizeC = cosize(SmemLayoutC{}) * sizeof(ComputeTypeC);
  static constexpr int kShmSizeO = cosize(SmemLayoutO{}) * sizeof(OutType);

  static constexpr int kShmSize = cute::max(kShmSizeA + kShmSizeB + kShmSizeC, kShmSizeO);
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

template <int M,
          int N,
          int K,
          typename OutType,
          typename ComputeTypeA,
          typename ComputeTypeB,
          typename ComputeTypeC = OutType>
torch::Tensor run_block_copy(const torch::Tensor a, const torch::Tensor b, std::optional<torch::Tensor> _c) {

  at::cuda::CUDAGuard device_guard{a.get_device()};
  auto stream = at::cuda::getCurrentCUDAStream().stream();

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

  using Spec = spec::KernelSpec<OutType, ComputeTypeA, ComputeTypeB, ComputeTypeC, M, N, K>;

  dim3 block = Spec::kThreadNum;
  dim3 grid((N + Spec::kBlockN - 1) / Spec::kBlockN, (M + Spec::kBlockM - 1) / Spec::kBlockM);
  int shm_size = Spec::kShmSize;

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
      cudaFuncSetAttribute(block_copy<Spec, IsGemm, IsCvtPrecision>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           shm_size);
    }
    block_copy<Spec, IsGemm, IsCvtPrecision>
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
  m.def("block_copy_bf16_bf16_bf16_fp32",
        &(run_block_copy<128, 128, 64, cute::bfloat16_t, cute::bfloat16_t, cute::bfloat16_t, float>),
        "Run a mixed-precision half 16x8x8 MMA operation.");
}
