#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <torch/extension.h>
#include <torch/types.h>

template <typename Spec, bool IsGemm, bool IsCvtPrecision>
__global__ void tiled_copy(void *Cptr, const void *Aptr, const void *Bptr, int m, int n, int k, void *Outptr) {
  using namespace cute;

  using X = Underscore;
  using OutType = typename Spec::OutType;
  using ComputeTypeA = typename Spec::ComputeTypeA;
  using ComputeTypeB = typename Spec::ComputeTypeB;
  using ComputeTypeC = typename Spec::ComputeTypeC;
  using TiledMMA = typename Spec::TiledMMA;
  using TiledCopyA = typename Spec::TiledCopyA;
  using TiledCopyB = typename Spec::TiledCopyB;
  using TiledCopyC = typename Spec::TiledCopyC;
  using TiledCopyO = typename Spec::TiledCopyO;

  constexpr int kTileM = Spec::kTileM;
  constexpr int kTileN = Spec::kTileN;
  constexpr int kTileK = Spec::kTileK;

  int tid = threadIdx.x;

  Tensor mA = make_tensor(make_gmem_ptr((ComputeTypeA *)Aptr), make_shape(m, k), make_stride(k, Int<1>{})); // (M, K)
  Tensor mB = make_tensor(make_gmem_ptr((ComputeTypeB *)Bptr), make_shape(n, k), make_stride(k, Int<1>{})); // (N, K)
  Tensor mC = make_tensor(make_gmem_ptr((ComputeTypeC *)Cptr), make_shape(m, n), make_stride(n, Int<1>{})); // (M, N)
  Tensor mO = make_tensor(make_gmem_ptr((OutType *)Outptr), make_shape(m, n), make_stride(n, Int<1>{}));    // (M, N)

  auto tiler = make_tile(Int<kTileM>{}, Int<kTileN>{}, Int<kTileK>{});
  auto coord = make_coord(0, 0, 0);

  // Define the block global tensors (static)
  Tensor gA = local_tile(mA, tiler, coord, Step<_1, X, _1>{}); // (kTileM, kTileK)
  Tensor gB = local_tile(mB, tiler, coord, Step<X, _1, _1>{}); // (kTileN, kTileK)
  Tensor gC = local_tile(mC, tiler, coord, Step<_1, _1, X>{}); // (kTileM, kTileN)
  Tensor gO = local_tile(mO, tiler, coord, Step<_1, _1, X>{}); // (kTileM, kTileN)

  TiledMMA tiled_mma;
  ThrMMA thr_mma = tiled_mma.get_slice(tid);

  Tensor tCgA = thr_mma.partition_A(gA); // (MMA, MMA_M, MMA_K)
  Tensor tCgB = thr_mma.partition_B(gB); // (MMA, MMA_N, MMA_K)
  Tensor tCgC = thr_mma.partition_C(gC); // (MMA, MMA_M, MMA_N)

  Tensor tCrA = thr_mma.partition_fragment_A(gA); // (MMA, MMA_M, MMA_K)
  Tensor tCrB = thr_mma.partition_fragment_B(gB); // (MMA, MMA_N, MMA_K)
  Tensor tCrC = thr_mma.partition_fragment_C(gC); // (MMA, MMA_M, MMA_N)

  TiledCopyA g2r_tiled_copy_a;
  ThrCopy g2r_thr_copy_a = g2r_tiled_copy_a.get_slice(tid);
  Tensor tAgA = g2r_thr_copy_a.retile_S(tCgA); // (CPY, CPY_M, CPY_K)
  // Equivalent to:
  // Tensor tAgA = g2r_thr_copy_a.partition_S(gA);    // (CPY, CPY_M, CPY_K)
  Tensor tArA = g2r_thr_copy_a.retile_D(tCrA); // (CPY, CPY_M, CPY_K)

  TiledCopyB g2r_tiled_copy_b;
  ThrCopy g2r_thr_copy_b = g2r_tiled_copy_b.get_slice(tid);
  Tensor tBgB = g2r_thr_copy_b.retile_S(tCgB); // (CPY, CPY_N, CPY_K)
  // Equivalent to:
  // Tensor tBgB = g2r_thr_copy_b.partition_S(gB);  // (CPY, CPY_N, CPY_K)
  Tensor tBrB = g2r_thr_copy_b.retile_D(tCrB); // (CPY, CPY_N, CPY_K)

  // if (thread0()) {
  //   print(tCgA); printf("\n");
  //   print(tCgB); printf("\n");
  //   print(tCgC); printf("\n");
  //   print(tCrA); printf("\n");
  //   print(tCrB); printf("\n");
  //   print(tCrC); printf("\n");
  //   print(tAgA); printf("\n");
  //   print(tArA); printf("\n");
  //   print(tBgB); printf("\n");
  //   print(tBrB); printf("\n");
  // }

  copy(g2r_tiled_copy_a, tAgA, tArA);
  copy(g2r_tiled_copy_b, tBgB, tBrB);

  if constexpr (IsGemm) {
    clear(tCrC); // Set the accumulators to zero
  } else {
    TiledCopyC g2r_tiled_copy_c;
    ThrCopy g2r_thr_copy_c = g2r_tiled_copy_c.get_slice(tid);
    Tensor tCgC_g2r = g2r_thr_copy_c.retile_S(tCgC); // (CPY, CPY_M, CPY_N)
    Tensor tCrC_g2r = g2r_thr_copy_c.retile_D(tCrC); // (CPY, CPY_M, CPY_N)
    copy(g2r_tiled_copy_c, tCgC_g2r, tCrC_g2r);
  }

  gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);

  TiledCopyO r2g_tiled_copy_o;
  if constexpr (!IsCvtPrecision) {
    ThrCopy r2g_thr_copy_o = r2g_tiled_copy_o.get_slice(tid);
    Tensor tCrC_r2g = r2g_thr_copy_o.retile_S(tCrC); // (CPY, CPY_M, CPY_N)
    Tensor tCgC_r2g = r2g_thr_copy_o.retile_D(tCgC); // (CPY, CPY_M, CPY_N)
    copy(r2g_tiled_copy_o, tCrC_r2g, tCgC_r2g);
  } else {
    Tensor tCgO = thr_mma.partition_C(gO); // (MMA, MMA_M, MMA_N)
    auto t = make_tensor_like<OutType>(tCrC);
    copy(tCrC, t); // Convert precision

    ThrCopy r2g_thr_copy_o = r2g_tiled_copy_o.get_slice(tid);
    Tensor tCrC_r2g = r2g_thr_copy_o.retile_S(t);    // (CPY, CPY_M, CPY_N)
    Tensor tCgO_r2g = r2g_thr_copy_o.retile_D(tCgO); // (CPY, CPY_M, CPY_N)
    copy(r2g_tiled_copy_o, tCrC_r2g, tCgO_r2g);
  }
}

namespace spec {

using namespace cute;

template <typename OutType_,
          typename ComputeTypeA_,
          typename ComputeTypeB_,
          typename ComputeTypeC_,
          int kTileM_ = 32,
          int kTileN_ = 32,
          int kTileK_ = 16>
struct KernelSpec {
  using OutType = OutType_;
  using ComputeTypeA = ComputeTypeA_;
  using ComputeTypeB = ComputeTypeB_;
  using ComputeTypeC = ComputeTypeC_;

  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;

  using MMA_op = SM80_16x8x8_F32BF16BF16F32_TN;
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

  using Copy_op = AutoVectorizingCopy;

  using CopyA_atom = Copy_Atom<Copy_op, ComputeTypeA>;
  using CopyB_atom = Copy_Atom<Copy_op, ComputeTypeB>;
  using CopyC_atom = Copy_Atom<Copy_op, ComputeTypeC>;
  using CopyO_atom = Copy_Atom<Copy_op, OutType>;

  using TiledCopyA = decltype(make_tiled_copy_A(CopyA_atom{}, TiledMMA{}));
  using TiledCopyB = decltype(make_tiled_copy_B(CopyB_atom{}, TiledMMA{}));
  using TiledCopyC = decltype(make_tiled_copy_C(CopyC_atom{}, TiledMMA{}));
  using TiledCopyO = decltype(make_tiled_copy_C(CopyO_atom{}, TiledMMA{}));

  static constexpr int kThreadNum = size(TiledMMA{});
  static constexpr int kShmSize = 0;
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
torch::Tensor run_tiled_copy(const torch::Tensor a, const torch::Tensor b, std::optional<torch::Tensor> _c) {

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

  // cute::print(typename Spec::TiledMMA{});
  // cute::print(typename Spec::TiledCopyA{});
  // cute::print_latex(typename Spec::TiledMMA{});
  // cute::print_latex(typename Spec::TiledCopyA{});
  // cute::print_latex(typename Spec::TiledCopyB{});
  // cute::print_latex(typename Spec::TiledCopyC{});
  // cute::print_latex(typename Spec::TiledCopyO{});

  dim3 block = Spec::kThreadNum;
  dim3 grid((N + Spec::kTileN - 1) / Spec::kTileN, (M + Spec::kTileM - 1) / Spec::kTileM);
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
    tiled_copy<Spec, IsGemm, IsCvtPrecision>
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
  m.def("tiled_copy_bf16_bf16_bf16_fp32",
        &(run_tiled_copy<32, 32, 16, cute::bfloat16_t, cute::bfloat16_t, cute::bfloat16_t, float>),
        "Run a mixed-precision bf16 32x32x16 MMA operation.");
}
