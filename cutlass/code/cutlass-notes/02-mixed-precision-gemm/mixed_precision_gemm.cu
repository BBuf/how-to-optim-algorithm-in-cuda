#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <torch/extension.h>
#include <torch/types.h>

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890) && ((__CUDACC_VER_MAJOR__ >= 12) && (__CUDACC_VER_MINOR__ >= 4)))
#define CUTE_ARCH_MMA_SM89_ENABLED
#endif

#if defined(__CUDA_ARCH__)
#define CUTE_INVALID_CONTROL_PATH(x)                                                                                  \
  assert(0 && x);                                                                                                     \
  printf(x);                                                                                                          \
  __brkpt()
#else
#define CUTE_INVALID_CONTROL_PATH(x)                                                                                  \
  assert(0 && x);                                                                                                     \
  printf(x)
#endif

template <typename Spec, bool IsGemm, bool IsCvtPrecision>
__global__ void
mixed_precision_gemm(void *Cptr, const void *Aptr, const void *Bptr, int m, int n, int k, void *Outptr) {
  using namespace cute;

  using X = Underscore;
  using OutType = typename Spec::OutType;
  using ComputeTypeA = typename Spec::ComputeTypeA;
  using ComputeTypeB = typename Spec::ComputeTypeB;
  using ComputeTypeC = typename Spec::ComputeTypeC;
  using TiledMMA = typename Spec::TiledMMA;

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

  auto copy_atom = AutoVectorizingCopy{};

  copy(copy_atom, tCgA, tCrA);
  copy(copy_atom, tCgB, tCrB);

  if constexpr (IsGemm)
    clear(tCrC); // Set the accumulators to zero
  else
    copy(copy_atom, tCgC, tCrC);

  gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);

  if constexpr (!IsCvtPrecision) {
    copy(copy_atom, tCrC, tCgC);
  } else {
    auto tCrO = make_tensor_like<OutType>(tCrC);
    copy(tCrC, tCrO); // Convert precision

    Tensor tCgO = thr_mma.partition_C(gO); // (MMA, MMA_M, MMA_N)
    copy(copy_atom, tCrO, tCgO);

    // Equivalent to:
    // Tensor tCgO = thr_mma.partition_C(gO);  // (MMA, MMA_M, MMA_N)
    // copy(copy_atom, tCrC, tCgO);

    // Equivalent to:
    // Tensor tCgO = thr_mma.partition_C(gO);  // (MMA, MMA_M, MMA_K)
    // Tensor tCrO = thr_mma.partition_fragment_C(gO);  // (MMA, MMA_M, MMA_N)
    // copy(copy_atom, tCrC, tCrO);  // Convert precision
    // copy(copy_atom, tCrO, tCgO);
  }
}

namespace cute {

struct SM90_16x8x32_F32E4M3E5M2F32_TN {
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void fma(float &d0,
                                   float &d1,
                                   float &d2,
                                   float &d3,
                                   uint32_t const &a0,
                                   uint32_t const &a1,
                                   uint32_t const &a2,
                                   uint32_t const &a3,
                                   uint32_t const &b0,
                                   uint32_t const &b1,
                                   float const &c0,
                                   float const &c1,
                                   float const &c2,
                                   float const &c3) {
#if defined(CUTE_ARCH_MMA_SM89_ENABLED)
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32 "
                 "{%0,  %1,  %2,  %3},"
                 "{%4,  %5,  %6,  %7},"
                 "{%8,  %9},"
                 "{%10, %11, %12, %13};\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM90_16x8x32_F32E4M3E5M2F32_TN without CUTE_ARCH_MMA_SM89_ENABLED");
#endif
  }
};

template <> struct MMA_Traits<SM90_16x8x32_F32E4M3E5M2F32_TN> {
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _8, _32>;
  using ThrID = Layout<_32>;
  using ALayout = Layout<Shape<Shape<_4, _8>, Shape<_4, _2, _2>>, Stride<Stride<_64, _1>, Stride<_16, _8, _256>>>;
  using BLayout = Layout<Shape<Shape<_4, _8>, Shape<_4, _2>>, Stride<Stride<_32, _1>, Stride<_8, _128>>>;
  using CLayout = Layout<Shape<Shape<_4, _8>, Shape<_2, _2>>, Stride<Stride<_32, _1>, Stride<_16, _8>>>;
};

} // namespace cute

namespace spec {

using namespace cute;

template <typename OutType_,
          typename ComputeTypeA_,
          typename ComputeTypeB_,
          typename ComputeTypeC_,
          int kTileM_ = 16,
          int kTileN_ = 8,
          int kTileK_ = 8>
struct KernelSpec {
  using OutType = OutType_;
  using ComputeTypeA = ComputeTypeA_;
  using ComputeTypeB = ComputeTypeB_;
  using ComputeTypeC = ComputeTypeC_;

  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;

  using MMA_op = std::conditional_t<
      std::is_same_v<ComputeTypeA, bfloat16_t> && std::is_same_v<ComputeTypeB, bfloat16_t> &&
          std::is_same_v<ComputeTypeC, float>,
      SM80_16x8x8_F32BF16BF16F32_TN,
      std::conditional_t<std::is_same_v<ComputeTypeA, float_e4m3_t> && std::is_same_v<ComputeTypeB, float_e5m2_t> &&
                             std::is_same_v<ComputeTypeC, float>,
                         SM90_16x8x32_F32E4M3E5M2F32_TN,
                         void>>;

  static_assert(!std::is_same_v<MMA_op, void>, "Unsupported MMA op!");

  using TiledMMA = decltype(make_tiled_mma(MMA_op{}));

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
torch::Tensor run_mixed_precision_gemm(const torch::Tensor a, const torch::Tensor b, std::optional<torch::Tensor> _c) {

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

  dim3 block = Spec::kThreadNum;
  dim3 grid((N + Spec::kTileN - 1) / Spec::kTileN, (M + Spec::kTileM - 1) / Spec::kTileM);
  int shm_size = Spec::kShmSize;

  printf("Block Size: (%d, %d, %d) | Grid Size: (%d, %d, %d) | Shared Memory Size: %d Bytes\n", block.x, block.y,
         block.z, grid.x, grid.y, grid.z, shm_size);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  auto get_data_ptr = [](const torch::Tensor &tensor) -> void * {
    return tensor.defined() ? tensor.data_ptr() : nullptr;
  };
  void *out_ptr = get_data_ptr(out);

  cudaDeviceSynchronize();

  // Kernel launch
  BOOL_SWITCH(is_gemm, IsGemm, [&] {
    cudaEventRecord(start, stream);
    mixed_precision_gemm<Spec, IsGemm, IsCvtPrecision>
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
  m.def("mixed_precision_gemm_fp32_bf16_bf16_fp32",
        &(run_mixed_precision_gemm<16, 8, 8, float, cute::bfloat16_t, cute::bfloat16_t>),
        "Run a mixed-precision 16x8x8 MMA operation.");
  m.def("mixed_precision_gemm_bf16_bf16_bf16_fp32",
        &(run_mixed_precision_gemm<16, 8, 8, cute::bfloat16_t, cute::bfloat16_t, cute::bfloat16_t, float>),
        "Run a mixed-precision 16x8x8 MMA operation.");
  m.def("mixed_precision_gemm_fp32_e4m3_e5m2_fp32",
        &(run_mixed_precision_gemm<16, 8, 32, float, cute::float_e4m3_t, cute::float_e5m2_t>),
        "Run a mixed-precision fp8 16x8x32 MMA operation.");
  m.def("mixed_precision_gemm_bf16_e4m3_e5m2_fp32",
        &(run_mixed_precision_gemm<16, 8, 32, cute::bfloat16_t, cute::float_e4m3_t, cute::float_e5m2_t, float>),
        "Run a mixed-precision fp8 16x8x32 MMA operation.");
}
