#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/types.h>

#define CUTLASS_CHECK(status)                                                                                         \
  {                                                                                                                   \
    cutlass::Status error = status;                                                                                   \
    if (error != cutlass::Status::kSuccess) {                                                                         \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ << std::endl;        \
      exit(EXIT_FAILURE);                                                                                             \
    }                                                                                                                 \
  }

namespace spec {

using namespace cute;

template <typename OutType_,
          typename ComputeTypeA_,
          typename ComputeTypeB_,
          typename ComputeTypeC_,
          typename AccType_,
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

  static constexpr int kThreadNum = size(TiledMMA{});
  static constexpr int kThreadsPerWarp = 32;
  static constexpr int kTileM_Copy = cute::min(kThreadsPerWarp, kTileM);
  static constexpr int kTileN_Copy = cute::min(kThreadsPerWarp, kTileN);

  static constexpr int kAlignedCopyItemsA =
      cute::min(128 / 8 / sizeof(ComputeTypeA), kTileK *kTileM_Copy / kThreadNum);
  static constexpr int kAlignedCopyItemsB =
      cute::min(128 / 8 / sizeof(ComputeTypeB), kTileK *kTileN_Copy / kThreadNum);

  using Copy_G2S_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using CopyA_G2S_atom = Copy_Atom<Copy_G2S_op, ComputeTypeA>;
  using CopyB_G2S_atom = Copy_Atom<Copy_G2S_op, ComputeTypeB>;

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

  using Copy_S2R_op_A = std::conditional_t<sizeof(ComputeTypeA) == 2, SM75_U32x4_LDSM_N, AutoVectorizingCopy>;
  using Copy_S2R_op_B = std::conditional_t<sizeof(ComputeTypeB) == 2, SM75_U32x4_LDSM_N, AutoVectorizingCopy>;

  using CopyA_S2R_atom = Copy_Atom<Copy_S2R_op_A, ComputeTypeA>;
  using CopyB_S2R_atom = Copy_Atom<Copy_S2R_op_B, ComputeTypeB>;

  using SmemLayoutAtomA = decltype(composition(Swizzle<3, 3, 3>{},
                                               make_layout(make_shape(Int<8>{}, Int<cute::min(64, kTileK)>{}),
                                                           make_stride(Int<cute::min(64, kTileK)>{}, Int<1>{}))));
  using SmemLayoutAtomB = SmemLayoutAtomA;

  //////////////////////////////////////////////////////////////////////////////////

  // A matrix configuration
  using ElementA = ComputeTypeA_;
  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 16 / sizeof(ElementA);
  // static constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;

  // B matrix configuration
  using ElementB = ComputeTypeB_;
  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 16 / sizeof(ElementB);

  // C matrix configuration
  using ElementC = ComputeTypeC_;
  using LayoutC = cutlass::layout::RowMajor;
  static constexpr int AlignmentC = 16 / sizeof(ElementC);

  // D matrix configuration
  using ElementD = OutType_;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 16 / sizeof(ElementD);

  // Core kernel configurations
  using ElementAccumulator = AccType_;
  using ElementCompute = AccType_;
  using ArchTag = cutlass::arch::Sm80;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using TileShape = Shape<Int<kTileM_>, Int<kTileN_>, Int<kTileK_>>;

  using DispatchPolicy = cutlass::gemm::MainloopSm80CpAsync<G2S_Stages>;

  // Mainloop
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<DispatchPolicy,
                                                                      TileShape,
                                                                      ElementA,
                                                                      cutlass::gemm::TagToStrideA_t<LayoutA>,
                                                                      ElementB,
                                                                      cutlass::gemm::TagToStrideB_t<LayoutB>,
                                                                      TiledMMA,
                                                                      TiledCopyA_G2S,
                                                                      SmemLayoutAtomA,
                                                                      CopyA_S2R_atom,
                                                                      cute::identity, // A
                                                                      TiledCopyB_G2S,
                                                                      SmemLayoutAtomB,
                                                                      CopyB_S2R_atom,
                                                                      cute::identity // B
                                                                      >;

  // Epilogue
  using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
      ElementC,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutD>,
      cutlass::epilogue::thread::LinearCombination<ElementD,
                                                   1,
                                                   ElementAccumulator,
                                                   ElementCompute,
                                                   cutlass::epilogue::thread::ScaleType::Default,
                                                   cutlass::FloatRoundStyle::round_to_nearest,
                                                   ElementC>,
      cutlass::gemm::EpilogueDefault>;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int>, CollectiveMainloop, CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  static void run(void *Aptr, void *Bptr, void *Cptr, void *Dptr, int M, int N, int K, cudaStream_t stream = nullptr) {
    // Instantiate CUTLASS kernel depending on templates
    Gemm gemm;

    // Make strides
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
    cutlass::KernelHardwareInfo kernel_hw_info;

    // Change device_id to another value if you are running on a machine with multiple GPUs and wish
    // to use a GPU other than that with device ID 0.
    kernel_hw_info.device_id = 0;
    kernel_hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(kernel_hw_info.device_id);

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {(ElementA *)Aptr, stride_A, (ElementB *)Bptr, stride_B},
        {{(ElementAccumulator)1.f, (ElementAccumulator)1.f}, (ElementC *)Cptr, stride_C, (ElementD *)Dptr, stride_D},
        kernel_hw_info};

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Check if the problem size is supported or not
    CUTLASS_CHECK(gemm.can_implement(arguments));

    // Initialize CUTLASS kernel with arguments and workspace pointer
    CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

    // Correctness / Warmup iteration
    CUTLASS_CHECK(gemm.run(stream));
  }
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
          typename ComputeTypeC = OutType,
          typename AccType = ComputeTypeC>
torch::Tensor run_gemm_api(const torch::Tensor a, const torch::Tensor b, std::optional<torch::Tensor> _c) {

  at::cuda::CUDAGuard device_guard{a.get_device()};
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
    c = torch::zeros({M, N}, options);
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
    out = torch::zeros({M, N}, options);

    CHECK_TORCH_TENSOR_DTYPE(out, torch_compute_type_out)
    CHECK_TORCH_TENSOR_SHAPE(out, M, N)
  }

  using Spec =
      spec::KernelSpec<OutType, ComputeTypeA, ComputeTypeB, ComputeTypeC, AccType, kTileM, kTileN, kTileK, G2S_Stages>;

  void *out_ptr = IsCvtPrecision ? out.data_ptr() : c.data_ptr();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaDeviceSynchronize();

  cudaEventRecord(start, stream);
  Spec::run(a.data_ptr(), b.data_ptr(), c.data_ptr(), out_ptr, M, N, K);
  cudaEventRecord(stop, stream);

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
  m.def("gemm_api_fp16_fp16_fp16_fp16", &(run_gemm_api<128, 128, 64, 5, cute::half_t, cute::half_t, cute::half_t>),
        "Run a mixed-precision half 16x8x8 MMA operation.");
  m.def("gemm_api_fp16_fp16_fp16_fp32",
        &(run_gemm_api<128, 128, 64, 5, cute::half_t, cute::half_t, cute::half_t, float>),
        "Run a mixed-precision half 16x8x8 MMA operation.");
  m.def("gemm_api_bf16_bf16_bf16_fp32",
        &(run_gemm_api<128, 128, 64, 5, cute::bfloat16_t, cute::bfloat16_t, cute::bfloat16_t, float>),
        "Run a mixed-precision half 16x8x8 MMA operation.");
}
