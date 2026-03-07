#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

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
          int kTileK_>
struct KernelSpec {
  // A matrix configuration
  using ElementA = ComputeTypeA_;
  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 16 / sizeof(ElementA);
  // static constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;

  // B matrix configuration
  using ElementB = ComputeTypeB_;
  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 16 / sizeof(ElementB);

  // C/D matrix configuration
  using ElementC = ComputeTypeC_;
  using LayoutC = cutlass::layout::RowMajor;
  static constexpr int AlignmentC = 16 / sizeof(ElementC);

  using ElementD = OutType_;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 16 / sizeof(ElementD);

  // Core kernel configurations
  using ElementAccumulator = AccType_;
  using ElementCompute = AccType_;
  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using TileShape = Shape<Int<kTileM_>, Int<kTileN_>, Int<kTileK_>>;
  using ClusterShape = Shape<_2, _1, _1>;
  using StageCount = cutlass::gemm::collective::StageCountAuto;
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueTile = cutlass::epilogue::collective::EpilogueTileAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using FusionOperation =
      cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC, ElementCompute>;
  using TileScheduler = cutlass::gemm::PersistentScheduler;

  // template <
  //   class ArchTag, class OpClass,
  //   class TileShape_MNK, class ClusterShape_MNK,
  //   class EpilogueTileType,
  //   class ElementAccumulator, class ElementCompute,
  //   class ElementC, class GmemLayoutTagC, int AlignmentC,
  //   class ElementD, class GmemLayoutTagD, int AlignmentD,
  //   class EpilogueScheduleType,
  //   class FusionOpOrCallbacks =
  //   cutlass::epilogue::fusion::LinearCombination<ElementD,ElementCompute,ElementC,ElementCompute>, class Enable =
  //   void
  // >
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<ArchTag,
                                                                                       OperatorClass,
                                                                                       TileShape,
                                                                                       ClusterShape,
                                                                                       EpilogueTile,
                                                                                       ElementAccumulator,
                                                                                       ElementCompute,
                                                                                       ElementC,
                                                                                       LayoutC,
                                                                                       AlignmentC,
                                                                                       ElementD,
                                                                                       LayoutD,
                                                                                       AlignmentD,
                                                                                       EpilogueSchedule,
                                                                                       FusionOperation>::CollectiveOp;

  // template <
  //   class ArchTag, class OpClass,
  //   class ElementA, class GmemLayoutA, int AlignmentA,
  //   class ElementB, class GmemLayoutB, int AlignmentB,
  //   class ElementAccumulator,
  //   class TileShape_MNK, class ClusterShape_MNK,
  //   class StageCountType,
  //   class KernelScheduleType,
  //   class Enable = void
  // >
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      LayoutA,
      AlignmentA,
      ElementB,
      LayoutB,
      AlignmentB,
      ElementAccumulator,
      TileShape,
      ClusterShape,
      conditional_t<cute::is_same_v<StageCount, cutlass::gemm::collective::StageCountAuto>,
                    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
                        sizeof(typename CollectiveEpilogue::SharedStorage))>,
                    StageCount>,
      KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int>, // Indicates ProblemShape
                                                          CollectiveMainloop,
                                                          CollectiveEpilogue,
                                                          TileScheduler>;

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

    // Change device_id to another value if you are running on a machine with multiple GPUs and wish
    // to use a GPU other than that with device ID 0.
    int device_id = 0;
    cutlass::KernelHardwareInfo kernel_hw_info =
        cutlass::KernelHardwareInfo::make_kernel_hardware_info<typename Gemm::GemmKernel>(device_id);
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

  if (!_c.has_value()) {
    auto options = torch::TensorOptions().dtype(torch_compute_type_c).device(torch::kCUDA);
    c = torch::zeros({M, N}, options);
  } else {
    c = _c.value();
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

  using Spec = spec::KernelSpec<OutType, ComputeTypeA, ComputeTypeB, ComputeTypeC, AccType, kTileM, kTileN, kTileK>;

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
  m.def("gemm_api_fp16_fp16_fp16_fp16", &(run_gemm_api<128, 128, 64, cute::half_t, cute::half_t, cute::half_t>),
        "Run a mixed-precision half 16x8x8 MMA operation.");
  m.def("gemm_api_fp16_fp16_fp16_fp32", &(run_gemm_api<128, 128, 64, cute::half_t, cute::half_t, cute::half_t, float>),
        "Run a mixed-precision half 16x8x8 MMA operation.");
  m.def("gemm_api_bf16_bf16_bf16_fp32",
        &(run_gemm_api<128, 128, 64, cute::bfloat16_t, cute::bfloat16_t, cute::bfloat16_t, float>),
        "Run a mixed-precision half 16x8x8 MMA operation.");
}
