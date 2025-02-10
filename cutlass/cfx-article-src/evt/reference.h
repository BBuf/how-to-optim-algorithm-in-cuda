#include <iostream>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/tensor_ref.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Element>
bool verify_tensor(thrust::host_vector<Element> vector_Input,
                   thrust::host_vector<Element> vector_Input_Ref,
                   bool printValues = false, int64_t verify_length = -1) {

  int64_t size = (vector_Input.size() < vector_Input_Ref.size())
                     ? vector_Input.size()
                     : vector_Input_Ref.size();
  size = (verify_length == -1) ? size : verify_length;

  // 0.05 for absolute error
  float abs_tol = 5e-2f;
  // 10% for relative error
  float rel_tol = 1e-1f;
  for (int64_t i = 0; i < size; ++i) {
    if (printValues)
      std::cout << vector_Input[i] << " " << vector_Input_Ref[i] << std::endl;
    float diff = (float)(vector_Input[i] - vector_Input_Ref[i]);
    float abs_diff = fabs(diff);
    float abs_ref = fabs((float)vector_Input_Ref[i] + 1e-5f);
    float relative_diff = abs_diff / abs_ref;
    if ((isnan(vector_Input_Ref[i]) || isnan(abs_diff) || isinf(abs_diff)) ||
        (abs_diff > abs_tol && relative_diff > rel_tol)) {
      printf("[%d/%d] diff = %f, rel_diff = %f, {computed=%f, ref=%f}.\n",
             int(i), int(size), abs_diff, relative_diff,
             (float)(vector_Input[i]), (float)(vector_Input_Ref[i]));
      return false;
    }
  }

  return true;
}

/// Command line options parsing
struct Options {

  bool help;
  bool error;
  bool printVal;

  int m, n, k, l;
  float alpha, beta;
  unsigned seed;

  Options()
      : help(false), error(false), m(2048), n(2048), k(2048), l(1), alpha(1.f),
        beta(0.f), seed(0u) {}

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m, 2048);
    cmd.get_cmd_line_argument("n", n, 2048);
    cmd.get_cmd_line_argument("k", k, 2048);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("seed", seed, 0u);
    cmd.get_cmd_line_argument("print", printVal, false);
  }

  /// Prints the usage statement.
  std::ostream &print_usage(std::ostream &out) const {

    out << "evt_gemm_cute \n\n"
        << "  This example showcases the use of CUTLASS's Custom Epilogue Visitor Trees (EVT)\n"
        << "  for developing fused gemm+epilogue kernel targeting NVIDIA's Hopper architecture.\n\n"
        << "Options:\n\n"
        << "  --help                      If specified, displays this usage "
           "statement\n\n"
        << "  --m=<int>                   Sets the M extent of the GEMM\n"
        << "  --n=<int>                   Sets the N extent of the GEMM\n"
        << "  --k=<int>                   Sets the K extent of the GEMM\n"
        << "  --seed=<uint>               Set random seed (0: randomize from clock)\n"
        << "  --alpha=<f32>               Epilogue scalar alpha\n"
        << "  --beta=<f32>                Epilogue scalar beta\n\n"
        << "  --print                     Print the output and ref values\n\n";

    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

// Wrapper to construct, run, and verify a GEMM. This example showcases
// CUTLASS's collective operation builders by specializing the GEMM only on the
// kernel schedule it will use and the number of pipeline stages.
//
// One can use a special `Auto` type that tells the CollectiveBuilder
// to select an appropriate value on its own. The CollectiveBuilder will attempt
// to select configurations that will result in the most-performant kernel, but
// this is not a guarantee.
//
// If relying on 'Auto' schedules, all builders must use the 'Auto' schedule to
// ensure compatiblity. For example, if `KernelScheduleAuto` is used for the
// mainloop builder, `EpilogueScheduleAuto` must be used for the epilogue
// builder.
//
// Furthermore, if an override schedule is selected, both epilogue and mainloop
// schedules must be specifically opt into a compatible selection.
//
// Behavior of the CollectiveBuilder with `Auto` types is subject to change in
// future releases
// -- do not rely on `Auto` if you require a specific scheduling policy.
template <class CustomEVT,
          class ElementA,
          class ElementB,
          class ElementC,
          class ElementCompute,
          // Type of kernel schedule to generate
          class MainloopScheduleType =
              cutlass::gemm::collective::KernelScheduleAuto,
          // Type of epilogue schedule to generate
          class EpilogueScheduleType =
              cutlass::epilogue::collective::EpilogueScheduleAuto,
          // Number of pipeline stages to use
          class StageCountType = cutlass::gemm::collective::StageCountAuto,
          // Type of tile scheduler to use
          class TileSchedulerType = cutlass::gemm::PersistentScheduler,
          // Do we use custom epilogue visitor tree (EVT) fusion
          bool UseCustomEVT = false>
struct ExampleRunner {

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::ColumnMajor;

  using ElementD = ElementC;
  using ElementAccumulator = ElementCompute;
  using ElementScalar = ElementCompute;

  // 16B alignment lets us use TMA
  static constexpr int AlignmentA = 16 / sizeof(ElementA);
  static constexpr int AlignmentB = 16 / sizeof(ElementB);
  static constexpr int AlignmentC = 16 / sizeof(ElementC);
  static constexpr int AlignmentD = 16 / sizeof(ElementD);

  static_assert(
      not UseCustomEVT ||
          (cute::is_same_v<EpilogueScheduleType,
                           cutlass::epilogue::TmaWarpSpecialized> ||
           cute::is_same_v<EpilogueScheduleType,
                           cutlass::epilogue::TmaWarpSpecializedCooperative>),
      "Epilogue visitor trees are currently only supported by the TMA "
      "warp-specialized epilogue");
  static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;

  // A predefined set of fusion operations (implemented with EVT) are supported
  // by the TMA warp-specialized epilogue. Users can select one of these
  // operations by passing one of the tags defined in
  // include/cutlass/epilogue/fusion/operations.hpp to the CollectiveBuilder.
  // This frees the user from having to compute additional parameters such as
  // stage counts and copy atoms/layouts. These tags also provide additional
  // metadata that can be queried at compile time.
  using DefaultOperation = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementScalar, RoundStyle>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
          Shape<_128, _128, _64>, Shape<_1, _1, _1>,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementCompute, ElementC, LayoutC, AlignmentC, ElementD, LayoutD,
          AlignmentD, EpilogueScheduleType,
          cute::conditional_t<UseCustomEVT, CustomEVT,
                              DefaultOperation>>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, ElementA,
          LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
          ElementAccumulator, Shape<_128, _128, _64>, Shape<_2, _1, _1>,
          cute::conditional_t<
              cute::is_same_v<StageCountType,
                              cutlass::gemm::collective::StageCountAuto>,
              cutlass::gemm::collective::StageCountAutoCarveout<static_cast<
                  int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
              StageCountType>,
          MainloopScheduleType>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      TileSchedulerType>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using LayoutTagA = cutlass::gemm::detail::StrideToLayoutTagA_t<StrideA>;
  using LayoutTagB = cutlass::gemm::detail::StrideToLayoutTagB_t<StrideB>;
  using LayoutTagC = cutlass::gemm::detail::StrideToLayoutTagC_t<StrideC>;
  using LayoutTagD = cutlass::gemm::detail::StrideToLayoutTagC_t<StrideD>;

  //
  // Data members
  //

  /// Initialization
  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;
  uint64_t seed = 0;

  //
  // Methods
  //

  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize(const ProblemShapeType &problem_size) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    stride_A =
        cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    stride_B =
        cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    stride_C =
        cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    stride_D =
        cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));
  }

  template <class Epi_Args, class TA, class TB, class TC>
  bool run(const Options &options, const cutlass::KernelHardwareInfo &hw_info,
           TA *A, TB *B, TC *C, TC *D, const Epi_Args &epi_args) {
    ProblemShapeType problem_size =
        ProblemShapeType{options.m, options.n, options.k, options.l};

    initialize(problem_size);

    typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                                       problem_size,
                                       {A, stride_A, B, stride_B},
                                       {{}, // epilogue.thread
                                        C,
                                        stride_C,
                                        D,
                                        stride_D},
                                       hw_info};

    // Custom EVT fusions will have nested unnamed args, the structure of which
    // can be deduced from the type definition of the EVT.
    // Each node's arguments has the recursive structure of
    // {first_child_args, ..., last_child_args, op_args},
    // For more complex examples of EVT initialization please refer to
    // include/cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp
    if constexpr (UseCustomEVT) {
      arguments.epilogue.thread = epi_args;
    }
    // Pre-defined fusions will have flat, named args for user-friendlyness
    else {
      arguments.epilogue.thread.alpha = options.alpha;
      arguments.epilogue.thread.beta = options.beta;
    }

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "This kernel is not supported. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return false;
    }

    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
      std::cerr
          << "Failed to initialize the CUTLASS kernel. Last CUDA error is: "
          << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return false;
    }

    // Run the GEMM
    status = gemm_op.run();
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to launch the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return false;
    }

    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Error running the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(result) << std::endl;
      return false;
    }

    return true;
  }
};

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to print a description of the example run and its result
void print_result(const std::string &description, bool passed) {
  std::cout << description << ": " << (passed ? "Passed" : "Failed")
            << std::endl;
}
