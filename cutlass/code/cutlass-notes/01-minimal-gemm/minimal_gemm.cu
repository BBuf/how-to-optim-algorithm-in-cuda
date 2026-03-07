#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <torch/extension.h>
#include <torch/types.h>

template <typename Spec, bool IsGemm>
__global__ void minimal_gemm(void *Cptr, const void *Aptr, const void *Bptr, int m, int n, int k) {
  using namespace cute;

  using X = Underscore;
  using T = typename Spec::T;
  using TiledMMA = typename Spec::TiledMMA;

  constexpr int kTileM = Spec::kTileM;
  constexpr int kTileN = Spec::kTileN;
  constexpr int kTileK = Spec::kTileK;

  int tid = threadIdx.x;

  Tensor mA = make_tensor(make_gmem_ptr((T *)Aptr), make_shape(m, k), make_stride(k, Int<1>{})); // (M, K)
  Tensor mB = make_tensor(make_gmem_ptr((T *)Bptr), make_shape(n, k), make_stride(k, Int<1>{})); // (N, K)
  Tensor mC = make_tensor(make_gmem_ptr((T *)Cptr), make_shape(m, n), make_stride(n, Int<1>{})); // (M, N)

  auto tiler = make_tile(Int<kTileM>{}, Int<kTileN>{}, Int<kTileK>{});
  auto coord = make_coord(0, 0, 0);

  // Define the block global tensors (static)
  Tensor gA = local_tile(mA, tiler, coord, Step<_1, X, _1>{}); // (kTileM, kTileK)
  Tensor gB = local_tile(mB, tiler, coord, Step<X, _1, _1>{}); // (kTileN, kTileK)
  Tensor gC = local_tile(mC, tiler, coord, Step<_1, _1, X>{}); // (kTileM, kTileN)

  // Equivalent to:
  // Tensor gA = local_tile(mA, make_shape(Int<kTileM>{}, Int<kTileK>{}), make_coord(0, 0));  // (kTileM, kTileK)
  // Tensor gB = local_tile(mB, make_shape(Int<kTileN>{}, Int<kTileK>{}), make_coord(0, 0));  // (kTileN, kTileK)
  // Tensor gC = local_tile(mC, make_shape(Int<kTileM>{}, Int<kTileN>{}), make_coord(0, 0));  // (kTileM, kTileN)

  TiledMMA tiled_mma;
  ThrMMA thr_mma = tiled_mma.get_slice(tid);

  Tensor tCgA = thr_mma.partition_A(gA); // (MMA, MMA_M, MMA_K)
  Tensor tCgB = thr_mma.partition_B(gB); // (MMA, MMA_N, MMA_K)
  Tensor tCgC = thr_mma.partition_C(gC); // (MMA, MMA_M, MMA_N)

  Tensor tCrA = thr_mma.partition_fragment_A(gA); // (MMA, MMA_M, MMA_K)
  Tensor tCrB = thr_mma.partition_fragment_B(gB); // (MMA, MMA_N, MMA_K)
  Tensor tCrC = thr_mma.partition_fragment_C(gC); // (MMA, MMA_M, MMA_N)

  // Equivalent to:
  // Tensor tCrA = thr_mma.make_fragment_A(tCgA);  // (MMA, MMA_M, MMA_K)
  // Tensor tCrB = thr_mma.make_fragment_B(tCgB);  // (MMA, MMA_N, MMA_K)
  // Tensor tCrC = thr_mma.make_fragment_C(tCgC);  // (MMA, MMA_M, MMA_N)

  auto copy_atom = AutoVectorizingCopy{};

  copy(copy_atom, tCgA, tCrA);
  copy(copy_atom, tCgB, tCrB);

  if constexpr (IsGemm)
    clear(tCrC); // Set the accumulators to zero
  else
    copy(copy_atom, tCgC, tCrC);

  gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);

  copy(copy_atom, tCrC, tCgC);

  // if (thread0()) {
  //   print_latex(tiled_mma); printf("\n");
  //   print(tCgA); printf("\n");
  //   print(tCgB); printf("\n");
  //   print(tCgC); printf("\n");
  //   print(tCrA); printf("\n");
  //   print(tCrB); printf("\n");
  //   print(tCrC); printf("\n");
  // }
}

namespace spec {

using namespace cute;

template <typename T_, int kTileM_ = 16, int kTileN_ = 8, int kTileK_ = 8> struct KernelSpec {
  using T = T_;

  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;

  using MMA_op = SM80_16x8x8_F16F16F16F16_TN;
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

template <typename ComputeType, typename AccType = ComputeType>
torch::Tensor run_minimal_gemm(const torch::Tensor &a, const torch::Tensor &b, std::optional<torch::Tensor> &_c) {

  at::cuda::CUDAGuard device_guard{a.get_device()};
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  const int M = 16;
  const int N = 8;
  const int K = 8;

  auto torch_compute_type = [] {
    if constexpr (std::is_same_v<ComputeType, cute::half_t>) return torch::kHalf;
    throw std::runtime_error("Unsupported ComputeType!");
  }();

  auto torch_acc_type = [] {
    if constexpr (std::is_same_v<AccType, cute::half_t>) return torch::kHalf;
    throw std::runtime_error("Unsupported AccType!");
  }();

  torch::Tensor c;
  bool is_gemm;

  if (!_c.has_value()) {
    auto options = torch::TensorOptions().dtype(torch_acc_type).device(torch::kCUDA);
    c = torch::empty({M, N}, options);
    is_gemm = true;
  } else {
    c = _c.value();
    is_gemm = false;
  }

  CHECK_TORCH_TENSOR_DTYPE(a, torch_compute_type)
  CHECK_TORCH_TENSOR_DTYPE(b, torch_compute_type)
  CHECK_TORCH_TENSOR_DTYPE(c, torch_acc_type)

  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, N, K)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

  using Spec = spec::KernelSpec<ComputeType, M, N, K>;

  // cute::print(typename Spec::TiledMMA{});

  dim3 block = Spec::kThreadNum;
  dim3 grid((N + Spec::kTileN - 1) / Spec::kTileN, (M + Spec::kTileM - 1) / Spec::kTileM);
  int shm_size = Spec::kShmSize;

  printf("Block Size: (%d, %d, %d) | Grid Size: (%d, %d, %d) | Shared Memory Size: %d Bytes\n", block.x, block.y,
         block.z, grid.x, grid.y, grid.z, shm_size);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaDeviceSynchronize();

  // Kernel launch
  BOOL_SWITCH(is_gemm, IsGemm, [&] {
    cudaEventRecord(start, stream);
    minimal_gemm<Spec, IsGemm><<<grid, block, shm_size, stream>>>(
        reinterpret_cast<AccType *>(c.data_ptr()), reinterpret_cast<ComputeType *>(a.data_ptr()),
        reinterpret_cast<ComputeType *>(b.data_ptr()), M, N, K);
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

  return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("minimal_gemm", &(run_minimal_gemm<cute::half_t>), "Run a single 16x8x8 MMA operation.");
}
