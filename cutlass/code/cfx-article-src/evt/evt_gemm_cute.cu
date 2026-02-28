/***************************************************************************************************
 * Copyright (c) 2024 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>

#include <cute/tensor.hpp>

#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/epilogue/thread/linear_combination.h>

#include <cutlass/arch/mma_sm90.h>
#include <cutlass/device_kernel.h>
#include <cutlass/util/helper_cuda.hpp>
#include <cutlass/util/print_error.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <cutlass/util/cublas_wrappers.hpp>

#include "reference.h"
using namespace cute;

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto
convert_type(Tensor<Engine, Layout> const &tensor) {
  using From_type = typename Engine::value_type;
  constexpr int numel = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
  // HACK: this requires tensor to be "contiguous"
  auto frag =
      convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(
          tensor.data()));
  return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
  // Tensor out = make_tensor_like<To_type>(tensor);
  // cute::copy(make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout()),
  // out); return out;
}

// Functor classes allowing us to use log in EVT
template <typename T>
struct FastLog {
  CUTLASS_HOST_DEVICE
  T operator()(T const &rhs) const {
    return cutlass::fast_log(rhs);
  }
};
template <typename T, int N>
struct FastLog<cutlass::Array<T, N>> {
  CUTLASS_HOST_DEVICE
  cutlass::Array<T, N> operator()(cutlass::Array<T, N> const &rhs) const {

    FastLog<T> fast_op;
    cutlass::Array<T, N> y;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      y[i] = fast_op(rhs[i]);
    }

    return y;
  }
};

template <class ElementA, class ElementB,
          class SmemLayoutA, // (M,K,P)
          class SmemLayoutB> // (N,K,P)
struct SharedStorage {
  array_aligned<ElementA, cosize_v<SmemLayoutA>> smem_A;
  array_aligned<ElementB, cosize_v<SmemLayoutB>> smem_B;

  uint64_t tma_barrier[size<2>(SmemLayoutA{})];
  uint64_t mma_barrier[size<2>(SmemLayoutA{})];
};

template <class ProblemShape, class CtaTiler, class TA, class SmemLayoutA,
          class TmaA, class TB, class SmemLayoutB, class TmaB, class TC,
          class CStride, class TD, class DStride,
          class TiledMma, class Alpha, class Beta, class EVT>
__global__ static __launch_bounds__(
    decltype(size(TiledMma{}))::
        value) void gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
                                TA const *A,
                                CUTLASS_GRID_CONSTANT TmaA const tma_a,
                                TB const *B,
                                CUTLASS_GRID_CONSTANT TmaB const tma_b, TC *C,
                                CStride dC, TD *D, DStride dD,
                                TiledMma mma, Alpha alpha,
                                Beta beta,
                                CUTE_GRID_CONSTANT
                                typename EVT::Params const epi_params) {
  using namespace cute;

  EVT evt(epi_params, {});
  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{}); // (M, N, K)
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{}); // (BLK_M, BLK_N, BLK_K)

  static_assert(is_static<SmemLayoutA>::value);
  static_assert(is_static<SmemLayoutB>::value);

  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutA{}) == size<0>(cta_tiler)); // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutB{}) == size<1>(cta_tiler)); // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutA{}) == size<2>(cta_tiler)); // BLK_K
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutB{}) == size<2>(cta_tiler)); // BLK_K

  CUTE_STATIC_ASSERT_V(congruent(select<0, 1>(shape_MNK), dC)); // (M, N)
  CUTE_STATIC_ASSERT_V(congruent(select<0, 1>(shape_MNK), dD)); // (M, N)

  //
  // Full and Tiled Tensors
  //

  // Represent the full tensors
  auto [M, N, K] = shape_MNK;
  Tensor mA = tma_a.get_tma_tensor(make_shape(M, K)); // (M,K) TMA Tensor
  Tensor mB = tma_b.get_tma_tensor(make_shape(N, K)); // (N,K) TMA Tensor
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC); // (M,N)
  Tensor mD = make_tensor(make_gmem_ptr(D), make_shape(M, N), dD); // (M,N)

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _); // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord,
                         Step<_1, X, _1>{}); // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord,
                         Step<X, _1, _1>{}); // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord,
                         Step<_1, _1, X>{}); // (BLK_M,BLK_N)
  Tensor gD = local_tile(mD, cta_tiler, cta_coord,
                         Step<_1, _1, X>{}); // (BLK_M,BLK_N)

  // Shared memory tensors
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB>;
  SharedStorage &smem = *reinterpret_cast<SharedStorage *>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.smem_A.data()),
                          SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.smem_B.data()),
                          SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

  //
  // Partition the copying of A and B tiles
  //

  auto [tAgA, tAsA] =
      tma_partition(tma_a, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sA),
                    group_modes<0, 2>(gA)); // (TMA,k) and (TMA,PIPE)

  auto [tBgB, tBsB] =
      tma_partition(tma_b, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sB),
                    group_modes<0, 2>(gB)); // (TMA,k) and (TMA,PIPE)

  // The TMA is responsible for copying everything in mode-0 of tAsA and tBsB
  constexpr int kTmaTransactionBytes =
      CUTE_STATIC_V(size<0>(tAsA)) * sizeof(TA) +
      CUTE_STATIC_V(size<0>(tBsB)) * sizeof(TB);

  //
  // PREFETCH
  //

  auto K_PIPE_MAX = size<1>(tAsA);

  // Total count of tiles
  int k_tile_count = size<1>(tAgA);
  // Current tile index in gmem to read from
  int k_tile = 0;

  // Initialize Barriers
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();
  uint64_t *producer_mbar = smem.tma_barrier;
  uint64_t *consumer_mbar = smem.mma_barrier;

  using ProducerBarType = cutlass::arch::ClusterTransactionBarrier; // TMA
  using ConsumerBarType = cutlass::arch::ClusterBarrier;            // MMA
  CUTE_UNROLL
  for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
    if ((warp_idx == 0) && lane_predicate) {
      ProducerBarType::init(&producer_mbar[pipe], 1);
      ConsumerBarType::init(&consumer_mbar[pipe], 256 /*128*/);
    }
  }
  // Ensure barrier init is complete on all CTAs
  cluster_sync();

  // Start async loads for all pipes
  CUTE_UNROLL
  for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
    if ((warp_idx == 0) && lane_predicate) {
      // Set expected Tx Bytes after each reset / init
      ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe],
                                            kTmaTransactionBytes);
      copy(tma_a.with(producer_mbar[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
      copy(tma_b.with(producer_mbar[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
    }
    --k_tile_count;
    ++k_tile;
  }

  //
  // Define A/B partitioning and D accumulators
  //

  ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
  Tensor tDsA = thr_mma.partition_A(sA); // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tDsB = thr_mma.partition_B(sB); // (MMA,MMA_N,MMA_K,PIPE)
  Tensor tCgC = thr_mma.partition_C(gC); // (MMA,MMA_M,MMA_N)
  Tensor tDgD = thr_mma.partition_C(gD); // (MMA,MMA_M,MMA_N)

  // Allocate accumulators and clear them
  Tensor tDrAcc = partition_fragment_C(mma, select<0, 1>(cta_tiler));
  clear(tDrAcc);
  Tensor tDrAcc_ep = partition_fragment_C(mma, select<0, 1>(cta_tiler));
  clear(tDrAcc_ep);

  // Allocate "fragments"
  Tensor tDrA = thr_mma.make_fragment_A(tDsA); // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tDrB = thr_mma.make_fragment_B(tDsB); // (MMA,MMA_N,MMA_K,PIPE)

  //
  // PIPELINED MAIN LOOP
  //
  auto write_state = cutlass::PipelineState<K_PIPE_MAX>(); // TMA writes
  auto read_state = cutlass::PipelineState<K_PIPE_MAX>();  // MMA  reads

  CUTE_NO_UNROLL
  while (k_tile_count > -K_PIPE_MAX) {
    // Wait for Producer to complete
    int read_pipe = read_state.index();
    ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

    // MMAs to cover 1 K_TILE
    warpgroup_arrive();
    gemm(mma, tDrA(_, _, _, read_pipe), tDrB(_, _, _, read_pipe),
         tDrAcc); // (V,M) x (V,N) => (V,M,N)
    warpgroup_commit_batch();

    // Wait for all MMAs in a K_TILE to complete
    warpgroup_wait<0>();

    // Notify that consumption is done
    ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
    ++read_state;

    if ((warp_idx == 0) && lane_predicate) {
      int pipe = write_state.index();
      // Wait for Consumer to complete consumption
      ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());
      // Set expected Tx Bytes after each reset / init
      ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe],
                                            kTmaTransactionBytes);
      copy(tma_a.with(producer_mbar[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
      copy(tma_b.with(producer_mbar[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
      ++write_state;
    }
    --k_tile_count;
    ++k_tile;
  }

  //
  // Epilogue (unpredicated)
  //

  // Make sure all warpgroups have finished mma
  cutlass::arch::NamedBarrier::sync(size(mma), 0);

  using SmemCopyAtomD = Copy_Atom<cute::SM90_U32x4_STSM_N, TC>;

  auto smem_tiled_copy_D = make_tiled_copy_C(SmemCopyAtomD{}, mma);
  auto smem_thr_copy_D = smem_tiled_copy_D.get_thread_slice(threadIdx.x);

  // EVT: Set up callbacks
  constexpr bool RefSrc =
      true; // Register tensors reference R2S copy src layout
  auto shape_MNKL = append<4>(shape_MNK, _1{});
  auto tile_coord_MNKL = append<4>(cta_coord, _0{});

  // Fragments for C copy -- same size as accumulator fragments
  Tensor tCrC = partition_fragment_C(mma, select<0, 1>(cta_tiler));
  clear(tCrC);

  // OOB predication for tile quantization "residue"
  // Absolute coordinate tensors (dynamic)
  Tensor mD_crd = make_identity_tensor(make_shape(M, N)); // (M,N)
  Tensor cD_mn =
      local_tile(mD_crd, take<0, 2>(cta_tiler),
                 make_coord(blockIdx.x, blockIdx.y)); // (CTA_M,CTA_N)
  Tensor tRS_cD_mn = smem_thr_copy_D.partition_S(flat_divide(
      cD_mn, take<0, 2>(cta_tiler))); // (R2S,R2S_M,R2S_N,EPI_M,EPI_N)
  // Relative coordinate tensors (static)
  Tensor cD = make_counting_tensor(cD_mn.layout()); // (CTA_M,CTA_N)
  Tensor tRS_cD =
      make_counting_tensor(tRS_cD_mn.layout()); // (R2S,R2S_M,R2S_N,EPI_M,EPI_N)
  // Subtract the global "bottom right" corner from the local "top left" corner
  // to get the max relative coordinate
  auto residue_cD = make_coord(M, N) - cD_mn(_0{});         // (m,n)
  auto residue_tRS_cD = make_coord(M, N) - tRS_cD_mn(_0{}); // (m,n)

  auto cst_args = cutlass::epilogue::fusion::detail::ConsumerStoreArgs(
      shape_MNKL, cta_tiler, tile_coord_MNKL, mma,
      take<0, 2>(cta_tiler), // epilogue tiles = CTA tiles
      smem_tiled_copy_D, cD, residue_cD, tRS_cD, residue_tRS_cD, tCrC,
      threadIdx.x);
  auto cst_callbacks = evt.get_consumer_store_callbacks<RefSrc>(cst_args);

  cst_callbacks.begin();

  if (evt.is_C_load_needed()) {
    copy(tCgC, tCrC);
  }


  // Vectorized fragment view
  constexpr int FragmentSize = 32;
  Tensor tDrAcc_frg = recast<cutlass::Array<float, FragmentSize>>(tDrAcc);
  Tensor tDrAcc_ep_frg = recast<cutlass::Array<float, FragmentSize>>(tDrAcc_ep);
  for (int epi_v = 0; epi_v < size(tDrAcc_frg); ++epi_v) {
    tDrAcc_ep_frg(epi_v) = cst_callbacks.visit(tDrAcc_frg(epi_v), epi_v, 0, 0);
  }


  // include this convert utility
  Tensor tDrAcc_out = convert_type<TC>(tDrAcc_ep);

  copy(tDrAcc_out, tDgD);

  cst_callbacks.end();
}

// Setup params for a TN GEMM
template <class EVT, class TA, class TB, class TC, class TD, class Alpha, class Beta>
void gemm_tn(int m, int n, int k, Alpha alpha, TA const *A, int ldA,
             TB const *B, int ldB, Beta beta, TC *C, int ldC,
             TD *D, int ldD,
             typename EVT::Params &epi_params, cudaStream_t stream = 0) {
  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K); // (M, N, K)

  // Define TN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{}); // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{}); // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC); // (dM, dN)
  auto dD = make_stride(Int<1>{}, ldD); // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<64>{};
  auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<3>{};                      // Pipeline

  // Define the smem layouts (static)
  auto sA =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM, bK, bP));
  auto sB =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(bN, bK, bP));

  // Define the MMA
  TiledMMA tiled_mma = make_tiled_mma(
      SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{},
      Layout<Shape<_2, _1, _1>>{});
  // TiledMMA tiled_mma =
  // make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::K,GMMA::Major::K>{});

  // Define the TMAs
  // Create Global memory tensors for TMA inspection
  Tensor mA = make_tensor(A, make_shape(M, K), dA);
  Tensor mB = make_tensor(B, make_shape(N, K), dB);

  // Create TMA Atoms with the desired copy operation on the source and
  // destination
  Copy_Atom tmaA =
      make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_, _, 0), make_shape(bM, bK));
  Copy_Atom tmaB =
      make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_, _, 0), make_shape(bN, bK));

  //
  // Setup and Launch
  //

  // Launch parameter setup
  int smem_size =
      int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));
  dim3 dimBlock(size(tiled_mma));
  dim3 dimCluster(1, 1, 1);
  dim3 dimGrid(round_up(size(ceil_div(m, bM)), dimCluster.x),
               round_up(size(ceil_div(n, bN)), dimCluster.y));
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster,
                                         smem_size};

  void const *kernel_ptr = reinterpret_cast<void const *>(
      &gemm_device<decltype(prob_shape), decltype(cta_tiler), TA, decltype(sA),
                   decltype(tmaA), TB, decltype(sB), decltype(tmaB), TC,
                   decltype(dC), TD, decltype(dD),
                   decltype(tiled_mma), decltype(alpha),
                   decltype(beta), EVT>);

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  // Kernel Launch
  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      params, kernel_ptr, prob_shape, cta_tiler, A, tmaA, B, tmaB, C, dC,
      D, dD,
      tiled_mma, alpha, beta, epi_params);
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: Failed at kernel Launch" << std::endl;
  }
}

int main(int argc, char const **argv) {

  cudaDeviceProp props;
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: "
              << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (props.major != 9) {
    std::cout << "This example requires NVIDIA's Hopper Architecture GPU with "
                 "compute capability 90a\n"
              << std::endl;
    return 0;
  }

#if !defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  std::cout << "CUTLASS_ARCH_MMA_SM90_SUPPORTED must be enabled, but it is "
               "not. Test is waived \n"
            << std::endl;
#endif


  Options options;

  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }
  int m = options.m;
  int n = options.n;
  int k = options.k;
  float alpha = options.alpha;
  float beta = options.beta;
  if (options.seed == 0)
    srand(time(NULL));
  else
    srand(options.seed);

  print("M N K = [%d %d %d], alpha = %f, beta = %f\n", m, n, k, alpha, beta);

  using TA = cute::half_t;
  using TB = cute::half_t;
  using TC = float;

  thrust::host_vector<TA> h_A(m * k);
  thrust::host_vector<TB> h_B(n * k);
  thrust::host_vector<TC> h_C(m * n);
  thrust::host_vector<TC> h_D(m * n);
  thrust::host_vector<TC> h_DRef(m * n);
  thrust::host_vector<TC> h_DRefBlas(m * n);

  for (int j = 0; j < m * k; ++j) {
    float val = 2 * (rand() / double(RAND_MAX)) - 1;
    h_A[j] = static_cast<TA>(val);
  }
  for (int j = 0; j < n * k; ++j) {
    float val = 2 * (rand() / double(RAND_MAX)) - 1;
    h_B[j] = static_cast<TB>(val);
  }
  for (int j = 0; j < m * n; ++j)
  {
    float val = 2 * (rand() / double(RAND_MAX)) - 1;
    h_C[j] = static_cast<TC>(val);
  }

  for (int j = 0; j < m * n; ++j) {
    h_D[j] = TC(0);
    h_DRef[j] = TC(0);
    h_DRefBlas[j] = TC(0);
  }

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TB> d_B = h_B;
  thrust::device_vector<TC> d_C = h_C;
  thrust::device_vector<TC> d_D = h_D;
  thrust::device_vector<TC> d_DRef = h_DRef;

  // TN GEMM
  int ldA = k, ldB = k, ldC = m, ldD = m;

  //////////////// Example 1: alpha * acc + beta * C //////////////////////////

  static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
  using EVTType = float;

  using namespace cutlass::epilogue::fusion;
  using CustomEVT = // alpha * acc + beta * C
      Sm90EVT<
          Sm90Compute<
              cutlass::homogeneous_multiply_add, TC, EVTType,
              RoundStyle>, // beta * C + (alpha * acc)
          Sm90ScalarBroadcast<EVTType>, // beta
          Sm90SrcFetch<EVTType>,        // C
          Sm90EVT<
              Sm90Compute<
                  cutlass::multiplies, EVTType, EVTType, RoundStyle>, // alpha * acc
              Sm90ScalarBroadcast<EVTType>, // alpha
              Sm90AccFetch                  // acc
              >>;

  CustomEVT::Arguments epi_args{
      // ternary op : beta * C + (alpha * acc)
      {{EVTType(beta)}}, // leaf op+args : beta
      {},                        // leaf op+args : C
      {
          // binary op : alpha * acc
          {{EVTType(alpha)}}, // leaf op+args : alpha
          {},                         // leaf op+args : acc
          {}                          // binary args : multiplies
      },                              // end binary op
      {}                              // ternary args : multiply_add
  };                                  // end

  auto prob_shape = make_shape(m, n, k); // (M, N, K)

  typename CustomEVT::Params epi_params =
      CustomEVT::to_underlying_arguments(prob_shape, epi_args, nullptr);

  // Run CuTe example with fused EVT (standard epilogue: alpha * AB + beta * C)
  gemm_tn<CustomEVT>(m, n, k, options.alpha, d_A.data().get(), ldA,
                     d_B.data().get(), ldB, options.beta, d_C.data().get(), ldC,
                     d_D.data().get(), ldD, epi_params);
  CUTE_CHECK_LAST();
  h_D = d_D;

  // Run CUTLASS example for reference (standard epilogue: alpha * AB + beta * C)
  cutlass::KernelHardwareInfo hw_info;

  hw_info.device_id = 0;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  ExampleRunner<CustomEVT, TA, TB, TC, EVTType,
                cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                cutlass::epilogue::TmaWarpSpecializedCooperative,
                cutlass::gemm::collective::StageCountAuto,
                cutlass::gemm::PersistentScheduler, true>
      ws_cooperative_schedule_auto_stage_custom_evt_runner;
  bool valid = false;
  valid = ws_cooperative_schedule_auto_stage_custom_evt_runner.run(
      options, hw_info, d_A.data().get(), d_B.data().get(), d_C.data().get(),
      d_DRef.data().get(), epi_args);
  h_DRef = d_DRef;
  bool passed = false;
  // Run BLAS for reference (standard epilogue: alpha * AB + beta * C)
  cublasHandle_t handle;
  cublasCreate(&handle);
  // Note this is going to accumulate into d_C
  blam::cublas::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                     d_A.data().get(), k, d_B.data().get(), k,
                     &beta, d_C.data().get(), m);

  h_DRefBlas = d_C;

  passed = verify_tensor(h_D, h_DRef, options.printVal);
  print_result("Custom kernel with epilogue visitor tree (Reference against CUTLASS)",
               valid && passed);
  passed = verify_tensor(h_D, h_DRefBlas, false);
  print_result("Custom kernel with epilogue visitor tree (Reference against BLAS)",
               valid && passed);

  //////////////// Example 2: BCE Loss //////////////////////////

  using CtaTileShapeMNK = Shape<_128, _128, _64>;
  using ElementCompute = EVTType;
  using ElementScalar = TC;
  using ElementBias = TC;

  using CMinus1 =
    Sm90EVT<
      Sm90Compute<cutlass::minus, ElementCompute, ElementCompute, RoundStyle>,
      Sm90SrcFetch<TC>,
      Sm90ScalarBroadcast<ElementScalar>
    >;

  using MatmulPlusBias =
    Sm90EVT<
      Sm90Compute<cutlass::plus, ElementCompute, ElementCompute, RoundStyle>,
      Sm90ColBroadcast<0, CtaTileShapeMNK, ElementBias, Stride<_1, _0, _0>>,
      Sm90AccFetch
    >;

  using TopoVisitor =
    Sm90TopologicalVisitor<
      ElementCompute,
      cute::tuple<
        cute::seq<>,
        cute::seq<>,
        cute::seq<0, 1>,
        cute::seq<0>,
        cute::seq<3>,
        cute::seq<4>,
        cute::seq<2, 5>,
      >,
      MatmulPlusBias,
      CMinus1,
      Sm90Compute<cutlass::multiplies, ElementCompute, ElementCompute, RoundStyle>,
      Sm90Compute<cutlass::epilogue::thread::Sigmoid, ElementCompute, ElementCompute,
                  RoundStyle>,
      Sm90Compute<cutlass::epilogue::thread::Clamp, ElementCompute, ElementCompute,
                  RoundStyle>,
      Sm90Compute<FastLog, ElementCompute, ElementCompute, RoundStyle>,
      Sm90Compute<cutlass::plus, ElementCompute, ElementCompute, RoundStyle>
    >;
  using BCELossEVT =
    Sm90EVT<
      Sm90ScalarReduction<
        cutlass::plus,       // register reduce function
        cutlass::atomic_add, // GMEM reduce function
          ElementScalar, ElementCompute, RoundStyle,
          Stride<_0, _0, _0>>, // no batching here
      TopoVisitor
    >;

  // Set aside space for bias vector and result
  thrust::host_vector<ElementBias> h_bias_BCE(n);
  for (size_t i = 0; i < n; ++i)
    h_bias_BCE[i] = static_cast<ElementBias>(2 * (rand() / double(RAND_MAX)) - 1);
  thrust::device_vector<ElementBias> d_bias_BCE = h_bias_BCE;
  auto stride_bias_BCE = make_stride(_1{}, _0{}, _0{});
  TC* h_result = (TC*) malloc(sizeof(TC));
  TC* h_resultRef = (TC*) malloc(sizeof(TC));
  TC* d_result;
  cudaMalloc(&d_result, sizeof(TC));
  cudaMemset(d_result, 0, sizeof(TC));
  auto stride_result = make_stride(_0{}, _0{}, _0{});

  // Reinitialize C to a matrix of 0s and 1s (labels)
  for (size_t i = 0; i < m * n; ++i)
    h_C[i] = static_cast<TC>(rand() % 2);
  d_C = h_C;

  BCELossEVT::Arguments args_BCE =
  {
    { // TopoVisitor [(C - 1) * (bias + AB) + log(clamp(sigmoid(bias + AB)))]
      { // args to MatmulPlusBias = bias + AB (node 0)
        {d_bias_BCE.data().get(), 0, stride_bias_BCE}, // args to ColBroadcast
        {},  // args to AccFetch
        {}   // op args: plus
      },
      { // args to CMinus1 = C - 1 (node 1)
        {}, // args to SrcFetch
        {{ElementScalar(1.0)}}, // args to ScalarBroadcast
        {}  // op args: minus
      },
      {}, // op args: multiplies (node 2)
      {}, // op args: sigmoid (node 3)
      {0.001f, 0.999f},   // op args: clamp (node 4)
      {}, // op args: log (node 5)
      {}, // op args: plus (node 6)
    },
    {d_result, 0, stride_result} // args to ScalarReduction
  };

  typename BCELossEVT::Params params_BCE =
      BCELossEVT::to_underlying_arguments(prob_shape, args_BCE, nullptr);
  gemm_tn<BCELossEVT>(m, n, k, options.alpha, d_A.data().get(), ldA,
                     d_B.data().get(), ldB, options.beta, d_C.data().get(), ldC,
                     d_D.data().get(), ldD, params_BCE);
  h_D = d_D;
  cudaMemcpy(h_result, d_result, sizeof(TC), cudaMemcpyDeviceToHost);
  cudaMemset(d_result, 0, sizeof(TC));
  CUTE_CHECK_LAST();

  ExampleRunner<BCELossEVT, TA, TB, TC, EVTType,
                cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                cutlass::epilogue::TmaWarpSpecializedCooperative,
                cutlass::gemm::collective::StageCountAuto,
                cutlass::gemm::PersistentScheduler, true>
      ws_cooperative_schedule_auto_stage_BCELoss_evt_runner;
  valid = ws_cooperative_schedule_auto_stage_BCELoss_evt_runner.run(
      options, hw_info, d_A.data().get(), d_B.data().get(), d_C.data().get(),
      d_DRef.data().get(), args_BCE);
  h_DRef = d_DRef;
  passed = verify_tensor(h_D, h_DRef, options.printVal);
  print_result("Custom kernel with BCE loss epilogue (Reference against CUTLASS)",
               valid && passed);
  cudaMemcpy(h_resultRef, d_result, sizeof(TC), cudaMemcpyDeviceToHost);
  CUTE_CHECK_LAST();
  std::cout << "BCE loss result (custom kernel fusion): "
            << *h_result / ((TC) -m * n)
            << std::endl;
  std::cout << "BCE loss result (CUTLASS example): "
            << *h_resultRef / ((TC) -m * n)
            << std::endl;

  // auto min_D = thrust::min_element(thrust::host, h_D.begin(), h_D.end());
  // auto max_D = thrust::max_element(thrust::host, h_D.begin(), h_D.end());
  // std::cout << "Smallest element: " << *min_D << std::endl;
  // std::cout << "Largest element:  " << *max_D << std::endl;

  cudaFree(d_result);
  free(h_result);

  return 0;
}
