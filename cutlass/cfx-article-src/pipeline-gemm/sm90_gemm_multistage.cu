/***************************************************************************************************
 * Copyright (c) 2024 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/cluster_launch.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/device_kernel.h"

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

using namespace cute;

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: this requires tensor to be "contiguous"
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
    // Tensor out = make_tensor_like<To_type>(tensor);
    // cute::copy(make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout()), out);
    // return out;
}

template <class ElementA,
          class ElementB,
          class ElementC,
          class SmemLayoutA,  // (M,K,P)
          class SmemLayoutB,  // (N,K,P)
          class SmemLayoutC>  
struct SharedStorage
{
  array_aligned<ElementA, cosize_v<SmemLayoutA>> smem_A;
  // put in smem_C
  union {
    array_aligned<ElementB, cosize_v<SmemLayoutB>> smem_B;
    array_aligned<ElementC, cosize_v<SmemLayoutC>> smem_C;
  };

  uint64_t tma_barrier[size<2>(SmemLayoutA{})]; // size = P
  uint64_t mma_barrier[size<2>(SmemLayoutA{})];

  
};

template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TB, class SmemLayoutB, class TmaB,
          class TC, class SmemLayoutC, class TmaC,
          class TiledMma>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
            TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
            TC      * C, CUTLASS_GRID_CONSTANT TmaC const tma_store,
            TiledMma mma)
{   
  
  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

  static_assert(is_static<SmemLayoutA>::value);
  static_assert(is_static<SmemLayoutB>::value);

  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutA{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutB{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutA{}) == size<2>(cta_tiler));  // BLK_K
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutB{}) == size<2>(cta_tiler));  // BLK_K

  //
  // Full and Tiled Tensors
  //

  // Represent the full tensors
  auto [M, N, K] = shape_MNK;
  Tensor mA = tma_a.get_tma_tensor(make_shape(M,K));                   // (M,K) TMA Tensor
  Tensor mB = tma_b.get_tma_tensor(make_shape(N,K));                   // (N,K) TMA Tensor
//   Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M,N), dC);      // (M,N)

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
//   Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

  // Shared memory tensors
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TA, TB, TC, SmemLayoutA, SmemLayoutB, SmemLayoutC>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.smem_A.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.smem_B.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

  //
  // Partition the copying of A and B tiles
  //
  // TUTORIAL:
  //   These are TMA partitionings, which have a dedicated custom partitioner.
  //   The Int<0>, Layout<_1> indicates that the TMAs are not multicasted.
  //     Any multicasting must be in conformance with tma_x constructed with make_tma_atom on host.
  //   The group_modes<0,2> transforms the (X,Y,Z)-shaped tensors into ((X,Y),Z)-shaped tensors
  //     with the understanding that the TMA is responsible for everything in mode-0.
  //   The tma_partition reorders and offsets mode-0 according to the tma_x atom and the multicast info.
  //

  auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                    group_modes<0,2>(sA), group_modes<0,2>(gA));  // (TMA,k) and (TMA,PIPE)

  auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                    group_modes<0,2>(sB), group_modes<0,2>(gB));  // (TMA,k) and (TMA,PIPE)

  // The TMA is responsible for copying everything in mode-0 of tAsA and tBsB
  constexpr int kTmaTransactionBytes = CUTE_STATIC_V(size<0>(tAsA)) * sizeof(TA) +
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
  uint64_t* producer_mbar = smem.tma_barrier;
  uint64_t* consumer_mbar = smem.mma_barrier;

  using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;  // TMA
  using ConsumerBarType = cutlass::arch::ClusterBarrier;             // MMA
  CUTE_UNROLL
  for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
    if ((warp_idx == 0) && lane_predicate) {
      ProducerBarType::init(&producer_mbar[pipe],   1);
      ConsumerBarType::init(&consumer_mbar[pipe], size(mma));
    }
  }
  // Ensure barrier init is complete on all CTAs
  cluster_sync();

  // Start async loads for all pipes
  CUTE_UNROLL
  for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe)
  {
    if ((warp_idx == 0) && lane_predicate)
    {
      // Set expected Tx Bytes after each reset / init
      ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], kTmaTransactionBytes);
      copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
      copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
    }
    --k_tile_count;
    ++k_tile;
  }

  //
  // Define A/B partitioning and C accumulators
  //
  // TUTORIAL:
  //   The tCrA and tCrB are actually Tensors of MMA Descriptors constructed as views of SMEM.
  //   The MMA Descriptor generation is automatic via inspection and validation of the SMEM Layouts.
  //   Because the MMA reads directly from SMEM and the fragments are descriptors rather than registers,
  //     there is no need for copy(tCsA, tCrA) in the mainloop.
  //

  ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);                               // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K,PIPE)
//   Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

  // Allocate accumulators and clear them
//   Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)
  Tensor tCrC = partition_fragment_C(mma, select<0, 1>(cta_tiler));
  clear(tCrC);

  // Allocate descriptor iterators
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);                         // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);                         // (MMA,MMA_N,MMA_K,PIPE)

  //
  // PIPELINED MAIN LOOP
  //
  // TUTORIAL:
  //   Rather than interleaving the stages and instructions like in SM70 and SM80,
  //     the SM90 mainloops rely on explicit producer-consumer synchronization
  //     on the purely async instructions TMA and MMA.
  //   More advanced pipeline and warp-specialization strategies are available in CUTLASS mainloops.
  //

  // A PipelineState is a circular pipe index [.index()] and a pipe phase [.phase()]
  //   that flips each cycle through K_PIPE_MAX.
  auto write_state = cutlass::PipelineState<K_PIPE_MAX>();             // TMA writes
  auto read_state  = cutlass::PipelineState<K_PIPE_MAX>();             // MMA  reads

  CUTE_NO_UNROLL
  while (k_tile_count > -K_PIPE_MAX)
  {
    // Wait for Producer to complete
    int read_pipe = read_state.index();
    ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

    // MMAs to cover 1 K_TILE
    warpgroup_arrive(); 
    gemm(mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);     // (V,M) x (V,N) => (V,M,N)
    warpgroup_commit_batch();

    // Wait for all MMAs in a K_TILE to complete
    warpgroup_wait<0>();

    // Notify that consumption is done
    ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
    ++read_state;

    if ((warp_idx == 0) && lane_predicate)
    {
      int pipe = write_state.index();
      // Wait for Consumer to complete consumption
      ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());
      // Set expected Tx Bytes after each reset / init
      ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], kTmaTransactionBytes);
      copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
      copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
      ++write_state;
    }
    --k_tile_count;
    ++k_tile;
  }

  //
  // Epilogue
  //

#if 1

    // Make sure all warpgroups have finished mma
    cutlass::arch::NamedBarrier::sync(size(mma), 0);

    using SmemCopyAtomC = Copy_Atom<cute::SM90_U32x4_STSM_N, TC>;
    
    Tensor sC = make_tensor(make_smem_ptr(smem.smem_C.data()), SmemLayoutC{}); // (BLK_M,BLK_N)

    auto smem_tiled_copy_C = make_tiled_copy_C(SmemCopyAtomC{}, mma);
    auto smem_thr_copy_C = smem_tiled_copy_C.get_thread_slice(threadIdx.x);
    // include this convert utility
    Tensor tCrC_out = convert_type<TC>(tCrC);
    Tensor taccCrC = smem_thr_copy_C.retile_S(tCrC_out);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccCsC = smem_thr_copy_C.partition_D(sC);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
    cute::copy(smem_tiled_copy_C, taccCrC, taccCsC);
    cutlass::arch::fence_view_async_shared(); // ensure smem writes are visible to TMA

    // NumThreads == size(mma)
    cutlass::arch::NamedBarrier::arrive(size(mma) + cutlass::NumThreadsPerWarp,
        cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

    // Prepare TMA store

    // auto [M, N, K] = shape_MNK;
    Tensor mC = tma_store.get_tma_tensor(make_shape(M,N));
    Tensor gC = local_tile(mC, select<0,1>(cta_tiler), make_coord(blockIdx.x, blockIdx.y));
    auto block_tma_store = tma_store.get_slice(_0{}); // CTA slice
    Tensor tCgC = block_tma_store.partition_D(gC);  // (TMA, TMA_M, TMA_K)
    Tensor tCsC = block_tma_store.partition_S(sC);  // (TMA, TMA_M, TMA_K)

    // TMA STORE: SMEM -> GMEM
    // int const warp_idx = cutlass::canonical_warp_idx_sync();
    // int const lane_predicate = cute::elect_one_sync();
    if (warp_idx == 0) {
        // Ensure RMEM -> SMEM copy completes before issuing TMA store
        cutlass::arch::NamedBarrier::sync(
            size(mma) + cutlass::NumThreadsPerWarp, 
            cutlass::arch::ReservedNamedBarriers::EpilogueBarrier
        );
    }

    if (warp_idx == 0 && lane_predicate) {
        cute::copy(tma_store, tCsC, tCgC);
        // tma_store_arrive();
    }

    // tma_store_wait<0>();

#endif

}

// Setup params for a TN GEMM
template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm_tn(int m, int n, int k,
        Alpha alpha,
        TA const* A, int ldA,
        TB const* B, int ldB,
        Beta beta,
        TC      * C, int ldC,
        cudaStream_t stream = 0)
{
//   printf("gemm tn.\n");
  
  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define TN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{});                      // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{});                      // (dN, dK)
//   auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)
  auto dC = make_stride(ldC, Int<1>{});                      // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<256>{};
  auto bN = Int<192>{};
  auto bK = Int<128>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<2>{};  // Pipeline
  using AtomLayoutMNK = Layout<Shape<_2, _1, _1>>;

  // Define the smem layouts (static)
  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
  auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(bN,bK,bP));

  // Define the Tiled MMA
  TiledMMA tiled_mma = make_tiled_mma(SM90_64x192x16_F32F16F16_SS<GMMA::Major::K,GMMA::Major::K>{},
    AtomLayoutMNK{});

  // For FP16 ACCUM example, enable these tile sizes and variable definitions.
  // auto bM = Int<256>{};
  // auto bN = Int<256>{};
  // auto bK = Int<96>{};
  // auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
  // auto bP = Int<2>{};  // Pipeline
  // using AtomLayoutMNK = Layout<Shape<_4, _1, _1>>;

  // auto sA = tile_to_shape(GMMA::Layout_K_SW64_Atom<TA>{}, make_shape(bM,bK,bP));
  // auto sB = tile_to_shape(GMMA::Layout_K_SW64_Atom<TB>{}, make_shape(bN,bK,bP));

  // TiledMMA tiled_mma = make_tiled_mma(SM90_64x256x16_F16F16F16_SS<GMMA::Major::K,GMMA::Major::K>{},
  //   AtomLayoutMNK{});

  //
  // Back to common setting
  //
  
  // Define the smem layout for the output (static)
  auto sC = tile_to_shape(GMMA::Layout_K_SW128_Atom<TC>{}, make_shape(bM,bN));

  // Define the TMAs
  // Create Global memory tensors for TMA inspection
  Tensor mA = make_tensor(A, make_shape(M,K), dA);
  Tensor mB = make_tensor(B, make_shape(N,K), dB);
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);

  // Create TMA Atoms with the desired copy operation on the source and destination
  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM,bK));
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN,bK));

  auto tma_store = make_tma_copy(SM90_TMA_STORE{}, mC, sC, make_shape(bM, bN), _1{});

  //
  // Setup and Launch
  //

  // Launch parameter setup
  int smem_size = int(sizeof(SharedStorage<TA, TB, TC, decltype(sA), decltype(sB), decltype(sC)>));

//   std::cout << "smem size is " << smem_size << std::endl;



  dim3 dimBlock(size(tiled_mma));
  // dim3 dimCluster(1, 1, 1);
  dim3 dimCluster(1, 2, 1);
  dim3 dimGrid(round_up(size(ceil_div(m, bM)), dimCluster.x),
               round_up(size(ceil_div(n, bN)), dimCluster.y));
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

  void const* kernel_ptr = reinterpret_cast<void const*>(
                              &gemm_device<decltype(prob_shape), decltype(cta_tiler),
                                           TA, decltype(sA), decltype(tmaA),
                                           TB, decltype(sB), decltype(tmaB),
                                           TC, decltype(sC), decltype(tma_store),
                                           decltype(tiled_mma)>);

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
    kernel_ptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size));

#if 1
  // Kernel Launch
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
                                                             prob_shape, cta_tiler,
                                                             A, tmaA,
                                                             B, tmaB,
                                                             C, tma_store,
                                                             tiled_mma);
  CUTE_CHECK_LAST();


  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: Failed at kernel Launch" << std::endl;
  }

#endif
}

template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm(char transA, char transB, int m, int n, int k,
     Alpha alpha,
     TA const* A, int ldA,
     TB const* B, int ldB,
     Beta beta,
     TC      * C, int ldC,
     cudaStream_t stream = 0)
{
  if (transA == 'N' && transB == 'T') {
    // This example doesn't implement NT gemm for simplicity.
    // return gemm_nt(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
  } else
  if (transA == 'T' && transB == 'N') {
    return gemm_tn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
  }
  assert(false && "Not implemented");
}

int main(int argc, char** argv)
{

  cudaDeviceProp props;
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (props.major != 9) {
    std::cout << "This example requires NVIDIA's Hopper Architecture GPU with compute capability 90a\n" << std::endl;
    return 0;
  }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

  int m = 8192;
  if (argc >= 2)
    sscanf(argv[1], "%d", &m);

  int n = 8192;
  if (argc >= 3)
    sscanf(argv[2], "%d", &n);

  int k = 8192;
  if (argc >= 4)
    sscanf(argv[3], "%d", &k);

  char transA = 'T';
//   if (argc >= 5)
//     sscanf(argv[4], "%c", &transA);

  char transB = 'N';
//   if (argc >= 6)
//     sscanf(argv[5], "%c", &transB);

  print("M N K = [%d %d %d].\n", m, n, k);

  using TA = cute::half_t;
  using TB = cute::half_t;
  using TC = cute::half_t;
  using TI = cute::half_t;

  TI alpha = TI(1.0f);
  TI beta  = TI(0.0f);

  thrust::host_vector<TA> h_A(m*k);
  thrust::host_vector<TB> h_B(n*k);
  thrust::host_vector<TC> h_C(m*n);

  // Initialize the tensors
  // for (int j = 0; j < m*k; ++j) h_A[j] = TA(int((rand() % 2) ? 1 : -1));
  // for (int j = 0; j < n*k; ++j) h_B[j] = TB(int((rand() % 2) ? 1 : -1));
  for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TB>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < m*n; ++j) h_C[j] = TC(0);

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TB> d_B = h_B;
  thrust::device_vector<TC> d_C = h_C;

  double gflops = (2.0*m*n*k) * 1e-9;

  const int timing_iterations = 10;
  GPU_Clock timer;

  int ldA = 0, ldB = 0, ldC = n;

  if (transA == 'N') {
    ldA = m;
  } else if (transA == 'T') {
    ldA = k;
  } else {
    assert(false);
  }

  if (transB == 'N') {
    ldB = k;
  } else if (transB == 'T') {
    ldB = n;
  } else {
    assert(false);
  }

  // Run once
  d_C = h_C;
  gemm(transA, transB, m, n, k,
       alpha,
       d_A.data().get(), ldA,
       d_B.data().get(), ldB,
       beta,
       d_C.data().get(), ldC);
  CUTE_CHECK_LAST();
  thrust::host_vector<TC> cute_result = d_C;

  // Timing iterations
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    gemm(transA, transB, m, n, k,
         alpha,
         d_A.data().get(), ldA,
         d_B.data().get(), ldB,
         beta,
         d_C.data().get(), ldC);
  }
  double cute_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("CUTE_GEMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time, cute_time*1000);

#else

  std::cout << "CUTLASS_ARCH_MMA_SM90_SUPPORTED must be enabled, but it is not. Test is waived \n" << std::endl;
#endif

 return 0;

}
