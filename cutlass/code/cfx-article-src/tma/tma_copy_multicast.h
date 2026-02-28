#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <chrono>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cutlass/numeric_types.h"
#include <cute/arch/cluster_sm90.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

#include "cutlass/detail/layout.hpp"

#include "cuda_launch.hpp"
#include "shared_storage.h"
#include "smem_helper.hpp"

template <typename _TiledCopyS, typename _TiledCopyD, typename _GmemLayout,
          typename _GmemLayoutOut, typename _SmemLayout, typename _TileShape,
          typename _ClusterShape>
struct ParamsMulticast {
  using TiledCopyS = _TiledCopyS;
  using TiledCopyD = _TiledCopyD;
  using GmemLayout = _GmemLayout;
  using GmemLayoutOut = _GmemLayoutOut;
  using SmemLayout = _SmemLayout;
  using TileShape = _TileShape;
  using ClusterShape = _ClusterShape;

  TiledCopyS const tmaLoad;
  TiledCopyD const tmaStore;
  GmemLayout const gmemLayout;
  GmemLayoutOut const gmemLayoutOut;
  SmemLayout const smemLayout;
  TileShape const tileShape;
  ClusterShape const cluster_shape;

  ParamsMulticast(_TiledCopyS const &tmaLoad, _TiledCopyD const &tmaStore,
                  _GmemLayout const &gmemLayout,
                  _GmemLayoutOut const &gmemLayoutOut,
                  _SmemLayout const &smemLayout, _TileShape const &tileShape,
                  _ClusterShape const &cluster_shape)
      : tmaLoad(tmaLoad), tmaStore(tmaStore), gmemLayout(gmemLayout),
        gmemLayoutOut(gmemLayoutOut), smemLayout(smemLayout),
        tileShape(tileShape), cluster_shape(cluster_shape) {}
};

template <int kNumThreads, class Element, class Params>
__global__ static void __launch_bounds__(kNumThreads, 1)
    copyTMAKernelMulticast(CUTE_GRID_CONSTANT Params const params) {
  using namespace cute;

  //
  // Get layouts and tiled copies from Params struct
  //
  using GmemLayout = typename Params::GmemLayout;
  using GmemLayoutOut = typename Params::GmemLayoutOut;
  using SmemLayout = typename Params::SmemLayout;
  using TileShape = typename Params::TileShape;
  using ClusterShape = typename Params::ClusterShape;

  auto &tmaLoad = params.tmaLoad;
  auto &tmaStore = params.tmaStore;
  auto &gmemLayout = params.gmemLayout;
  auto &gmemLayoutOut = params.gmemLayoutOut;
  auto &smemLayout = params.smemLayout;
  auto &tileShape = params.tileShape;
  auto &cluster_shape = params.cluster_shape;

  uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();

  constexpr uint32_t cluster_size = size(ClusterShape{});

  uint16_t tma_mcast_mask = ((uint16_t(1) << cluster_size) - 1);

  // Use Shared Storage structure to allocate aligned SMEM addresses.
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorageTMA<Element, SmemLayout>;
  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(shared_memory);

  // Define smem tensor
  Tensor sS =
      make_tensor(make_smem_ptr(shared_storage.smem.data()), smemLayout);

  // Get mbarrier object and its value type
  auto &mbarrier = shared_storage.mbarrier;
  using BarrierType = cutlass::arch::ClusterTransactionBarrier::ValueType;
  static_assert(cute::is_same_v<BarrierType, uint64_t>,
                "Value type of mbarrier is uint64_t.");

  // Constants used for TMA
  const int warp_idx = cutlass::canonical_warp_idx_sync();
  const bool lane_predicate = cute::elect_one_sync();
  constexpr int kTmaTransactionBytes =
      sizeof(ArrayEngine<Element, size(SmemLayout{})>);

  // Prefetch TMA descriptors for load and store
  if (warp_idx == 0 && lane_predicate) {
    prefetch_tma_descriptor(tmaLoad.get_tma_descriptor());
    prefetch_tma_descriptor(tmaStore.get_tma_descriptor());
  }

  // Get CTA view of gmem tensor
  Tensor mS = tmaLoad.get_tma_tensor(shape(gmemLayout));
  auto blkCoord = make_coord(blockIdx.x, blockIdx.y);
  Tensor gS = local_tile(mS, tileShape, blkCoord);

  auto cta_tmaS = tmaLoad.get_slice(block_rank_in_cluster);
  auto tSgSX = cta_tmaS.partition_S(gS);
  auto tSgS = group_modes<1, rank(tSgSX)>(tSgSX);
  auto tSsSX = cta_tmaS.partition_D(sS);
  auto tSsS = group_modes<1, rank(tSsSX)>(tSsSX);

  if (warp_idx == 0 and lane_predicate) {
    mbarrier.init(1 /* arrive count */);
  }
  __syncthreads();
  cute::cluster_sync();
  cutlass::arch::fence_barrier_init();

  if (warp_idx == 0 and lane_predicate) {
    mbarrier.arrive_and_expect_tx(kTmaTransactionBytes);
    copy(
        tmaLoad.with(reinterpret_cast<BarrierType &>(mbarrier), tma_mcast_mask),
        tSgS(_, 0), tSsS(_, 0));
  }
  __syncthreads();

  mbarrier.wait(0 /* phase */);

  cutlass::arch::fence_view_async_shared();

  // Get CTA view of gmem out tensor
  auto blkCoordOut = make_coord(blockIdx.x, blockIdx.y, blockIdx.z);
  auto mD = tmaStore.get_tma_tensor(shape(gmemLayoutOut));
  auto gD = local_tile(mD, tileShape, blkCoordOut);

  auto cta_tmaD = tmaStore.get_slice(Int<0>{});

  if (warp_idx == 0 and lane_predicate) {
    cute::copy(tmaStore, cta_tmaD.partition_S(sS), cta_tmaD.partition_D(gD));
    // cute::tma_store_arrive();
  }
  // cute::tma_store_wait<0>();
  cute::cluster_sync();
}

template <int kNumThreads, class Element, class Params>
__global__ static void __launch_bounds__(kNumThreads, 1)
    copyTMAKernelNoMulticast(CUTE_GRID_CONSTANT Params const params) {
  using namespace cute;

  //
  // Get layouts and tiled copies from Params struct
  //
  using GmemLayout = typename Params::GmemLayout;
  using GmemLayoutOut = typename Params::GmemLayoutOut;
  using SmemLayout = typename Params::SmemLayout;
  using TileShape = typename Params::TileShape;

  auto &tmaLoad = params.tmaLoad;
  auto &tmaStore = params.tmaStore;
  auto &gmemLayout = params.gmemLayout;
  auto &gmemLayoutOut = params.gmemLayoutOut;
  auto &smemLayout = params.smemLayout;
  auto &tileShape = params.tileShape;
  auto &cluster_shape = params.cluster_shape;

  // Use Shared Storage structure to allocate aligned SMEM addresses.
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorageTMA<Element, SmemLayout>;
  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(shared_memory);

  // Define smem tensor
  Tensor sS =
      make_tensor(make_smem_ptr(shared_storage.smem.data()), smemLayout);

  // Get mbarrier object and its value type
  auto &mbarrier = shared_storage.mbarrier;
  using BarrierType = cutlass::arch::ClusterTransactionBarrier::ValueType;
  static_assert(cute::is_same_v<BarrierType, uint64_t>,
                "Value type of mbarrier is uint64_t.");

  // Constants used for TMA
  const int warp_idx = cutlass::canonical_warp_idx_sync();
  const bool lane_predicate = cute::elect_one_sync();
  constexpr int kTmaTransactionBytes =
      sizeof(ArrayEngine<Element, size(SmemLayout{})>);

  // Prefetch TMA descriptors for load and store
  if (warp_idx == 0 && lane_predicate) {
    prefetch_tma_descriptor(tmaLoad.get_tma_descriptor());
    prefetch_tma_descriptor(tmaStore.get_tma_descriptor());
  }

  // Get CTA view of gmem tensor
  Tensor mS = tmaLoad.get_tma_tensor(shape(gmemLayout));
  auto blkCoord = make_coord(blockIdx.x, blockIdx.y);
  Tensor gS = local_tile(mS, tileShape, blkCoord);

  auto cta_tmaS = tmaLoad.get_slice(0);
  auto tSgSX = cta_tmaS.partition_S(gS);
  auto tSgS = group_modes<1, rank(tSgSX)>(tSgSX);
  auto tSsSX = cta_tmaS.partition_D(sS);
  auto tSsS = group_modes<1, rank(tSsSX)>(tSsSX);

  if (warp_idx == 0 and lane_predicate) {
    mbarrier.init(1 /* arrive count */);
  }
  __syncthreads();
  cutlass::arch::fence_barrier_init();

  if (warp_idx == 0 and lane_predicate) {
    mbarrier.arrive_and_expect_tx(kTmaTransactionBytes);
    copy(tmaLoad.with(reinterpret_cast<BarrierType &>(mbarrier)), tSgS(_, 0),
         tSsS(_, 0));
  }
  __syncthreads();

  mbarrier.wait(0 /* phase */);

  cutlass::arch::fence_view_async_shared();

  // Get CTA view of gmem out tensor
  auto blkCoordOut = make_coord(blockIdx.x, blockIdx.y, blockIdx.z);
  auto mD = tmaStore.get_tma_tensor(shape(gmemLayoutOut));
  auto gD = local_tile(mD, tileShape, blkCoordOut);

  auto cta_tmaD = tmaStore.get_slice(Int<0>{});

  if (warp_idx == 0 and lane_predicate) {
    cute::copy(tmaStore, cta_tmaD.partition_S(sS), cta_tmaD.partition_D(gD));
    // cute::tma_store_arrive();
  }
  // cute::tma_store_wait<0>();
}

template <bool use_multicast = true, int COPYN = 2, int TILE_M = 128,
          int TILE_N = 128, int THREADS = 128>
int copy_host_tma_load_and_store_kernel_multicast(int M, int N,
                                                  int iterations = 1) {
  using namespace cute;

  std::cout << "Deep copy " << COPYN << "X." << std::endl;

  if constexpr (use_multicast)
    printf("Copy with TMA Multicast load and store.\n");
  else
    printf("Copy with TMA load and store, NO multicast.\n");

  using Element = float;

  using ClusterShape = Shape<_1, _1, Int<COPYN>>;
  ClusterShape cluster_shape;

  auto tensor_shape = make_shape(M, N);
  auto tensor_shape_out = make_shape(M, N, Int<COPYN>{});

  // Allocate and initialize
  thrust::host_vector<Element> h_S(size(tensor_shape));     // (M, N)
  thrust::host_vector<Element> h_D(size(tensor_shape_out)); // (M, N, COPYN)

  for (size_t i = 0; i < h_S.size(); ++i)
    h_S[i] = static_cast<Element>(float(i));

  thrust::device_vector<Element> d_S = h_S;
  thrust::device_vector<Element> d_D = h_D;

  //
  // Make tensors
  //

  auto gmemLayoutS = make_layout(tensor_shape, LayoutRight{});
  auto gmemLayoutD = make_ordered_layout(tensor_shape_out, Step<_1, _0, _2>{});
  //   print(gmemLayoutD);

  Tensor tensor_S = make_tensor(
      make_gmem_ptr(thrust::raw_pointer_cast(d_S.data())), gmemLayoutS);
  Tensor tensor_D = make_tensor(
      make_gmem_ptr(thrust::raw_pointer_cast(d_D.data())), gmemLayoutD);

  using bM = Int<TILE_M>;
  using bN = Int<TILE_N>;

  auto tileShape = make_shape(bM{}, bN{});
  // NOTE: same smem layout for TMA load and store
  auto smemLayout =
      tile_to_shape(cfx::getSmemLayoutK<Element, TILE_N>(), tileShape);
  auto tma_load = make_tma_copy(SM90_TMA_LOAD_MULTICAST{}, tensor_S, smemLayout,
                                tileShape, size(cluster_shape));
  auto tma_load_no_multicast =
      make_tma_copy(SM90_TMA_LOAD{}, tensor_S, smemLayout, tileShape, _1());

  // print(tma_load);
  auto tma_store = make_tma_copy(SM90_TMA_STORE{}, tensor_D, smemLayout,
                                 tileShape, Int<1>{});
  // print(tma_store);

  ParamsMulticast params(tma_load, tma_store, gmemLayoutS, gmemLayoutD,
                         smemLayout, tileShape, cluster_shape);
  ParamsMulticast params_no_multicast(tma_load_no_multicast, tma_store,
                                      gmemLayoutS, gmemLayoutD, smemLayout,
                                      tileShape, cluster_shape);

  dim3 gridDim(ceil_div(M, TILE_M), ceil_div(N, TILE_N), COPYN);
  dim3 blockDim(THREADS);
  dim3 cluster_dims(size<0>(cluster_shape), size<1>(cluster_shape),
                    size<2>(cluster_shape));

  int smem_size = int(sizeof(SharedStorageTMA<Element, decltype(smemLayout)>));
  printf("smem size: %d.\n", smem_size);

  void const *kernel;
  if constexpr (use_multicast)
    kernel = (void const *)
        copyTMAKernelMulticast<THREADS, Element, decltype(params)>;
  else
    kernel =
        (void const *)copyTMAKernelNoMulticast<THREADS, Element,
                                               decltype(params_no_multicast)>;
  cfk::utils::set_smem_size(smem_size, kernel);

  // Define the cluster launch parameter structure.
  cutlass::ClusterLaunchParams launch_params{gridDim, blockDim, cluster_dims,
                                             smem_size};

  for (int i = 0; i < iterations; i++) {
    auto t1 = std::chrono::high_resolution_clock::now();
    if constexpr (use_multicast)
      cutlass::Status status =
          cutlass::launch_kernel_on_cluster(launch_params, kernel, params);
    else
      cutlass::Status status = cutlass::launch_kernel_on_cluster(
          launch_params, kernel, params_no_multicast);
    cudaError result = cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    if (result != cudaSuccess) {
      std::cerr << "CUDA Runtime error: " << cudaGetErrorString(result)
                << std::endl;
      return -1;
    }
    std::chrono::duration<double, std::milli> tDiff = t2 - t1;
    double time_ms = tDiff.count();
    std::cout << "Trial " << i << " Completed in " << time_ms << "ms ("
              << (COPYN + 1) * 1e-6 * M * N * sizeof(Element) / time_ms
              << " GB/s)" << std::endl;
  }

  //
  // Verify
  //

  h_D = d_D;

  int good = 0, bad = 0;

  int offset = size(tensor_shape);

  for (size_t i = 0; i < h_S.size(); ++i) {
    for (int j = 0; j < COPYN; ++j) {
      if (h_D[i + j * offset] == h_S[i])
        good++;
      else
        bad++;
    }
  }

  std::cout << "Success " << good << ", Fail " << bad << std::endl;

  return 0;
}
