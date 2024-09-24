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
          typename _SmemLayout, typename _TileShape>
struct Params {
  using TiledCopyS = _TiledCopyS;
  using TiledCopyD = _TiledCopyD;
  using GmemLayout = _GmemLayout;
  using SmemLayout = _SmemLayout;
  using TileShape = _TileShape;

  TiledCopyS const tmaLoad;
  TiledCopyD const tmaStore;
  GmemLayout const gmemLayout;
  SmemLayout const smemLayout;
  TileShape const tileShape;

  Params(_TiledCopyS const &tmaLoad, _TiledCopyD const &tmaStore,
         _GmemLayout const &gmemLayout, _SmemLayout const &smemLayout,
         _TileShape const &tileShape)
      : tmaLoad(tmaLoad), tmaStore(tmaStore), gmemLayout(gmemLayout),
        smemLayout(smemLayout), tileShape(tileShape) {}
};

template <int kNumThreads, class Element, class Params>
__global__ static void __launch_bounds__(kNumThreads, 1)
    copyTMAKernel(CUTE_GRID_CONSTANT Params const params) {
  using namespace cute;

  //
  // Get layouts and tiled copies from Params struct
  //
  using GmemLayout = typename Params::GmemLayout;
  using SmemLayout = typename Params::SmemLayout;
  using TileShape = typename Params::TileShape;

  auto &tmaLoad = params.tmaLoad;
  auto &tmaStore = params.tmaStore;
  auto &gmemLayout = params.gmemLayout;
  auto &smemLayout = params.smemLayout;
  auto &tileShape = params.tileShape;

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

  auto cta_tmaS = tmaLoad.get_slice(Int<0>{});

  if (warp_idx == 0 and lane_predicate) {
    mbarrier.init(1 /* arrive count */);
    mbarrier.arrive_and_expect_tx(kTmaTransactionBytes);
    copy(tmaLoad.with(reinterpret_cast<BarrierType &>(mbarrier)),
         cta_tmaS.partition_S(gS), cta_tmaS.partition_D(sS));
  }
  __syncthreads();

  mbarrier.wait(0 /* phase */);
  
  cutlass::arch::fence_view_async_shared();

  // Get CTA view of gmem out tensor
  auto mD = tmaStore.get_tma_tensor(shape(gmemLayout));
  auto gD = local_tile(mD, tileShape, blkCoord);

  auto cta_tmaD = tmaStore.get_slice(Int<0>{});

  if (warp_idx == 0 and lane_predicate) {
    cute::copy(tmaStore, cta_tmaD.partition_S(sS), cta_tmaD.partition_D(gD));
    // cute::tma_store_arrive();
  }
  // cute::tma_store_wait<0>();
}

template <int TILE_M = 128, int TILE_N = 128, int THREADS = 32>
int copy_host_tma_load_and_store_kernel(int M, int N, int iterations = 1) {
  using namespace cute;

  printf("Copy with TMA load and store -- no swizzling.\n");

  using Element = float;

  auto tensor_shape = make_shape(M, N);

  // Allocate and initialize
  thrust::host_vector<Element> h_S(size(tensor_shape)); // (M, N)
  thrust::host_vector<Element> h_D(size(tensor_shape)); // (M, N)

  for (size_t i = 0; i < h_S.size(); ++i)
    h_S[i] = static_cast<Element>(float(i));

  thrust::device_vector<Element> d_S = h_S;
  thrust::device_vector<Element> d_D = h_D;

  //
  // Make tensors
  //

  auto gmemLayoutS = make_layout(tensor_shape, LayoutRight{});
  auto gmemLayoutD = make_layout(tensor_shape, LayoutRight{});
  Tensor tensor_S = make_tensor(
      make_gmem_ptr(thrust::raw_pointer_cast(d_S.data())), gmemLayoutS);
  Tensor tensor_D = make_tensor(
      make_gmem_ptr(thrust::raw_pointer_cast(d_D.data())), gmemLayoutD);

  using bM = Int<TILE_M>;
  using bN = Int<TILE_N>;

  auto tileShape = make_shape(bM{}, bN{});
  // NOTE: same smem layout for TMA load and store
  auto smemLayout = make_layout(tileShape, LayoutRight{});
  auto tma_load =
      make_tma_copy(SM90_TMA_LOAD{}, tensor_S, smemLayout);
  // print(tma_load);

  auto tma_store = make_tma_copy(SM90_TMA_STORE{}, tensor_D, smemLayout);
  // print(tma_store);

  Params params(tma_load, tma_store, gmemLayoutS, smemLayout, tileShape);

  dim3 gridDim(ceil_div(M, TILE_M), ceil_div(N, TILE_N));
  dim3 blockDim(THREADS);

  int smem_size = int(sizeof(SharedStorageTMA<Element, decltype(smemLayout)>));
  printf("smem size: %d.\n", smem_size);

  void const *kernel =
      (void const *)copyTMAKernel<THREADS, Element, decltype(params)>;
  cfk::utils::set_smem_size(smem_size, kernel);

  dim3 cluster_dims(1);

  // Define the cluster launch parameter structure.
  cutlass::ClusterLaunchParams launch_params{gridDim, blockDim, cluster_dims,
                                             smem_size};

  for (int i = 0; i < iterations; i++) {
    auto t1 = std::chrono::high_resolution_clock::now();    
    cutlass::Status status =
        cutlass::launch_kernel_on_cluster(launch_params, kernel, params);
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
              << 2e-6 * M * N * sizeof(Element) / time_ms << " GB/s)"
              << std::endl;
  }

  //
  // Verify
  //

  h_D = d_D;

  int good = 0, bad = 0;

  for (size_t i = 0; i < h_D.size(); ++i) {
    if (h_D[i] == h_S[i])
      good++;
    else
      bad++;
  }

  std::cout << "Success " << good << ", Fail " << bad << std::endl;

  return 0;
}
