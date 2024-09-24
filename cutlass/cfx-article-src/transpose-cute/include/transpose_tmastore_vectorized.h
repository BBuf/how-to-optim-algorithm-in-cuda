#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

#include "cutlass/detail/layout.hpp"

#include "shared_storage.h"
#include "smem_helper.hpp"

using namespace cute;

template <class TensorS, class SmemLayout, class TiledCopyS, class TiledCopyD,
          class GmemLayoutD, class TileShapeD, class ThreadLayoutM,
          class SmemLayoutM>
__global__ static void __launch_bounds__(256)
    transposeKernelTMA(TensorS const S, SmemLayout const smemLayout,
                       TiledCopyS const tiled_copy_S,
                       CUTE_GRID_CONSTANT TiledCopyD const tmaStoreD,
                       GmemLayoutD const gmemLayoutD,
                       TileShapeD const tileShapeD, ThreadLayoutM const tM,
                       SmemLayoutM const smemLayoutM) {
  using namespace cute;
  using Element = typename TensorS::value_type;

  int lane_predicate = cute::elect_one_sync();
  int warp_idx = cutlass::canonical_warp_idx_sync();
  bool leaderWarp = warp_idx == 0;

  // Use Shared Storage structure to allocate aligned SMEM addresses.
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorageTranspose<Element, SmemLayout>;
  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(shared_memory);
  Tensor sM =
      make_tensor(make_smem_ptr(shared_storage.smem.data()), smemLayoutM);

  Tensor gS = S(make_coord(_, _), blockIdx.x, blockIdx.y); // (bM, bN)
  auto thr_copy_S = tiled_copy_S.get_thread_slice(threadIdx.x);

  Tensor tSgS = thr_copy_S.partition_S(gS); // (CopyOp, CopyM, CopyN)
  Tensor tSrS = make_fragment_like(tSgS);   // (CopyOp, CopyM, CopyN)
  Tensor tMsM = local_partition(sM, tM, threadIdx.x);

  // Copy from GMEM to RMEM to SMEM
  copy(tiled_copy_S, tSgS, tSrS);
  copy(tSrS, tMsM);

  auto synchronize = [&]() {
    cutlass::arch::NamedBarrier::sync(size(ThreadLayoutM{}), 0);
  };
  cutlass::arch::fence_view_async_shared();
  synchronize();

  // Issue the TMA store.
  Tensor mD = tmaStoreD.get_tma_tensor(shape(gmemLayoutD));
  auto blkCoordD = make_coord(blockIdx.y, blockIdx.x);
  Tensor gD = local_tile(mD, tileShapeD, blkCoordD);
  Tensor sD = make_tensor(make_smem_ptr(shared_storage.smem.data()),
                          smemLayout); // (bN, bM)

  auto cta_tmaD = tmaStoreD.get_slice(0);

  Tensor tDgDX = cta_tmaD.partition_D(gD);
  Tensor tDgD = group_modes<1, rank(tDgDX)>(tDgDX); // (TMA,REST)
  assert(size<1>(tDgD) == 1);

  Tensor tDsDX = cta_tmaD.partition_S(sD);
  Tensor tDsD = group_modes<1, rank(tDsDX)>(tDsDX); // (TMA,REST)
  static_assert(size<1>(tDsD) == 1);

  if (leaderWarp and lane_predicate) {
    copy(tmaStoreD, tDsD, tDgD);
  }
  // Wait for TMA store to complete.
  tma_store_wait<0>();
}

template <typename Element> void transpose_tma(TransposeParams<Element> params) {
//  printf("Vectorized load into registers, write out via TMA Store\n");
//  printf("Profiler reports uncoalesced smem accesses\n");

  auto tensor_shape = make_shape(params.M, params.N);
  auto tensor_shape_trans = make_shape(params.N, params.M);
  auto gmemLayoutS = make_layout(tensor_shape, LayoutRight{});
  auto gmemLayoutD = make_layout(tensor_shape_trans, LayoutRight{});
  Tensor tensor_S = make_tensor(make_gmem_ptr(params.input), gmemLayoutS);
  Tensor tensor_D = make_tensor(make_gmem_ptr(params.output), gmemLayoutD);

  //
  // Tile tensors
  //

  using bM = Int<32>;
  using bN = Int<32>;

  auto block_shape = make_shape(bM{}, bN{});       // (bM, bN)
  auto block_shape_trans = make_shape(bN{}, bM{}); // (bN, bM)

  Tensor tiled_tensor_S = tiled_divide(tensor_S, block_shape); // ((bM, bN), m', n')
  Tensor tiled_tensor_D = tiled_divide(tensor_D, block_shape_trans); // ((bN, bM), n', m')

  auto threadLayoutS =
      make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{});
  auto vecLayoutS = make_layout(make_shape(Int<1>{}, Int<4>{}));
  using AccessTypeS = cutlass::AlignedArray<Element, size(vecLayoutS)>;
  using AtomS = Copy_Atom<UniversalCopy<AccessTypeS>, Element>;
  auto tiled_copy_S = make_tiled_copy(AtomS{}, threadLayoutS, vecLayoutS);

  auto tileShapeD = block_shape_trans;
  auto smemLayoutD =
      tile_to_shape(cfx::getSmemLayoutK<Element, bM{}>(),
                    make_shape(shape<0>(tileShapeD), shape<1>(tileShapeD)));
  // TMA only supports certain swizzles
  // https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/copy_traits_sm90_tma_swizzle.hpp
  auto tmaD = make_tma_copy(SM90_TMA_STORE{}, tensor_D, smemLayoutD, tileShapeD,
                            Int<1>{});

  auto tileShapeM = make_shape(Int<4>{}, Int<8>{}, Int<32>{});
  auto smemLayoutM = composition(smemLayoutD, make_layout(tileShapeM));
  auto threadLayoutM = make_layout(make_shape(Int<1>{}, Int<8>{}, Int<32>{}),
                                   make_stride(Int<1>{}, Int<1>{}, Int<8>{}));

  size_t smem_size =
      int(sizeof(SharedStorageTranspose<Element, decltype(smemLayoutD)>));

  //
  // Determine grid and block dimensions
  //

  dim3 gridDim(
      size<1>(tiled_tensor_S),
      size<2>(tiled_tensor_S)); // Grid shape corresponds to modes m' and n'
  dim3 blockDim(size(threadLayoutS));

  transposeKernelTMA<<<gridDim, blockDim, smem_size>>>(
      tiled_tensor_S, smemLayoutD, tiled_copy_S, tmaD, gmemLayoutD, tileShapeD,
      threadLayoutM, smemLayoutM);
}
