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
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

#include "cutlass/detail/layout.hpp"

#include "shared_storage.h"
#include "util.h"

using namespace cute;

template <class TensorS, class TensorD, class ThreadLayoutS, class ThreadLayoutD>
__global__ static void __launch_bounds__(256, 1)
transposeKernelNaive(TensorS const S, TensorD const DT,
                ThreadLayoutS const tS, ThreadLayoutD const tD) {
  using Element = typename TensorS::value_type;

  Tensor gS = S(make_coord(_, _), blockIdx.x, blockIdx.y);   // (bM, bN)
  Tensor gDT = DT(make_coord(_, _), blockIdx.x, blockIdx.y); // (bN, bM)

  Tensor tSgS = local_partition(gS, tS, threadIdx.x); // (ThrValM, ThrValN)
  Tensor tDgDT = local_partition(gDT, tD, threadIdx.x);

  Tensor rmem = make_tensor_like(tSgS);

  copy(tSgS, rmem);
  copy(rmem, tDgDT);
}

template <typename Element> void transpose_naive(TransposeParams<Element> params) {
  
  //
  // Make Tensors
  //
  auto tensor_shape = make_shape(params.M, params.N);
  auto tensor_shape_trans = make_shape(params.N, params.M);
  auto gmemLayoutS = make_layout(tensor_shape, LayoutRight{});
  auto gmemLayoutD = make_layout(tensor_shape_trans, LayoutRight{});
  Tensor tensor_S = make_tensor(make_gmem_ptr(params.input), gmemLayoutS);
  Tensor tensor_D = make_tensor(make_gmem_ptr(params.output), gmemLayoutD);
  
  // Make a transposed view of the output
  auto gmemLayoutDT = make_layout(tensor_shape, GenColMajor{});
  Tensor tensor_DT = make_tensor(make_gmem_ptr(params.output), gmemLayoutDT);
  
  //
  // Tile tensors
  //
  
  using bM = Int<64>;
  using bN = Int<64>;
  
  auto block_shape = make_shape(bM{}, bN{});       // (bM, bN)
  auto block_shape_trans = make_shape(bN{}, bM{}); // (bN, bM)
  
  Tensor tiled_tensor_S = tiled_divide(tensor_S, block_shape); // ((bM, bN), m', n')
  Tensor tiled_tensor_DT = tiled_divide(tensor_DT, block_shape_trans); // ((bN, bM), n', m')
  
  auto threadLayoutS =
      make_layout(make_shape(Int<8>{}, Int<32>{}), LayoutRight{});
  auto threadLayoutD =
      make_layout(make_shape(Int<8>{}, Int<32>{}), LayoutRight{});
  
  dim3 gridDim(
      size<1>(tiled_tensor_S),
      size<2>(tiled_tensor_S)); // Grid shape corresponds to modes m' and n'
  dim3 blockDim(size(threadLayoutS)); // 256 threads
  transposeKernelNaive<<<gridDim, blockDim>>>(tiled_tensor_S, tiled_tensor_DT,
                                            threadLayoutS, threadLayoutD);
};
