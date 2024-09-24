#pragma once
/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// copy kernel adapted from https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/tiled_copy.cu

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

template <class TensorS, class TensorD, class ThreadLayout, class VecLayout>
__global__ static void __launch_bounds__(256, 1)
    copyKernel(TensorS const S, TensorD const D, ThreadLayout, VecLayout) {
  using namespace cute;
  using Element = typename TensorS::value_type;

  Tensor gS = S(make_coord(_, _), blockIdx.x, blockIdx.y);   // (bM, bN)
  Tensor gD = D(make_coord(_, _), blockIdx.x, blockIdx.y); // (bN, bM)

  // Define `AccessType` which controls the size of the actual memory access.
  using AccessType = cutlass::AlignedArray<Element, size(VecLayout{})>;

  // A copy atom corresponds to one hardware memory access.
  using Atom = Copy_Atom<UniversalCopy<AccessType>, Element>;

  // Construct tiled copy, a tiling of copy atoms.
  //
  // Note, this assumes the vector and thread layouts are aligned with contigous data
  // in GMEM. Alternative thread layouts are possible but may result in uncoalesced
  // reads. Alternative vector layouts are also possible, though incompatible layouts
  // will result in compile time errors.
  auto tiled_copy =
    make_tiled_copy(
      Atom{},                       // access size
      ThreadLayout{},               // thread layout
      VecLayout{});                 // vector layout (e.g. 4x1)

  // Construct a Tensor corresponding to each thread's slice.
  auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);

  Tensor tSgS = thr_copy.partition_S(gS);             // (CopyOp, CopyM, CopyN)
  Tensor tDgD = thr_copy.partition_D(gD);             // (CopyOp, CopyM, CopyN)

  Tensor rmem = make_tensor_like(tSgS);               // (ThrValM, ThrValN)

  copy(tSgS, rmem);
  copy(rmem, tDgD);
}

template <typename T> void copy_baseline(TransposeParams<T> params) {

  using Element = float;
  using namespace cute;

  //
  // Make tensors
  //
  auto tensor_shape = make_shape(params.M, params.N);
  auto gmemLayoutS = make_layout(tensor_shape, LayoutRight{});
  auto gmemLayoutD = make_layout(tensor_shape, LayoutRight{});
  Tensor tensor_S = make_tensor(make_gmem_ptr(params.input), gmemLayoutS);
  Tensor tensor_D = make_tensor(make_gmem_ptr(params.output), gmemLayoutD);
 
  //
  // Tile tensors
  //
  using bM = Int<32>;
  using bN = Int<1024>;

  auto block_shape = make_shape(bM{}, bN{});       // (bM, bN)

  Tensor tiled_tensor_S =
      tiled_divide(tensor_S, block_shape); // ((bM, bN), m', n')
  Tensor tiled_tensor_D =
      tiled_divide(tensor_D, block_shape); // ((bN, bM), n', m')

  auto threadLayout =
      make_layout(make_shape(Int<8>{}, Int<32>{}), LayoutRight{});

  auto vec_layout = make_layout(make_shape(Int<4>{}, Int<1>{}));

  //
  // Determine grid and block dimensions
  //

  dim3 gridDim(
      size<1>(tiled_tensor_S),
      size<2>(tiled_tensor_S)); // Grid shape corresponds to modes m' and n'
  dim3 blockDim(size(threadLayout)); // 256 threads

  copyKernel<<<gridDim, blockDim>>>(tiled_tensor_S, tiled_tensor_D,
                                       threadLayout,  vec_layout);
}
