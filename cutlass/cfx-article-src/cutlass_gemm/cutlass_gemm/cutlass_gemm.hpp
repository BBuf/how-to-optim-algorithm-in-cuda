/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#ifndef COMPILE_3X_HOPPER

// CUTLASS 2.X syntax GEMM
// Adapted from https://github.com/NVIDIA/cutlass/blob/main/examples/00_basic_gemm/basic_gemm.cu

#include <cutlass/gemm/device/gemm.h>

template<typename DataType, typename OutputType> void cutlass_gemm_wrapper(int M, int N, int K, DataType const* ptrA, DataType const* ptrB, OutputType* ptrC) {
  using Gemm = cutlass::gemm::device::Gemm<
    DataType,                     // ElementA
    cutlass::layout::RowMajor,    // LayoutA
    DataType,                     // ElementB
    cutlass::layout::RowMajor,    // LayoutB
    OutputType,                     // ElementOutput
    cutlass::layout::RowMajor,    // LayoutOutput
    float                         // ElementAccumulator
  >;

  float alpha = 1.0f;
  float beta = 0.0f;

  int lda = M;
  int ldb = K;
  int ldc = M;

  Gemm gemm_op;
  gemm_op({
    {M, N, K},
    {ptrA, lda},            // TensorRef to A device tensor
    {ptrB, ldb},            // TensorRef to B device tensor
    {ptrC, ldc},            // TensorRef to C device tensor
    {ptrC, ldc},            // TensorRef to D device tensor - may be the same as C
    {alpha, beta}           // epilogue operation arguments
  });
}

#else

// CUTLASS 3.X syntax GEMM
// Adapted from https://github.com/NVIDIA/cutlass/blob/main/examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

template<typename DataType, typename OutputType> void cutlass_gemm_wrapper(int M, int N, int K, DataType const* ptrA, DataType const* ptrB, OutputType* ptrC) {


  // A matrix configuration
  using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
  constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<DataType>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

  // B matrix configuration
  using         LayoutB     = cutlass::layout::RowMajor;                   // Layout type for B matrix operand
  constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<DataType>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

  // C/D matrix configuration
  using         LayoutC     = cutlass::layout::RowMajor;                   // Layout type for C and D matrix operands
  constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<OutputType>::value;

  // Core kernel configurations
  using ElementAccumulator  = float;                                          // Element type for internal accumulation
  using ArchTag             = cutlass::arch::Sm90;                            // Tag indicating the minimum SM that supports the intended feature
  using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
  using TilesShape          = Shape<_128,_128,_64>;                           // Threadblock-level tile size
  using ClusterShape        = Shape<_1,_2,_1>;                                // Shape of the threadblocks in a cluster
  using StageCountType = cutlass::gemm::collective::StageCountAuto;           // Stage count maximized based on the tile size
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;       // Kernel to launch based on the default setting in the Collective Builder 

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      DataType, LayoutA, AlignmentA,
      DataType, LayoutB, AlignmentB,
      ElementAccumulator,
      TilesShape, ClusterShape,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TilesShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    OutputType, LayoutC, AlignmentC,
    OutputType, LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int>, // Indicates ProblemShape
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  Gemm gemm_op;
  cutlass::Status status;

  //
  // Define the problem size
  //

  float alpha = 1.00f;
  float beta = 0.0f;

  //
  // Allocate device memory
  //
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, Int<1>{}));
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, Int<1>{}));
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, Int<1>{}));
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, Int<1>{}));

  //
  // Launch GEMM on the device
  //
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {ptrA, stride_A, ptrB, stride_B},
    {{alpha, beta}, ptrC, stride_C, ptrC, stride_D}
  };
  
  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  gemm_op.can_implement(arguments);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  gemm_op.initialize(arguments, workspace.get());

  // Correctness / Warmup iteration
  gemm_op.run();

}
#endif





