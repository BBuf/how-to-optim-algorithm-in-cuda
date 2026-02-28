/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

//
// Composable warp-specialized kernel design adapted from FlashAttention-3 code.
// 

#pragma once

#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"

using namespace cute;

// writeout RMEM -> GMEM
template <int kStages, class ElementA, class ElementB,
          class SmemLayoutA, class SmemLayoutB>
struct SharedStorageAB {
    
    array_aligned<ElementA, cosize_v<SmemLayoutA>> smem_A;
    array_aligned<ElementB, cosize_v<SmemLayoutB>> smem_B;
    
    struct {
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline;
    };
};

// writeout RMEM -> SMEM -> GMEM
template <int kStages, class ElementA, class ElementB, class ElementC,
          class SmemLayoutA, class SmemLayoutB, class SmemLayoutC>
struct SharedStorageABC {
    
    array_aligned<ElementA, cosize_v<SmemLayoutA>> smem_A;
    union {
        array_aligned<ElementB, cosize_v<SmemLayoutB>> smem_B;
        array_aligned<ElementC, cosize_v<SmemLayoutC>> smem_C;
    };

    struct {
        cutlass::arch::ClusterBarrier barrier_C;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline;
        int tile_count_semaphore;
    };
};


template<int kBlockM_, int kBlockN_, int kBlockK_,
         int kNWarps_, int kStages_, int kClusterM_ = 1, int kClusterN_ = 1,
         typename Element_ = cutlass::half_t, typename OutputType_ = Element_, bool fp32_accum = true>
struct Kernel_traits {
    using Element = Element_;
    // using ElementAccum = float;
    using ElementAccum = std::conditional_t<fp32_accum, float, Element>;
    using OutputType = OutputType_;
    using index_t = int64_t;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNumThreads = kNWarps * cutlass::NumThreadsPerWarp;
    static constexpr int NumMmaThreads = kNumThreads - 128;
    // Use one warp in producer warpgroup for TMA
    static constexpr int NumProducerThreads = cutlass::NumThreadsPerWarp;

    static constexpr int Num_WG_MMA = (kNWarps / 4) - 1;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kBlockK = kBlockK_;
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kBlockK>>;

    static constexpr int kClusterM = kClusterM_;
    static constexpr int kClusterN = kClusterN_;
    using ClusterShape_MNK = Shape<Int<kClusterM>, Int<kClusterN>, _1>;

    static constexpr int kStages = kStages_;

    using AtomLayoutMNK = Layout<Shape<Int<Num_WG_MMA>, _1, _1>>;
    // using AtomLayoutMNK = Layout<Shape<Int<kBlockM/64>, _1, _1>>;
    using TiledMma = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK>(),
        AtomLayoutMNK{}));
    // using TiledMma =
    //     decltype(make_tiled_mma(SM90_64x128x16_F16F16F16_SS<GMMA::Major::K,GMMA::Major::K>{},
    //         Layout<Shape<_2,_1,_1>>{}));

    using SmemLayoutAtomA =
        decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
            decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutA =
        decltype(tile_to_shape(SmemLayoutAtomA{},
            make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));

    using SmemLayoutAtomB =
        decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
            decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutB =
        decltype(tile_to_shape(SmemLayoutAtomA{},
            make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));

    using SmemLayoutAtomC =
        decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, OutputType,
            decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<1>(TileShape_MNK{}))>());
    using SmemLayoutC =
        decltype(tile_to_shape(SmemLayoutAtomC{}, select<0, 1>(TileShape_MNK{})));

    using SmemCopyAtomC = Copy_Atom<cute::SM90_U32x4_STSM_N, OutputType>;
    
    using SharedStorage =
        SharedStorageABC<kStages, Element, Element, OutputType,
            SmemLayoutA, SmemLayoutB, SmemLayoutC>;

    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
    using PipelineState = typename cutlass::PipelineState<kStages>;
    using BarrierType = typename MainloopPipeline::ProducerBarrierType;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
