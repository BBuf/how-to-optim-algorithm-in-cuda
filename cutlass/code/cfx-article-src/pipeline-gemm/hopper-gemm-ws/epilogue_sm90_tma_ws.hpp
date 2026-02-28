/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

//
// Composable warp-specialized kernel design adapted from FlashAttention-3 code.
// 

#pragma once

#include <cutlass/cutlass.h>
#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"

#include "convert_util.h"

namespace cfx {

using namespace cute;

template <typename Ktraits>
struct CollectiveEpilogue {

    using Element = typename Ktraits::OutputType;    
    // static constexpr int kBlockM = Ktraits::kBlockM;
    // static constexpr int kBlockN = Ktraits::kBlockN;
    // static constexpr int kBlockK = Ktraits::kBlockK;
    // using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kBlockK>>;
    using TileShape_MNK = typename Ktraits::TileShape_MNK;

    static constexpr int kNWarps = Ktraits::kNWarps;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;    

    static constexpr int NumCopyThreads = cutlass::NumThreadsPerWarpGroup;
    // static constexpr int NumMmaThreads = kNThreads - NumCopyThreads;
     static constexpr int NumMmaThreads = Ktraits::NumMmaThreads;

    using ShapeT = cute::Shape<int32_t, int32_t>;
    using StrideT = cute::Shape<int32_t, _1>;
    using LayoutT = cute::Layout<ShapeT, StrideT>;

    using SmemLayoutC = typename Ktraits::SmemLayoutC;

    // for RMEM -> SMEM
    using SmemCopyAtomC = typename Ktraits::SmemCopyAtomC;

    using TMA_C = decltype(make_tma_copy(
        cute::SM90_TMA_STORE{},
        make_tensor(
            make_gmem_ptr(static_cast<Element*>(nullptr)), 
            ShapeT{}, 
            StrideT{}
        ),
        SmemLayoutC{},
        select<0, 1>(TileShape_MNK{}),
        _1{}));  // no mcast for O

    // Host side kernel arguments
    struct Arguments {
        Element* ptr_C;
        LayoutT const layout_C;
    };

    // Device side kernel params
    struct Params {
        Element* ptr_C;
        LayoutT const layout_C;
        TMA_C tma_store;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        Tensor mC = make_tensor(make_gmem_ptr(args.ptr_C), args.layout_C);
        TMA_C tma_store = make_tma_copy(
            cute::SM90_TMA_STORE{},
            mC,
            SmemLayoutC{},
            select<0, 1>(TileShape_MNK{}),
            _1{}); // no mcast for C
        return {args.ptr_C, args.layout_C, tma_store};
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& epilogue_params) {
        cute::prefetch_tma_descriptor(epilogue_params.tma_store.get_tma_descriptor());
    }

    template <typename SharedStorage, typename FrgTensorC, typename TiledMma>
    CUTLASS_DEVICE void
    store(Params const& epilogue_params,
          FrgTensorC const& tCrC,
          SharedStorage& shared_storage,
          TiledMma tiled_mma,
          int thread_idx,
          cute::tuple<int32_t, int32_t, int32_t> const& block_coord
          ) {

        // RMEM -> SMEM COPY
        auto [m_block, n_block, bidb] = block_coord;

        Tensor sC = make_tensor(make_smem_ptr(shared_storage.smem_C.data()), SmemLayoutC{});
        auto smem_tiled_copy_C = make_tiled_copy_C(SmemCopyAtomC{}, tiled_mma);
        auto smem_thr_copy_C = smem_tiled_copy_C.get_thread_slice(thread_idx);
        
        // cast from AccumType to OutputType
        Tensor tCrC_out = convert_type<Element>(tCrC);
        Tensor taccCrC = smem_thr_copy_C.retile_S(tCrC_out);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccCsC = smem_thr_copy_C.partition_D(sC);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
        cute::copy(smem_tiled_copy_C, taccCrC, taccCsC);
        cutlass::arch::fence_view_async_shared(); // ensure smem writes are visible to TMA
        cutlass::arch::NamedBarrier::arrive(NumMmaThreads + cutlass::NumThreadsPerWarp,
                                            cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

        // Prepare TMA store

        Tensor mC = epilogue_params.tma_store.get_tma_tensor(epilogue_params.layout_C.shape());
        Tensor gC = local_tile(mC, select<0,1>(TileShape_MNK{}), make_coord(m_block, n_block));
        
        auto block_tma_store = epilogue_params.tma_store.get_slice(_0{}); // CTA slice
        Tensor tCgC = block_tma_store.partition_D(gC);  // (TMA, TMA_M, TMA_K)
        Tensor tCsC = block_tma_store.partition_S(sC);  // (TMA, TMA_M, TMA_K)
        
        // TMA STORE: SMEM -> GMEM
        int write_warp_idx = kNWarps - 1;
        int const warp_idx = cutlass::canonical_warp_idx_sync();
        int const lane_predicate = cute::elect_one_sync();
        if (warp_idx == write_warp_idx) {
            // Ensure RMEM -> SMEM copy completes before issuing TMA store
            cutlass::arch::NamedBarrier::sync(
                NumMmaThreads + cutlass::NumThreadsPerWarp, 
                cutlass::arch::ReservedNamedBarriers::EpilogueBarrier
            );
        }
        if (warp_idx == write_warp_idx && lane_predicate) {
            cute::copy(epilogue_params.tma_store, tCsC, tCgC);
            tma_store_arrive();
        }
        // TODO: overlap epilogue with next CTA load in persistent kernel
        // tma_store_wait<0>();
    }

    CUTLASS_DEVICE void
    store_tail() {
        tma_store_wait<0>();
    }

};

} // namespace cfx
