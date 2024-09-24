/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

//
// Composable warp-specialized kernel design adapted from FlashAttention-3 code.
// 

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"

namespace cfx {

using namespace cute;

template <typename Ktraits>
struct CollectiveMainloop {

    using Element = typename Ktraits::Element;
    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using ClusterShape = typename Ktraits::ClusterShape_MNK;
    using BarrierType = typename Ktraits::BarrierType;

    static constexpr int kStages = Ktraits::kStages;
    
    using SmemLayoutA = typename Ktraits::SmemLayoutA;
    using SmemLayoutB = typename Ktraits::SmemLayoutB;

    using ShapeT = cute::Shape<int32_t, int32_t>;
    using StrideT = cute::Shape<int32_t, _1>;
    using LayoutT = cute::Layout<ShapeT, StrideT>;
    
    using TMA_A = decltype(make_tma_copy(
        cute::SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr(static_cast<Element const*>(nullptr)), 
            ShapeT{}, 
            StrideT{}
        ),
        take<0, 2>(SmemLayoutA{}),
        select<0, 2>(TileShape_MNK{}),
        _1{}));  // no mcast

    using TMA_B = decltype(make_tma_copy(
        cute::SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr(static_cast<Element const*>(nullptr)), 
            ShapeT{}, 
            StrideT{}
        ),
        take<0, 2>(SmemLayoutB{}),
        select<1, 2>(TileShape_MNK{}),
        _1{})); // no mcast

    // static constexpr int NumMmaThreads = size(typename Ktraits::TiledMma{});
    static constexpr int NumMmaThreads = Ktraits::NumMmaThreads;
    using MainloopPipeline = typename Ktraits::MainloopPipeline;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;

    // Set the bytes transferred in this TMA transaction (may involve multiple issues)
    static constexpr uint32_t TmaTransactionBytesA =
        static_cast<uint32_t>(size(take<0, 2>(SmemLayoutA{})) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesB =
        static_cast<uint32_t>(size(take<0, 2>(SmemLayoutB{})) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytes = TmaTransactionBytesA + TmaTransactionBytesB;
    
    // Host side kernel arguments
    struct Arguments {
        Element const* ptr_A;
        LayoutT layout_A;
        Element const* ptr_B;
        LayoutT layout_B;
    };

    // Device side kernel params
    struct Params {
        LayoutT layout_A;
        LayoutT layout_B;
        TMA_A tma_load_A;
        TMA_B tma_load_B;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        Tensor mA = make_tensor(make_gmem_ptr(args.ptr_A), args.layout_A);
        Tensor mB = make_tensor(make_gmem_ptr(args.ptr_B), args.layout_B);
        TMA_A tma_load_A = make_tma_copy(
            cute::SM90_TMA_LOAD{},
            mA,
            SmemLayoutA{}(_, _, _0{}),
            select<0, 2>(TileShape_MNK{}),
            _1{}); // no mcast
        TMA_B tma_load_B = make_tma_copy(
            cute::SM90_TMA_LOAD{},
            mB,
            SmemLayoutB{}(_, _, _0{}),
            select<1, 2>(TileShape_MNK{}),
            _1{}); // no mcast
        return {args.layout_A, args.layout_B,
                tma_load_A, tma_load_B};
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& mainloop_params) {
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_A.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_B.get_tma_descriptor());
    }

    template <typename Scheduler, typename SharedStorage>
    CUTLASS_DEVICE void
    load(Params const& mainloop_params,
         MainloopPipeline pipeline,
         PipelineState& smem_pipe_write,
         SharedStorage &shared_storage,
         Scheduler& scheduler,
         typename Scheduler::Params const& scheduler_params,
         typename Scheduler::WorkTileInfo& work_tile_info,
         cute::tuple<int32_t, int32_t, int32_t> block_coord,
         int work_idx,
         int k_tile_count
         ) {

        // Fetch logical block coordinates
        auto [m_block, n_block, bidb] = block_coord;

        // Define SMEM tensors
        Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem_A.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
        Tensor sB = make_tensor(make_smem_ptr(shared_storage.smem_B.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

        // Define GMEM tensors as TMA tensors
        Tensor mA = mainloop_params.tma_load_A.get_tma_tensor(mainloop_params.layout_A.shape());
        Tensor mB = mainloop_params.tma_load_B.get_tma_tensor(mainloop_params.layout_B.shape());

        // Get CTA views of GMEM
        Tensor gA = local_tile(mA, select<0, 2>(TileShape_MNK{}), make_coord(blockIdx.x, _));  // (BLK_M,BLK_K,k)
        Tensor gB = local_tile(mB, select<1, 2>(TileShape_MNK{}), make_coord(blockIdx.y, _));  // (BLK_N,BLK_K,k)

        // Partition the copying of A and B tiles
        auto [tAgA, tAsA] = tma_partition(mainloop_params.tma_load_A, Int<0>{}, Layout<_1>{},
                                group_modes<0,2>(sA), group_modes<0,2>(gA));  // (TMA,k) and (TMA,PIPE)
        auto [tBgB, tBsB] = tma_partition(mainloop_params.tma_load_B, Int<0>{}, Layout<_1>{},
                                group_modes<0,2>(sB), group_modes<0,2>(gB));  // (TMA,k) and (TMA,PIPE)

        // DO TMA LOAD from a single thread
        // VERSION 1: no prologue-epilogue overlapping
        int lane_predicate = cute::elect_one_sync();
        if(lane_predicate) {
            // MAINLOOP LOADS
            CUTLASS_PRAGMA_NO_UNROLL
            for(int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
                pipeline.producer_acquire(smem_pipe_write);
                BarrierType *tmaBar = pipeline.producer_get_barrier(smem_pipe_write);
                auto stage = smem_pipe_write.index();
                copy(mainloop_params.tma_load_A.with(*tmaBar, 0), tAgA(_, k_tile), tAsA(_, stage));
                copy(mainloop_params.tma_load_B.with(*tmaBar, 0), tBgB(_, k_tile), tBsB(_, stage));
                ++smem_pipe_write;
            }
        }

        // TODO VERSION 2: overlap prologue B load with epilogue
        // DO first TMA load A.
        // shared_storage.barrier_C.wait((work_idx + 1) % 2);
        // In mainloop, DO TMA load B then next TMA load A.
        // scheduler.prefetch_next_work(scheduler_params, work_tile_info);
        // In last iteration, DO final TMA load B.
        // scheduler.broadcast_next_work(work_tile_info);
    }

    /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
    CUTLASS_DEVICE void
    load_tail(MainloopPipeline pipeline,
              PipelineState& smem_pipe_write) {
        int lane_predicate = cute::elect_one_sync();
        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        // Issue the epilogue waits
        if (warp_idx_in_warpgroup == 0 && lane_predicate) {
          /* This helps avoid early exit of blocks in Cluster
          * Waits for all stages to either be released (all Consumer UNLOCKs), or if the stage was never used
          * then would just be acquired since the phase was still inverted from make_producer_start_state
          */
          pipeline.producer_tail(smem_pipe_write);
        }
    }

    template <typename SharedStorage, typename FrgTensorC>
    CUTLASS_DEVICE void
    mma(Params const& mainloop_params,
        MainloopPipeline pipeline,
        PipelineState& smem_pipe_read,
        FrgTensorC& tCrC,
        int thread_idx,
        int work_idx,
        SharedStorage& shared_storage,
        int k_tile_count
        ) {
        
        Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem_A.data()), SmemLayoutA{});
        Tensor sB = make_tensor(make_smem_ptr(shared_storage.smem_B.data()), SmemLayoutB{});

        typename Ktraits::TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
        
        Tensor tCsA = thr_mma.partition_A(sA);  // (MMA,MMA_M,MMA_K,PIPE)
        Tensor tCsB = thr_mma.partition_B(sB);  // (MMA,MMA_N,MMA_K,PIPE)

        // Allocate "fragments" -- these are WGMMA matrix descriptors
        Tensor tCrA = thr_mma.make_fragment_A(tCsA);  // (MMA,MMA_M,MMA_K,PIPE)
        Tensor tCrB = thr_mma.make_fragment_B(tCsB);  // (MMA,MMA_N,MMA_K,PIPE)

        // TODO VERSION 2
        // call arrive on barrier_C for prologue-epilogue overlapping
        // if (cutlass::canonical_warp_idx_sync() == Ktraits::kNWarps - 1 && lane_predicate) {
        //     tma_store_wait<0>();
        //     shared_storage.barrier_C.arrive(0, lane_predicate);
        // }
        // int k_tile = k_tile_count-1;

        // MAINLOOP MMA
        CUTLASS_PRAGMA_NO_UNROLL
        for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
            // Wait for TMA to load this stage of the pipeline
            pipeline.consumer_wait(smem_pipe_read);
            auto stage = smem_pipe_read.index();
        
            warpgroup_arrive();
            // WGMMA with dispatch mode (V,M,K) x (V,N,K) => (V,M,N)
            gemm(tiled_mma, tCrA(_,_,_,stage), tCrB(_,_,_,stage), tCrC);
            warpgroup_commit_batch();
        
            // Wait for all MMAs in a K_TILE to complete
            warpgroup_wait<0>();

            // Release the stage of the pipeline for TMA
            pipeline.consumer_release(smem_pipe_read);
            ++smem_pipe_read;
        }

        // Make sure all warpgroups have finished mma
        cutlass::arch::NamedBarrier::sync(NumMmaThreads, 0);

    }

};

} // namespace cfx
