/******************************************************************************
 * Copyright (c) 2024 Colfax Research                                         *
 ******************************************************************************/
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include <cute/layout.hpp>
#include <cute/pointer.hpp>
#include <cute/tensor.hpp>
#include <cute/container/tuple.hpp>
#include <cute/util/print.hpp>
#include <cutlass/array.h>
#include <cutlass/barrier.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/detail/helper_macros.hpp>
#include <cutlass/fast_math.h>
#include <cutlass/gemm_coord.h>

#include "kernel_traits.h"

namespace cfx {

///////////////////////////////////////////////////////////////////////////////

class SingleTileScheduler {

public:
    // Host side kernel arguments
    struct Arguments {
        int const num_blocks_m, num_blocks_n, num_blocks_k;
        void* workspace = nullptr;
    };

    // Device side kernel params
    struct Params {
        int mn_blocks;
        int num_blocks_k;
    };

    Params const params;

    static Params
    to_underlying_arguments(Arguments const& args) {
        return {args.num_blocks_m * args.num_blocks_n, args.num_blocks_k};
    }

    static dim3
    get_grid_dim(Arguments const& args, int num_sm) {
        return {uint32_t(args.num_blocks_m), uint32_t(args.num_blocks_n)};
    }

    static size_t get_workspace_size(int, int) { return 0; }
    static cudaError_t initialize_workspace(int, int, void*, cudaStream_t) { return cudaSuccess; }

    struct WorkTileInfo {
        int M_idx = 0;
        int N_idx = 0;
        bool valid = false;

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            return {M_idx, N_idx, 0, params.num_blocks_k};
        }

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const { return valid; }

        CUTLASS_DEVICE
        bool constexpr
        compute_epilogue(Params const& params) const { return true; }

        CUTLASS_DEVICE
        bool constexpr
        needs_fixup(Params const& params) const { return false; }

    };

    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work() const {
        return {int(blockIdx.x), int(blockIdx.y), true};
    }

    template<bool IsProducer=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(WorkTileInfo const& current_work) const {
        return {-1, -1, false};
    }

    CUTLASS_DEVICE
    SingleTileScheduler(Params const& params_) : params(params_) {}


    template <class AccumTensor>
    CUTLASS_DEVICE
    void
    fixup(WorkTileInfo const& worktile,
          AccumTensor& accumulators,
          uint32_t barrier_idx,
          uint32_t thread_idx) const {}
};

///////////////////////////////////////////////////////////////////////////////

class DataParallelPersistentTileScheduler {

public:
    // Host side kernel arguments
    struct Arguments {
        int const num_blocks_m, num_blocks_n, num_blocks_k;
        void* workspace = nullptr;
    };

    // Device side kernel params
    struct Params {
        int mn_blocks;
        int num_blocks_k;
        cutlass::FastDivmod m_block_divmod;
    };

    Params const params;

    static Params
    to_underlying_arguments(Arguments const& args) {
        return {args.num_blocks_m * args.num_blocks_n,
                args.num_blocks_k,
                cutlass::FastDivmod(args.num_blocks_m)};
    }

    static dim3
    get_grid_dim(Arguments const& args, int num_sm) {
        return {uint32_t(num_sm)};
    }

    static size_t get_workspace_size(int, int) { return 0; }
    static cudaError_t initialize_workspace(int, int, void*, cudaStream_t) { return cudaSuccess; }

    struct WorkTileInfo {
        int tile_idx;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return tile_idx < params.mn_blocks;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            int m_block, n_block;
            params.m_block_divmod(n_block, m_block, tile_idx);
            return {m_block, n_block, 0, params.num_blocks_k};
        }

        CUTLASS_DEVICE
        bool constexpr
        compute_epilogue(Params const& params) const { return true; }

        CUTLASS_DEVICE
        bool constexpr
        needs_fixup(Params const& params) const { return false; }

    };

    CUTLASS_DEVICE
    DataParallelPersistentTileScheduler(Params const& params_) : params(params_) {}

    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work() const {
        return {int(blockIdx.x)};
    }

    template<bool IsProducer=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(WorkTileInfo const& current_work) const {
        return {current_work.tile_idx + int(gridDim.x)};
    }

    template <class AccumTensor>
    CUTLASS_DEVICE
    void
    fixup(WorkTileInfo const& worktile,
          AccumTensor& accumulators,
          uint32_t barrier_idx,
          uint32_t thread_idx) const {}

};

////////////////////////////////////////////////////////////////////

template <class KernelTraits, bool Heuristic = false>
class StreamKPersistentTileScheduler {

public:
    using ElementType = typename KernelTraits::Element;
    using ElementAccum = typename KernelTraits::ElementAccum;
    using TileShape = typename KernelTraits::TileShape_MNK;
    using BarrierType = typename cutlass::NamedBarrierManager<1>::T;

    // Minimum size of a stream-K work assignment
    static constexpr int min_k_blocks = 8;

    // per-CTA data members
    int sk_k_block_cta_start;
    int sk_k_block_cta_stop;

    // Host side kernel arguments
    struct Arguments {
        int const num_blocks_m, num_blocks_n, num_blocks_k;
        void* workspace;
    };

    // Device side kernel params
    struct Params {
        int const mn_blocks;
        cutlass::FastDivmod k_block_divmod, major_block_divmod;
        int const dp_tiles;
        int const sk_work_k_block_offset;
        int const sk_k_blocks_per_cta;
        bool const raster_along_m;
        void* const workspace;
    };

    Params const params;

    static Params
    to_underlying_arguments(Arguments const& args) {
        int device;
        cudaGetDevice(&device);
        int num_sms;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);

        int mn_blocks = args.num_blocks_m * args.num_blocks_n;
        int sk_tiles = get_num_sk_tiles(mn_blocks, num_sms);
        int dp_tiles = mn_blocks - sk_tiles;

        bool raster_along_m = (args.num_blocks_m < args.num_blocks_n);
        int major_block = raster_along_m ? args.num_blocks_m : args.num_blocks_n;

        // We linearly index all stream-K K-blocks, and divide them
        // evenly between the SMs.
        int sk_work_k_block_offset = dp_tiles * args.num_blocks_k;
        int sk_k_blocks = sk_tiles * args.num_blocks_k;
        int sk_k_blocks_per_cta = max(cutlass::ceil_div(sk_k_blocks, num_sms), min_k_blocks);

        return {mn_blocks,
                cutlass::FastDivmod(args.num_blocks_k),
                cutlass::FastDivmod(major_block),
                dp_tiles,
                sk_work_k_block_offset,
                sk_k_blocks_per_cta,
                raster_along_m,
                args.workspace};
    }

    static int
    get_num_sk_tiles(int mn_blocks, int num_sms) {
        // We implement the hybrid schedule, where the last <2 full waves
        // are done as stream-K work, and the rest as data-parallel work.
        int full_waves = mn_blocks / num_sms;
        int dp_waves = (full_waves <= 1) ? 0 : full_waves - 1;
        int dp_tiles = dp_waves * num_sms;
        int sk_tiles = mn_blocks - dp_tiles;
        if constexpr (Heuristic) {
          // fall back to data-parallel if the tail wave is > half full,
          // or if sk_tiles is divisible by num_sms.
          if (sk_tiles % num_sms == 0 || (sk_tiles % num_sms > num_sms / 2)) {
              sk_tiles = 0;
          }
        }
        return sk_tiles;
    }

    static dim3
    get_grid_dim(Arguments const& args, int num_sm) {
        return {uint32_t(num_sm)};
    }


    static size_t
    get_barrier_workspace_size(int sk_tiles) {
        // For each SK output tile and each consumer warpgroup, warpgroups
        // collaborating on that tile synchronize through a barrier.
        return sizeof(BarrierType) * sk_tiles * KernelTraits::Num_WG_MMA;
    }

    static size_t
    CUTLASS_HOST_DEVICE
    get_reduction_workspace_size(int sk_tiles) {
        // For each SK output tile, a bM x bN workspace tensor holds partial
        // results for that tile. (Different consumer warpgroups in the same CTA
        // write to different parts of this tile.)
        return sk_tiles * cute::size<0>(TileShape{}) * cute::size<1>(TileShape{}) * sizeof(ElementAccum);
    }

    static size_t
    get_workspace_size(int mn_blocks, int num_sms) {
        int sk_tiles = get_num_sk_tiles(mn_blocks, num_sms);
        return get_barrier_workspace_size(sk_tiles) + get_reduction_workspace_size(sk_tiles);
    }

    static cudaError_t
    initialize_workspace(int mn_blocks, int num_sms, void* workspace, cudaStream_t stream) {
        // Barriers have to be initialized to zero. We can ignore the reduction workspace.
        // We place the barrier workspace after the reduction workspace.
        int sk_tiles = get_num_sk_tiles(mn_blocks, num_sms);
        size_t reduction_workspace_size = get_reduction_workspace_size(sk_tiles);
        uint8_t* barrier_workspace = reinterpret_cast<uint8_t*>(workspace) + reduction_workspace_size;
        size_t barrier_workspace_size = get_barrier_workspace_size(sk_tiles);
        return cudaMemsetAsync(barrier_workspace, 0, barrier_workspace_size, stream);
    }

    struct WorkTileInfo {
        int m_block;
        int n_block;
        int k_block_start;
        int k_block_stop;

        CUTLASS_DEVICE
        int
        mn_block_idx(Params const& params) const {
          if (params.raster_along_m)
            return (n_block * params.major_block_divmod.divisor + m_block);
          else
            return (m_block * params.major_block_divmod.divisor + n_block);
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
          return {m_block, n_block, k_block_start, k_block_stop};
        }

        CUTLASS_DEVICE
        WorkTileInfo(Params const& params, int dp_work_idx) {
            // Constructor for an entire tile of data-parallel work
            if (params.raster_along_m)
              params.major_block_divmod(n_block, m_block, dp_work_idx);
            else
              params.major_block_divmod(m_block, n_block, dp_work_idx);
            k_block_start = 0;
            k_block_stop = params.k_block_divmod.divisor;
        }

        CUTLASS_DEVICE
        WorkTileInfo(Params const& params, int sk_block_idx, int sk_k_block_cta_start, int sk_k_block_cta_stop) {

            if (sk_block_idx >= sk_k_block_cta_stop) {
                m_block = n_block = k_block_start = k_block_stop = -1;
            } else {
                int mn_block;
                params.k_block_divmod(mn_block, k_block_start, sk_block_idx);
                if (params.raster_along_m)
                  params.major_block_divmod(n_block, m_block, mn_block);
                else
                  params.major_block_divmod(m_block, n_block, mn_block);

                int num_k_blocks = params.k_block_divmod.divisor;
                k_block_stop = min(num_k_blocks, k_block_start + (sk_k_block_cta_stop - sk_block_idx));
            }

            // if (threadIdx.x == 128) {
            //   if (is_valid())
            //     print("Defined valid worktile at (%d, %d, %d, %d), block index %d, CTA %d\n",
            //             m_block, n_block,
            //             k_block_start, k_block_stop, sk_block_idx, blockIdx.x);
            //   else
            //     print("Defined invalid worktile at (%d, %d, %d, %d), block index %d, CTA %d\n",
            //             m_block, n_block,
            //             k_block_start, k_block_stop, sk_block_idx, blockIdx.x);
            // }
        }

        CUTLASS_DEVICE
        WorkTileInfo()
            // Constructor for invalid worktiles
            : m_block(-1), n_block(-1), k_block_start(-1), k_block_stop(-1) {}

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return k_block_start >= 0;
        }

        CUTLASS_DEVICE
        bool
        compute_epilogue(Params const& params) const {
            // The CTA working on the FIRST k-block in a tile is assigned to compute the epilogue.
            // This avoids some synchronization overhead, since the rest of the tile belongs to
            // the BEGINNING of another CTA's work assignment.
            return k_block_start == 0;
        }

        CUTLASS_DEVICE
        bool
        needs_fixup(Params const& params) const {
            // Work units that represent a complete worktile can skip the fixup method.
            return k_block_stop - k_block_start < params.k_block_divmod.divisor;
        }

    };

    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work() const {
        if (blockIdx.x < params.dp_tiles)
          return {params, static_cast<int>(blockIdx.x)}; // data-parallel, total worktile
        else if (sk_k_block_cta_stop > sk_k_block_cta_start)
            return {params, sk_k_block_cta_start, sk_k_block_cta_start, sk_k_block_cta_stop};
        return {};
    }

    template<bool IsProducer=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(WorkTileInfo const& current_work) {
      int current_mn_block = current_work.mn_block_idx(params);
      if (current_mn_block + static_cast<int>(gridDim.x) < params.dp_tiles)
          // next work in data-parallel region
          return {params, current_mn_block + static_cast<int>(gridDim.x)};
      else if (current_mn_block < params.dp_tiles)
          // last work in data-parallel region, continue to first one in stream-K region
          return {params, sk_k_block_cta_start, sk_k_block_cta_start, sk_k_block_cta_stop};
      else {
          // already in stream-K region, continue to next work in stream_K region
          int current_mnk_block = current_mn_block * params.k_block_divmod.divisor + current_work.k_block_stop;
          return {params, current_mnk_block, sk_k_block_cta_start, sk_k_block_cta_stop};
      }
    }

    CUTLASS_DEVICE
    StreamKPersistentTileScheduler(Params const& params_) : params(params_) {
      sk_k_block_cta_start =
          params.sk_work_k_block_offset
          + params.sk_k_blocks_per_cta * blockIdx.x;
      sk_k_block_cta_stop = sk_k_block_cta_start + params.sk_k_blocks_per_cta;

      // Adjust for too-small work assignments. If a work assignment starts or ends
      // close to the end of a block, transfer that piece to the adjacent assignment.
      int num_k_blocks = params.k_block_divmod.divisor;
      int k_block_start, k_block_stop;
      params.k_block_divmod.divmod(k_block_start, sk_k_block_cta_start);
      params.k_block_divmod.divmod(k_block_stop, sk_k_block_cta_stop);
      if (num_k_blocks - k_block_start < min_k_blocks && k_block_start > 0) {
        // transfer to previous
        sk_k_block_cta_start += num_k_blocks - k_block_start;
        params.k_block_divmod.divmod(k_block_start, sk_k_block_cta_start);
      }
      if (num_k_blocks - k_block_stop < min_k_blocks && k_block_stop != 0) {
        // absorb from next
        sk_k_block_cta_stop += num_k_blocks - k_block_stop;
        params.k_block_divmod.divmod(k_block_stop, sk_k_block_cta_stop);
      }
      if (k_block_stop < min_k_blocks && k_block_stop != 0)
        // transfer to next
        sk_k_block_cta_stop -= k_block_stop;
      if (k_block_start < min_k_blocks && k_block_start > 0)
        // absorb from previous
        sk_k_block_cta_start -= k_block_start;

      int total_blocks = params.mn_blocks * params.k_block_divmod.divisor;
      sk_k_block_cta_stop = min(sk_k_block_cta_stop, total_blocks);
      sk_k_block_cta_start = min(sk_k_block_cta_start, total_blocks);
    }

    template <class AccumTensor>
    CUTLASS_DEVICE
    void
    fixup(WorkTileInfo const& worktile,
          AccumTensor& accumulators,
          uint32_t barrier_idx,
          uint32_t thread_idx) const {
        /* Note that since the only operation we do with the workspace is elementwise
         addition, we can treat it as a raw array rather than a tensor. The actual
         organization of the partial outputs among the CTA's threads doesn't matter,
         as long as they end up in the same places for the CTA which computes the
         epilogue. */
        using namespace cute;

        if (!worktile.needs_fixup(params)) // total worktile, skip fixup
            return;

        static constexpr int num_barriers = KernelTraits::Num_WG_MMA;
        using BarrierManager = cutlass::NamedBarrierManager<
            cutlass::NumThreadsPerWarpGroup,
                static_cast<uint32_t>(cutlass::arch::ReservedNamedBarriers::StreamkBarrier0),
                num_barriers>;
        static constexpr uint32_t ThreadCount = BarrierManager::ThreadCount; // 1 warpgroup
        size_t reduction_workspace_size = get_reduction_workspace_size(params.mn_blocks - params.dp_tiles);
        BarrierType* barrier_workspace = reinterpret_cast<BarrierType*>(
            reinterpret_cast<uint8_t*>(params.workspace) + reduction_workspace_size);
        size_t tile_idx = worktile.mn_block_idx(params) - params.dp_tiles;
        int32_t lock_idx = tile_idx * num_barriers + barrier_idx;
        uint32_t barrier_group_thread_idx = thread_idx % BarrierManager::ThreadCount;

        using ElementAccumulator = typename AccumTensor::value_type;
        // using AccumulatorArray = cutlass::Array<ElementAccumulator, size(AccumTensor{})>;
        ElementAccumulator* reduction_workspace = reinterpret_cast<ElementAccumulator*>(params.workspace);
        ElementAccumulator* reduction_tile =
            reduction_workspace
            + tile_idx * cute::size<0>(TileShape{}) * cute::size<1>(TileShape{});

        // Because of how work assignments are constructed, we expect partial worktiles
        // with large k-coordinates to finish first. So we have the barriers count backwards:
        // each arriving worker increments the barrier by the number of k-blocks it computed,
        // and each worker waits until the barrier count reaches the number of k-blocks
        // after its k_block_stop.
        // Note that thread_idx counts threads in all MMA warpgroups, starting from 0
        if (worktile.k_block_stop == params.k_block_divmod.divisor) {
            // Last tile in k-mode = expected first tile to arrive.
            // write to workspace, then arrive at barrier
            CUTLASS_PRAGMA_UNROLL
            for (uint i = 0; i < size(AccumTensor{}); ++i)
                reduction_tile[(ThreadCount * num_barriers * i) + thread_idx] = accumulators[i];
            BarrierManager::arrive_inc(barrier_idx, barrier_workspace, barrier_group_thread_idx,
                                       lock_idx, worktile.k_block_stop - worktile.k_block_start);
        } else if (!worktile.compute_epilogue(params)) {
            // 1st, ..., (n-1)st tiles: wait at barrier for previous tile,
            // reduce into workspace, arrive at barrier
            BarrierManager::wait_eq(barrier_idx, barrier_workspace, barrier_group_thread_idx,
                                    lock_idx, params.k_block_divmod.divisor - worktile.k_block_stop);
            CUTLASS_PRAGMA_UNROLL
            for (uint i = 0; i < size(AccumTensor{}); ++i)
                reduction_tile[(ThreadCount * num_barriers * i) + thread_idx] += accumulators[i];
            BarrierManager::arrive_inc(barrier_idx, barrier_workspace, barrier_group_thread_idx,
                                       lock_idx, worktile.k_block_stop - worktile.k_block_start);
        } else {
            // nth tile: wait at barrier for previous tile, reduce into registers
            BarrierManager::wait_eq(barrier_idx, barrier_workspace, barrier_group_thread_idx,
                                    lock_idx, params.k_block_divmod.divisor - worktile.k_block_stop);
            CUTLASS_PRAGMA_UNROLL
            for (uint i = 0; i < size(AccumTensor{}); ++i)
                accumulators[i] += reduction_tile[(ThreadCount * num_barriers * i) + thread_idx];
        }
    }

};

} // namespace cfx

//////////////////////////////////////////////////////////////////////////////

template <class TileScheduler>
__global__
void print_work_schedule_device(typename TileScheduler::Params scheduler_params, int* dummy) {
  TileScheduler scheduler(scheduler_params);
  // if (threadIdx.x == 0)
  //   cute::print("%d blocks per SM, this block from block %d to %d", scheduler.k_blocks_per_SM, scheduler.k_block_cta_start, scheduler.k_block_cta_stop);
  for (typename TileScheduler::WorkTileInfo work = scheduler.get_initial_work();
       work.is_valid(scheduler_params);
       work = scheduler.get_next_work(work)) {
    auto [m_block, n_block, k_block_start, k_block_stop] = work.get_block_coord(scheduler_params);
    if (threadIdx.x == 0) {
            dummy += blockIdx.x;
            cute::print("CTA at (%d, %d, %d): work tile (%d, %d), k block start %d, k block stop %d, %s, %s\n",
                        blockIdx.x, blockIdx.y, blockIdx.z, m_block, n_block, k_block_start, k_block_stop,
                        work.needs_fixup(scheduler_params)? "needs fixup" : "does not need fixup",
                        work.compute_epilogue(scheduler_params) ? "compute epilogue" : "do not compute epilogue");
    }
    __syncthreads();
  }
}

template <class TileScheduler_>
void print_work_schedule(int m, int n, int k, int num_sms) {
  // using KernelTraits = Kernel_traits<256, 192, 128, 12, 2, /*ClusterM=*/1, /*ClusterN=*/1, cute::half_t, cute::half_t>;
  // using TileScheduler = cfx::StreamKPersistentTileScheduler<KernelTraits>;
  using TileScheduler = TileScheduler_;
  int num_blocks_m = cutlass::ceil_div(m, 256);
  int num_blocks_n = cutlass::ceil_div(n, 192);
  int num_blocks_k = cutlass::ceil_div(k, 128);
  typename TileScheduler::Arguments scheduler_args {num_blocks_m, num_blocks_n, num_blocks_k, nullptr};
  typename TileScheduler::Params scheduler_params = TileScheduler::to_underlying_arguments(scheduler_args);
  dim3 grid_dim = TileScheduler::get_grid_dim(scheduler_args, num_sms);
  dim3 block_dim(384, 1, 1); // 12 warps
  cute::print("grid dim (%d, %d, %d)\n", grid_dim.x, grid_dim.y, grid_dim.z);
  cute::print("%d blocks_m, %d blocks_n, %d output tiles\n", num_blocks_m, num_blocks_n, scheduler_params.mn_blocks);
  int *dummy;
  cudaMalloc(&dummy, sizeof(int));
  CUTE_CHECK_LAST();
  print_work_schedule_device<TileScheduler><<<grid_dim, block_dim>>>(scheduler_params, dummy);
  CUTE_CHECK_LAST();
  cudaFree(dummy);
  CUTE_CHECK_LAST();
}

template <class TileScheduler_>
void print_sk_work_schedule(int m, int n, int k, int num_sms) {
  // using KernelTraits = Kernel_traits<256, 192, 128, 12, 2, /*ClusterM=*/1, /*ClusterN=*/1, cute::half_t, cute::half_t>;
  // using TileScheduler = cfx::StreamKPersistentTileScheduler<KernelTraits>;
  using TileScheduler = TileScheduler_;
  int num_blocks_m = cutlass::ceil_div(m, 256);
  int num_blocks_n = cutlass::ceil_div(n, 192);
  int num_blocks_k = cutlass::ceil_div(k, 128);
  typename TileScheduler::Arguments scheduler_args {num_blocks_m, num_blocks_n, num_blocks_k, nullptr};
  typename TileScheduler::Params scheduler_params = TileScheduler::to_underlying_arguments(scheduler_args);
  dim3 grid_dim = TileScheduler::get_grid_dim(scheduler_args, num_sms);
  dim3 block_dim(384, 1, 1); // 12 warps
  int sk_tiles = scheduler_params.mn_blocks - scheduler_params.dp_tiles;
  cute::print("grid dim (%d, %d, %d)\n", grid_dim.x, grid_dim.y, grid_dim.z);
  cute::print("%d blocks_m, %d blocks_n, %d blocks_k, %d output tiles, raster along %s\n",
              num_blocks_m, num_blocks_n,
              scheduler_params.k_block_divmod.divisor, scheduler_params.mn_blocks,
              scheduler_params.raster_along_m ? "M" : "N");
  cute::print("%d DP tiles, %d SK tiles, SK work starts at k-block %d, %d k-blocks per CTA\n",
              scheduler_params.dp_tiles, sk_tiles, scheduler_params.sk_work_k_block_offset,
              scheduler_params.sk_k_blocks_per_cta);
  int *dummy;
  cudaMalloc(&dummy, sizeof(int));
  CUTE_CHECK_LAST();
  print_work_schedule_device<TileScheduler><<<grid_dim, block_dim>>>(scheduler_params, dummy);
  CUTE_CHECK_LAST();
  cudaFree(dummy);
  CUTE_CHECK_LAST();
}
