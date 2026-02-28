/******************************************************************************
 * Copyright (c) 2024 Colfax Research                                         *
 ******************************************************************************/
#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/numeric_types.h"

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

#include "cli_options.h"
#include "kernel_traits.h"
#include "tile_scheduler.hpp"
#include "mainloop_sm90_tma_gmma_ws.hpp"
#include "epilogue_sm90_tma_ws.hpp"
#include "hopper_gemm_kernel.h"

template <class TA, class TB, class TC,
          class Alpha, class Beta,
          class TileScheduler,
          bool validation = false,
          bool print_layouts_only = false>
void gemm_tn(int M, int N, int K,
    Alpha alpha,
    TA const* A, int ldA,
    TB const* B, int ldB,
    Beta beta,
    TC      * C, int ldC,
    int iters = 0,
    bool csv = false,
    cudaStream_t stream = 0) {
    // LEGEND:
    // kBlockM, kBlockN, kBlockK, kNWarps, kStages, ClusterM, ClusterN
    // operand dtype, output dtype, use fp32 accum

    // fp16 accum optimal sizes
    // using Kernel_traits = std::conditional_t<!validation,
    //   Kernel_traits<256, 256, 128-32, 20, 2, /*ClusterM=*/1, /*ClusterN=*/2, TA, TC, false>,
    //   // Kernel_traits<128 + 64, 256, 128, 12 + 4, 2, 1, 1, TA, TC, false>,
    //   Kernel_traits<128, 256, 128, 12, 2, 1, 1, TA, TC, false> // validation params
    // >;

    // fp32 accum optimal sizes
    using Kernel_traits = std::conditional_t<!validation,
    // bM, bN, bK, warps, stages, ClusterM, ClusterN, TA, TC
      Kernel_traits<256, 192, 128, 12, 2, /*ClusterM=*/1, /*ClusterN=*/1, TA, TC>,
      // Kernel_traits<128, 256, 128, 12, 2, 2, 1, TA, TC>,
      // Kernel_traits<128, 256, 64, 12, 4, 2, 1, TA, TC>,
      // Kernel_traits<128, 256, 96, 12, 3, 2, 1, TA, TC>,
      // Kernel_traits<192, 192, 128, 16, 2, 1, 2, TA, TC>,
      Kernel_traits<128, 256, 128, 12, 2, 1, 1, TA, TC> // validation params
    >;

    // std::cout << "Num threads = " << Kernel_traits::kNumThreads << std::endl;
    // std::cout << "Num Mma threads = " << Kernel_traits::NumMmaThreads << std::endl;
    // using TiledMMA = typename Kernel_traits::TiledMma;
    // std::cout << "Size of tiled mma = " << size(TiledMMA{}) << std::endl;

    auto smem_layout_A = typename Kernel_traits::SmemLayoutA{};
    auto smem_layout_B = typename Kernel_traits::SmemLayoutB{};
    auto smem_layout_C = typename Kernel_traits::SmemLayoutC{};

    using TileShape_MNK = typename Kernel_traits::TileShape_MNK;
    using ClusterShape = typename Kernel_traits::ClusterShape_MNK;

    if (K % Kernel_traits::kBlockK != 0) {
        print("K not divisible by bK = %d\n", Kernel_traits::kBlockK);
        return;
    }

      using CollectiveMainloop = cfx::CollectiveMainloop<Kernel_traits>;
    using CollectiveEpilogue = cfx::CollectiveEpilogue<Kernel_traits>;
    using Scheduler = std::conditional_t<validation,
      cfx::SingleTileScheduler,
      TileScheduler>;

    typename CollectiveMainloop::Params mainloop_params =
        CollectiveMainloop::to_underlying_arguments({
            A,
            make_layout(make_shape(M, K), make_stride(ldA, Int<1>{})),  // layout_A
            B,
            make_layout(make_shape(N, K), make_stride(ldB, Int<1>{})),  // layout_B
        });

    typename CollectiveEpilogue::Params epilogue_params =
        CollectiveEpilogue::to_underlying_arguments({
            C,
            make_layout(make_shape(M, N), make_stride(ldC, Int<1>{}))
        });


    int num_blocks_m = cutlass::ceil_div(M, Kernel_traits::kBlockM);
    int num_blocks_n = cutlass::ceil_div(N, Kernel_traits::kBlockN);
    int num_blocks_k = cutlass::ceil_div(K, Kernel_traits::kBlockK);
    // round if using clusters
    num_blocks_m = cutlass::ceil_div(num_blocks_m, size<0>(ClusterShape{})) * size<0>(ClusterShape{});
    num_blocks_n = cutlass::ceil_div(num_blocks_n, size<1>(ClusterShape{})) * size<1>(ClusterShape{});

    int device;
    cudaGetDevice(&device);
    CUTE_CHECK_LAST();
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
    CUTE_CHECK_LAST();

    auto workspace_size = Scheduler::get_workspace_size(num_blocks_m * num_blocks_n, num_sms);
    void* workspace;
    cudaMalloc(&workspace, workspace_size);
    CUTE_CHECK_LAST();

    Scheduler::initialize_workspace(num_blocks_m * num_blocks_n, num_sms, workspace, stream);
    CUTE_CHECK_LAST();

    typename Scheduler::Arguments scheduler_args = {num_blocks_m, num_blocks_n, num_blocks_k, workspace};
    typename Scheduler::Params scheduler_params = Scheduler::to_underlying_arguments(scheduler_args);

    // Get the ptr to kernel function.
    void *kernel;
    kernel = (void *)cfx::hopper_gemm_ws<Kernel_traits, Scheduler>;
    int smem_size = sizeof(typename Kernel_traits::SharedStorage);
    if (smem_size >= 48 * 1024) {
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
      CUTE_CHECK_LAST();
    }

    dim3 grid_dims = Scheduler::get_grid_dim(scheduler_args, num_sms);
    static constexpr int ctaSize = Kernel_traits::kNWarps * 32;
    dim3 block_dims(ctaSize);
    dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
    cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size, stream};

    if constexpr(print_layouts_only) {
      auto layout_A = mainloop_params.layout_A;
      auto tma_A = mainloop_params.tma_load_A;
      auto layout_B = mainloop_params.layout_B;
      auto tma_B = mainloop_params.tma_load_B;
      auto layout_C = epilogue_params.layout_C;
      auto tma_store = epilogue_params.tma_store;
      print("GMEM Layout A: "); print(layout_A); print("\n");
      print("GMEM Layout B: "); print(layout_B); print("\n");
      print("GMEM Layout C: "); print(layout_C); print("\n");
      print("TMA Load A   : "); print(tma_A); print("\n");
      print("TMA Load B   : "); print(tma_B); print("\n");
      print("TMA Store    : "); print(tma_store); print("\n");
      print("SMEM Layout A: "); print(smem_layout_A); print("\n");
      print("SMEM Layout B: "); print(smem_layout_B); print("\n");
      print("SMEM Layout C: "); print(smem_layout_C); print("\n");
      int smem_size_A = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_A));
      int smem_size_B = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_B));
      int smem_size_C = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_C));
      print("SMEM size    : A = %d, B = %d, C = %d, total = %d\n",
            smem_size_A, smem_size_B, smem_size_C, smem_size);
      print("Grid size    : (%d, %d, %d)\n", grid_dims.x, grid_dims.y, grid_dims.z);
      print("Logical grid : (%d, %d)\n", num_blocks_m, num_blocks_n);
      print("Cluster size : (%d, %d, %d)\n", cluster_dims.x, cluster_dims.y, cluster_dims.z);
      print("Block size   : (%d, %d, %d)\n", block_dims.x, block_dims.y, block_dims.z);
      print("SMs          : %d\n", num_sms);
      return;
    }

    cutlass::launch_kernel_on_cluster(
        launch_params, kernel,
        mainloop_params, epilogue_params, scheduler_params);


    GPU_Clock timer;

    if (iters > 0) {
        timer.start();
        for (int i = 0; i < iters; ++i) {
            Scheduler::initialize_workspace(num_blocks_m * num_blocks_n, num_sms, workspace, stream);

            cutlass::launch_kernel_on_cluster(
                launch_params, kernel,
                mainloop_params, epilogue_params, scheduler_params);
        }
        int worktiles = (M / Kernel_traits::kBlockM) * (N / Kernel_traits::kBlockN);
        int waves = (worktiles + num_sms - 1) / num_sms;
        int partial_wave_size = worktiles % num_sms;

        double avg_time = timer.seconds() / iters;
        double tflops = (2.0 * M * N * K) * 1e-12;
        if (csv)
            print("%d,%d,%d,%d,%d,%d,%.4f,%.1f,%d\n",
                  M, N, K, worktiles, waves, partial_wave_size, avg_time * 1000, tflops / avg_time, iters);
        else {
            print("TN GEMM, M x N x K = %d x %d x %d\n", M, N, K);
            print("Time:     [%6.1f]TFlop/s  %6.4fms  (average of %d iterations)\n",
                  tflops / avg_time, avg_time*1000, iters);
        }
    }
    CUTE_CHECK_LAST();
    cudaFree(workspace);
    CUTE_CHECK_LAST();
}

template <class TA, class TB, class TC, class TI, bool validation=false, bool print_layouts_only=false>
void dispatch_on_scheduler(int M, int N, int K, TI alpha,
                           TA const* A, int ldA,
                           TB const* B, int ldB,
                           TI beta,
                           TC* C, int ldC,
                           int scheduler_num,
                           int iters = 0,
                           bool csv = false,
                           cudaStream_t stream = 0) {
  switch (scheduler_num) {
  case 0:
    gemm_tn<TA, TB, TC, TI, TI, cfx::SingleTileScheduler, validation, print_layouts_only>(M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC, iters, csv, stream);
    break;
  case 1:
    gemm_tn<TA, TB, TC, TI, TI, cfx::DataParallelPersistentTileScheduler, validation, print_layouts_only>(M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC, iters, csv, stream);
    break;
  case 2:
    using KernelTraits = Kernel_traits<256, 192, 128, 12, 2, /*ClusterM=*/1, /*ClusterN=*/1, TA, TC>;
    gemm_tn<TA, TB, TC, TI, TI, cfx::StreamKPersistentTileScheduler<KernelTraits>, validation, print_layouts_only>(M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC, iters, csv, stream);
    break;
  case 3:
    using KernelTraits = Kernel_traits<256, 192, 128, 12, 2, /*ClusterM=*/1, /*ClusterN=*/1, TA, TC>;
    gemm_tn<TA, TB, TC, TI, TI, cfx::StreamKPersistentTileScheduler<KernelTraits, true>, validation, print_layouts_only>(M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC, iters, csv, stream);
    break;
  default:
    std::cout << "Unknown tile scheduler number " << scheduler_num << std::endl;
  }
}


template <class TA, class TB, class TC, class TI>
void time_gemm_and_print(Options const& options, bool csv = false) {
  int m = options.m;
  int n = options.n;
  int k = options.k;
  TI alpha = options.alpha;
  TI beta = options.beta;
  int scheduler_num = options.scheduler_num;

  thrust::host_vector<TA> h_A(m*k);
  thrust::host_vector<TB> h_B(n*k);
  thrust::host_vector<TC> h_C(m*n);

  for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TB>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < m*n; ++j) h_C[j] = TC(0);

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TB> d_B = h_B;
  thrust::device_vector<TC> d_C = h_C;

  int ldA = k, ldB = k, ldC = n;

  d_C = h_C;

  dispatch_on_scheduler<TA, TB, TC, TI>
      (m, n, k, alpha, d_A.data().get(), ldA, d_B.data().get(), ldB, beta, d_C.data().get(), ldC,
       scheduler_num, options.timing_iterations, csv);

}

template <class TA, class TB, class TC, class TI>
void run_one(Options const& options) {
  int m = options.m;
  int n = options.n;
  int k = options.k;
  TI alpha = options.alpha;
  TI beta = options.beta;
  int scheduler_num = options.scheduler_num;

  thrust::host_vector<TA> h_A(m*k);
  thrust::host_vector<TB> h_B(n*k);
  thrust::host_vector<TC> h_C(m*n);

  for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TB>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < m*n; ++j) h_C[j] = static_cast<TC>(0);

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TB> d_B = h_B;
  thrust::device_vector<TC> d_C = h_C;

  int ldA = k, ldB = k, ldC = n;

  d_C = h_C;
  dispatch_on_scheduler(m, n, k, alpha, d_A.data().get(), ldA,
                        d_B.data().get(), ldB, beta, d_C.data().get(), ldC,
                        scheduler_num);
  CUTE_CHECK_LAST();
}

template <class TA, class TB, class TC, class TI>
void validate(Options const& options) {
  int m = options.m;
  int n = options.n;
  int k = options.k;
  TI alpha = options.alpha;
  TI beta = options.beta;
  int scheduler_num = options.scheduler_num;

  thrust::host_vector<TA> h_A(m*k);
  thrust::host_vector<TB> h_B(n*k);
  thrust::host_vector<TC> h_C(m*n);
  thrust::host_vector<TC> h_Cref(m * n);

  for (int j = 0; j < m * k; ++j) h_A[j] = static_cast<TA>(2 * (rand() / double(RAND_MAX)) - 1);
  for (int j = 0; j < n * k; ++j) h_B[j] = static_cast<TB>(2 * (rand() / double(RAND_MAX)) - 1);
  // for (int j = 0; j < m * k; ++j) h_A[j] = static_cast<TA>(1.0);
  // for (int j = 0; j < n * k; ++j) h_B[j] = static_cast<TB>(1.0);
  for (int j = 0; j < m * n; ++j) h_C[j] = static_cast<TC>(0);
  for (int j = 0; j < m * n; ++j) h_Cref[j] = static_cast<TC>(0);

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TB> d_B = h_B;
  thrust::device_vector<TC> d_C = h_C;
  thrust::device_vector<TC> d_Cref = h_Cref;

  int ldA = k, ldB = k, ldC = n;

  dispatch_on_scheduler(m, n, k, alpha, d_A.data().get(), ldA,
                        d_B.data().get(), ldB, beta, d_C.data().get(), ldC,
                        scheduler_num);

  CUTE_CHECK_LAST();

  h_C = d_C;

  dispatch_on_scheduler<TA, TB, TC, TI, true>(m, n, k, alpha, d_A.data().get(), ldA,
                        d_B.data().get(), ldB, beta, d_Cref.data().get(), ldC,
                        scheduler_num);

  CUTE_CHECK_LAST();

  h_Cref = d_Cref;

  TC eps = std::numeric_limits<TC>::epsilon();
  float size_factor = std::log(k);
  float max_abs_diff = 0;
  float max_rel_diff = 0;
  int abs_index = 0, rel_index = 0;
  for (int j = 0; j < m*n; ++j) {
    float abs_diff = std::abs(h_C[j] - h_Cref[j]);
    float rel_diff = abs_diff / (std::abs(h_Cref[j]) + eps);
    if (abs_diff > max_abs_diff) { max_abs_diff = abs_diff; abs_index = j;}
    if (rel_diff > max_rel_diff) { max_rel_diff = rel_diff; rel_index = j;}
  }

  if(max_abs_diff > eps * size_factor || max_rel_diff > eps) {
    std::cout << "Validation error.\n";
    std::cout << "max abs diff = " << max_abs_diff << " at index " << abs_index
              << ", selected = " << h_C[abs_index]
              << ", reference = " << h_Cref[abs_index] << std::endl;
    std::cout << "max rel diff = " << max_rel_diff << " at index " << rel_index
              << ", selected = " << h_C[rel_index] << ", reference = " << h_Cref[rel_index]
              << std::endl;
    std::cout << "First 20 elements:\n";
    std::cout << "Selected: \n";
    for (int i = 0; i < 20; ++i)
      std::cout << h_C[i] << " ";
    std::cout << "\nReference: \n";
    for (int i = 0; i < 20; ++i)
      std::cout << h_Cref[i] << " ";
    std::cout << std::endl;
  } else {
    std::cout << "Validation passed!\n";
  }

}

template <class TA, class TB, class TC>
void print_layouts(Options const& options) {
  int m = options.m;
  int n = options.n;
  int k = options.k;
  int scheduler_num = options.scheduler_num;

  thrust::host_vector<TA> h_A(m*k);
  thrust::host_vector<TB> h_B(n*k);
  thrust::host_vector<TC> h_C(m*n);
  thrust::host_vector<TC> h_Cref(m * n);

  for (int j = 0; j < m * k; ++j) h_A[j] = static_cast<TA>(2 * (rand() / double(RAND_MAX)) - 1);
  for (int j = 0; j < n * k; ++j) h_B[j] = static_cast<TB>(2 * (rand() / double(RAND_MAX)) - 1);
  for (int j = 0; j < m * n; ++j) h_C[j] = static_cast<TC>(0);
  for (int j = 0; j < m * n; ++j) h_Cref[j] = static_cast<TC>(0);

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TB> d_B = h_B;
  thrust::device_vector<TC> d_C = h_C;
  thrust::device_vector<TC> d_Cref = h_Cref;

  int ldA = k, ldB = k, ldC = n;

  using TI = float;
  TI alpha {1.0f};
  TI beta  {0.0f};

  dispatch_on_scheduler<TA, TB, TC, TI, false, true>(m, n, k, alpha, d_A.data().get(), ldA,
                        d_B.data().get(), ldB, beta, d_C.data().get(), ldC,
                        scheduler_num);
}
