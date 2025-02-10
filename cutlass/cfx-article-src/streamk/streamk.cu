/******************************************************************************
 * Copyright (c) 2024 Colfax Research                                         *
 ******************************************************************************/

#include "cli_options.h"
#include "hopper_gemm_kernel_launch.h"
#include "kernel_traits.h"
#include "tile_scheduler.hpp"

int main(int argc, char const* argv[]) {
  cudaDeviceProp props;
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (props.major != 9) {
    std::cout << "This example requires NVIDIA's Hopper Architecture GPU with compute capability 90a\n" << std::endl;
    return 0;
  }
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  Options options;

  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }
  if (options.seed == 0)
    srand(time(NULL));
  else
    srand(options.seed);

  using TA = cute::half_t;
  using TB = cute::half_t;
  using TC = cute::half_t;
  using TI = float;




  if (options.task == "profile")
    run_one<TA, TB, TC, TI>(options);
  else if (options.task == "validate")
    validate<TA, TB, TC, TI>(options);
  else if (options.task == "time")
    time_gemm_and_print<TA, TB, TC, TI>(options, false);
  else if (options.task == "print_layouts")
    print_layouts<TA, TB, TC>(options);
  else if (options.task == "print_schedule") {
    switch (options.scheduler_num) {
      case 0:
        print_work_schedule<cfx::SingleTileScheduler>(options.m, options.n, options.k, 114);
        break;
      case 1:
        print_work_schedule<cfx::DataParallelPersistentTileScheduler>(options.m, options.n, options.k, 114);
        break;
      case 2:
        using KernelTraits = Kernel_traits<256, 192, 128, 12, 2, /*ClusterM=*/1, /*ClusterN=*/1, TA, TC>;
        print_sk_work_schedule<cfx::StreamKPersistentTileScheduler<KernelTraits>>(options.m, options.n, options.k, 114);
        break;
      case 3:
        using KernelTraits = Kernel_traits<256, 192, 128, 12, 2, /*ClusterM=*/1, /*ClusterN=*/1, TA, TC>;
        print_sk_work_schedule<cfx::StreamKPersistentTileScheduler<KernelTraits, true>>(options.m, options.n, options.k, 114);
        break;
      default:
        std::cout << "Unknown tile scheduler number " << options.scheduler_num << std::endl;
    }
  } else if (options.task == "quantization_csv") {
    std::cout << "m,n,k,tiles,waves,partial_wave_size,time,tflops,iters\n";
    int m = 1024;
    for (int n = 192; n < 114 * 192; n += 192) {
      for (int k = 2048; k <= 8192; k += 2048) {
        options.m = m;
        options.n = n;
        options.k = k;
        time_gemm_and_print<TA, TB, TC, TI>(options, true);
      }
    }
  }
  else
    options.print_usage(std::cout);

#else

  std::cout << "CUTLASS_ARCH_MMA_SM90_SUPPORTED must be enabled, but it is not. Test is waived \n" << std::endl;
#endif

  return 0;
}
