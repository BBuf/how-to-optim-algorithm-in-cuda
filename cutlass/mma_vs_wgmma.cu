// Origin code from https://github.com/HydraQYH/NVGPUMicroBenchmark/blob/master/mma_vs_wgmma.cu
#include <cstdio>
#include <random>
#include <chrono>
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/device_kernel.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>

 // 这个文件对比两种路径：
 // 1) 经典 warp-level MMA（SM80 tensor core）直接从 global 读到寄存器 fragment 再做 gemm
 // 2) SM90 WGMMA：使用 TMA 把 A/B 搬到 shared memory，并用 cluster transaction barrier 同步后进行 warp-group MMA
 // 代码主要用于 micro-benchmark/正确性校验，不是一个完整的 GEMM kernel。

template <
    typename T_IN,
    typename T_OUT,
    typename TiledMma,
    typename SharedMemoryALayout,
    typename SharedMemoryBLayout,
    typename TensorA,
    typename TensorB,
    typename TensorC,
    typename TensorD,
    typename TmaA,
    typename TmaB,
    bool Accumulate = true>
__global__ void wgmma_kernel(
    TiledMma mma,
    TensorA gA,
    TensorB gB,
    TensorC gC,
    TensorD gD,
    CUTLASS_GRID_CONSTANT TmaA const tmaA,
    CUTLASS_GRID_CONSTANT TmaB const tmaB) {
  using namespace cute;

   // WGMMA 路径：
   // - 先用 TMA 把 gmem 的 A/B 复制到 shared memory 的 sA/sB
   // - 通过 ClusterTransactionBarrier 等待 TMA 完成
   // - 每个线程从 shared memory 分区到自己的 fragment，然后 warp-group 执行 GMMA

  extern __shared__ __align__(128) uint8_t shared_memory[];
   // 动态 shared memory 被手动切分为：
   // [sA | sB | mbar]
   // 其中 mbar 是 cluster transaction barrier 的存储（uint64_t）。
  Tensor sA = make_tensor(make_smem_ptr(reinterpret_cast<T_IN*>(shared_memory)), SharedMemoryALayout{});
  Tensor sB = make_tensor(make_smem_ptr(reinterpret_cast<T_IN*>(shared_memory) + cosize(SharedMemoryALayout{})), SharedMemoryBLayout{});
  uint64_t* mbar = reinterpret_cast<uint64_t*>(shared_memory + sizeof(T_IN) * (cosize(SharedMemoryALayout{}) + cosize(SharedMemoryBLayout{})));

   // 由 TMA descriptor 推导“可被 TMA 访问”的张量视图（mA/mB），shape 与 gA/gB 一致。
  Tensor mA = tmaA.get_tma_tensor(shape(gA));
  Tensor mB = tmaB.get_tma_tensor(shape(gB));

   // 这里只取 slice(0)：本例 dimGrid=1 且不做多 CTA 切片。
  auto cta_tmaA = tmaA.get_slice(0);
  auto cta_tmaB = tmaB.get_slice(0);
   // partition_S：把 source（global/TMA tensor）按当前 CTA/线程的视角分区
   // partition_D：把 destination（shared tensor）按当前 CTA/线程的视角分区
  Tensor tAgA = cta_tmaA.partition_S(mA);
  Tensor tAsA = cta_tmaA.partition_D(sA);
  Tensor tBgB = cta_tmaB.partition_S(mB);
  Tensor tBsB = cta_tmaB.partition_D(sB);

  // The TMA is responsible for copying everything in mode-0 of tAsA and tBsB
   // barrier 需要知道一次 TMA transaction 会写入多少字节，用于 arrive_and_expect_tx。
  constexpr int tma_transaction_bytes = sizeof(make_tensor_like(tensor<0>(tAsA))) + sizeof(make_tensor_like(tensor<0>(tBsB)));

#ifdef VERBOSE
  if (thread0()) {
    printf("TMA Data Struct:\n");
    print(mA);
    printf("\n");
    print(mB);
    printf("\n");
    print(sA);
    printf("\n");
    print(sB);
    printf("\n");
    print(tAgA);
    printf("\n");
    print(tAsA);
    printf("\n");
    print(tBgB);
    printf("\n");
    print(tBsB);
    printf("\n");
    printf("tma_transaction_bytes: %d\n", tma_transaction_bytes);
  }
#endif

  // Initialize Barriers
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();
  using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;
   // 只让 warp0 的一个 lane 初始化 barrier，避免重复 init。
  if ((warp_idx == 0) && lane_predicate) {
    ProducerBarType::init(mbar, 1);
  }
  cluster_sync();
  if ((warp_idx == 0) && lane_predicate) {
    // Set expected Tx Bytes after each reset / init
     // arrive_and_expect_tx：声明“接下来会有 tma_transaction_bytes 的写入”，wait 会等待计数归零。
    ProducerBarType::arrive_and_expect_tx(mbar, tma_transaction_bytes);
     // 通过 tmaA/tmaB 把 global 的分区复制到 shared 的分区，barrier 会在 transaction 完成时被满足。
    copy(tmaA.with(*mbar), tAgA, tAsA);
    copy(tmaB.with(*mbar), tBgB, tBsB);
  }
   // 等待 TMA 复制完成；此时 sA/sB 中的数据对后续 warp-group MMA 可见。
  ProducerBarType::wait(mbar, 0);

   // 每个线程从 tiled_mma 中拿到自己的 thread slice，并把 shared/global 的张量映射到寄存器可访问的分区。
  ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCgD = thr_mma.partition_C(gD);

  // Allocate accumulators and clear them
   // tCrAcc：寄存器 accumulator fragment（输出 tile 对应的投影/打包形式）
  Tensor tCrAcc = thr_mma.make_fragment_C(tCgD);

  if constexpr (Accumulate) {
    for (int i = 0; i < size(tCrAcc); ++i) {
      tCrAcc(i) = tCgC(i);
    }
  } else {
    clear(tCrAcc);
  }

  // Allocate "fragments" -- Creat Descriptor
   // tCrA/tCrB：为 WGMMA 构建的寄存器 fragment/descriptor（其底层仍然指向 shared 分区 tCsA/tCsB）。
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);

#ifdef VERBOSE
  if (thread0()) {
    printf("WGMMA Data Struct:\n");
    print(tCsA);
    printf("\n");
    print(tCsB);
    printf("\n");
    print(tCgC);
    printf("\n");
    print(tCgD);
    printf("\n");
    print(tCrA);
    printf("\n");
    print(tCrB);
    printf("\n");
  }
#endif
  // warp-group 级别的 GMMA：需要显式的 arrive/commit/wait 来保证指令批次与依赖。
  warpgroup_arrive();
  if constexpr (Accumulate) {
    mma.accumulate_ = GMMA::ScaleOut::One;
  }
  gemm(mma, tCrA(_, _, _), tCrB(_, _, _), tCrAcc);
  warpgroup_commit_batch();

  // Wait for all MMAs in a K_TILE to complete
  warpgroup_wait<0>();
  // 将 accumulator fragment 写回到 global D（tCgD 是对 gD 的分区视图）。
  for (int i = 0; i < size(tCrAcc); ++i) {
    tCgD(i) = tCrAcc(i);
  }
}

template<
    typename T_IN,
    typename T_OUT,
    typename MMA,
    typename TensorA,
    typename TensorB,
    typename TensorC,
    typename TensorD,
    bool Accumulate = false>
__global__ void mma_kernel(
    MMA mma,
    TensorA gA,
    TensorB gB,
    TensorC gC,
    TensorD gD) {
  using namespace cute;
  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  // MMA 路径：不使用 shared memory / TMA。
  // 每个线程直接从 global 的分区读取到寄存器 fragment（tCrA/tCrB），然后执行 gemm。
  // Allocate registers for pipelining
  Tensor tCgA = thr_mma.partition_A(gA);             // (MMA,MMA_M,MMA_K)
  Tensor tCgB = thr_mma.partition_B(gB);             // (MMA,MMA_N,MMA_K)
  Tensor tCrA = thr_mma.partition_fragment_A(gA);    // (MMA,MMA_M,MMA_K)
  Tensor tCrB = thr_mma.partition_fragment_B(gB);    // (MMA,MMA_N,MMA_K)
  Tensor tCgC = thr_mma.partition_C(gC);             // (MMA,MMA_M,MMA_N)
  Tensor tCgD = thr_mma.partition_C(gD);             // (MMA,MMA_M,MMA_N)

  // Allocate the accumulators -- same size as the projected data
  Tensor tCrAcc = thr_mma.make_fragment_C(tCgC); 
#ifdef VERBOSE
  if (thread0()) {
    printf("MMA Data Struct:\n");
    print(tCgA);
    printf("\n");
    print(tCgB);
    printf("\n");
    print(tCrA);
    printf("\n");
    print(tCrB);
    printf("\n");
    print(tCgC);
    printf("\n");
    print(tCgD);
    printf("\n");
    print(tCrAcc);
    printf("\n");
  }
#endif
  // Prepare A：把该线程负责的 A 分区搬到寄存器 fragment
  for (int i = 0; i < size(tCgA); ++i) {
    tCrA(i) = tCgA(i);
  }
  // Prepare B：把该线程负责的 B 分区搬到寄存器 fragment
  for (int i = 0; i < size(tCgB); ++i) {
    tCrB(i) = tCgB(i);
  }
  // Prepare C
  if constexpr (Accumulate) {
    for (int i = 0; i < size(tCrAcc); ++i) {
      tCrAcc(i) = tCgC(i);
    }
  } else {
    clear(tCrAcc);
  }
  // 经典 warp-level MMA（SM80）
  gemm(mma, tCrA, tCrB, tCrAcc);
  // Store D
  // 将 accumulator 写回到 global D 的分区
  for (int i = 0; i < size(tCrAcc); ++i) {
    tCgD(i) = tCrAcc(i);
  }
}

template<typename T_IN, typename T_OUT>
void init_host(
    thrust::host_vector<T_IN>& h_A,
    thrust::host_vector<T_IN>& h_B,
    thrust::host_vector<T_OUT>& h_C,
    bool accumulate = false) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> std_normal(0.0, 1.0);

  size_t size_A = h_A.size();
  T_IN* ptr_A = h_A.data();
  for (size_t i = 0; i < size_A; i++) {
    ptr_A[i] = (T_IN)(std_normal(gen));
  }

  size_t size_B = h_B.size();
  T_IN* ptr_B = h_B.data();
  for (size_t i = 0; i < size_B; i++) {
    ptr_B[i] = (T_IN)(std_normal(gen));
  }
  
  size_t size_C = h_C.size();
  T_OUT* ptr_C = h_C.data();
  for (size_t i = 0; i < size_C; i++) {
    if (accumulate) {
      ptr_C[i] = (T_OUT)(std_normal(gen));
    } else {
      ptr_C[i] = (T_OUT)(0.0f);
    }
  }
}

template<typename T_IN, typename T_OUT>
void gemm_reference(
    thrust::host_vector<T_IN>& h_A,
    thrust::host_vector<T_IN>& h_B,
    thrust::host_vector<T_OUT>& h_C,
    thrust::host_vector<T_OUT>& h_D,
    int M, int N, int K) {
  // CPU 参考实现（按 C + A*B 计算），用于校验 MMA/WGMMA 的数值正确性。
  T_IN* ptr_A = h_A.data();
  T_IN* ptr_B = h_B.data();
  T_OUT* ptr_C = h_C.data();
  T_OUT* ptr_D = h_D.data();
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float acc = static_cast<float>(ptr_C[m * N + n]);
      for (int k = 0; k < K; k++) {
        int offset_A = m * K + k;
        int offset_B = n * K + k;
        acc += static_cast<float>(ptr_A[offset_A]) * static_cast<float>(ptr_B[offset_B]);
      }
      ptr_D[m * N + n] = static_cast<T_OUT>(acc);
    }
  }
}

template<typename T>
void print_matrix(thrust::host_vector<T>& matrix, int row, int col) {
  T* ptr = matrix.data();
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      float v = static_cast<float>(ptr[i * col + j]);
      printf("%.3f", v);
      if (j < col - 1) {
        printf("\t");
      }
    }
    printf("\n");
  }
}

template<typename T>
void check_matrix(thrust::host_vector<T>& matrix, thrust::host_vector<T>& ref_matrix,int row, int col) {
  T* ptr = matrix.data();
  T* ref_ptr = ref_matrix.data();
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      float v = static_cast<float>(ptr[i * col + j]);
      float v_ref = static_cast<float>(ref_ptr[i * col + j]);
      if (std::abs(v - v_ref) > 1e-5) {
        printf("Mismatch %f vs %f in (%d, %d)\n", v, v_ref, i, j);
      }
    }
  }
}

template<typename T>
void check_bit_align(
    thrust::host_vector<T>& mma_matrix,
    thrust::host_vector<T>& wgmma_matrix,
    thrust::host_vector<T>& ref_matrix,
    int row, int col) {
  T* ref_ptr = ref_matrix.data();
  T* mma_ptr = mma_matrix.data();
  T* wgmma_ptr = wgmma_matrix.data();

  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      T v_ref = ref_ptr[i * col + j];
      T v_mma = mma_ptr[i * col + j];
      T v_wgmma = wgmma_ptr[i * col + j];
      if constexpr (std::is_same_v<T, float>) {
        // printf("Diff(MMA vs. WGMMA) %f vs. %f\n", v_mma - v_ref, v_wgmma - v_ref);
        uint32_t* raw_mma = reinterpret_cast<uint32_t*>(&v_mma);
        uint32_t* raw_wgmma = reinterpret_cast<uint32_t*>(&v_wgmma);
        uint32_t bit_diff = (*raw_mma) ^ (*raw_wgmma);
        if (bit_diff != 0) {
          printf("Bit align for (%d, %d): %x\n", i, j, bit_diff);
        }
      }
    }
  }
}

template<typename T_IN, typename T_OUT, typename TensorA, typename TensorB, typename TensorC, typename TensorD>
void mma(TensorA tensor_a, TensorB tensor_b, TensorC tensor_c, TensorD tensor_d, bool accumulate = false) {
  using namespace cute;
  // 构建 SM80 tiled MMA（64x128x16 tile），用于对比/基线。
  TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{},
                                Layout<Shape<_2,_2>>{},
                                Tile<_64,_128,_16>{});
#ifdef VERBOSE
  print(mma);
#endif
  dim3 dimBlock(size(mma));
  dim3 dimGrid(1, 1, 1);
  if (accumulate) {
    auto kernel_fptr = mma_kernel<
                        T_IN, T_OUT,
                        decltype(mma),
                        decltype(tensor_a), decltype(tensor_b), decltype(tensor_c), decltype(tensor_d), true>;
    kernel_fptr<<<dimGrid, dimBlock, 0, NULL>>>(mma, tensor_a, tensor_b, tensor_c, tensor_d);
  } else {
    auto kernel_fptr = mma_kernel<
                        T_IN, T_OUT,
                        decltype(mma),
                        decltype(tensor_a), decltype(tensor_b), decltype(tensor_c), decltype(tensor_d), false>;
    kernel_fptr<<<dimGrid, dimBlock, 0, NULL>>>(mma, tensor_a, tensor_b, tensor_c, tensor_d);
  }
  CUTE_CHECK_ERROR(cudaDeviceSynchronize());
}

template<typename T_IN, typename T_OUT, typename TensorA, typename TensorB, typename TensorC, typename TensorD>
void wgmma(TensorA tensor_a, TensorB tensor_b, TensorC tensor_c, TensorD tensor_d, bool accumulate = false) {
  using namespace cute;
  // 构建 SM90 WGMMA：
  // - 用 GMMA::Layout_K_SW32_Atom 定义 shared memory 的 swizzle 布局
  // - 用 TMA 将 gmem 复制到 swizzled smem
  // - 使用 SM90_64x128x16 GMMA 指令完成一个 tile
  constexpr int bM = 64;
  constexpr int bN = 128;
  constexpr int bK = 16;
  auto M = shape<0>(tensor_a);
  auto N = shape<0>(tensor_b);
  auto K = shape<1>(tensor_a);

  static_assert(is_static_v<decltype(M)>);
  static_assert(is_static_v<decltype(N)>);
  static_assert(is_static_v<decltype(K)>);
  static_assert(M == 64);
  static_assert(N == 128);
  static_assert(K == 16);

  using SMEM_ATOM_LAYOUT = GMMA::Layout_K_SW32_Atom<T_IN>;
  auto smem_atom_layout = SMEM_ATOM_LAYOUT{};
#ifdef VERBOSE
  printf("smem_atom_layout:\n");
  print(smem_atom_layout);
  printf("\n");
#endif
  // 将 atom swizzle 布局扩展到整个 tile 的形状；这会决定 sA/sB 的 shared memory 排布。
  auto layout_sA = tile_to_shape(smem_atom_layout, make_shape(Int<bM>{}, Int<bK>{}));
  auto layout_sB = tile_to_shape(smem_atom_layout, make_shape(Int<bN>{}, Int<bK>{}));
#ifdef VERBOSE
  print(layout_sA);
  printf("\n");
  print(layout_sB);
  printf("\n");
#endif
  // 为 A/B 构建 TMA load descriptor：描述如何从 gmem 读、如何写入 swizzled smem。
  auto tmaA = make_tma_copy(SM90_TMA_LOAD{}, tensor_a, layout_sA);
  auto tmaB = make_tma_copy(SM90_TMA_LOAD{}, tensor_b, layout_sB);

#ifdef VERBOSE
  print(tmaA);
  print(tmaB);
#endif

  TiledMMA tiled_mma = make_tiled_mma(SM90_64x128x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{});

  // Launch parameter setup
  // smem_size：A tile + B tile + barrier storage（uint64_t）
  int smem_size = int(cosize(layout_sA) + cosize(layout_sB)) * sizeof(T_IN) + sizeof(uint64_t);
  printf("WGMMA Shared Memory Size: %d\n", smem_size);
  dim3 dimBlock(size(tiled_mma));
  dim3 dimCluster(1, 1, 1);
  dim3 dimGrid(1, 1, 1);
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};
  
  void const* acc_kernel_ptr = reinterpret_cast<void const*>(&wgmma_kernel<
                                T_IN, T_OUT,
                                decltype(tiled_mma),
                                decltype(layout_sA), decltype(layout_sB),
                                decltype(tensor_a), decltype(tensor_b), decltype(tensor_c), decltype(tensor_d),
                                decltype(tmaA), decltype(tmaB), true>);
  void const* bypass_kernel_ptr = reinterpret_cast<void const*>(&wgmma_kernel<
                                    T_IN, T_OUT,
                                    decltype(tiled_mma),
                                    decltype(layout_sA), decltype(layout_sB),
                                    decltype(tensor_a), decltype(tensor_b), decltype(tensor_c), decltype(tensor_d),
                                    decltype(tmaA), decltype(tmaB), false>);
  void const* kernel_ptr = accumulate ? acc_kernel_ptr : bypass_kernel_ptr;

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
    kernel_ptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size));

  // Kernel Launch
  cutlass::Status status = cutlass::launch_kernel_on_cluster(
    params, kernel_ptr,
    tiled_mma,
    tensor_a, tensor_b, tensor_c, tensor_d,
    tmaA, tmaB);
  CUTE_CHECK_ERROR(cudaDeviceSynchronize());
}

int main(void) {
  using namespace cute;
  using T_IN = cutlass::bfloat16_t;
  using T_OUT = float;
  constexpr int bM = 64;
  constexpr int bN = 128;
  constexpr int bK = 16;
  constexpr bool accumulate = true;
  thrust::host_vector<T_IN> h_A(bM * bK);
  thrust::host_vector<T_IN> h_B(bN * bK);
  thrust::host_vector<T_OUT> h_C(bM * bN);
  thrust::host_vector<T_OUT> h_ref_D(bM * bN);
  init_host<T_IN, T_OUT>(h_A, h_B, h_C, accumulate);
  gemm_reference<T_IN, T_OUT>(h_A, h_B, h_C, h_ref_D, bM, bN, bK);

  thrust::device_vector<T_IN> d_A(h_A);
  thrust::device_vector<T_IN> d_B(h_B);
  thrust::device_vector<T_OUT> d_C(h_C);
  thrust::device_vector<T_OUT> d_D_mma(bM * bN);
  thrust::device_vector<T_OUT> d_D_wgmma(bM * bN);

  auto gA = make_tensor(make_gmem_ptr(d_A.data().get()),
    make_layout(make_shape(Int<bM>{}, Int<bK>{}), cute::LayoutRight{}));
  auto gB = make_tensor(make_gmem_ptr(d_B.data().get()),
    make_layout(make_shape(Int<bN>{}, Int<bK>{}), cute::LayoutRight{}));
  auto gC = make_tensor(make_gmem_ptr(d_C.data().get()),
    make_layout(make_shape(Int<bM>{}, Int<bN>{}), cute::LayoutRight{}));
  auto mma_gD = make_tensor(make_gmem_ptr(d_D_mma.data().get()),
    make_layout(make_shape(Int<bM>{}, Int<bN>{}), cute::LayoutRight{}));
  auto wgmma_gD = make_tensor(make_gmem_ptr(d_D_wgmma.data().get()),
    make_layout(make_shape(Int<bM>{}, Int<bN>{}), cute::LayoutRight{}));

  // 分别运行两条路径，并与 CPU reference 比较
  mma<T_IN, T_OUT>(gA, gB, gC, mma_gD, accumulate);
  wgmma<T_IN, T_OUT>(gA, gB, gC, wgmma_gD, accumulate);

  thrust::host_vector<T_OUT> h_D_mma(d_D_mma);
  thrust::host_vector<T_OUT> h_D_wgmma(d_D_wgmma);

  printf("Check Diff:\n");
  check_matrix<T_OUT>(h_D_mma, h_ref_D, bM, bN);
  check_matrix<T_OUT>(h_D_wgmma, h_ref_D, bM, bN);
  printf("Check Diff Done!\n");

#ifdef VERBOSE
  printf("WGMMA Output:\n");
  print_matrix<T_OUT>(h_D_wgmma, bM, bN);
  printf("MMA Output:\n");
  print_matrix<T_OUT>(h_D_mma, bM, bN);
  printf("Ref Output:\n");
  print_matrix<T_OUT>(h_ref_D, bM, bN);
#endif
  check_bit_align<T_OUT>(h_D_mma, h_D_wgmma, h_ref_D, bM, bN);
  return 0;
}