#pragma once

#include <cute/atom/mma_traits_sm90_gmma.hpp>

namespace cfx {

using namespace cute;

// Helper functions for retrieving optimal swizzled layouts
template <typename PrecType, int DIM> constexpr auto getSmemLayoutK() {

  constexpr int headSizeBytes = sizeof(PrecType) * DIM;

  if constexpr (headSizeBytes == 16) {
    return GMMA::Layout_K_INTER_Atom<PrecType>{};
  } else if constexpr (headSizeBytes == 32) {
    return GMMA::Layout_K_SW32_Atom<PrecType>{};
  } else if constexpr (headSizeBytes == 64) {
    return GMMA::Layout_K_SW64_Atom<PrecType>{};
  } else {
    return GMMA::Layout_K_SW128_Atom<PrecType>{};
  }
}

template <typename PrecType, int DIM> constexpr auto getSmemLayoutMN() {

  constexpr int headSizeBytes = sizeof(PrecType) * DIM;

  if constexpr (headSizeBytes == 16) {
    return GMMA::Layout_MN_INTER_Atom<PrecType>{};
  } else if constexpr (headSizeBytes == 32) {
    return GMMA::Layout_MN_SW32_Atom<PrecType>{};
  } else if constexpr (headSizeBytes == 64) {
    return GMMA::Layout_MN_SW64_Atom<PrecType>{};
  } else {
    return GMMA::Layout_MN_SW128_Atom<PrecType>{};
  }
}

// 这个函数 set_smem_size 用于设置 CUDA kernel的动态共享内存大小。具体来说，它执行以下操作：
void set_smem_size(int smem_size, void const *kernel) {
  // account for dynamic smem capacity if needed
  // 函数首先检查传入的共享内存大小 smem_size 是否大于等于 48 KB (48 << 10 表示 48 乘以 1024，即 48 KB)。
  if (smem_size >= (48 << 10)) {
    // 如果 smem_size 大于等于 48 KB，函数调用 cudaFuncSetAttribute 来设置内核的最大动态共享内存大小。
    // cudaFuncSetAttribute 是 CUDA 运行时 API，用于设置内核函数的属性。这里设置的属性是 
    // cudaFuncAttributeMaxDynamicSharedMemorySize，值为 smem_size。
    cudaError_t result = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    // 如果 cudaFuncSetAttribute 返回的结果不是 cudaSuccess，则表示设置共享内存属性失败。
    // 函数会调用 cudaGetLastError 来清除错误标志，并输出错误信息。
    if (cudaSuccess != result) {
      result = cudaGetLastError(); // to clear the error bit
      std::cout << "  Shared Memory Allocation Failed " << std::endl
                << " cudaFuncSetAttribute() returned error: "
                << cudaGetErrorString(result) << std::endl;
    }
  }
}

} // namespace cfx