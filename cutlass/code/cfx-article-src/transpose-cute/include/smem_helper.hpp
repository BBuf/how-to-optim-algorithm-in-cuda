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

void set_smem_size(int smem_size, void const *kernel) {
  // account for dynamic smem capacity if needed
  if (smem_size >= (48 << 10)) {
    cudaError_t result = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (cudaSuccess != result) {
      result = cudaGetLastError(); // to clear the error bit
      std::cout << "  Shared Memory Allocation Failed " << std::endl
                << " cudaFuncSetAttribute() returned error: "
                << cudaGetErrorString(result) << std::endl;
    }
  }
}

} // namespace cfx