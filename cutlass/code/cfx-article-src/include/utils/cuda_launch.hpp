#pragma once

namespace cfk {
namespace utils {
inline void set_smem_size(int smem_size, void const *kernel) {
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
} // namespace utils
} // namespace cfk