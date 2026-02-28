#pragma once
#include <stdint.h>

/**
 * \tparam S: SShift, right shift the addr for swizzling
 * \tparam B: BShift, bits to be swizzled
 * \tparam M: MBase, bits keep the same
 */
template <uint32_t S, uint32_t B, uint32_t M>
__device__ __forceinline__ uint32_t swizzle(uint32_t addr) {
    constexpr auto Bmask = ((1 << B) - 1) << M;
    return ((addr >> S) & Bmask) ^ addr;
}
