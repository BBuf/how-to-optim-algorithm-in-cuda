#pragma once

#include "cutlass/detail/layout.hpp"

// Shared Storage for aligned addresses
template <class Element, class SmemLayout> struct SharedStorageTranspose {
  cute::array_aligned<Element, cute::cosize_v<SmemLayout>,
                      cutlass::detail::alignment_for_swizzle(SmemLayout{})>
      smem;
};