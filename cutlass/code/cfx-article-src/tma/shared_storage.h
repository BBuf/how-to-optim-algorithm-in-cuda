#pragma once

#include "cutlass/detail/layout.hpp"

template <class Element, class SmemLayout> struct SharedStorageTMA {
  cute::array_aligned<Element, cute::cosize_v<SmemLayout>,
                      cutlass::detail::alignment_for_swizzle(SmemLayout{})>
      smem;
  // alignas(16) uint64_t tma_load_mbar[1];
  cutlass::arch::ClusterTransactionBarrier mbarrier;
};