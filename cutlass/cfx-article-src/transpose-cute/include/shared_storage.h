#pragma once

#include "cutlass/detail/layout.hpp"

// 共享存储，用于对齐地址
// 该结构体模板用于定义共享存储，其中包含一个对齐的数组。
// 模板参数：
// - Element: 数组中元素的类型。
// - SmemLayout: 共享内存的布局类型。
template <class Element, class SmemLayout> struct SharedStorageTranspose {
  // 使用cute库中的array_aligned定义一个对齐的数组。
  // cute::cosize_v<SmemLayout> 用于计算Layout的codomain的大小。可以参考 https://github.com/NVIDIA/cutlass/issues/944 这个issue的解释
  cute::array_aligned<Element, cute::cosize_v<SmemLayout>,
                      cutlass::detail::alignment_for_swizzle(SmemLayout{})>
      smem;
};



// 获取组合布局中的Swizzle部分。
// 在代码中，B, M, S 是模板参数，用于定义 Swizzle 类的行为。具体来说：
// B：表示 Swizzle 操作中的一个维度的位移量。
// M：表示 Swizzle 操作中的另一个维度的位移量。
// S：表示 Swizzle 操作中的步长（stride）。
// 这些参数共同决定了 Swizzle 操作的具体方式，即如何在内存中重新排列数据。
// 通过这些参数，可以灵活地定义不同的 Swizzle 操作，以优化内存访问模式，提高 CUDA 程序的性能。
// template <int B, int M, int S, class Offset, class LayoutB>
// CUTE_HOST_DEVICE constexpr
// auto
// get_swizzle_portion(ComposedLayout<Swizzle<B,M,S>,Offset,LayoutB>)
// {
//   // 返回Swizzle部分
//   return Swizzle<B,M,S>{};
// }

// 非Swizzle布局的“Swizzle部分”是身份Swizzle。
// template <class Shape, class Stride>
// CUTE_HOST_DEVICE constexpr
// auto
// get_swizzle_portion(Layout<Shape,Stride>)
// {
//   // 返回默认的身份Swizzle
//   return Swizzle<0,4,3>{};
// }

// 计算Swizzle的对齐大小
// template<int B, int M, int S>
// CUTLASS_HOST_DEVICE constexpr
// size_t
// alignment_for_swizzle(cute::Swizzle<B, M, S>) {
//   // 确保B和M是非负数
//   static_assert(B >= 0 and M >= 0);
//   // 计算对齐大小，使用位移操作
//   return size_t(1) << size_t(B + M + cute::abs(S));
// }

// // 计算布局的对齐大小
// template<class Layout>
// CUTLASS_HOST_DEVICE constexpr
// size_t
// alignment_for_swizzle(Layout layout) {
//   // 获取布局的Swizzle部分并计算对齐大小
//   return alignment_for_swizzle(cute::detail::get_swizzle_portion(layout));
// }
