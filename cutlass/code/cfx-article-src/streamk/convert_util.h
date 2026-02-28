/******************************************************************************
 * Copyright (c) 2024 Colfax Research                                         *
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>
#include <cute/arch/cluster_sm90.hpp>  // For cute::elect_one_sync()

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: this requires tensor to be "contiguous"
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
    // Tensor out = make_tensor_like<To_type>(tensor);
    // cute::copy(make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout()), out);
    // return out;
}
