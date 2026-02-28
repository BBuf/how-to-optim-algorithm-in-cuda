# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from torchao.float8.float8_scaling_utils import (
    hp_tensor_to_float8_delayed,
    hp_tensor_to_float8_dynamic,
)

from torchao.float8.float8_tensor import (
    Float8Tensor,
    GemmInputRole,
    hp_tensor_and_scale_to_float8,
    LinearMMConfig,
)

from torchao.float8.float8_utils import e4m3_dtype, EPS
from torch._prims_common import suggest_memory_format


@torch.no_grad()
def precompute_float8_dynamic_scale_for_fsdp(module: nn.Module) -> None:
    """
    Calculate scale dynamically for all float8 parameters.
    This should be run after the optimizer step. It performs a single all-reduce to compute the
    scales for all float8 weights.
    Example usage:
        model(input).sum().backward()
        optim.step()
        precompute_float8_dynamic_scale_for_fsdp(model)
    """
    from torchao.float8.config import ScalingType
    from torchao.float8.float8_linear import Float8Linear
    from torch.distributed._tensor import DTensor

    if any(
        isinstance(m, Float8Linear) and m.scaling_type_weight is ScalingType.DELAYED
        for m in module.modules()
    ):
        raise NotImplementedError("Only supports delayed scaling")
    float8_linears: List[Float8Linear] = [
        m
        for m in module.modules()
        if isinstance(m, Float8Linear)
        and isinstance(m.weight, DTensor)
        and isinstance(m.weight._local_tensor, WeightWithDynamicFloat8CastTensor)
    ]
    weights: List[DTensor] = [float8_linear.weight for float8_linear in float8_linears]

    if not weights:
        return

    # inf-norm is equivalent to max(abs(w))
    max_weights = torch._foreach_norm(weights, ord=math.inf)  # Partial
    amax_tensor = torch.stack(max_weights)  # Partial
    # clamp is dispatched through DTensor
    # it will issue a single all-reduce
    amax_tensor = torch.clamp(amax_tensor, EPS)  # Replicate
    # keep consistent with float8_utils.amax_to_scale
    # torch.compile and eager show different numerics for 1.0 / float32,
    # upcast to float64 to ensure same numeric between compile and eager
    origin_dtype = amax_tensor.dtype
    amax_tensor = amax_tensor.to(torch.float64)
    scale_tensor = torch.finfo(torch.float8_e4m3fn).max / amax_tensor  # Replicate
    if origin_dtype is torch.float16:
        scale_tensor = torch.clamp(scale_tensor, max=torch.finfo(torch.float16).max)
    local_scale_tensor = scale_tensor.to_local().to(torch.float32)
    for i, float8_linear in enumerate(float8_linears):
        float8_linear.weight._local_tensor._precomputed_scale = local_scale_tensor[i]


# FSDP pads its local tensor on dim-0. The subclass should be preserved such
# that the padded local tensor (and any transformations like copying to GPU)
# is of the subclass as well.
_ops_to_preserve_subclass = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.copy_.default,
    torch.ops.aten.view.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten._pin_memory.default,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.clone.default,
}

# How Tensor Parallel (TP) and FSDP2 work

# Initialization: apply TP first then FSDP2
# nn.Linear(weight=torch.Tensor)
#      |
#      | apply float8 linear, `convert_to_float8_training`
#      |
# Float8Linear(weight=WeightWithDynamicFloat8CastTensor)
#      |
#      | apply tensor parallel, `parallelize_module` shards rowwise/colwise
#      |
# Float8Linear(weight=DTensor(local_tensor=WeightWithDynamicFloat8CastTensor,
#                             device_mesh=DeviceMesh([0, 1], mesh_dim_names=('tp',)),
#                             placements=(Shard(dim=0),)))
#      |
#      | apply FSDP2, `fully_shard` shards rowwise (dim=0)
#      |
# Float8Linear(weight=DTensor(local_tensor=WeightWithDynamicFloat8CastTensor,
#                             device_mesh=DeviceMesh([[0, 1], [2, 3]], mesh_dim_names=('dp', 'tp')),
#                             placements=(Shard(dim=0), Shard(dim=0))))

# Forward and backward: FSDP runs first then TP
# Float8Linear(weight=DTensor(local_tensor=WeightWithDynamicFloat8CastTensor,
#                             device_mesh=DeviceMesh([[0, 1], [2, 3]], mesh_dim_names=('dp', 'tp')),
#                             placements=(Shard(dim=0), Shard(dim=0))))
#      |
#      |   FSDP unshards parameters within dp mesh
#      |
# Float8Linear(weight=DTensor(local_tensor=WeightWithDynamicFloat8CastTensor,
#                             device_mesh=DeviceMesh([0, 1], mesh_dim_names=('tp',)),
#                             placements=(Shard(dim=0),)))
#      |
#      |   TP compute with torch.mm(input, weight)

class WeightWithDynamicFloat8CastTensor(torch.Tensor):
    @staticmethod
    def __new__(
        cls,
        tensor: torch.Tensor,
        linear_mm_config: LinearMMConfig,
        precomputed_scale: Optional[torch.Tensor] = None,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            memory_format=suggest_memory_format(tensor),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            pin_memory=tensor.is_pinned(),
            requires_grad=tensor.requires_grad,
        )

    def __init__(
        self,
        tensor: torch.Tensor,
        linear_mm_config: LinearMMConfig,
        precomputed_scale: Optional[torch.Tensor] = None,
    ):
        self._tensor = tensor
        self._linear_mm_config = linear_mm_config
        # for dynamic scaling
        # `precompute_float8_dynamic_scale_for_fsdp` calculates scales
        # for all float8 parameters after optimizer step
        self._precomputed_scale = precomputed_scale

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func == torch.ops.aten.detach.default:
            return WeightWithDynamicFloat8CastTensor(
                args[0]._tensor, args[0]._linear_mm_config
            )
        mm_config: Optional[LinearMMConfig] = None

        def unwrap(t):
            nonlocal mm_config
            if mm_config is None:
                mm_config = t._linear_mm_config
            else:
                assert t._linear_mm_config == mm_config
            return t._tensor

        args, kwargs = pytree.tree_map_only(
            WeightWithDynamicFloat8CastTensor, unwrap, (args, kwargs or {})
        )
        out = func(*args, **kwargs)
        if func not in _ops_to_preserve_subclass:
            return out
        return pytree.tree_map_only(
            torch.Tensor, lambda x: WeightWithDynamicFloat8CastTensor(x, mm_config), out
        )

    def __tensor_flatten__(self):
        if self._precomputed_scale:
            return ["_tensor", "_precomputed_scale"], self._linear_mm_config
        else:
            return ["_tensor"], self._linear_mm_config

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        mm_config = flatten_spec
        return WeightWithDynamicFloat8CastTensor(
            inner_tensors["_tensor"],
            mm_config,
            getattr(inner_tensors, "_precomputed_scale", None),
        )

    def __repr__(self):
        return f"WeightWithDynamicFloat8CastTensor(tensor={self._tensor}, linear_mm_config={self._linear_mm_config})"

    def fsdp_pre_all_gather(self, mesh):
        if self._precomputed_scale is not None:
            float8_tensor = hp_tensor_and_scale_to_float8(
                self._tensor,
                self._precomputed_scale,
                torch.float8_e4m3fn,
                self._linear_mm_config,
                GemmInputRole.WEIGHT,
            )
        else:
            float8_tensor = hp_tensor_to_float8_dynamic(
                self._tensor,
                e4m3_dtype,
                self._linear_mm_config,
                reduce_amax=True,
                gemm_input_role=GemmInputRole.WEIGHT,
                device_mesh=mesh,
            )
        return (float8_tensor._data,), (float8_tensor._scale,)

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[torch.Tensor] = None,
    ):
        (data,) = all_gather_outputs
        (scale,) = metadata
        if out is not None:
            from torch.distributed._tensor import DTensor
            if isinstance(out, Float8Tensor):
                out._scale = scale
            elif isinstance(out, DTensor) and isinstance(
                out._local_tensor, Float8Tensor
            ):
                out._local_tensor._scale = scale
            else:
                raise RuntimeError(
                    f"out must be a Float8Tensor or DTensor(_local_tensor=Float8Tensor), but got {out}"
                )
            return
        return Float8Tensor(
            data,
            scale,
            param_dtype,
            self._linear_mm_config,
            gemm_input_role=GemmInputRole.WEIGHT,
        ), (data,)


class WeightWithDelayedFloat8CastTensor(torch.Tensor):
    @staticmethod
    def __new__(
        cls,
        tensor: torch.Tensor,
        amax_buffer: torch.Tensor,
        amax_history_buffer: torch.Tensor,
        scale_buffer: torch.Tensor,
        linear_mm_config: LinearMMConfig,
        is_amax_initialized: bool,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            memory_format=suggest_memory_format(tensor),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            pin_memory=tensor.is_pinned(),
            requires_grad=tensor.requires_grad,
        )

    def __init__(
        self,
        tensor: torch.Tensor,
        amax_buffer: torch.Tensor,
        amax_history_buffer: torch.Tensor,
        scale_buffer: torch.Tensor,
        linear_mm_config: LinearMMConfig,
        is_amax_initialized: bool,
    ):
        self._tensor = tensor
        self._amax_buffer = amax_buffer
        self._amax_history_buffer = amax_history_buffer
        self._scale_buffer = scale_buffer
        self._linear_mm_config = linear_mm_config

        # Note: is_amax_initialized is not a buffer to avoid data dependent
        # control flow visible to dynamo
        # TODO(future PR): add serialization for this flag
        self.is_amax_initialized = is_amax_initialized

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func == torch.ops.aten.detach.default:
            return WeightWithDelayedFloat8CastTensor(
                args[0]._tensor,
                args[0]._amax_buffer,
                args[0]._amax_history_buffer,
                args[0]._scale_buffer,
                args[0]._linear_mm_config,
                args[0].is_amax_initialized,
            )
        mm_config: Optional[LinearMMConfig] = None
        amax_buffer: Optional[torch.Tensor] = None
        amax_history_buffer: Optional[torch.Tensor] = None
        scale_buffer: Optional[torch.Tensor] = None
        is_amax_initialized: Optional[bool] = None

        def unwrap(t):
            nonlocal mm_config
            if mm_config is None:
                mm_config = t._linear_mm_config
            else:
                assert t._linear_mm_config == mm_config
            nonlocal amax_buffer
            if amax_buffer is None:
                amax_buffer = t._amax_buffer
            nonlocal amax_history_buffer
            if amax_history_buffer is None:
                amax_history_buffer = t._amax_history_buffer
            nonlocal scale_buffer
            if scale_buffer is None:
                scale_buffer = t._scale_buffer
            nonlocal is_amax_initialized
            if is_amax_initialized is None:
                is_amax_initialized = t.is_amax_initialized
            return t._tensor

        args, kwargs = pytree.tree_map_only(
            WeightWithDelayedFloat8CastTensor, unwrap, (args, kwargs or {})
        )
        out = func(*args, **kwargs)
        if func not in _ops_to_preserve_subclass:
            return out
        return pytree.tree_map_only(
            torch.Tensor,
            lambda x: WeightWithDelayedFloat8CastTensor(
                x,
                amax_buffer,
                amax_history_buffer,
                scale_buffer,
                mm_config,
                is_amax_initialized,
            ),
            out,
        )

    def __tensor_flatten__(self):
        return (
            [
                "_tensor",
                "_amax_buffer",
                "_amax_history_buffer",
                "_scale_buffer",
            ],
            {
                "mm_config": self._linear_mm_config,
                "is_amax_initialized": self.is_amax_initialized,
            },
        )

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        return WeightWithDelayedFloat8CastTensor(
            inner_tensors["_tensor"],
            inner_tensors["_amax_buffer"],
            inner_tensors["_amax_history_buffer"],
            inner_tensors["_scale_buffer"],
            metadata["mm_config"],
            metadata["is_amax_initialized"],
        )

    def __repr__(self):
        return f"WeightWithDelayedFloat8CastTensor(tensor={self._tensor}, amax_buffer={self._amax_buffer}, scale_buffer={self._scale_buffer}, mm_config={self._linear_mm_config})"

    def fsdp_pre_all_gather(self, mesh):
        # initialize if needed
        # TODO(before land): ensure settings are consistent between Float8Linear and here
        if not self.is_amax_initialized:
            from torchao.float8.float8_linear import (
                _maybe_initialize_amaxes_scales_for_float8_cast,
            )

            _maybe_initialize_amaxes_scales_for_float8_cast(
                self._tensor,
                self._amax_buffer,
                self._amax_history_buffer,
                self._scale_buffer,
                "max",  # TODO(before land): read this from parent
                e4m3_dtype,
                self.is_amax_initialized,
                reduce_amax=True,
            )
            self.is_amax_initialized = True

        float8_tensor = hp_tensor_to_float8_delayed(
            self._tensor,
            self._scale_buffer,
            e4m3_dtype,
            self._amax_buffer,
            self._linear_mm_config,
            GemmInputRole.WEIGHT,
        )
        return (float8_tensor._data,), (float8_tensor._scale,)

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[torch.Tensor] = None,
    ):
        (data,) = all_gather_outputs
        (scale,) = metadata
        if out is not None:
            assert isinstance(out, Float8Tensor), f"{type(out)}"
            out._scale = scale
            return
        return Float8Tensor(
            data,
            scale,
            param_dtype,
            self._linear_mm_config,
            gemm_input_role=GemmInputRole.WEIGHT,
        ), (data,)


class WeightWithStaticFloat8CastTensor(torch.Tensor):
    @staticmethod
    def __new__(
        cls,
        tensor: torch.Tensor,
        static_scale: torch.Tensor,
        linear_mm_config: LinearMMConfig,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            memory_format=suggest_memory_format(tensor),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            pin_memory=tensor.is_pinned(),
            requires_grad=tensor.requires_grad,
        )

    def __init__(
        self,
        tensor: torch.Tensor,
        static_scale: torch.Tensor,
        linear_mm_config: LinearMMConfig,
    ):
        self._tensor = tensor
        self._static_scale = static_scale
        self._linear_mm_config = linear_mm_config

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func == torch.ops.aten.detach.default:
            return WeightWithStaticFloat8CastTensor(
                args[0]._tensor, args[0]._static_scale, args[0]._linear_mm_config
            )
        static_scale: Optional[torch.Tensor] = None
        mm_config: Optional[LinearMMConfig] = None

        def unwrap(t):
            nonlocal static_scale
            if static_scale is None:
                static_scale = t._static_scale
            nonlocal mm_config
            if mm_config is None:
                mm_config = t._linear_mm_config
            else:
                assert t._linear_mm_config == mm_config
            return t._tensor

        args, kwargs = pytree.tree_map_only(
            WeightWithStaticFloat8CastTensor, unwrap, (args, kwargs or {})
        )
        out = func(*args, **kwargs)
        if func not in _ops_to_preserve_subclass:
            return out
        return pytree.tree_map_only(
            torch.Tensor, 
            lambda x: WeightWithStaticFloat8CastTensor(x, static_scale, mm_config), 
            out,
        )

    def __tensor_flatten__(self):
        return ["_tensor", "_static_scale"], self._linear_mm_config

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        mm_config = flatten_spec
        return WeightWithStaticFloat8CastTensor(
            inner_tensors["_tensor"],
            inner_tensors["_static_scale"],
            mm_config,
        )

    def __repr__(self):
        return f"WeightWithStaticFloat8CastTensor(tensor={self._tensor}, static_scale={self._static_scale}, linear_mm_config={self._linear_mm_config})"

    def fsdp_pre_all_gather(self, mesh):
        float8_tensor = hp_tensor_and_scale_to_float8(
            self._tensor,
            self._static_scale,
            torch.float8_e4m3fn,
            self._linear_mm_config,
            GemmInputRole.WEIGHT,
        )
        return (float8_tensor._data,), (float8_tensor._scale,)

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[torch.Tensor] = None,
    ):
        (data,) = all_gather_outputs
        (scale,) = metadata
        if out is not None:
            from torch.distributed._tensor import DTensor
            if isinstance(out, Float8Tensor):
                out._scale = scale
            elif isinstance(out, DTensor) and isinstance(
                out._local_tensor, Float8Tensor
            ):
                out._local_tensor._scale = scale
            else:
                raise RuntimeError(
                    f"out must be a Float8Tensor or DTensor(_local_tensor=Float8Tensor), but got {out}"
                )
            return
        return Float8Tensor(
            data,
            scale,
            param_dtype,
            self._linear_mm_config,
            gemm_input_role=GemmInputRole.WEIGHT,
        ), (data,)
