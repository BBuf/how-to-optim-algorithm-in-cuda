# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
A simple module swap UX for a float8 version of `torch.nn.Linear`.
"""

import dataclasses
import enum

from typing import Optional

import torch

import torch.utils.checkpoint as checkpoint

from torchao.float8.config import Float8LinearConfig, ScalingType, ScalingGranularity

from torchao.float8.float8_scaling_utils import (
    _maybe_initialize_amaxes_scales_for_float8_cast,
    hp_tensor_to_float8_delayed,
    hp_tensor_to_float8_dynamic,
    hp_tensor_to_float8_static,
    get_maybe_axiswise_dim,
    NoopFwToFloat8E5M2BwDelayed,
    NoopFwToFloat8E5M2BwDynamic,
    NoopFwToFloat8E5M2BwStatic,
)

from torchao.float8.float8_tensor import (
    Float8Tensor,
    GemmInputRole,
    hp_tensor_and_scale_to_float8,
    LinearMMConfig,
    ScaledMMConfig,
)

from torchao.float8.float8_utils import (
    e4m3_dtype, 
    e5m2_dtype, 
    tensor_to_amax,
    tensor_to_scale,
)

from torchao.float8.fsdp_utils import (
    WeightWithDelayedFloat8CastTensor,
    WeightWithDynamicFloat8CastTensor,
    WeightWithStaticFloat8CastTensor,
)


@torch._dynamo.allow_in_graph
class manual_float8_matmul_with_args_in_float8(torch.autograd.Function):
    """
    Like torch.matmul, but with the arguments in float8

    Note: this function requires all arguments to already be Float8Tensor objects,
    which only supports tensorwise scaling granularity. The reason we didn't just make this
    function support axiswise scaling granularity is because that would need very
    careful testing of delayed scaling, as delayed scaling modifies buffers inplace.

    In the future we'll probably have to unify, just postponing that until a future PR.
    """

    @staticmethod
    def forward(
        ctx,
        input_fp8,
        weight_fp8_t,
    ):
        ctx.save_for_backward(input_fp8, weight_fp8_t)
        # the reshapes are needed in order to make the shapes compatible with
        # torch.mm
        orig_shape = input_fp8.shape
        input_fp8_reshaped = input_fp8.reshape(-1, orig_shape[-1])
        res_bits = torch.mm(input_fp8_reshaped, weight_fp8_t)
        res_bits = res_bits.reshape(*orig_shape[:-1], res_bits.shape[-1])
        return res_bits

    @staticmethod
    def backward(ctx, grad_output_fp8):
        input_fp8, weight_fp8_t = ctx.saved_tensors

        # the reshapes are needed in order to make the shapes compatible with
        # torch.mm
        grad_output_fp8_orig_shape = grad_output_fp8.shape
        grad_output_fp8_reshaped = grad_output_fp8.reshape(
            -1, grad_output_fp8_orig_shape[-1]
        )

        # calculate grad_input
        grad_input = torch.mm(
            grad_output_fp8_reshaped,
            weight_fp8_t.t(),
        )
        grad_input = grad_input.reshape(
            *grad_output_fp8_orig_shape[:-1], grad_input.shape[-1]
        )

        input_fp8_orig_shape = input_fp8.shape
        input_fp8_reshaped = input_fp8.reshape(-1, input_fp8_orig_shape[-1])

        # calculate grad_weight
        # Note: the variant below is slightly faster on LLaMa 3 8B pretraining
        # compared to than calculating `grad_weight_t = input_fp8_t @ grad_output_fp8_reshaped`
        grad_weight = torch.mm(
            grad_output_fp8_reshaped.t(),
            input_fp8_reshaped,
        )

        return grad_input, grad_weight.t()

@torch._dynamo.allow_in_graph
class manual_float8_matmul_with_args_in_hp(torch.autograd.Function):
    """
    Like torch.matmul, but with the arguments in high precision and the cast to float8
    defined inside of this function.

    Note: this function currently only supports dynamic scaling type and 
    axiswise granularity. We will have to unify this with other scaling types
    and other granularities in a separate PR.
    """

    @staticmethod
    def forward(
        ctx,
        input_hp: torch.Tensor,
        weight_hp_t: torch.Tensor,
        linear_mm_config: LinearMMConfig,
        config: Float8LinearConfig,
    ):
        ctx.save_for_backward(input_hp, weight_hp_t)
        ctx.linear_mm_config = linear_mm_config
        ctx.config = config

        c = config

        if c.cast_config_input.scaling_type is ScalingType.DISABLED:
            input_maybe_fp8 = input_hp
        else:
            input_maybe_fp8 = hp_tensor_to_float8_dynamic(
                input_hp, 
                e4m3_dtype, 
                linear_mm_config,
                gemm_input_role=GemmInputRole.INPUT,
                scaling_granularity=c.cast_config_input.scaling_granularity,
                axiswise_dim=get_maybe_axiswise_dim(-1, c.cast_config_input.scaling_granularity),
            )

        if c.cast_config_weight.scaling_type is ScalingType.DISABLED:
            weight_maybe_fp8_t = weight_hp_t
        else:
            weight_maybe_fp8_t = hp_tensor_to_float8_dynamic(
                weight_hp_t, 
                e4m3_dtype, 
                linear_mm_config,
                gemm_input_role=GemmInputRole.WEIGHT,
                scaling_granularity=c.cast_config_weight.scaling_granularity,
                axiswise_dim=get_maybe_axiswise_dim(0, c.cast_config_weight.scaling_granularity),
            )

        # the reshapes are needed in order to make the shapes compatible with
        # torch.mm
        orig_shape = input_maybe_fp8.shape
        input_maybe_fp8_reshaped = input_maybe_fp8.reshape(-1, orig_shape[-1])
        res_bits = torch.mm(input_maybe_fp8_reshaped, weight_maybe_fp8_t)
        res_bits = res_bits.reshape(*orig_shape[:-1], res_bits.shape[-1])
        return res_bits

    @staticmethod
    def backward(ctx, grad_output):
        input_hp, weight_hp_t = ctx.saved_tensors
        c = ctx.config

        # the reshapes are needed in order to make the shapes compatible with
        # torch.mm
        grad_output_orig_shape = grad_output.shape
        grad_output_reshaped = grad_output.reshape(
            -1, grad_output_orig_shape[-1]
        )

        #
        # calculate grad_input
        #

        if c.cast_config_grad_output.scaling_type is ScalingType.DISABLED:
            grad_output_reshaped_maybe_fp8_dim0 = grad_output_reshaped
        else:
            grad_output_reshaped_maybe_fp8_dim0 = hp_tensor_to_float8_dynamic(
                grad_output_reshaped,
                e5m2_dtype,
                ctx.linear_mm_config,
                gemm_input_role=GemmInputRole.GRAD_OUTPUT,
                scaling_granularity=c.cast_config_grad_output.scaling_granularity,
                axiswise_dim=get_maybe_axiswise_dim(-1, c.cast_config_grad_output.scaling_granularity),
            )
        
        if c.cast_config_weight_for_grad_input.scaling_type is ScalingType.DISABLED:
            weight_t_maybe_fp8_dim0 = weight_hp_t
        else:
            # Note: we need https://github.com/pytorch/pytorch/issues/136267 
            # to be solved to have a chance to reuse max(abs(weight, dim=...)) 
            # from the forward to get max(abs(weight)) here without reading 
            # the entire tensor.
            weight_t_maybe_fp8_dim0 = hp_tensor_to_float8_dynamic(
                weight_hp_t,
                e4m3_dtype,
                ctx.linear_mm_config,
                gemm_input_role=GemmInputRole.WEIGHT,
                scaling_granularity=c.cast_config_weight_for_grad_input.scaling_granularity,
                axiswise_dim=get_maybe_axiswise_dim(-1, c.cast_config_weight_for_grad_input.scaling_granularity),
            )

        grad_input = torch.mm(
            grad_output_reshaped_maybe_fp8_dim0,
            weight_t_maybe_fp8_dim0.t(),
        )
        grad_input = grad_input.reshape(
            *grad_output_orig_shape[:-1], grad_input.shape[-1]
        )

        input_hp_orig_shape = input_hp.shape
        input_hp_reshaped = input_hp.reshape(-1, input_hp_orig_shape[-1])

        #
        # calculate grad_weight
        #

        if c.cast_config_grad_output_for_grad_weight.scaling_type is ScalingType.DISABLED:
            grad_output_reshaped_maybe_fp8_dim1 = grad_output_reshaped
        else:
            grad_output_reshaped_maybe_fp8_dim1 = hp_tensor_to_float8_dynamic(
                grad_output_reshaped,
                e5m2_dtype,
                ctx.linear_mm_config,
                gemm_input_role=GemmInputRole.GRAD_OUTPUT,
                scaling_granularity=c.cast_config_grad_output_for_grad_weight.scaling_granularity,
                axiswise_dim=get_maybe_axiswise_dim(0, c.cast_config_grad_output_for_grad_weight.scaling_granularity),
            )
        
        if c.cast_config_input_for_grad_weight.scaling_type is ScalingType.DISABLED:
            input_reshaped_maybe_fp8_dim1 = input_hp_reshaped
        else:
            input_reshaped_maybe_fp8_dim1 = hp_tensor_to_float8_dynamic(
                input_hp_reshaped,
                e4m3_dtype,
                ctx.linear_mm_config,
                gemm_input_role=GemmInputRole.INPUT,
                scaling_granularity=c.cast_config_input_for_grad_weight.scaling_granularity,
                axiswise_dim=get_maybe_axiswise_dim(0, c.cast_config_input_for_grad_weight.scaling_granularity),
            )

        grad_weight = torch.mm(
            grad_output_reshaped_maybe_fp8_dim1.t(),
            input_reshaped_maybe_fp8_dim1,
        )

        empty_grads = None, None

        return grad_input, grad_weight.t(), *empty_grads


class Float8Linear(torch.nn.Linear):
    """
    Note: this is **not** a public API and is only intended to be used
    inside of this repository. Please file an issue if you would benefit
    from this being a public API.

    A wrapper around a `torch.nn.Linear` module which does fp8 compute, and tracks
    scales in way friendly to delayed scaling.
    """

    def __init__(self, *args, **kwargs):
        """
        Additional arguments on top of `torch.nn.Linear`'s arguments:
        * `config`: Float8LinearConfig
        """

        # Amax scales should always be kept as float32.
        self.always_float32_buffers = set()
        config = kwargs.pop("config")
        emulate = config.emulate
        super().__init__(*args, **kwargs)

        # Defines the scaling behavior of input, weight, grad_output
        self.scaling_type_input = config.cast_config_input.scaling_type
        self.scaling_type_weight = config.cast_config_weight.scaling_type
        self.scaling_type_grad_output = config.cast_config_grad_output.scaling_type
        # Convenience flag to skip code related to delayed scaling
        self.has_any_delayed_scaling = (
            self.scaling_type_input is ScalingType.DELAYED
            or self.scaling_type_weight is ScalingType.DELAYED
            or self.scaling_type_grad_output is ScalingType.DELAYED
        )

        self.config = config

        self.create_buffers()

        self.linear_mm_config = LinearMMConfig(
            # output
            ScaledMMConfig(
                emulate,
                self.config.gemm_config_output.use_fast_accum,
                False,
                self.config.pad_inner_dim,
            ),
            # grad_input
            ScaledMMConfig(
                emulate,
                self.config.gemm_config_grad_input.use_fast_accum,
                False,
                self.config.pad_inner_dim,
            ),
            # grad_weight
            ScaledMMConfig(
                emulate,
                self.config.gemm_config_grad_weight.use_fast_accum,
                False,
                self.config.pad_inner_dim,
            ),
        )

        # Note: is_amax_initialized is not a buffer to avoid data dependent
        # control flow visible to dynamo
        # TODO(future PR): add serialization for this flag
        self.is_amax_initialized = not self.config.enable_amax_init

        # Syncing of amaxes and scales happens outside of this function. This
        # flag is here to enforce that the user does not forget to do this.
        self.amax_and_scale_synced = not self.config.enable_amax_init

        # This is needed to properly handle autocast in the amax/scale
        # update function for torch.float16
        self.last_seen_input_dtype = None

        # pre_forward and post_forward are currently broken with FSDP
        # and torch.compile, this option can disable them
        # Note that when using `self.config.enable_pre_and_post_forward = False`,
        # it's recommended to also set `self.config.enable_amax_init = False`.
        # Otherwise, the amax buffer would never be marked as initialized and
        # would be initialized in every iteration.
        self.enable_pre_and_post_forward = self.config.enable_pre_and_post_forward

    def create_buffers(self):
        # Default values for history buffers, see above TODO
        history_len = self.config.delayed_scaling_config.history_len
        device = self.weight.device
        # TODO(future PR): dtype values below don't have the other float8
        # flavors, fix it
        default_input = torch.finfo(torch.float8_e4m3fn).max
        default_weight = torch.finfo(torch.float8_e4m3fn).max
        default_grad_output = torch.finfo(torch.float8_e5m2).max

        # Note: for now, create all the buffers if any are needed, to postpone
        # the work to make the scale and amax syncing and history calculation
        # handle a heterogeneous setup. We can do that work later if benchmarks
        # show it is worth doing.
        if self.has_any_delayed_scaling:
            self.register_always_float32_buffer(
                "fp8_amax_input", torch.tensor([default_input], device=device)
            )
            self.register_always_float32_buffer(
                "fp8_amax_history_input", torch.zeros(history_len, device=device)
            )
            self.register_always_float32_buffer(
                "fp8_scale_input", torch.tensor([1.0], device=device)
            )
            self.register_always_float32_buffer(
                "fp8_amax_weight", torch.tensor([default_weight], device=device)
            )
            self.register_always_float32_buffer(
                "fp8_amax_history_weight", torch.zeros(history_len, device=device)
            )
            self.register_always_float32_buffer(
                "fp8_scale_weight", torch.tensor([1.0], device=device)
            )
            self.register_always_float32_buffer(
                "fp8_amax_grad_output",
                torch.tensor([default_grad_output], device=device),
            )
            self.register_always_float32_buffer(
                "fp8_amax_history_grad_output", torch.zeros(history_len, device=device)
            )
            self.register_always_float32_buffer(
                "fp8_scale_grad_output", torch.tensor([1.0], device=device)
            )

        if self.config.cast_config_input.static_scale is not None:
            self.register_always_float32_buffer(
                "fp8_static_scale_input",
                self.config.cast_config_input.static_scale.to(device),
            )
        if self.config.cast_config_weight.static_scale is not None:
            self.register_always_float32_buffer(
                "fp8_static_scale_weight",
                self.config.cast_config_weight.static_scale.to(device),
            )
        if self.config.cast_config_grad_output.static_scale is not None:
            self.register_always_float32_buffer(
                "fp8_static_scale_grad_output",
                self.config.cast_config_grad_output.static_scale.to(device),
            )

    def register_always_float32_buffer(
        self, name: str, tensor: Optional[torch.Tensor], persistent: bool = True
    ) -> None:
        self.register_buffer(name=name, tensor=tensor, persistent=persistent)
        self.always_float32_buffers.add(name)

    def _apply(self, fn, recurse=True):
        ret = super()._apply(fn, recurse)
        self.convert_amax_buffer_to_float32()
        return ret

    def convert_amax_buffer_to_float32(self):
        for key in self.always_float32_buffers:
            if self._buffers[key] is not None:
                self._buffers[key] = self._buffers[key].to(torch.float32)

    def cast_input_to_float8(
        self, input: torch.Tensor, is_amax_initialized: bool
    ) -> torch.Tensor:
        # Duplicate the autocast logic for F.linear, so that the output
        # of our module has the right original precision
        if torch.is_autocast_enabled():
            # For now, hardcode to GPU's autocast dtype
            # if we need CPU support in the future, we can add it
            autocast_dtype = torch.get_autocast_gpu_dtype()
            input = input.to(autocast_dtype)

        if self.scaling_type_input is ScalingType.DELAYED:
            scale_fn_name = self.config.delayed_scaling_config.scale_fn_name
            _maybe_initialize_amaxes_scales_for_float8_cast(
                input,
                self.fp8_amax_input,
                self.fp8_amax_history_input,
                self.fp8_scale_input,
                scale_fn_name,
                e4m3_dtype,
                is_amax_initialized,
                reduce_amax=True,
            )
            input_fp8 = hp_tensor_to_float8_delayed(
                input,
                self.fp8_scale_input,
                e4m3_dtype,
                self.fp8_amax_input,
                linear_mm_config=self.linear_mm_config,
                gemm_input_role=GemmInputRole.INPUT,
            )
        elif self.scaling_type_input is ScalingType.DYNAMIC:
            input_fp8 = hp_tensor_to_float8_dynamic(
                input, 
                e4m3_dtype, 
                self.linear_mm_config,
                gemm_input_role=GemmInputRole.INPUT,
            )
        else:
            assert self.scaling_type_input is ScalingType.STATIC
            input_fp8 = hp_tensor_to_float8_static(
                input, self.fp8_static_scale_input, e4m3_dtype, self.linear_mm_config
            )

        return input_fp8

    def get_weight_scale(self, weight: torch.Tensor) -> Optional[torch.Tensor]:
        if isinstance(weight, Float8Tensor):
            return None
        if self.scaling_type_weight is ScalingType.DELAYED:
            scale_fn_name = self.config.delayed_scaling_config.scale_fn_name
            _maybe_initialize_amaxes_scales_for_float8_cast(
                weight,
                self.fp8_amax_weight,
                self.fp8_amax_history_weight,
                self.fp8_scale_weight,
                scale_fn_name,
                e4m3_dtype,
                self.is_amax_initialized,
                reduce_amax=True,
            )
            self.fp8_amax_weight.fill_(tensor_to_amax(weight))
            return self.fp8_scale_weight
        elif self.scaling_type_weight is ScalingType.DYNAMIC:
            return tensor_to_scale(weight, e4m3_dtype)
        else:
            assert self.scaling_type_weight is ScalingType.STATIC
            return self.fp8_static_scale_weight

    def cast_weight_to_float8_t(
        self,
        weight: torch.Tensor,
        is_amax_initialized: bool,
        weight_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(weight, Float8Tensor):
            return weight.t()
        weight_fp8 = hp_tensor_and_scale_to_float8(
            weight,
            weight_scale,
            e4m3_dtype,
            self.linear_mm_config,
            gemm_input_role=GemmInputRole.WEIGHT,
        )
        return weight_fp8.t()

    def cast_weight_to_original_t(self, weight: torch.Tensor):
        if isinstance(weight, Float8Tensor):
            return weight.to_original_precision().t()
        else:
            return weight.t()

    def cast_output_to_float8_in_bw(self, output: torch.Tensor) -> torch.Tensor:
        if self.scaling_type_grad_output is ScalingType.DELAYED:
            scale_fn_name = self.config.delayed_scaling_config.scale_fn_name
            output = NoopFwToFloat8E5M2BwDelayed.apply(
                output,
                self.fp8_amax_grad_output,
                self.fp8_amax_history_grad_output,
                self.fp8_scale_grad_output,
                scale_fn_name,
                self.is_amax_initialized,
                self.linear_mm_config,
            )
        elif self.scaling_type_grad_output is ScalingType.DYNAMIC:
            output = NoopFwToFloat8E5M2BwDynamic.apply(output, self.linear_mm_config)
        else:
            assert self.scaling_type_grad_output is ScalingType.STATIC
            output = NoopFwToFloat8E5M2BwStatic.apply(
                output,
                self.fp8_static_scale_grad_output,
                self.linear_mm_config,
            )
        return output

    def float8_pre_forward(self, input):
        if not self.enable_pre_and_post_forward:
            return
        if (
            self.is_amax_initialized
            and (not self.amax_and_scale_synced)
            and torch.is_grad_enabled()
        ):
            raise AssertionError(
                "amaxes and scales not synced, please call `sync_float8_amax_and_scale_history` before forward"
            )
        self.last_seen_input_dtype = input.dtype

    def float8_post_forward(self):
        if not self.enable_pre_and_post_forward:
            return
        # Ensure that calling forward again will fail until the user syncs
        # amaxes and scales
        self.is_amax_initialized = True
        self.amax_and_scale_synced = False

    def forward_fp8_matmul(self, input: torch.Tensor) -> torch.Tensor:
        has_any_axiswise_scaling = (
            self.config.cast_config_input.scaling_granularity is ScalingGranularity.AXISWISE or
            self.config.cast_config_weight.scaling_granularity is ScalingGranularity.AXISWISE or
            self.config.cast_config_grad_output.scaling_granularity is ScalingGranularity.AXISWISE or
            self.config.cast_config_input_for_grad_weight.scaling_granularity is ScalingGranularity.AXISWISE or
            self.config.cast_config_weight_for_grad_input.scaling_granularity is ScalingGranularity.AXISWISE or
            self.config.cast_config_grad_output_for_grad_weight.scaling_granularity is ScalingGranularity.AXISWISE
        )

        if not has_any_axiswise_scaling:
            input_fp8 = self.cast_input_to_float8(input, self.is_amax_initialized)
            # If force_recompute_fp8_weight_in_bwd, we only recompute the fp8 weight,
            # weight_scale should be saved.
            weight_scale = self.get_weight_scale(self.weight)

            if self.config.force_recompute_fp8_weight_in_bwd:
                weight_fp8_t = checkpoint.checkpoint(
                    self.cast_weight_to_float8_t,
                    self.weight,
                    self.is_amax_initialized,
                    weight_scale,
                )
            else:
                weight_fp8_t = self.cast_weight_to_float8_t(
                    self.weight, self.is_amax_initialized, weight_scale
                )

            output = manual_float8_matmul_with_args_in_float8.apply(input_fp8, weight_fp8_t)

            # Cast grad_output to float8_e5m2 during backward
            output = self.cast_output_to_float8_in_bw(output)

        else:
            # for now, axiswise path is separate
            # TODO(future PR): unify to support mix and match
            output = manual_float8_matmul_with_args_in_hp.apply(
                input, 
                self.weight.t(),
                self.linear_mm_config,
                self.config,
            )
        return output
    
    def forward_original_precision_matmul(self, input: torch.Tensor) -> torch.Tensor:
        if self.config.force_recompute_fp8_weight_in_bwd:
            orig_weight_t = checkpoint.checkpoint(self.cast_weight_to_original_t, self.weight)
        else:
            orig_weight_t = self.cast_weight_to_original_t(self.weight)

        output = torch.matmul(input, orig_weight_t)
        return output

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.has_any_delayed_scaling:
            self.float8_pre_forward(input)

        if self.config.use_fp8_all_gather_only:
            output = self.forward_original_precision_matmul(input)
        else:
            output = self.forward_fp8_matmul(input)

        if self.bias is not None:
            output = output + self.bias.to(output.dtype)

        if self.has_any_delayed_scaling:
            self.float8_post_forward()
        return output

    def extra_repr(self):
        c = self.config
        ci = f"i:{c.cast_config_input.short_str()}"
        cw = f"w:{c.cast_config_weight.short_str()}"
        cgo = f"go:{c.cast_config_grad_output.short_str()}"
        parts = [ci, cw, cgo]
        if c.cast_config_input_for_grad_weight != c.cast_config_input:
            parts.append(f"i_gw:{c.cast_config_input_for_grad_weight.short_str()}") 
        if c.cast_config_weight_for_grad_input != c.cast_config_weight:
            parts.append(f"w_gi:{c.cast_config_weight_for_grad_input.short_str()}") 
        if c.cast_config_grad_output_for_grad_weight != c.cast_config_grad_output:
            parts.append(f"go_gw:{c.cast_config_grad_output_for_grad_weight.short_str()}") 
        cast_config_str = ",".join(parts)
        s = f'{super().extra_repr()}, cast_configs={cast_config_str}"'
        return s

    @classmethod
    def from_float(
        cls,
        mod,
        config: Optional[Float8LinearConfig] = None,
    ):
        """
        Create an nn.Linear with fp8 compute from a regular nn.Linear

        Args:
            mod (torch.nn.Linear): nn.Linear to convert
            config (Optional[Float8LinearConfig]): configuration for conversion to float8
        """
        if config is None:
            config = Float8LinearConfig()
        with torch.device("meta"):
            new_mod = cls(
                mod.in_features,
                mod.out_features,
                bias=False,
                config=config,
            )
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        # need to create buffers again when moving from meta device to
        # real device
        new_mod.create_buffers()

        # If FSDP float8 all-gather is on, wrap the weight in a float8-aware
        # tensor subclass. This must happen last because:
        # 1. weight needs to be on the correct device to create the buffers
        # 2. buffers need to be already created for the delayed scaling version
        #    of the weight wrapper to be initialized
        if config.enable_fsdp_float8_all_gather:
            if config.cast_config_weight.scaling_type is ScalingType.DYNAMIC:
                new_mod.weight = torch.nn.Parameter(
                    WeightWithDynamicFloat8CastTensor(
                        new_mod.weight,
                        new_mod.linear_mm_config,
                    )
                )
            elif config.cast_config_weight.scaling_type is ScalingType.DELAYED:
                new_mod.weight = torch.nn.Parameter(
                    WeightWithDelayedFloat8CastTensor(
                        new_mod.weight,
                        new_mod.fp8_amax_weight,
                        new_mod.fp8_amax_history_weight,
                        new_mod.fp8_scale_weight,
                        new_mod.linear_mm_config,
                        new_mod.is_amax_initialized,
                    )
                )
            else:
                assert config.cast_config_weight.scaling_type is ScalingType.STATIC
                new_mod.weight = torch.nn.Parameter(
                    WeightWithStaticFloat8CastTensor(
                        new_mod.weight,
                        new_mod.fp8_static_scale_weight,
                        new_mod.linear_mm_config,
                    )
                )

        return new_mod
