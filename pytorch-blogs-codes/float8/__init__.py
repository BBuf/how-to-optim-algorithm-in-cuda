# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# Lets define a few top level things here
from torchao.float8.config import (
    CastConfig,
    DelayedScalingConfig,
    Float8GemmConfig,
    Float8LinearConfig,
    ScalingType,
)
from torchao.float8.float8_linear import Float8Linear, WeightWithDelayedFloat8CastTensor
from torchao.float8.float8_linear_utils import (
    convert_to_float8_training,
    linear_requires_sync,
    sync_float8_amax_and_scale_history,
)
from torchao.float8.float8_tensor import (
    Float8Tensor,
    GemmInputRole,
    LinearMMConfig,
    ScaledMMConfig,
)
from torchao.float8.inference import Float8MMConfig
from torchao.float8.fsdp_utils import precompute_float8_dynamic_scale_for_fsdp

from torchao.utils import TORCH_VERSION_AT_LEAST_2_5


if TORCH_VERSION_AT_LEAST_2_5:
    # Needed to load Float8Tensor with weights_only = True
    from torch.serialization import add_safe_globals

    add_safe_globals(
        [
            Float8Tensor,
            ScaledMMConfig,
            GemmInputRole,
            LinearMMConfig,
            Float8MMConfig,
            WeightWithDelayedFloat8CastTensor,
        ]
    )

__all__ = [
    # configuration
    "DelayedScalingConfig",
    "ScalingType",
    "Float8GemmConfig",
    "Float8LinearConfig",
    "CastConfig",
    # top level UX
    "convert_to_float8_training",
    "linear_requires_sync",
    "sync_float8_amax_and_scale_history",
    "precompute_float8_dynamic_scale_for_fsdp",
    # note: Float8Tensor and Float8Linear are not public APIs
]
