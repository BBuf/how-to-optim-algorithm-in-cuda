# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
This file defines the Python functions for float8 which expect inputs
of class `Float8Tensor`. This is a thin wrapper on top of the aten API
to simplify the product code.
"""

from typing import Optional

import torch


# [Note] Usage of scales
# The meaning of scale in this library can be found in the definition of the Float8Tensor
# Cublas defines scale to always mean a multiplicative factor for the respective matrices
# For a,b going from fp8 -> fp32 we multiple by the inverse of the scale
# For output going from fp32 -> fp8 we multiply by the scale
def addmm_float8_unwrapped(
    a_data: torch.Tensor,
    a_scale: torch.Tensor,
    b_data: torch.Tensor,
    b_scale: torch.tensor,
    output_dtype: torch.dtype,
    output_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    use_fast_accum: bool = False,
) -> torch.Tensor:
    """
    This is the unwrapped version of addmm_float8, which does not take in Float8Tensors
    as inputs. This is used to standardize the logic between subclassed and non subclassed
    versions of the linear module.
    """
    a_inverse_scale = a_scale.reciprocal()
    b_inverse_scale = b_scale.reciprocal()

    if output_dtype == torch.float32 and bias is not None:
        # Bias is not supported by _scaled_mm when output is fp32
        output = torch._scaled_mm(
            a_data,
            b_data,
            scale_a=a_inverse_scale,
            scale_b=b_inverse_scale,
            scale_result=output_scale,
            out_dtype=output_dtype,
            use_fast_accum=use_fast_accum,
        )
        output += bias
        return output
    output = torch._scaled_mm(
        a_data,
        b_data,
        scale_a=a_inverse_scale,
        scale_b=b_inverse_scale,
        bias=bias,
        scale_result=output_scale,
        out_dtype=output_dtype,
        use_fast_accum=use_fast_accum,
    )
    return output
