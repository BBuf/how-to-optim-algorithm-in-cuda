# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
Defines an nn module designed to be used during inference
"""

from typing import NamedTuple, Optional, Tuple

import torch

from torchao.float8.float8_utils import is_row_major, pad_tensor_for_matmul

Tensor = torch.Tensor


class Float8MMConfig(NamedTuple):
    """
    Configuration for the scaled_mm in the forward and backward pass.

    Attributes:
        emulate (bool): Whether to emulate the matmuls in fp32.
        use_fast_accum (bool): Whether to use the fast-accumulation option for scaled_mm.
        pad_inner_dim (bool): Whether to pad the inner dimension of a and b with 0s.
                              This is needed for matmuls not aligned to 16.
    """

    emulate: bool = False
    use_fast_accum: bool = False
    pad_inner_dim: bool = False


def preprocess_data(
    a_data: Tensor,
    b_data: Tensor,
    scaled_mm_config: Float8MMConfig,
) -> Tuple[Tensor, Tensor]:
    """Preprocess the inner fp8 data tensors for admmm
    Args:
        a_data: Input tensor A.
        b_data: Input tensor B.
        scaled_mm_config: Configuration for _scaled_mm.
    Returns:
        Preprocessed tensors A and B in the format for _scaled_mm.
    """
    if scaled_mm_config.pad_inner_dim:
        assert a_data.size(1) == b_data.size(
            0
        ), f"Inner dims must match for mm, got {a_data.size(1)} and {b_data.size(0)}"
        a_data = pad_tensor_for_matmul(a_data, dims=1)
        b_data = pad_tensor_for_matmul(b_data, dims=0)
    if not is_row_major(a_data.stride()):
        a_data = a_data.contiguous()
    if is_row_major(b_data.stride()):
        b_data = b_data.t().contiguous().t()
    return a_data, b_data


def addmm_float8_unwrapped_inference(
    a_data: Tensor,
    a_scale: Tensor,
    b_data: Tensor,
    b_scale: Tensor,
    output_dtype: torch.dtype,
    output_scale: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    use_fast_accum: bool = False,
) -> Tensor:
    """
    This is the unwrapped version of addmm_float8, which does not take in Float8Tensors
    as inputs. This is used to standardize the logic between subclassed and non subclassed
    versions of the linear module.
    """

    if output_dtype == torch.float32 and bias is not None:
        # Bias is not supported by _scaled_mm when output is fp32
        output = torch._scaled_mm(
            a_data,
            b_data,
            scale_a=a_scale,
            scale_b=b_scale,
            scale_result=output_scale,
            out_dtype=output_dtype,
            use_fast_accum=use_fast_accum,
        )
        output += bias
        return output
    output = torch._scaled_mm(
        a_data,
        b_data,
        scale_a=a_scale,
        scale_b=b_scale,
        bias=bias,
        scale_result=output_scale,
        out_dtype=output_dtype,
        use_fast_accum=use_fast_accum,
    )
    return output


def _is_rowwise_scaled(x) -> bool:
    """Checks if an AQT tensor is rowwise scaled
    Args:
        x: AffineQuantizedTensor tensor
    """
    return x.block_size == (1,) * (x.dim() - 1) + (x.shape[-1],)
