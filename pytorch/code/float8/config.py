# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import enum
import logging
from dataclasses import dataclass
from typing import Optional

import torch

logger: logging.Logger = logging.getLogger()

class ScalingType(enum.Enum):
    """定义Float8缩放类型的枚举类"""
    # 延迟缩放 - 使用历史amax值来计算缩放因子
    DELAYED = "delayed"
    # 动态缩放 - 根据当前张量动态计算缩放因子
    DYNAMIC = "dynamic"  
    # 静态缩放 - 使用固定的缩放因子
    STATIC = "static"
    # 禁用缩放 - 跳过缩放,保持原始精度
    DISABLED = "disabled"

    def short_str(self):
        """返回缩放类型的简短字符串表示"""
        if self is ScalingType.DELAYED:
            return "del"
        elif self is ScalingType.DYNAMIC:
            return "dyn"
        elif self is ScalingType.STATIC:
            return "sta"
        else:
            assert self is ScalingType.DISABLED
            return "dis"


class ScalingGranularity(enum.Enum):
    """定义Float8缩放粒度的枚举类"""

    # 对整个张量使用单个缩放因子
    TENSORWISE = "tensorwise"
    # 沿张量的一个轴计算缩放因子,将其降为大小1
    AXISWISE = "axiswise"

    def short_str(self):
        """返回缩放粒度的简短字符串表示"""
        if self is ScalingGranularity.TENSORWISE:
            return "ten"
        else:
            assert self is ScalingGranularity.AXISWISE
            return "axs"


@dataclass(frozen=True)
class CastConfig:
    """单个张量转换为Float8的配置类"""

    # 缩放类型,默认为动态缩放
    scaling_type: ScalingType = ScalingType.DYNAMIC
    # 缩放粒度,默认为张量级
    scaling_granularity: ScalingGranularity = ScalingGranularity.TENSORWISE
    # 静态缩放因子,仅在静态缩放时使用
    static_scale: Optional[torch.Tensor] = None

    def short_str(self):
        """返回配置的简短字符串表示"""
        return f"{self.scaling_type.short_str()}_{self.scaling_granularity.short_str()}"

    def __post_init__(self):
        """初始化后的验证"""
        if self.scaling_type is ScalingType.STATIC:
            assert (
                self.static_scale is not None
            ), "static_scale must be specified for static scaling"
        if self.scaling_granularity is ScalingGranularity.AXISWISE:
            assert self.scaling_type is ScalingType.DYNAMIC, \
                "only dynamic scaling type is supported for axiswise scaling granularity"

@dataclass(frozen=True)
class DelayedScalingConfig:
    """延迟缩放的配置类"""

    # 控制amax缓冲区的历史长度
    history_len: int = 16

    # 控制如何从amax历史计算当前缩放因子的函数名
    scale_fn_name: str = "max"

    def __post_init__(self):
        """初始化后的验证"""
        assert (
            self.scale_fn_name == "max"
        ), f"{self.scale_fn_name} is not implemented yet. Only max is supported for now."


@dataclass
class Float8TypeConfig:
    """Float8类型配置类,用于选择首选的Float8类型对"""

    # 首选的e4m3类型
    e4m3_dtype = torch.float8_e4m3fn

    # 首选的e5m2类型
    e5m2_dtype = torch.float8_e5m2

    def __post_init__(self):
        """初始化后的处理,为ROCm平台设置特定的类型"""
        if torch.version.hip and torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            MI300_ARCH = ("gfx940", "gfx941", "gfx942")
            if prop.gcnArchName.split(":")[0] in MI300_ARCH:
                self.e4m3_dtype = torch.float8_e4m3fnuz
                self.e5m2_dtype = torch.float8_e5m2fnuz


@dataclass(frozen=True)
class Float8GemmConfig:
    """Float8 GEMM操作的配置类"""

    # 是否使用低精度的快速累加
    # 注意:当使用模拟时此标志无效
    use_fast_accum: bool = False


@dataclass(frozen=True)
class Float8LinearConfig:
    """
    用于将torch.nn.Linear模块转换为float8进行训练的配置类。
    """

    #
    # 用于转换input、weight、grad_output的每个张量的配置,
    # 这些张量用于计算output、grad_weight和grad_input的gemm操作。
    #
    # 注意:
    # 1. 如果cast_config_input_for_grad_weight为None,则
    #    cast_config_input用于缩放两个使用input的gemm中的input
    # 2. 如果指定了cast_config_input_for_grad_weight,则
    #    a. cast_config_input用于缩放计算output的gemm中的input
    #    b. cast_config_input_for_grad_weight用于缩放计算grad_weight的gemm中的input
    # 3. 同样的行为也适用于cast_config_weight和cast_config_grad_output
    #
    # input相关配置
    cast_config_input: CastConfig = CastConfig()
    cast_config_input_for_grad_weight: Optional[CastConfig] = None
    # weight相关配置  
    cast_config_weight: CastConfig = CastConfig()
    cast_config_weight_for_grad_input: Optional[CastConfig] = None
    # grad_output相关配置
    cast_config_grad_output: CastConfig = CastConfig()
    cast_config_grad_output_for_grad_weight: Optional[CastConfig] = None

    #
    # 用于计算output、grad_input和grad_weight的每个gemm操作的配置
    # TODO:如果fast_accum为False与axiswise缩放一起使用时抛出警告
    #
    gemm_config_output: Float8GemmConfig = Float8GemmConfig(use_fast_accum=True)
    gemm_config_grad_input: Float8GemmConfig = Float8GemmConfig()
    gemm_config_grad_weight: Float8GemmConfig = Float8GemmConfig()

    #
    # 每个linear层的配置
    #

    # 如果为True,在Float8Linear的第一次迭代时,amaxes将使用输入数据初始化。
    # 截至2023-12-30,这在autocast + torch.compile + FSDP下不起作用。
    # 启用此选项对测试有帮助,但对实际训练任务不是必需的。
    enable_amax_init: bool = True

    # 如果为True,将运行pre-forward和post-forward函数。
    # 截至2023-12-30,这在autocast + torch.compile + FSDP下不起作用。
    # 启用此选项对安全性有帮助,但不是严格必需的。
    enable_pre_and_post_forward: bool = True

    # 如果为True,则使用float8 linear模块权重的张量子类,
    # 该子类实现pre/post-all-gather方法以在FSDP2中进行float8 all-gather。
    enable_fsdp_float8_all_gather: bool = False

    # 如果为True,在执行fp8缩放的矩阵乘法之前,
    # 我们将用0填充a(维度1)和b(维度2)的内部维度。
    # 这对于_scaled_mm的矩阵乘法是必需的,因为它有一个强约束:
    # 对于M,N,K,N和K必须是16的倍数。
    # 但这可能导致内存峰值,因此默认关闭。
    pad_inner_dim: bool = False

    # 如果为True,使用模拟而不是硬件加速的gemm
    emulate: bool = False

    # 延迟缩放的配置
    # 注意:这实际上是按张量应用的,但目前只支持对模型中的所有张量和层使用相同的配置。
    # 如果将来我们添加对更细粒度配置的支持,这个字段可能会移到每个张量的配置中。
    delayed_scaling_config: DelayedScalingConfig = DelayedScalingConfig()

    # 如果启用此选项,fp8_weight将始终在反向传播中重新计算。
    # 建议在使用FSDP时启用此标志。
    # 否则,可能会保存整个fp8_weight,而不是分片的权重。
    # 如果使用外部激活检查点上下文或SAC,您可以禁用此选项,
    # 并在自定义的AC上下文中处理fp8权重的重新计算。
    force_recompute_fp8_weight_in_bwd: bool = False

    # 如果为True,我们只使用fp8-all-gather来减少通信成本。
    # gemm计算仍然在原始精度下进行。
    # cast_config_weight用于决定如何将权重转换为fp8,
    # 其他转换配置将被忽略。
    use_fp8_all_gather_only: bool = False

    def __post_init__(self):
        # 如果用户未指定,则填充额外的转换覆盖
        # 注意:这通过使用object.__setattr__绕过了这个dataclass的frozen属性。
        # 这没问题,因为我们真正需要的是这个对象在__post_init__之后被冻结,
        # 以便torch.compile能够工作。
        if self.cast_config_input_for_grad_weight is None:
            object.__setattr__(self, "cast_config_input_for_grad_weight", self.cast_config_input)
        if self.cast_config_weight_for_grad_input is None:
            object.__setattr__(self, "cast_config_weight_for_grad_input", self.cast_config_weight)
        if self.cast_config_grad_output_for_grad_weight is None:
            object.__setattr__(self, "cast_config_grad_output_for_grad_weight", self.cast_config_grad_output)

        # float8 all-gather只支持tensorwise,将来可能支持blockwise
        if self.cast_config_weight.scaling_granularity != ScalingGranularity.TENSORWISE:
            assert not self.enable_fsdp_float8_all_gather, \
                f"enable_fsdp_float8_all_gather只支持tensorwise缩放粒度,得到{self.cast_config_weight.scaling_granularity}"

        # 在兼容性检查中保存一些字符
        cc_i = self.cast_config_input
        cc_w = self.cast_config_weight
        cc_go = self.cast_config_grad_output
        cc_i_gw = self.cast_config_input_for_grad_weight
        cc_w_gi = self.cast_config_weight_for_grad_input
        cc_go_gw = self.cast_config_grad_output_for_grad_weight
        # 目前,我们只有两个操作数要么都是高精度,要么都是float8的gemm内核。
        # 将来可能会放宽这个限制。
        # TODO:使用特定的dtypes使float8检查更精确。
        for cc1, cc2, gemm_name in (
            (cc_i, cc_w, "output"),
            (cc_go, cc_w_gi, "grad_input"),
            (cc_i_gw, cc_go_gw, "grad_weight"),
        ):
            is_disabled_1 = cc1.scaling_type is ScalingType.DISABLED
            is_disabled_2 = cc1.scaling_type is ScalingType.DISABLED
            assert is_disabled_1 == is_disabled_2, \
                f"{gemm_name}的操作数精度不兼容"
        
        if self.use_fp8_all_gather_only:
            assert self.enable_fsdp_float8_all_gather, "use_fp8_all_gather_only需要enable_fsdp_float8_all_gather为True"
            
        # 有关此警告的更多详细信息,请参见force_recompute_fp8_weight_in_bwd周围的注释。
        if (
            self.enable_fsdp_float8_all_gather
            and not self.force_recompute_fp8_weight_in_bwd
        ):
            logger.warning(
                "使用FSDP时,建议启用config.force_recompute_fp8_weight_in_bwd。"
            )
               


# 预制的常用配置方案
# TODO(未来PR): 对此进行一轮设计,最终作为顶层公共API暴露出去
class Float8LinearRecipeName(enum.Enum):
    # 所有操作使用tensorwise缩放粒度
    ALL_TENSORWISE = "all_tensorwise"  
    # 所有操作使用axiswise缩放粒度
    ALL_AXISWISE = "all_axiswise"
    # 前向使用axiswise缩放,权重梯度使用高精度计算
    LW_AXISWISE_WITH_GW_HP = "lw_axiswise_with_gw_hp"


def recipe_name_to_linear_config(
    recipe_name: Float8LinearRecipeName,
) -> Float8LinearConfig:
    """
    输入: `Float8LinearRecipeName` 枚举值
    输出: 根据配方名称返回对应的 `Float8LinearConfig` 配置
    """

    if recipe_name is Float8LinearRecipeName.ALL_TENSORWISE:
        # 默认配置,使用cuBLAS tensorwise内核进行动态per-tensor缩放
        return Float8LinearConfig()

    elif recipe_name is Float8LinearRecipeName.ALL_AXISWISE:
        # 使用CUTLASS rowwise内核进行动态axiswise缩放
        cc_i = CastConfig(scaling_granularity=ScalingGranularity.AXISWISE)  # 输入使用axiswise缩放
        cc_w = CastConfig(scaling_granularity=ScalingGranularity.AXISWISE)  # 权重使用axiswise缩放
        cc_go = CastConfig(scaling_granularity=ScalingGranularity.AXISWISE) # 梯度输出使用axiswise缩放

        # 当前torch._scaled_mm中的rowwise CUTLASS内核只有在use_fast_accum=True时才快
        # 注意rowwise缩放比tensorwise缩放更精确,所以考虑这个标志时,
        # tensorwise和rowwise对精度的整体影响会有所不同
        gc_o = Float8GemmConfig(use_fast_accum=True)   # 输出gemm配置
        gc_gi = Float8GemmConfig(use_fast_accum=True)  # 梯度输入gemm配置  
        gc_gw = Float8GemmConfig(use_fast_accum=True)  # 梯度权重gemm配置

        return Float8LinearConfig(
            cast_config_input=cc_i,
            cast_config_weight=cc_w,
            cast_config_grad_output=cc_go,
            gemm_config_output=gc_o,
            gemm_config_grad_input=gc_gi,
            gemm_config_grad_weight=gc_gw,
        )

    elif recipe_name is Float8LinearRecipeName.LW_AXISWISE_WITH_GW_HP:

        # lw对all-axiswise的修改配方:
        #
        #   output_hp = input_fp8_axiswise_dim0 @ weight_t_axiswise_dim1
        #   grad_input_hp = grad_output_fp8_axiswise_dim0 @ weight_fp8_tensorwise
        #   grad_weight_hp = input_t_hp @ grad_output_hp
        #
        # 主要特点:
        #   * 提高了grad_weight的精度
        #   * 与普通的all-axiswise相比,input、weight和grad_output现在只需要在单个维度上进行axiswise缩放,
        #     这更适合快速内核

        # output_hp = input_fp8_axiswise_dim0 @ weight_t_axiswise_dim1
        cc_i = CastConfig(scaling_granularity=ScalingGranularity.AXISWISE)  # 输入使用axiswise缩放
        cc_w = CastConfig(scaling_granularity=ScalingGranularity.AXISWISE)  # 权重使用axiswise缩放

        # grad_input_hp = grad_output_fp8_axiswise_dim0 @ weight_fp8_tensorwise
        cc_go = CastConfig(scaling_granularity=ScalingGranularity.AXISWISE)     # 梯度输出使用axiswise缩放
        cc_w_gi = CastConfig(scaling_granularity=ScalingGranularity.TENSORWISE) # 用于计算梯度输入的权重使用tensorwise缩放

        # grad_weight_hp = input_t_hp @ grad_output_hp
        cc_i_gw = CastConfig(scaling_type=ScalingType.DISABLED)   # 用于计算梯度权重的输入禁用缩放
        cc_go_gw = CastConfig(scaling_type=ScalingType.DISABLED)  # 用于计算梯度权重的梯度输出禁用缩放

        # 当前torch._scaled_mm中的rowwise CUTLASS内核只有在use_fast_accum=True时才快
        # 注意rowwise缩放比tensorwise缩放更精确,所以考虑这个标志时,
        # tensorwise和rowwise对精度的整体影响会有所不同
        gc_o = Float8GemmConfig(use_fast_accum=True)   # 输出gemm配置
        gc_gi = Float8GemmConfig(use_fast_accum=True)  # 梯度输入gemm配置
        gc_gw = Float8GemmConfig(use_fast_accum=True)  # 梯度权重gemm配置

        return Float8LinearConfig(
            cast_config_input=cc_i,
            cast_config_weight=cc_w,
            cast_config_grad_output=cc_go,
            cast_config_input_for_grad_weight=cc_i_gw,
            cast_config_weight_for_grad_input=cc_w_gi,
            cast_config_grad_output_for_grad_weight=cc_go_gw,
            gemm_config_output=gc_o,
            gemm_config_grad_input=gc_gi,
            gemm_config_grad_weight=gc_gw,
        )

    else:
        raise AssertionError(f"未知的配方名称 {recipe_name}")
