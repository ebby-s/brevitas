# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module

import brevitas
from brevitas.core.function_wrapper import RoundSte
from brevitas.core.function_wrapper import TensorClamp
from brevitas.core.quant.delay import DelayWrapper
from brevitas.function.ops import max_float
from brevitas.function.ops_ste import floor_ste


class FloatQuant(brevitas.jit.ScriptModule):

    __constants__ = ['nan_e5m2', 'nan_e4m3']

    def __init__(
            self,
            nan_e5m2: bool,
            nan_e4m3: bool,
            float_round_impl: Module = RoundSte(),
            tensor_clamp_impl: Module = TensorClamp(),
            quant_delay_steps: int = 0):
        super(FloatQuant, self).__init__()
        self.float_round_impl = float_round_impl
        self.tensor_clamp_impl = tensor_clamp_impl
        self.nan_e5m2 = nan_e5m2
        self.nan_e4m3 = nan_e4m3
        self.delay_wrapper = DelayWrapper(quant_delay_steps)

    @brevitas.jit.script_method
    def to_minifloat(self, scale: Tensor, zero_point: Tensor, exp_width: Tensor, man_width: Tensor, x: Tensor) -> Tensor:
        y_quant = x / scale
        y_quant = y_quant + zero_point
        # Round to float.
        exp_nrm = floor_ste(torch.log2(torch.abs(y_quant) + ((y_quant == 0) * (2**-126)))) - man_width
        exp_nrm = torch.clamp_min(exp_nrm, 0)
        y = y_quant * torch.exp2(-exp_nrm)
        y = self.float_round_impl(y)
        y_rnd = y * torch.exp2(exp_nrm)
        # Clamp values
        max_int_val = self.max_float(exp_width, man_width)
        min_int_val = self.min_float(exp_width, man_width)
        y_clamp = self.tensor_clamp_impl(y_rnd, min_val=min_int_val, max_val=max_int_val)
        return y_clamp

    @brevitas.jit.script_method
    def min_float(self, exp_width, man_width):
        return -max_float(exp_width, man_width, -(man_width-1), self.nan_e5m2, self.nan_e4m3)

    @brevitas.jit.script_method
    def max_float(self, exp_width, man_width):
        return max_float(exp_width, man_width, -(man_width-1), self.nan_e5m2, self.nan_e4m3)

    @brevitas.jit.script_method
    def forward(self, scale: Tensor, zero_point: Tensor, exp_width: Tensor, man_width: Tensor, x: Tensor) -> Tensor:
        y_minifloat = self.to_minifloat(scale, zero_point, exp_width, man_width, x)
        y = y_minifloat - zero_point
        y = y * scale
        y = self.delay_wrapper(x, y)
        return y


class RescalingFloatQuant(brevitas.jit.ScriptModule):

    def __init__(
            self,
            int_quant: Module,
            scaling_impl: Module,
            int_scaling_impl: Module,
            zero_point_impl: Module,
            exp_width_impl: Module,
            man_width_impl: Module):
        super(RescalingFloatQuant, self).__init__()
        self.int_quant = int_quant
        self.scaling_impl = scaling_impl
        self.int_scaling_impl = int_scaling_impl
        self.zero_point_impl = zero_point_impl
        self.msb_clamp_exp_width_impl = exp_width_impl
        self.msb_clamp_man_width_impl = man_width_impl

    @brevitas.jit.script_method
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        exp_width = self.msb_clamp_exp_width_impl()
        man_width = self.msb_clamp_man_width_impl()
        bit_width = 2**exp_width + man_width-1 + 1 # Bits from exponent, bits from mantissa, bit for signed 2's cmpl.
        threshold = self.scaling_impl(x)
        int_threshold = self.int_scaling_impl(exp_width, man_width)         # Assuming FloatScaling
        scale = threshold / int_threshold
        zero_point = self.zero_point_impl(x, scale, bit_width)            # Assuming ZeroZeroPoint.
        y = self.int_quant(scale, zero_point, exp_width, man_width, x)
        return y, scale, zero_point, bit_width
