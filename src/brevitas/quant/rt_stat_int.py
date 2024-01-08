# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from brevitas.core.scaling import ScalingImplType
from brevitas.quant.base import *
from brevitas.quant.solver.act import ActQuantSolver
from brevitas.quant.solver.weight import WeightQuantSolver
from brevitas.inject.enum import StatsOp

__all__ = [
    'Int8WeightPerTensorFixedPointRtS',
    'Int8WeightPerChannelFixedPointRtS',
    'Int8WeightPerBlockFixedPointRtS',
    'Int8ActPerTensorFixedPointRtS',
    'Int8ActPerChannelFixedPointRtS',
    'Int8ActPerBlockFixedPointRtS',
    'Int8WeightPerTensorFixedPointPercentileRtS',
    'Int8WeightPerChannelFixedPointPercentileRtS',
    'Int8WeightPerBlockFixedPointPercentileRtS',
    'Int8ActPerTensorFixedPointPercentileRtS',
    'Int8ActPerChannelFixedPointPercentileRtS',
    'Int8ActPerBlockFixedPointPercentileRtS',
    'FullInt8WeightPerTensorFixedPointRtS',
    'FullInt8WeightPerChannelFixedPointRtS',
    'FullInt8WeightPerBlockFixedPointRtS',
    'FullInt8ActPerTensorFixedPointRtS',
    'FullInt8ActPerChannelFixedPointRtS',
    'FullInt8ActPerBlockFixedPointRtS']


class Int8WeightPerTensorFixedPointRtS(NarrowIntQuant,
                                    MaxStatsScaling,
                                    PerTensorPoTScaling8bit,
                                    WeightQuantSolver):
    """
    """
    scaling_impl_type = ScalingImplType.DYNAMIC_STATS
    pass


class Int8ActPerTensorFixedPointRtS(NarrowIntQuant,
                                 MaxStatsScaling,
                                 PerTensorPoTScaling8bit,
                                 ActQuantSolver):
    """
    """
    scaling_impl_type = ScalingImplType.DYNAMIC_STATS
    pass


class Int8WeightPerChannelFixedPointRtS(NarrowIntQuant,
                                     MaxStatsScaling,
                                     PerChannelPoTScaling8bit,
                                     WeightQuantSolver):
    """
    """
    scaling_impl_type = ScalingImplType.DYNAMIC_STATS
    pass


class Int8ActPerChannelFixedPointRtS(NarrowIntQuant,
                                 MaxStatsScaling,
                                 PerChannelPoTScaling8bit,
                                 ActQuantSolver):
    """
    """
    scaling_impl_type = ScalingImplType.DYNAMIC_STATS
    pass


class Int8WeightPerBlockFixedPointRtS(NarrowIntQuant,
                                     MaxStatsScaling,
                                     PerBlockPoTScaling8bit,
                                     WeightQuantSolver):
    """
    """
    scaling_impl_type = ScalingImplType.DYNAMIC_STATS
    stats_reduce_dim = 1
    pass


class Int8ActPerBlockFixedPointRtS(NarrowIntQuant,
                                 MaxStatsScaling,
                                 PerBlockPoTScaling8bit,
                                 ActQuantSolver):
    """
    """
    scaling_impl_type = ScalingImplType.DYNAMIC_STATS
    stats_reduce_dim = 1
    pass


class Int8WeightPerTensorFixedPointPercentileRtS(NarrowIntQuant,
                                    PercentileStatsScaling,
                                    PerTensorPoTScaling8bit,
                                    WeightQuantSolver):
    """
    """
    scaling_impl_type = ScalingImplType.DYNAMIC_STATS
    pass


class Int8ActPerTensorFixedPointPercentileRtS(NarrowIntQuant,
                                 PercentileStatsScaling,
                                 PerTensorPoTScaling8bit,
                                 ActQuantSolver):
    """
    """
    scaling_impl_type = ScalingImplType.DYNAMIC_STATS
    pass


class Int8WeightPerChannelFixedPointPercentileRtS(NarrowIntQuant,
                                     PercentileStatsScaling,
                                     PerChannelPoTScaling8bit,
                                     WeightQuantSolver):
    """
    """
    scaling_impl_type = ScalingImplType.DYNAMIC_STATS
    pass


class Int8ActPerChannelFixedPointPercentileRtS(NarrowIntQuant,
                                 PercentileStatsScaling,
                                 PerChannelPoTScaling8bit,
                                 ActQuantSolver):
    """
    """
    scaling_impl_type = ScalingImplType.DYNAMIC_STATS
    pass


class Int8WeightPerBlockFixedPointPercentileRtS(NarrowIntQuant,
                                     PercentileStatsScaling,
                                     PerBlockPoTScaling8bit,
                                     WeightQuantSolver):
    """
    """
    scaling_impl_type = ScalingImplType.DYNAMIC_STATS
    stats_reduce_dim = 1
    pass


class Int8ActPerBlockFixedPointPercentileRtS(NarrowIntQuant,
                                 PercentileStatsScaling,
                                 PerBlockPoTScaling8bit,
                                 ActQuantSolver):
    """
    """
    scaling_impl_type = ScalingImplType.DYNAMIC_STATS
    stats_reduce_dim = 1
    pass


class FullInt8WeightPerTensorFixedPointRtS(IntQuant,
                                    MaxStatsScaling,
                                    PerTensorPoTScaling8bit,
                                    WeightQuantSolver):
    """
    """
    scaling_impl_type = ScalingImplType.DYNAMIC_STATS
    pass


class FullInt8ActPerTensorFixedPointRtS(IntQuant,
                                 MaxStatsScaling,
                                 PerTensorPoTScaling8bit,
                                 ActQuantSolver):
    """
    """
    scaling_impl_type = ScalingImplType.DYNAMIC_STATS
    pass


class FullInt8WeightPerChannelFixedPointRtS(IntQuant,
                                     MaxStatsScaling,
                                     PerChannelPoTScaling8bit,
                                     WeightQuantSolver):
    """
    """
    scaling_impl_type = ScalingImplType.DYNAMIC_STATS
    pass


class FullInt8ActPerChannelFixedPointRtS(IntQuant,
                                 MaxStatsScaling,
                                 PerChannelPoTScaling8bit,
                                 ActQuantSolver):
    """
    """
    scaling_impl_type = ScalingImplType.DYNAMIC_STATS
    pass


class FullInt8WeightPerBlockFixedPointRtS(IntQuant,
                                     MaxStatsScaling,
                                     PerBlockPoTScaling8bit,
                                     WeightQuantSolver):
    """
    """
    scaling_impl_type = ScalingImplType.DYNAMIC_STATS
    stats_reduce_dim = 1
    pass


class FullInt8ActPerBlockFixedPointRtS(IntQuant,
                                 MaxStatsScaling,
                                 PerBlockPoTScaling8bit,
                                 ActQuantSolver):
    """
    """
    scaling_impl_type = ScalingImplType.DYNAMIC_STATS
    stats_reduce_dim = 1
    pass
