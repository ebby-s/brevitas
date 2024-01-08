from brevitas.core.function_wrapper import TensorClamp
from brevitas.core.scaling import ScalingImplType
from brevitas.quant.base import *
from brevitas.quant.solver.act import ActQuantSolver
from brevitas.quant.solver.bias import BiasQuantSolver
from brevitas.quant.solver.trunc import TruncQuantSolver
from brevitas.quant.solver.weight import WeightQuantSolver

__all__ = [
    'E3M2WeightPerTensorFixedPoint',
    'E3M2ActPerTensorFixedPoint',
    'E3M2WeightPerChannelFixedPoint',
    'E3M2ActPerChannelFixedPoint',
    'E3M2WeightPerBlockFixedPoint',
    'E3M2ActPerBlockFixedPoint']


class E3M2WeightPerTensorFixedPoint(FloatQuant,
                                    MaxStatsScaling,
                                    PerTensorFPPoTScalingE3M2,
                                    WeightQuantSolver):
    """
    """
    scaling_impl_type = ScalingImplType.DYNAMIC_STATS
    pass


class E3M2ActPerTensorFixedPoint(FloatQuant,
                                 MaxStatsScaling,
                                 PerTensorFPPoTScalingE3M2,
                                 ActQuantSolver):
    """
    """
    scaling_impl_type = ScalingImplType.DYNAMIC_STATS
    pass


class E3M2WeightPerChannelFixedPoint(FloatQuant,
                                    MaxStatsScaling,
                                    PerChannelFPPoTScalingE3M2,
                                    WeightQuantSolver):
    """
    """
    scaling_impl_type = ScalingImplType.DYNAMIC_STATS
    pass


class E3M2ActPerChannelFixedPoint(FloatQuant,
                                 MaxStatsScaling,
                                 PerChannelFPPoTScalingE3M2,
                                 ActQuantSolver):
    """
    """
    scaling_impl_type = ScalingImplType.DYNAMIC_STATS
    # per_channel_broadcastable_shape = (1,2,1,1) # Zero all but channel dim.
    # scaling_stats_permute_dims = (1,0,2,3)      # Make channel dim first.


class E3M2WeightPerBlockFixedPoint(FloatQuant,
                                    MaxStatsScaling,
                                    PerBlockFPPoTScalingE3M2,
                                    WeightQuantSolver):
    """
    """
    scaling_impl_type = ScalingImplType.DYNAMIC_STATS
    stats_reduce_dim = 1


class E3M2ActPerBlockFixedPoint(FloatQuant,
                                 MaxStatsScaling,
                                 PerBlockFPPoTScalingE3M2,
                                 ActQuantSolver):
    """
    """
    scaling_impl_type = ScalingImplType.DYNAMIC_STATS
    # scaling_shape = (1,2,3,3)
    stats_reduce_dim = 1
