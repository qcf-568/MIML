# Copyright (c) OpenMMLab. All rights reserved.
from .fcn_head import FCNHead
from .uper_head import UPerHead
from .sep_aspp_head import DepthwiseSeparableASPPHead

__all__ = [
    'FCNHead', 'UPerHead', 'DepthwiseSeparableASPPHead'
]
