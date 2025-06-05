# Copyright (c) OpenMMLab. All rights reserved.
from .fcn_head import FCNHead
from .psp_head import PSPHead
from .sep_aspp_head2  import DepthwiseSeparableASPPHead2
from .test_rfb1ha1c import TestRFB1HA1CLab
from .decode_seg import Decode_seg 
__all__ = [
    'FCNHead', 'PSPHead', 'DepthwiseSeparableASPPHead2', 'TestRFB1HA1CLab', 'Decode_seg'
]
