# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class TamperDatajpgpng(CustomDataset):
    CLASSES = ('background', 'modify')
    PALETTE = [[0, 0, 0], [255, 0, 0]]
    def __init__(self, **kwargs):
        super(TamperDatajpgpng, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            **kwargs)
        assert osp.exists(self.img_dir),self.img_dir
