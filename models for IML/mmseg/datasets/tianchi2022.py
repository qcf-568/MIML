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


@DATASETS.register_module()
class TamperCOCO(CustomDataset):
    CLASSES = ('background', 'modify')
    PALETTE = [[0, 0, 0], [255, 0, 0]]
    def __init__(self, **kwargs):
        super(TamperCOCO, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            **kwargs)
        assert osp.exists(self.img_dir),self.img_dir
        with open('coco.pk', 'rb') as f:
            self.img_infos = pickle.load(f)

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        results = dict(img_info=dict(filename=os.path.join(self.img_dir, img_info['p'])), ann_info=None, gt_bboxes=img_info['b'], mask=img_info['m'])
        self.pre_pipeline(results)
        return self.pipeline(results)

    def __getitem__(self, idx):
      try:
        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)
      except:
        print('error')
        return self.__getitem__(random.randint(0,self.__len__()))
