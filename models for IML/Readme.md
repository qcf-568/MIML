### This is the official implement of the APSC-Net

---

The model is available at [Google Drive](https://drive.google.com/file/d/1fTFUnn1mCO9w-YG3wa9Xqqkdn2PsSwmZ/view?usp=sharing) and [Baidu Drive](https://pan.baidu.com/s/1Y4qJOa6GWD_9MDBXmkOWBg?pwd=apsc)

---

#### Data structure

```
-----this_dir
   |
   -------Python files of this dir (e.g. apscnet.py)
   |
   -------APSC-Net.pth
   |
   -------mmseg/...
   |
   -------mmcv_custom/...
   |
   -------test_data/
              |
              -------------CASIA1/
              |               |
              |               ---------imgs/
              |               |
              |               ---------masks/
              | 
              -------------Columbia/... (the same as CASIA1 dir)
              |
              -------------Coverage/... (the same as CASIA1 dir)
              |
              -------------NIST16/...  (the same as CASIA1 dir)
              |
              -------------IMD20/... (the same as CASIA1 dir)
```

---
#### Command for evaluation
```
CASIAv1:
CUDA_VISIBLE_DEVICES=0 python casia1_infer.py --cfg apscnet.py --pth APSC-Net.pth

NIST16:
CUDA_VISIBLE_DEVICES=0 python nist_infer.py --cfg apscnet.py --pth APSC-Net.pth

IMD20:
CUDA_VISIBLE_DEVICES=0 python imd20_infer.py --cfg apscnet.py --pth APSC-Net.pth

Columbia:
CUDA_VISIBLE_DEVICES=0 python colu_infer.py --cfg apscnet.py --pth APSC-Net.pth

Coverage:
CUDA_VISIBLE_DEVICES=0 python cover_infer.py --cfg apscnet.py --pth APSC-Net.pth
```
#### Final output format of the inference code:
```
IoU Precision Recall F-score
```
#### Training dataset preparation
1. For each training dataset, prepare images and the corresponding masks. The masks should be normalized to 0-255 instead of 0-1. For example, if the mask is binary, 0 should represent authentic region and 255 should represent tampered region.
2. Rename the images' filename suffix to '.jpg' and rename the masks' filename suffix to '.png'. The prefix should be the same for a pair of image and mask. For example, a mask should be named "1.png" for its corresponding image "1.jpg".

3. Arrange the collected images and masks to below structure:
```
[Your dataset name, e.g. CASIA2]
             |
             ---------imgs
             |         |
             |         |---------1.jpg
             |         |---------2.jpg
             |
             |--------masks
                       |
                       |---------1.png
                       |---------2.png
```
For example, if the dataset name is CASIA2, all the images should be placed in 'CASIA2/imgs/' dir and all the masks should be placed in 'CASIA2/masks/' dir.
4. Run the get_pks.py to get dataset pks, specify the args "--dataset" to your dataset path:
```
python get_pks.py --dataset [your dataset path, e.g. "normed/CASIA2"]
```
After that, you will get a dir named "pks/" and a pickle file (e.g. named "CASIA2.pk") in it. The "CASIA2.pk" is a list, each item is a pair of image and mask: {'filename': '1.jpg', 'ann': {'seg_map': '1.png'}}

5. Add this dataset to the config file.
For the above example, see [this line](https://github.com/qcf-568/MIML/blob/main/models%20for%20IML/apscnet.py#L120). The [img_dir](https://github.com/qcf-568/MIML/blob/main/models%20for%20IML/apscnet.py#L126) is the path of your images dir and the [ann_dir](https://github.com/qcf-568/MIML/blob/main/models%20for%20IML/apscnet.py#L127) is the path of their masks.
Then add this dataset variable name to train/val/test list as [line 178](https://github.com/qcf-568/MIML/blob/main/models%20for%20IML/apscnet.py#L178).

---
#### Command for training
```
bash tools/dist_train.sh apscnet.py 10
```
#### Pre-trained weights for initilize before training
The ADE-20k weights are downloaded from https://github.com/facebookresearch/ConvNeXt/tree/main/semantic_segmentation
Just need to run the below 2 commands to get the "convnext_ade.pth"
```
wget https://dl.fbaipublicfiles.com/convnext/ade20k/upernet_convnext_base_22k_640x640.pth upernet_convnext_base_22k_640x640.pth
python cvt_conv.py
```


