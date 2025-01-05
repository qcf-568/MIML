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
#### FInal output format of the inference code:
```
IoU Precision Recall F-score
```
---
#### Command for training
```
bash tools/dist_train.sh apscnet.py 8
```


