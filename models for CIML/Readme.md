### This part is under construction...

The classifiers are available at Google Drive and Baidu Drive.

The DASS model is available at Google Drive and [Baidu Drive](https://pan.baidu.com/s/1lmksoTe2b2xObGkhUbd5-A?pwd=DASS).

The SAFM model is available at Google Drive and [Baidu Drive](https://pan.baidu.com/s/1PnLepP7bAd-8L5NcUGBx4A?pwd=SAFM).


CUDA_VISIBLE_DEVICES=0 python classify_convxl.py

CUDA_VISIBLE_DEVICES=0 python Auto_Annotate_SPG.py --pth DASS.pth

CUDA_VISIBLE_DEVICES=0 python Auto_Annotate_SDG.py --pth SAFM.pth
