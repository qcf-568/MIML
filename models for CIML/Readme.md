### This part is under construction...

The classifiers are available at Google Drive and [Baidu Drive](https://pan.baidu.com/s/1-NidYwgVZUA0Pi0KE3ngGw?pwd=conv).

The DASS model is available at Google Drive and [Baidu Drive](https://pan.baidu.com/s/1lmksoTe2b2xObGkhUbd5-A?pwd=DASS).

The SAFM model is available at Google Drive and [Baidu Drive](https://pan.baidu.com/s/1PnLepP7bAd-8L5NcUGBx4A?pwd=SAFM).


Commands to run the classifier:
```
CUDA_VISIBLE_DEVICES=0 python classify_convxl.py
```


Commands to run the DASS to auto-annotate the image pairs in SPG:
```
CUDA_VISIBLE_DEVICES=0 python Auto_Annotate_SPG.py --pth DASS.pth
```


Commands to run the SACM to auto-annotate the image pairs in SDG:

```
CUDA_VISIBLE_DEVICES=0 python Auto_Annotate_SDG.py --pth SAFM.pth
```
