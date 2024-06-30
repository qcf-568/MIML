### This is the official implement of Category-Aware Auto-Annotation (CAAA)


![CAAA](https://github.com/qcf-568/MIML/blob/main/models%20for%20CIML/CAAA_OK.png)


The classifiers are available at [Google Drive](https://drive.google.com/file/d/1OMGtuzqhjwcvDaP3OO1njPfAS_2s0vg8/view?usp=sharing) and [Baidu Drive](https://pan.baidu.com/s/1-NidYwgVZUA0Pi0KE3ngGw?pwd=conv).

The DASS model is available at [Google Drive](https://drive.google.com/file/d/1PXL9e8XiRGlSIcGhhppLXJtVG2rdQh5a/view?usp=sharing) and [Baidu Drive](https://pan.baidu.com/s/1lmksoTe2b2xObGkhUbd5-A?pwd=DASS).

The SACM model is available at [Google Drive](https://drive.google.com/file/d/1_C5gATKv8Mh7SyKNE_ubSpXlEASkEYja/view?usp=sharing) and [Baidu Drive](https://pan.baidu.com/s/1PnLepP7bAd-8L5NcUGBx4A?pwd=SAFM).



To leverage the CAAA for auto-annotation, you should first categorize the image pairs (each pair contains a forged image and its authentic image) into aligned SPG and SDG. Then construct the dir structure as follows:

```
        roots (dir of SPG or SDG pairs)
            |
            |---dir1
            |     |----0.jpg (authentic image)
            |     |----1.jpg (manipulated image)
            |
            |---dir2
            |     |----0.jpg (authentic image)
            |     |----1.jpg (manipulated image)
            |
      ..........
 ```

Then run the scripts for auto-annotation.


Commands to run the classifier to catogerize the image pairs into SPG or SDG:
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
