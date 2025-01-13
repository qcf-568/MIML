# Towards Modern Image Manipulation Localization: A Large-Scale Dataset and Novel Methods

### This is the official implementation of the paper Towards Modern Image Manipulation Localization: A Large-Scale Dataset and Novel Methods.  [paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Qu_Towards_Modern_Image_Manipulation_Localization_A_Large-Scale_Dataset_and_Novel_CVPR_2024_paper.pdf)<br/>

---

### The main contributions of this work are as follows:
* <font size=10>We propose to harness Constrained Image Manipulation Localization (CIML) models to automatically annotate the numerous unlabelled manually forged images from the web (e.g. those from [PhotoshopBattles](https://www.reddit.com/r/photoshopbattles/)). Thereby addressing the severe scarcity of non-synthetic data for image manipulation localization.</font>
* <font size=10>We propose a novel and effective paradigm [CAAA](https://github.com/qcf-568/MIML/tree/main/models%20for%20CIML) for constrained image manipulation localization, which significantly improves the accuracy of the automatic annotations. We believe that this is the best paradigm for CIML-based auto-annotation.</font>
* <font size=10>We propose a novel metric [QES](https://github.com/qcf-568/MIML/blob/main/other%20scripts%20for%20the%20dataset's%20construction/QES.py) to automatically to filter out the possible bad annotations, which is crucial to ensure the quality of an automatically annotated dataset. This metric is quite effective in reflecting the quality of the predictions during the construction of the dataset, where the ground truth is not available.</font>
* <font size=10>Based on the above techniques, we construct a large-scale dataset, termed as [MIML](https://pan.baidu.com/s/1f6IpxTvBZNeFnvGMfURj1A?pwd=MIML), with 123,150 manually forged images and pixel-level annotations denoting the forged regions. The MIML dataset can significantly improve the generalization of different forgery localization models, especially on modern-style images (as those in IMD20 dataset).</font> <br/> <br/> <br/>

<font size=10>We are confirmed that large-scale non-synthetic data is vital for deep image manipulation localization models. We sincerely hope that our methods and our dataset can shed light on the community and promote the real-world applications of deep image forensic models. </font><br/>


<font size=10>This work is an initial attempt of automatic annotation for IML, futher improvements could be made. We are glad to witnessed the development of this field together with the community. </font><br/>

---

### The Modern Image Manipulation Localization (MIML) dataset is now publicly available at [Kaggle](https://kaggle.com/datasets/0bdf8fbe72a76c53f4a40cb5f8d4ebe7f3c11d5fa47eb30134f96a4fe927dbc1) and [Baidu Drive](https://pan.baidu.com/s/1f6IpxTvBZNeFnvGMfURj1A?pwd=MIML). 

<font size=10>Researchers are welcome 😃 to apply for this dataset by sending an email to  202221012612@mail.scut.edu.cn (with institution email address) and explaining:</font><br/>
1. Who you are and your institution.
2. Who is your supervisor/mentor.
---

### Code release plan

#### Evaluation code of CIML and IML: Within June, 2024 (Done)
#### Training code of CIML: Within Sep, 2024 (Done)
#### Training code of IML: Within 2024 (Done)

If you find any bug or question, please report it to me.

---

### Environment package version
```
Python3.9
torch==1.13.1+cu117
mmcv==1.6.0
mmcv-full==1.6.0
albumentations==1.3.1
```
---
#### A much more powerful V2 version of this work will be released in a few weeks. 
---

<font size=10>Any question about this work please contact 202221012612@mail.scut.edu.cn.</font><br/>

---

### Reference

```
@inproceedings{qu2024towards,
  title={Towards Modern Image Manipulation Localization: A Large-Scale Dataset and Novel Methods},
  author={Qu, Chenfan and Zhong, Yiwu and Liu, Chongyu and Xu, Guitao and Peng, Dezhi and Guo, Fengjun and Jin, Lianwen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10781--10790},
  year={2024}
}
```
