[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spinnet-learning-a-general-surface-descriptor/point-cloud-registration-on-3dmatch-benchmark)](https://paperswithcode.com/sota/point-cloud-registration-on-3dmatch-benchmark?p=spinnet-learning-a-general-surface-descriptor)
[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
[![arXiv](https://img.shields.io/badge/arXiv-2011.12149-b31b1b.svg)](https://arxiv.org/abs/2011.12149)
# SpinNet: Learning a General Surface Descriptor for 3D Point Cloud Registration (CVPR 2021)

This is the official repository of **SpinNet** ([[Arxiv report](https://arxiv.org/abs/2011.12149)]), a conceptually simple neural architecture to extract local 
features which are rotationally invariant whilst sufficiently informative to enable accurate registration. For technical details, please refer to:

**SpinNet: Learning a General Surface Descriptor for 3D Point Cloud Registration**  <br />
[Sheng Ao*](http://scholar.google.com/citations?user=cvS1yuMAAAAJ&hl=zh-CN), [Qingyong Hu*](https://www.cs.ox.ac.uk/people/qingyong.hu/), [Bo Yang](https://yang7879.github.io/), [Andrew Markham](https://www.cs.ox.ac.uk/people/andrew.markham/), [Yulan Guo](http://yulanguo.me/). <br />
(* *indicates equal contribution*)

**[[Paper](https://arxiv.org/abs/2011.12149)] [Video] [Project page]** <br />


### (1) Overview

<p align="center"> <img src="figs/Fig2.png" width="100%"> </p>

<p align="center"> <img src="figs/Fig3.png" width="100%"> </p>

<p align="center"> <img src="figs/Fig1.png" width="65%"> </p>



### (2) Results on Public Datasets


- #### Comparisons with the State-of-the-arts.
<p align="center"> <img src="figs/Table1.png" width="65%"> </p>

- #### Performance under Different Number of Sampled Points
<p align="center"> <img src="figs/Table2.png" width="65%"> </p>

- #### Performance under Different Error Thresholds
<p align="center"> <img src="figs/Fig4.png" width="70%"> </p>


### (3) Generalization Performance
- #### Generalization From 3DMatch to ETH
<p align="center"> <img src="figs/Table4.png" width="65%"> </p>

- #### Generalization From KITTI to 3DMatch 
<p align="center"> <img src="figs/Table5.png" width="65%"> </p>

- #### Generalization From 3DMatch to KITTI
<p align="center"> <img src="figs/Table6.png" width="65%"> </p>

### (4) Qualitative Results
<p align="center"> <img src="figs/Fig5.png" width="100%"> </p>



### Citation
If you find our work useful in your research, please consider citing:

    @inproceedings{ao2020SpinNet,
      title={SpinNet: Learning a General Surface Descriptor for 3D Point Cloud Registration},
      author={Ao, Sheng and Hu, Qingyong and Yang, Bo and Markham, Andrew and Guo, Yulan},
      booktitle={arXiv preprint arXiv:2011.12149},
      year={2021}
    }

### Updates
* 01/03/2021: This paper has been accepted by CVPR 2021!
* 25/11/2020: Initial release!


## Related Repos
1. [RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds](https://github.com/QingyongHu/RandLA-Net) ![GitHub stars](https://img.shields.io/github/stars/QingyongHu/RandLA-Net.svg?style=flat&label=Star)
2. [SoTA-Point-Cloud: Deep Learning for 3D Point Clouds: A Survey](https://github.com/QingyongHu/SoTA-Point-Cloud) ![GitHub stars](https://img.shields.io/github/stars/QingyongHu/SoTA-Point-Cloud.svg?style=flat&label=Star)
3. [3D-BoNet: Learning Object Bounding Boxes for 3D Instance Segmentation on Point Clouds](https://github.com/Yang7879/3D-BoNet) ![GitHub stars](https://img.shields.io/github/stars/Yang7879/3D-BoNet.svg?style=flat&label=Star)
4. [SensatUrban: Learning Semantics from Urban-Scale Photogrammetric Point Clouds](https://github.com/QingyongHu/SpinNet) ![GitHub stars](https://img.shields.io/github/stars/QingyongHu/SensatUrban.svg?style=flat&label=Star)













