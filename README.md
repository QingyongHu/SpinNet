# SpinNet: Learning a General Surface Descriptor for 3D Point Cloud Registration

This is the repository of **SpinNet** ([[Arxiv report](https://arxiv.org/abs/2011.12149)]), a conceptually simple neural architecture to extract local 
features which are rotationally invariant whilst sufficiently informative to enable accurate registration. For technical details, please refer to:


**SpinNet: Learning a General Surface Descriptor for 3D Point Cloud Registration**  <br />
[Sheng Ao*](http://scholar.google.com/citations?user=cvS1yuMAAAAJ&hl=zh-CN), [Qingyong Hu*](https://www.cs.ox.ac.uk/people/qingyong.hu/), [Bo Yang](https://yang7879.github.io/), [Andrew Markham](https://www.cs.ox.ac.uk/people/andrew.markham/), [Yulan Guo](http://yulanguo.me/). <br />
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

    @article{ao2020SpinNet,
      title={SpinNet: Learning a General Surface Descriptor for 3D Point Cloud Registration},
      author={Ao, Sheng and Hu, Qingyong and Yang, Bo and Markham, Andrew and Guo, Yulan},
      journal={arXiv preprint arXiv:2011.12149},
      year={2020}
    }


### Updates
* 25/11/2020: Initial release











