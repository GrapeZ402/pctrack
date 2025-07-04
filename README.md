# PCTrack &#x1F345;
<br/>

> PCTrack: Accurate Object Tracking for Live Video Analytics on Resource-Constrained Edge Devices
> 
> [Xinyi Zhang](https://github.com/GrapeZ402), [Haoran Xu](https://scholar.google.com/citations?user=UOwYW7gAAAAJ&hl=en), [Chenyun Yu](https://scholar.google.com/citations?user=xlnfZAcAAAAJ&hl=en&oi=ao), [Guang Tan*](https://scholar.google.com/citations?user=JerZls4AAAAJ&hl=en&oi=ao)

## News
- [2024/12/27]: The paper is accepted by IEEE TCSVT
- [2024/02/18]: The paper is under review for IEEE TCSVT.
- [2023/11/27]: Dataset release. 
- [2023/11/13]: Initial code. 
- [2023/11/11]: Demo release.

## Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry:
```
@ARTICLE{zhang2025pctrack,
  author={Zhang, Xinyi and Xu, Haoran and Yu, Chenyun and Tan, Guang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={PCTrack: Accurate Object Tracking for Live Video Analytics on Resource-Constrained Edge Devices}, 
  year={2025},
  volume={35},
  number={5},
  pages={3969-3982},
}
```

## Introduction
The task of live video analytics relies on real-time object tracking that typically involves 
computationally expensive deep neural network (DNN) models. In practice, it has become essential 
to process video data on edge devices deployed near the cameras. However, these edge devices 
often have very limited computing resources and thus suffer from poor tracking accuracy. 
Through a measurement study, we identify three major factors contributing to the performance issue: 
outdated detection results, tracking error accumulation, and ignorance of new objects.

We introduce a novel approach, called **Predict & Correct based Tracking**, or **`PCTrack`**, to systematically address these problems. 
Our design incorporates three innovative components: 

- (1) a **_Predictive Detection Propagator_** that rapidly updates outdated object bounding boxes to match the current frame 
through a lightweight prediction model; 
- (2) a **_Frame Difference Corrector_** that refines the object bounding boxes based on frame difference information; and
- (3) a **_New Object Detector_** that efficiently discovers newly appearing objects during tracking. 

PCTrack Pipeline:
<p align='center'>
    <img src="https://github.com/GrapeZ402/pctrack/blob/main/vendors/framework.png" width="100%">
</p>


Experimental results show that our approach achieves remarkable accuracy improvements, ranging **from 19.4% to 34.7%**, 
across diverse traffic scenarios, compared to state of the art methods.

## Demonstration

The detection and tracking process of a segment of dataset OnRamp with PCTrack and MARLIN+IDP: 
<p align='center'>
    <img src="https://github.com/GrapeZ402/pctrack/blob/main/vendors/compare.png" width="100%">
</p>



## Repository requirements

- install Pytorch 1.10.0 in Jetson Nano or Jetson TX2 by following the 
[instruction](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)

- install preliminary packages:
```shell
pip install scikit-learn==0.24.2 opencv-python==4.8.1.78 scipy==1.5.4 numpy==1.19.5
pip install swig==4.1.1
pip install oapackage==2.4.4
```

## Dataset 
Download full dataset data [HERE](https://pan.baidu.com/s/1UVrEK7nM2D4YZ1dZqooIsA) (the password is `402C`). Folder structure:
```
pctrack
├── datasets/
│   ├── OnRamp/
│   ├── Intersect/
│   ├── Express/
│   ├── UrbMixed/
│   ├── labels/
│   │   ├── OnRamp_label/
│   │   ├── Intersect_label/
│   │   ├── Express_label/
│   │   ├── UrbMixed_label/
```


## PCTrack evaluation


### testing PCTrack
- running PCTrack by using:
```shell
python3 run.py
```
- the key parameters used in `run.py` are defined as follows:
  - `--data`: type=str, choices=['OnRamp', 'Intersect', 'Express', 'UrbMixed'], default='OnRamp'
  - `--delay`: type=int, choices=[25, 30, 35, 40, 45], default=30
  - `--method`: type=str, choices=['ours', 'base'], default='ours'
  - `--source`: type=str, default='datasets/OnRamp/'
  - `--label`: type=str, default='datasets/labels/OnRamp_label/'
  - `--save`: type=bool, default=False

### evaluation of F1 score
- running evaluation by using:
```shell
python3 eval.py
```
- the key parameters used in `eval.py` are defined as follows:
  - `--result_path`: type=str, example='datasets/OnRamp/'
  - `--gt_path`: type=str, example='datasets/labels/OnRamp_label/'

