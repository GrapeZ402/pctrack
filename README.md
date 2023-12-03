# PCTrack &#x1F345;
<br/>

> PCTrack: Accurate Object Tracking for Live Video Analytics on Resource-Constrained Edge Devices
> 
> [Xinyi Zhang](https://github.com/GrapeZ402), [Haoran Xu](https://scholar.google.com/citations?user=UOwYW7gAAAAJ&hl=en), [Chenyun Yu](https://scholar.google.com/citations?user=xlnfZAcAAAAJ&hl=en&oi=ao), [Guang Tan*](https://scholar.google.com/citations?user=JerZls4AAAAJ&hl=en&oi=ao)

## News
- [2023/11/24]: The paper is under review for ICDE-2024.
- [2023/11/17]: Trajectory prediction dataset release. 
- [2023/11/13]: Initial code. 
- [2023/11/11]: Demo release.

## Introduction



The task of live video analytics relies on real-time object tracking that typically involves 
computationally expensive deep neural network (DNN) models. In practice, it has become essential 
to process video data on edge devices deployed near the cameras. However, these edge devices 
often have very limited computing resources and thus suffer from poor tracking accuracy. 
Through a measurement study, we identify three major factors contributing to the performance issue: 
outdated detection results, tracking error accumulation, and ignorance of new objects.

We introduce a novel approach, called Predict & Correct based Tracking, or PCTrack, to systematically address these problems. 
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


## Repository requirements

- install Pytorch 1.10.0 in Jetson Nano or Jetson TX2 by following the 
[instruction](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)

- install preliminary packages:
```shell
pip install scikit-learn opencv-python scipy numpy
pip install swig==4.1.1
pip install oapackage==2.4.4
```

## PCTrack evaluation

- running PCTrack by using:
```shell
python3 test.py --data ${data} --delay ${delay} --method ${method}
```

