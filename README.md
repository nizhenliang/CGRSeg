## Context-Guided Spatial Feature Reconstruction for Efficient Semantic Segmentation
> Authors: Zhenliang Ni, Xinghao Chen, Yingjie Zhai, Yehui Tang, and Yunhe Wang
> 
> [arXiv:2405.06228](https://arxiv.org/abs/2405.06228)

## The overall architecture of CGRSeg
<img width="784" alt="cgrseg2" src="https://github.com/nizhenliang/CGRSeg/assets/48742170/eef8c502-599d-48aa-b05b-51a682ac7456">

The overall architecture of CGRSeg. The Rectangular Self-Calibration Module (RCM) is designed for spatial feature reconstruction and pyramid context extraction. 
The rectangular self-calibration attention (RCA) explicitly models the rectangular region and calibrates the attention shape. The Dynamic Prototype Guided (DPG) head
is proposed to improve the classification of the foreground objects via explicit class embedding.

## Results

#### ADE20K

<img width="539" alt="ade20k" src="https://github.com/nizhenliang/CGRSeg/assets/48742170/98e14385-8f41-417c-84d9-3cc6db0d32c1">

COCO-Stuff-10k

<img width="491" alt="coco" src="https://github.com/nizhenliang/CGRSeg/assets/48742170/9bf2487f-27d6-41d1-8e94-26f3fd994ce0">

Pascal Context

<img width="481" alt="pc" src="https://github.com/nizhenliang/CGRSeg/assets/48742170/d0b3f524-523f-4fc3-a809-691f4617ebb4">

## Environment

- ```shell
  conda create --name ssa python=3.8 -y
  conda activate ssa
  pip install torch==1.8.2+cu102 torchvision==0.9.2+cu102 torchaudio==0.8.2
  pip install timm==0.6.13
  pip install mmcv-full==1.6.1
  pip install opencv-python==4.1.2.30
  pip install "mmsegmentation==0.27.0"
  ```
  
  CGRSeg is built based on [mmsegmentation-0.27.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.27.0), which can be referenced for data preparation.

## Training & Testing

- Train
  
  ```shell
  # Single-gpu training
  python train.py local_configs/cgrseg/cgrseg-t_ade20k_160k.py
  
  # Multi-gpu (4-gpu) training
  bash dist_train.sh local_configs/cgrseg/cgrseg-t_ade20k_160k.py 4
  ```

- Test
  
  ```shell
  # Single-gpu testing
  python test.py local_configs/cgrseg/cgrseg-t_ade20k_160k.py ${CHECKPOINT_FILE} --eval mIoU
  
  # Multi-gpu (4-gpu) testing
  bash dist_test.sh local_configs/cgrseg/cgrseg-t_ade20k_160k.py ${CHECKPOINT_FILE} 4 --eval mIoU
  ```
