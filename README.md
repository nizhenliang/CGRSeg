## Paper: Context-Guided Spatial Feature Reconstruction for Efficient Semantic Segmentation
> Authors: Zhenliang Ni, Xinghao Chen, Yingjie Zhai, Yehui Tang, and Yunhe Wang

## The overall architecture of CGRSeg

![](E:\项目文件\轻量化分割模型预研\github\CGRSeg.PNG)

The overall architecture of CGRSeg. The Rectangular Self-Calibration Module (RCM) is designed for spatial feature reconstruction and pyramid context extraction. 

The rectangular self-calibration attention (RCA) explicitly models the rectangular region and calibrates the attention shape. The Dynamic Prototype Guided (DPG) head
is proposed to improve the classification of the foreground objects via explicit class embedding.

## Results

#### ADE20K

| Method         | mIoU | Flops(G) | Param(M) | Throughputs(img/s) |
|:--------------:|:----:|:--------:|:--------:|:------------------:|
| CGRSeg-T(Ours) | 43.6 | 4.0      | 9.4      | 138.4              |
| CGRSeg-B(Ours) | 45.5 | 7.6      | 18.1     | 98.4               |
| CGRSeg-L(Ours) | 48.3 | 14.9     | 35.7     | 73.0               |

COCO-Stuff-10k

| Method   | mIoU | Flops(G) | Param(M) |
|:--------:|:----:|:--------:|:--------:|
| CGRSeg-T | 42.2 | 4.0      | 9.4      |
| CGRSeg-B | 43.5 | 7.6      | 18.1     |
| CGRSeg-L | 46.0 | 14.9     | 35.7     |

Pascal Context

| Method   | mIoU(MS) | Flops(G) | Param(M) |
| -------- |:--------:|:--------:|:--------:|
| CGRSeg-T | 54.1     | 4.0      | 9.4      |
| CGRSeg-B | 56.5     | 7.6      | 18.1     |
| CGRSeg-L | 58.5     | 14.9     | 35.7     |

 Environment

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
