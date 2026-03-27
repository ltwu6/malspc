# MAL-UPC
This is the official implementation for the paper [Unsupervised 3D Point Cloud Completion via Multi-View Adversarial Learning](https://ieeexplore.ieee.org/abstract/document/10959142)(Accepted by IEEE TVCG)

![image](https://github.com/ltwu6/malspc/flowchart.png)

### Abstract
In real-world scenarios, scanned point clouds are often incomplete due to occlusion issues. The tasks of self-supervised and weakly-supervised point cloud completion involve reconstructing missing regions of these incomplete objects without the supervision of complete ground truth. Current methods either rely on multiple views of partial observations for supervision or overlook the intrinsic geometric similarity that can be identified and utilized from the given partial point clouds. In this paper, we propose MAL-UPC, a framework that effectively leverages both region-level and category-specific geometric similarities to complete missing structures. Our MAL-UPC does not require any 3D complete supervision and only necessitates single-view partial observations in the training set. Specifically, we first introduce a Pattern Retrieval Network to retrieve similar position and curvature patterns between the partial input and the predicted shape, then leverage these similarities to densify and refine the reconstructed results. Additionally, we render the reconstructed complete shape into multi-view depth maps and design an adversarial learning module to learn the geometry of the target shape from category-specific single-view depth images of the partial point clouds in the training set. To achieve anisotropic rendering, we design a density-aware radius estimation algorithm to improve the quality of the rendered images. Our MAL-UPC outperforms current state-of-the-art self-supervised methods and even some unpaired approaches.

## Environment
Python: 3.9  
PyTorch: 0.10.1  
Cuda: 11.1  

## Dataset
[Baiduyun](https://pan.baidu.com/s/1rkLvKgEaUc_YU1yIoA8uAQ)( 
code：pmvu)

## Training
```
CUDA_VISIBLE_DEVICES=0 python train_cfgan.py
```
## Cite this work
```
@article{wu2022crosspcc,
  author={Wu, Lintai and Cheng, Xianjing and Xu, Yong and Zeng, Huanqiang and Hou, Junhui},
  journal={IEEE Transactions on Visualization and Computer Graphics}, 
  title={Unsupervised 3D Point Cloud Completion via Multi-View Adversarial Learning}, 
  year={2025},
  volume={31},
  number={10},
  pages={7890-7905},
  doi={10.1109/TVCG.2025.3559340}}
}
```

