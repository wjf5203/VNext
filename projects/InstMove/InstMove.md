# InstMove: Instance Motion for Object-centric Video Segmentation

This is the official implementation of the paper : "[InstMove: Instance Motion for Object-centric Video Segmentation](https://arxiv.org/abs/2303.08132)".



## Introduction

In this work, we study the instance-level motion and present InstMove, which stands for **Inst**ance **M**otion for **O**bject-centric **V**ideo S**e**gmentation. In comparison to pixel-wise motion (optical flow), InstMove mainly relies on instance-level motion information that is free from image feature embeddings, and features physical interpretations, making it more accurate and robust toward occlusion and fast-moving objects. To better fit in with the video segmentation tasks, InstMove uses instance masks to model the physical presence of an object and learns the dynamic model through a memory network to predict its position and shape in the next frame. With only a few lines of code, InstMove can be integrated into current SOTA methods for three different video segmentation tasks and boost their performance. Specifically, we significantly improve the previous arts by 1.5 AP on OVIS dataset, which features heavy occlusions, and 4.9 AP on YouTubeVIS-Long dataset, which mainly contains fast moving objects. These results suggest that instance-level motion is robust and accurate, and hence serving as a powerful solution in complex scenarios for object-centric video segmentation.

### Usage

We provide the inference code combining InstMove and MinVIS as a reference, inserting the motion mask into the original MinVIS as an aid to the tracking process, and improving the performance of MinVIS on OVIS. InstMove can be directly transplanted and inserted into other existing model structures for use.



## InstMove & MinVIS

| OVIS       | AP   |
| ---------- | :--- |
| w/o motion | 26.2 |
| w/ motion  | 28.8 |

 First, install MinVIS following [INSTALL.md](./MinVIS_motion/INSTALL.md) , then download MinVIS weights and InstMove weights.

Evaluating on OVIS without motion:

```
python3 train_net_video.py --config-file configs/ovis/video_maskformer2_R50_bs32_8ep_frame.yaml --num-gpus 8   --eval-only MODEL.WEIGHTS minvis_ovis_R50.pth  OUTPUT_DIR ./MinVIS_OVIS_without_motion  MODEL.USE_MOTION False
```



Evaluating on OVIS with motion:

```
python3 train_net_video.py --config-file configs/ovis/video_maskformer2_R50_bs32_8ep_frame.yaml --num-gpus 8   --eval-only MODEL.WEIGHTS minvis_ovis_R50.pth  OUTPUT_DIR ./MinVIS_OVIS_motion  MODEL.USE_MOTION True
```





### <a name="CitingInstMove"></a>Citing InstMove

```BibTeX
@inproceedings{liu2023instmove,
  title={InstMove: Instance Motion for Object-centric Video Segmentation},
  author={Liu, Qihao and Wu, Junfeng and Jiang, Yi and Bai, Xiang and Yuille, Alan L and Bai, Song},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6344--6354},
  year={2023}
}
```

## Acknowledgement

This repo is largely based on MinVIS  (https://github.com/NVlabs/MinVIS).
