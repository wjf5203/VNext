# VNext: 



VNext is a **Next**-generation **V**ideo instance recognition framework on top of [Detectron2](https://github.com/facebookresearch/detectron2). Currently it provides advanced online and offline video instance segmentation algorithms. We will continue to update and improve it to provide a unified and efficient framework for the field of video instance recognition to nourish this field.



To date, VNext contains the official implementation of the following algorithms:

IDOL: In Defense of Online Models for Video Instance Segmentation

SeqFormer: Sequential Transformer for Video Instance Segmentation



## News:

- IDOL is accepted to ECCV 2022 as an **oral presentation**!
- SeqFormer is accepted to ECCV 2022 as an **oral presentation**!
- IDOL  won **first place** in the video instance segmentation track of the 4th Large-scale Video Object Segmentation Challenge (CVPR2022).


## Getting started

1. For Installation and data preparation, please refer to  to [INSTALL.md](./INSTALL.md) for more details.
2. For IDOL training and model zoo, please refer to [IDOL.md](./projects/IDOL/IDOL.md)

3. For SeqFormer training and model zoo, please refer to [SeqFormer.md](./projects/SeqFormer/SeqFormer.md)




## IDOL



### Abstract

In recent years, video instance segmentation (VIS) has been largely advanced by offline models, while online models gradually attracted less attention possibly due to their inferior performance. However, online methods have their inherent advantage in handling long video sequences and ongoing videos while offline models fail due to the limit of computational resources. Therefore, it would be highly desirable if online models can achieve comparable or even better performance than offline models. By dissecting current online models and offline models, we demonstrate that the main cause of the performance gap is the error-prone association between frames caused by the similar appearance among different instances in the feature space. Observing this, we propose an online framework based on contrastive learning that is able to learn more discriminative instance embeddings for association and fully exploit history information for stability. Despite its simplicity, our method outperforms all online and offline methods on three benchmarks. Specifically, we achieve 49.5 AP on YouTube-VIS 2019, a significant improvement of 13.2 AP and 2.1 AP over the prior online and offline art, respectively. Moreover, we achieve 30.2 AP on OVIS, a more challenging dataset with significant crowding and occlusions, surpassing the prior art by 14.8 AP.  The proposed method won **first place** in the video instance segmentation track of the 4th Large-scale Video Object Segmentation Challenge (CVPR2022). We hope the simplicity and effectiveness of our method, as well as our insight on current methods, could shed light on the exploration of VIS models.



<p align="center"><img src="assets/IDOL/arch.png" width="1000"/></p>

 

### Quantitative results

#### YouTube-VIS 2019



<p align="center"><img src="assets/IDOL/ytvis2019_results.png" width="1000"/></p>

 

#### OVIS 2021



<p align="center"><img src="assets/IDOL/ovis_results.png" width="1000"/></p>

 

## 

## SeqFormer

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/seqformer-a-frustratingly-simple-model-for/video-instance-segmentation-on-youtube-vis-1)](https://paperswithcode.com/sota/video-instance-segmentation-on-youtube-vis-1?p=seqformer-a-frustratingly-simple-model-for)



<p align="center"><img src="assets/SeqFormer/SeqFormer_sota.png" width="500"/></p>

[SeqFormer: Sequential Transformer for Video Instance Segmentation](https://arxiv.org/abs/2112.08275)

Junfeng Wu, Yi Jiang, Song Bai, Wenqing Zhang, Xiang Bai



### Abstract

In this work, we present SeqFormer for video instance segmentation. SeqFormer follows the principle of vision transformer that models instance relationships among video frames. Nevertheless, we observe that a stand-alone instance query suffices for capturing a time sequence of instances in a video, but attention mechanisms shall be done with each frame independently. To achieve this, SeqFormer locates an instance in each frame and aggregates temporal information to learn a powerful representation of a video-level instance, which is used to predict the mask sequences on each frame dynamically. Instance tracking is achieved naturally without tracking branches or post-processing. On YouTube-VIS, SeqFormer achieves 47.4 AP with a ResNet-50 backbone and 49.0 AP with a ResNet-101 backbone without bells and whistles. Such achievement significantly exceeds the previous state-of-the-art performance by 4.6 and 4.4, respectively. We hope SeqFormer could be a strong baseline that fosters future research in video instance segmentation, and in the meantime, advances this field with a more robust, accurate, neat model.



<p align="center"><img src="assets/SeqFormer/SeqFormer_arch.png" width="1000"/></p>

 

### Visualization results on YouTube-VIS 2019 valid set

 

<img src="assets/SeqFormer/vid_15.gif" width="400"/><img src="assets/SeqFormer/vid_78.gif" width="400"/>
<img src="assets/SeqFormer/vid_133.gif" width="400"/><img src="assets/SeqFormer/vid_210.gif" width="400"/>



### Quantitative results

#### YouTube-VIS 2019



<p align="center"><img src="assets/SeqFormer/ytvis2019_results.png" width="1000"/></p>

 

#### YouTube-VIS 2021



<p align="center"><img src="assets/SeqFormer/ytvis2021_results.png" width="1000"/></p>

 

#### 




## Citation

```
@inproceedings{seqformer,
  title={SeqFormer: Sequential Transformer for Video Instance Segmentation},
  author={Wu, Junfeng and Jiang, Yi and Bai, Song and Zhang, Wenqing and Bai, Xiang},
  booktitle={ECCV},
  year={2022},
}

@inproceedings{IDOL,
  title={In Defense of Online Models for Video Instance Segmentation},
  author={Wu, Junfeng and Liu, Qihao and Jiang, Yi and Bai, Song and Yuille, Alan and Bai, Xiang},
  booktitle={ECCV},
  year={2022},
}
```

## Acknowledgement

This repo is based on [detectron2](https://github.com/facebookresearch/detectron2), [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), [VisTR](https://github.com/Epiphqny/VisTR), and [IFC](https://github.com/sukjunhwang/IFC)  Thanks for their wonderful works.
