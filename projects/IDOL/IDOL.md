## IDOL 



### Model zoo

**Pretraining:**

We have at most three training steps for IDOL:

Step 1: pre-training the **instance segmentation pipeline** on COCO, following all other VIS methods.
Step 2: pre-training IDOL on pseudo key-reference pair from COCO. (This step forces the model to learn a position-insensitive contrastive embedding that relies on appearance of the object rather than the spatial position.)
Step 3: finetune our VIS method IDOL on VIS dataset (YTVIS19/YTVIS21/OVIS), following all other VIS methods.

In **Table 3,4,5**, all the IDOL results marked with **†** are obtained by **Step 1+2+3**, others **without †** are obtained by **Step 1+3**. 

Here we provide the pretrained weight from **Step1+2** on COCO.

Step1 need to train the instance segmentation pipeline on COCO, which requires removing some components from IDOL. Directly using step2 to pretrain on COCO will get inferior performance. We will update the weights of step1 in the near future.

| Backbone   | Step 1 | Step1+2                                                      |
| ---------- | ------ | ------------------------------------------------------------ |
| ResNet-50  | WIP    | [pretrain](https://drive.google.com/file/d/1ip-pxavMcyWOxfBcl_4cUIBKXRTA4wrO/view?usp=sharing) |
| ResNet-101 | WIP    | [pretrain](https://drive.google.com/file/d/1Gm162LthxorsS6pMX_XoVTn5iDAAMWbU/view?usp=sharing) |
| Swin-Large | WIP    | [pretrain](https://drive.google.com/file/d/1o-q4WIcMn_D5p1tSubJBWlPAnJLQ5Cbb/view?usp=sharing) |



Train on YouTube-VIS 2019, evaluate on YouTube-VIS 2019.

| Backbone                                                     | AP   | AP50 | AP75 | AR1  | AR10 | download                                                     |
| ------------------------------------------------------------ | ---- | ---- | ---- | ---- | ---- | ------------------------------------------------------------ |
| [R50](https://github.com/wjf5203/VNext/blob/main/projects/IDOL/configs/ytvis19_r50.yaml) | 49.3 | 74.0 | 52.8 | 47.6 | 58.7 | [model](https://drive.google.com/file/d/1FFbrfbK1oN4zTO5q_cw3zws2Z2Ppuzv7/view?usp=sharing) & [log](https://drive.google.com/file/d/16jqwbEzSkY-qcVZYYJeZ2vj8-Mffm30T/view?usp=share_link) |
| [R50](https://github.com/wjf5203/VNext/blob/main/projects/IDOL/configs/ytvis19_r50.yaml) | 50.2 | 75.1 | 53.6 | 48.4 | 59.3 | [model](https://drive.google.com/file/d/1BAZX99coA23RPrK8He3hDx59j11N7bUa/view?usp=sharing) |
| [R101](https://github.com/wjf5203/VNext/blob/main/projects/IDOL/configs/ytvis19_r101.yaml) | 50.1 | 73.1 | 56.1 | 47.0 | 57.9 | [model](https://drive.google.com/file/d/1T8S3_tZRcMJ1c5ioe3MGUNwKg9UP5ROW/view?usp=sharing) |
| [SwinL](https://github.com/wjf5203/VNext/blob/main/projects/IDOL/configs/ytvis19_swinL.yaml) | 64.3 | 87.5 | 71.0 | 55.6 | 69.1 | [model](https://drive.google.com/file/d/1Otlq8eqb_xg0eRF5dQHyxKvuEceOgwBk/view?usp=sharing) |



Train on YouTube-VIS 2021, evaluate on YouTube-VIS 2021.

| Backbone                                                     | AP   | AP50 | AP75 | AR1  | AR10 | download                                                     |
| ------------------------------------------------------------ | ---- | ---- | ---- | ---- | ---- | ------------------------------------------------------------ |
| [R50](https://github.com/wjf5203/VNext/blob/main/projects/IDOL/configs/ytvis21_r50.yaml) | 47.7 | 71.5 | 53.3 | 41.1 | 55.5 | [model](https://drive.google.com/file/d/1GqE7HEaJZCqUogsNi52vp_9NVEsiJwxL/view?usp=share_link) |
| [R101](https://github.com/wjf5203/VNext/blob/main/projects/IDOL/configs/ytvis21_r101.yaml) | 48.5 | 71.4 | 53.9 | 41.4 | 55.9 | [model](https://drive.google.com/file/d/1hYu32qR95QiZLKz40tsXrM_vsC9dcLLe/view?usp=share_link) |
| [SwinL](https://github.com/wjf5203/VNext/blob/main/projects/IDOL/configs/ytvis21_swinL.yaml) | 60.8 | 84.9 | 67.1 | 47.5 | 65.3 | [model](https://drive.google.com/file/d/17CSYz31VvaD-3RjLo3x5JjD-wL3zcDSq/view?usp=share_link) |



Train on OVIS, evaluate on OVIS.

| Backbone                                                     | AP   | AP50 | AP75 | AR1  | AR10 | download                                                     |
| ------------------------------------------------------------ | ---- | ---- | ---- | ---- | ---- | ------------------------------------------------------------ |
| [R50](https://github.com/wjf5203/VNext/blob/main/projects/IDOL/configs/ovis_r50.yaml) | 29.5 | 50.1 | 29.4 | 14.9 | 37.1 | [model](https://drive.google.com/file/d/1CalS2qLNm1MRlNdWkKhAKvS4CyM7vHTv/view?usp=share_link) |
| [SwinL](https://github.com/wjf5203/VNext/blob/main/projects/IDOL/configs/ovis_swin.yaml) | 41.4 | 64.1 | 44.0 | 17.6 | 48.4 | [model]( https://drive.google.com/file/d/1XXr5bx3CEwyurQvO4B8XMn2t6awYodGw/view?usp=share_link) |



 **Custom training  and validation split** 

We provide the custom split in the Oracle experiment: [oracle_split](https://drive.google.com/drive/folders/1veFOmJzKIr3tgGC8DI-hkBjUg1ivl5tc?usp=share_link)



### Training

To train SeqFormer on YouTube-VIS 2019 or OVIS with 8 GPUs , run:

```
python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/XXX.yaml --num-gpus 8 
```



### Inference & Evaluation



Evaluating on YouTube-VIS 2019 or OVIS:

```
python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/XXX.yaml --num-gpus 8 --eval-only
```



To get quantitative results, please zip the json file and upload to the [codalab server](https://competitions.codalab.org/competitions/20128#participate-submit_results) for YouTube-VIS 2019 and [server](https://codalab.lisn.upsaclay.fr/competitions/4763) for OVIS.



## Citation

```
@inproceedings{IDOL,
  title={In Defense of Online Models for Video Instance Segmentation},
  author={Wu, Junfeng and Liu, Qihao and Jiang, Yi and Bai, Song and Yuille, Alan and Bai, Xiang},
  booktitle={ECCV},
  year={2022},
}
```

## Acknowledgement

This repo is based on [detectron2](https://github.com/facebookresearch/detectron2), [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), [VisTR](https://github.com/Epiphqny/VisTR), and [IFC](https://github.com/sukjunhwang/IFC)  Thanks for their wonderful works.
