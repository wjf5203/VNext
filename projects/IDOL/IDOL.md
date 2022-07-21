## IDOL 



### Model zoo

Train on YouTube-VIS 2019, evaluate on YouTube-VIS 2019.

| Name      | AP   | AP50 | AP75 | AR1  | AR10 | download                                                     |
| --------- | ---- | ---- | ---- | ---- | ---- | ------------------------------------------------------------ |
| [R50]()   | 49.5 | 74.0 | 52.9 | 47.7 | 58.7 | [pretrain](https://drive.google.com/file/d/1ip-pxavMcyWOxfBcl_4cUIBKXRTA4wrO/view?usp=sharing) & [model](https://drive.google.com/file/d/1FFbrfbK1oN4zTO5q_cw3zws2Z2Ppuzv7/view?usp=sharing) |
| [R101]()  | 50.1 | 73.1 | 56.1 | 47.0 | 57.9 | [pretrain](https://drive.google.com/file/d/1Gm162LthxorsS6pMX_XoVTn5iDAAMWbU/view?usp=sharing) & [model](https://drive.google.com/file/d/1T8S3_tZRcMJ1c5ioe3MGUNwKg9UP5ROW/view?usp=sharing) |
| [SwinL]() | 64.3 | 87.5 | 71.0 | 55.6 | 69.1 | [pretrain](https://drive.google.com/file/d/1o-q4WIcMn_D5p1tSubJBWlPAnJLQ5Cbb/view?usp=sharing) & [model](https://drive.google.com/file/d/1Otlq8eqb_xg0eRF5dQHyxKvuEceOgwBk/view?usp=sharing) |



Model Zoo for YouTube-VIS 2021 and OVIS is coming soon.





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
