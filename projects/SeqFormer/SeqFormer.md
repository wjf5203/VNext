## SeqFormer 



This is a detectron2 version of [SeqFormer](https://arxiv.org/abs/2112.08275), which supports multi-card training and inference, and supports inference by clip-matching manner.  Joint-training with the COCO is not supported yet, and this will be added in a future version.  You can refer to the [original code](https://github.com/wjf5203/SeqFormer) of SeqFormer for more details about joint-training.





### Model zoo

Train on YouTube-VIS 2019, evaluate on YouTube-VIS 2019.

| Name  | AP   | AP50 | AP75 | AR1  | AR10 | download |
| ----- | ---- | ---- | ---- | ---- | ---- | -------- |
| R50   |      |      |      |      |      |          |
| R101  |      |      |      |      |      |          |
| SwinL |      |      |      |      |      |          |

Model Zoo of SeqFormer is in preparation.



### Training

We performed the experiment on NVIDIA Tesla V100 GPU. All models of SeqFormer are trained with total batch size of 32.

To train SeqFormer on YouTube-VIS 2019 with 8 GPUs , run:

```
python3 projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/XXX.yaml --num-gpus 8 
```



### Inference & Evaluation



Evaluating on YouTube-VIS 2019:

```
python3 projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/XXX.yaml --num-gpus 8 --eval-only
```

Evaluating on YouTube-VIS 2019 by a clip-matching manner:

```
python3 projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/XXX.yaml --num-gpus 8 --eval-only MODEL.SeqFormer.CLIP_MATCHING True  MODEL.SeqFormer.CLIP_LENGTH $CLIP_LENGTH  MODEL.SeqFormer.CLIP_STRIDE $CLIP_STRIDE  
```





To get quantitative results, please zip the json file and upload to the [codalab server](https://competitions.codalab.org/competitions/20128#participate-submit_results).



## Citation

```
@inproceedings{seqformer,
  title={SeqFormer: Sequential Transformer for Video Instance Segmentation},
  author={Wu, Junfeng and Jiang, Yi and Bai, Song and Zhang, Wenqing and Bai, Xiang},
  booktitle={ECCV},
  year={2022},
}
```

## Acknowledgement

This repo is based on [detectron2](https://github.com/facebookresearch/detectron2), [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), [VisTR](https://github.com/Epiphqny/VisTR), and [IFC](https://github.com/sukjunhwang/IFC)  Thanks for their wonderful works.
