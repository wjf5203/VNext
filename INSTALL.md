## Installation

First, clone the repository locally:

```bash
git clone https://github.com/wjf5203/VNext.git
cd VNext
```

Install dependencies and pycocotools for VIS:

```bash
pip install -r requirements.txt
pip install -e .
pip install shapely==1.7.1
pip install git+https://github.com/youtubevos/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"
```

Compiling Deformable DETR CUDA operators:

```bash
cd projects/IDOL/idol/models/ops/
sh make.sh
```





## Data Preparation



Download and extract 2019 version of YoutubeVIS train and val images with annotations from [CodeLab](https://competitions.codalab.org/competitions/20128#participate-get_data) or [YouTubeVIS](https://youtube-vos.org/dataset/vis/), download [OVIS](https://codalab.lisn.upsaclay.fr/competitions/4763#participate)  and COCO 2017 datasets. Then, link datasets:

```bash
cd datasets/
ln -s /path_to_coco_dataset coco
ln -s /path_to_YTVIS19_dataset ytvis_2019
ln -s /path_to_ovis_dataset ovis
```



Extract YouTube-VIS 2019, OVIS, COCO 2017 datasets, we expect the directory structure to be the following:

```
VNext
├── datasets
│   ├──ytvis_2019
│   ├──ovis 
│   ├──coco 
...
ytvis_2019
├── train
├── val
├── annotations
│   ├── instances_train_sub.json
│   ├── instances_val_sub.json
...
ovis
├── train
├── valid
├── annotations_train.json
├── annotations_valid.jso
...
coco
├── train2017
├── val2017
├── annotations
│   ├── instances_train2017.json
│   ├── instances_val2017.json
```


