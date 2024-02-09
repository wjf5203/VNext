# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import os

from .ytvis import (
    register_ytvis_instances,
    _get_ytvis_2019_instances_meta,
    _get_ytvis_2021_instances_meta,
    _get_ovis_instances_meta,
)

# ==== Predefined splits for YTVIS 2019 ===========
_PREDEFINED_SPLITS_YTVIS_2019 = {
    "ytvis_2019_train": ("ytvis_2019/train/JPEGImages",
                         "ytvis_2019/train.json"),
    "ytvis_2019_val": ("ytvis_2019/valid/JPEGImages",
                       "ytvis_2019/valid.json"),
    "ytvis_2019_test": ("ytvis_2019/test/JPEGImages",
                        "ytvis_2019/test.json"),
}


# ==== Predefined splits for YTVIS 2021 ===========
_PREDEFINED_SPLITS_YTVIS_2021 = {
    # "ytvis_2021_train": ("ytvis_2021/train/JPEGImages",
    #                      "ytvis_2021/train.json"),
    # "ytvis_2021_val": ("ytvis_2021/valid/JPEGImages",
    #                    "ytvis_2021/valid.json"),
    # "ytvis_2021_test": ("ytvis_2021/test/JPEGImages",
    #                     "ytvis_2021/test.json"),
    "ytvis_2022_val_full": ("ytvis_2022/val/JPEGImages",
                        "ytvis_2022/instances.json"),
    "ytvis_2022_val_sub": ("ytvis_2022/val/JPEGImages",
                       "ytvis_2022/instances_sub.json"),
}

# ==== Predefined splits for OVIS ===========
_PREDEFINED_SPLITS_OVIS = {
    "ovis_train": ("ovis/train",
                         "ovis/annotations_train.json"),
    "ovis_val": ("ovis/valid",
                       "ovis/annotations_valid.json"),
    "ovis_test": ("ovis/test",
                        "ovis/annotations_test.json"),
    "ovis_subtrain": ("ovis/train",
                         "ovis/ovis_train.json"),
    "ovis_subval_sparse1": ("ovis/train",
                       "ovis/ovis_val.json"),
    "ovis_subval_sparse3": ("ovis/train",
                       "ovis/ovis_val_sparse3.json"),
    "ovis_subval_sparse5": ("ovis/train",
                       "ovis/ovis_val_sparse5.json"),
    "ovis_subval_sparse7": ("ovis/train",
                       "ovis/ovis_val_sparse7.json"),
}


def register_all_ytvis_2019(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2019.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2019_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_ytvis_2021(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2021.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2021_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_ovis(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_OVIS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ovis_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    # register_all_ytvis_2019(_root)
    register_all_ytvis_2021(_root)
    register_all_ovis(_root)
