# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN


def add_seqformer_config(cfg):
    """
    Add config for SeqFormer.
    """
    cfg.MODEL.SeqFormer = CN()
    cfg.MODEL.SeqFormer.NUM_CLASSES = 80

    # DataLoader
    cfg.INPUT.PRETRAIN_TYPE = 'v1'
    cfg.INPUT.SAMPLING_FRAME_NUM = 1
    cfg.INPUT.SAMPLING_FRAME_RANGE = 10
    cfg.INPUT.SAMPLING_INTERVAL = 1
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
    cfg.INPUT.AUGMENTATIONS = [] # "brightness", "contrast", "saturation", "rotation"

    cfg.INPUT.COCO_PRETRAIN = False
    cfg.INPUT.PRETRAIN_SAME_CROP = False

    # LOSS
    cfg.MODEL.SeqFormer.MASK_WEIGHT = 2.0
    cfg.MODEL.SeqFormer.DICE_WEIGHT = 5.0
    cfg.MODEL.SeqFormer.GIOU_WEIGHT = 2.0
    cfg.MODEL.SeqFormer.L1_WEIGHT = 5.0
    cfg.MODEL.SeqFormer.CLASS_WEIGHT = 2.0
    cfg.MODEL.SeqFormer.DEEP_SUPERVISION = True
    cfg.MODEL.SeqFormer.MASK_STRIDE = 4
    cfg.MODEL.SeqFormer.MATCH_STRIDE = 4
    cfg.MODEL.SeqFormer.FOCAL_ALPHA = 0.25

    cfg.MODEL.SeqFormer.SET_COST_CLASS = 2
    cfg.MODEL.SeqFormer.SET_COST_BOX = 5
    cfg.MODEL.SeqFormer.SET_COST_GIOU = 2

    # TRANSFORMER
    cfg.MODEL.SeqFormer.NHEADS = 8
    cfg.MODEL.SeqFormer.DROPOUT = 0.1
    cfg.MODEL.SeqFormer.DIM_FEEDFORWARD = 1024
    cfg.MODEL.SeqFormer.ENC_LAYERS = 6
    cfg.MODEL.SeqFormer.DEC_LAYERS = 6

    cfg.MODEL.SeqFormer.HIDDEN_DIM = 256
    cfg.MODEL.SeqFormer.NUM_OBJECT_QUERIES = 300
    cfg.MODEL.SeqFormer.DEC_N_POINTS = 4
    cfg.MODEL.SeqFormer.ENC_N_POINTS = 4
    cfg.MODEL.SeqFormer.NUM_FEATURE_LEVELS = 4


    # Evaluation
    
    cfg.MODEL.SeqFormer.MERGE_ON_CPU = True
    cfg.MODEL.SeqFormer.MULTI_CLS_ON = True
    cfg.MODEL.SeqFormer.APPLY_CLS_THRES = 0.05
    # Clip-matching inference
    cfg.MODEL.SeqFormer.CLIP_MATCHING = False
    cfg.MODEL.SeqFormer.CLIP_LENGTH = 5
    cfg.MODEL.SeqFormer.CLIP_STRIDE = 1


    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    ## support Swin backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # find_unused_parameters
    cfg.FIND_UNUSED_PARAMETERS = True