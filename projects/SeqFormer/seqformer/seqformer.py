# ------------------------------------------------------------------------
# SeqFormer (https://github.com/wjf5203/SeqFormer)
# Copyright (c) 2021 ByteDance. All Rights Reserved.
# ------------------------------------------------------------------------


import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from typing import Dict, List
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
# from fvcore.nn import giou_loss, smooth_l1_loss
from .models.backbone import Joiner
from .models.deformable_detr import DeformableDETR, SetCriterion
from .models.matcher import HungarianMatcher
from .models.position_encoding import PositionEmbeddingSine
from .models.deformable_transformer import DeformableTransformer
from .models.segmentation_condInst import CondInst_segm
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.misc import NestedTensor
from .data.coco import convert_coco_poly_to_mask
import torchvision.ops as ops
from .util.misc import nested_tensor_from_tensor_list
from .models.clip_output import Videos, Clips

__all__ = ["SeqFormer"]




class MaskedBackbone(nn.Module):
    """ This is a thin wrapper around D2's backbone to provide padding masking"""

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        self.feature_strides = [backbone_shape[f].stride for f in backbone_shape.keys()]
        
        self.num_channels = [backbone_shape[f].channels for f in backbone_shape.keys()]

    def forward(self, tensor_list):
      
        xs = self.backbone(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out
        # return features

    def mask_out_padding(self, feature_shapes, image_sizes, device):
        masks = []
        assert len(feature_shapes) == len(self.feature_strides)
        for idx, shape in enumerate(feature_shapes):
            N, _, H, W = shape
            masks_per_feature_level = torch.ones((N, H, W), dtype=torch.bool, device=device)
            for img_idx, (h, w) in enumerate(image_sizes):
                masks_per_feature_level[
                    img_idx,
                    : int(np.ceil(float(h) / self.feature_strides[idx])),
                    : int(np.ceil(float(w) / self.feature_strides[idx])),
                ] = 0
            masks.append(masks_per_feature_level)
        return masks


@META_ARCH_REGISTRY.register()
class SeqFormer(nn.Module):
    """
    Implement SeqFormer
    """

    def __init__(self, cfg):
        super().__init__()

        self.num_frames = cfg.INPUT.SAMPLING_FRAME_NUM

        self.device = torch.device(cfg.MODEL.DEVICE)

        ### inference setting
        self.merge_on_cpu = cfg.MODEL.SeqFormer.MERGE_ON_CPU
        self.is_multi_cls = cfg.MODEL.SeqFormer.MULTI_CLS_ON
        self.apply_cls_thres = cfg.MODEL.SeqFormer.APPLY_CLS_THRES

        self.clip_matching = cfg.MODEL.SeqFormer.CLIP_MATCHING
        self.clip_stride = cfg.MODEL.SeqFormer.CLIP_STRIDE
        self.clip_length = cfg.MODEL.SeqFormer.CLIP_LENGTH

        self.is_coco = cfg.DATASETS.TEST[0].startswith("coco")
        self.num_classes = cfg.MODEL.SeqFormer.NUM_CLASSES
        self.mask_stride = cfg.MODEL.SeqFormer.MASK_STRIDE
        self.match_stride = cfg.MODEL.SeqFormer.MATCH_STRIDE
        self.mask_on = cfg.MODEL.MASK_ON

        self.coco_pretrain = cfg.INPUT.COCO_PRETRAIN
        hidden_dim = cfg.MODEL.SeqFormer.HIDDEN_DIM
        num_queries = cfg.MODEL.SeqFormer.NUM_OBJECT_QUERIES

        # Transformer parameters:
        nheads = cfg.MODEL.SeqFormer.NHEADS
        dropout = cfg.MODEL.SeqFormer.DROPOUT
        dim_feedforward = cfg.MODEL.SeqFormer.DIM_FEEDFORWARD
        enc_layers = cfg.MODEL.SeqFormer.ENC_LAYERS
        dec_layers = cfg.MODEL.SeqFormer.DEC_LAYERS
        enc_n_points = cfg.MODEL.SeqFormer.ENC_N_POINTS
        dec_n_points = cfg.MODEL.SeqFormer.DEC_N_POINTS
        num_feature_levels = cfg.MODEL.SeqFormer.NUM_FEATURE_LEVELS

        # Loss parameters:
        mask_weight = cfg.MODEL.SeqFormer.MASK_WEIGHT
        dice_weight = cfg.MODEL.SeqFormer.DICE_WEIGHT
        giou_weight = cfg.MODEL.SeqFormer.GIOU_WEIGHT
        l1_weight = cfg.MODEL.SeqFormer.L1_WEIGHT
        class_weight = cfg.MODEL.SeqFormer.CLASS_WEIGHT
        deep_supervision = cfg.MODEL.SeqFormer.DEEP_SUPERVISION
        # no_object_weight = cfg.MODEL.SeqFormer.NO_OBJECT_WEIGHT

        focal_alpha = cfg.MODEL.SeqFormer.FOCAL_ALPHA

        set_cost_class = cfg.MODEL.SeqFormer.SET_COST_CLASS
        set_cost_bbox = cfg.MODEL.SeqFormer.SET_COST_BOX
        set_cost_giou = cfg.MODEL.SeqFormer.SET_COST_GIOU


        N_steps = hidden_dim // 2
        d2_backbone = MaskedBackbone(cfg)
        backbone = Joiner(d2_backbone, PositionEmbeddingSine(N_steps, normalize=True))
        backbone.num_channels = d2_backbone.num_channels[1:]  # only take [c3 c4 c5] from resnet and gengrate c6 later
        backbone.strides = d2_backbone.feature_strides[1:]

        
        transformer = DeformableTransformer(
        d_model= hidden_dim,
        nhead=nheads,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_frames=self.num_frames,
        num_feature_levels=num_feature_levels,
        dec_n_points=dec_n_points,
        enc_n_points=enc_n_points,)
        
        
        model = DeformableDETR(
        backbone,
        transformer,
        num_classes=self.num_classes,
        num_frames=self.num_frames,
        num_queries=num_queries,
        num_feature_levels=num_feature_levels,
        aux_loss=deep_supervision,
        with_box_refine=True )

        self.detr = CondInst_segm(model, freeze_detr=False, rel_coord=True)
        
        self.detr.to(self.device)

        # building criterion
        matcher = HungarianMatcher(multi_frame=True, # True, False
                            cost_class=set_cost_class,
                            cost_bbox=set_cost_bbox,
                            cost_giou=set_cost_giou)

        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou":giou_weight}
        weight_dict["loss_mask"] = mask_weight
        weight_dict["loss_dice"] = dice_weight

        if deep_supervision:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)


        losses = ['labels', 'boxes', 'masks']
        

        self.criterion = SetCriterion(self.num_classes, matcher, weight_dict, losses, 
                             mask_out_stride=self.mask_stride,
                             focal_alpha=focal_alpha,
                             num_frames = self.num_frames)
        self.criterion.to(self.device)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

        self.merge_device = "cpu" if self.merge_on_cpu else self.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances: Instances
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """

        if self.training:
            images = self.preprocess_image(batched_inputs)
            clip_targets = self.prepare_targets(batched_inputs)
            output, loss_dict = self.detr(images, clip_targets, self.criterion, train=True)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            
            video_length = len(batched_inputs[0]['file_names'])
            if not self.clip_matching: # take the whole video as input
                images = self.preprocess_image(batched_inputs)
                self.detr.detr.num_frames = video_length
                output = self.detr.inference(images)
                ori_height = batched_inputs[0]['height']
                ori_width = batched_inputs[0]['width']
                video_output = self.whole_video_inference(output, (ori_height, ori_width), images.image_sizes[0])  # images.image_sizes[0] is resized size
            else: # clip matching used in IFC
                video_output = None
                is_last_clip = False
                for start_idx in range(0, video_length, self.clip_stride):
                    end_idx = start_idx + self.clip_length
                    if end_idx >= video_length:
                        is_last_clip = True
                        start_idx, end_idx = max(0, video_length - self.clip_length), video_length
                    frame_idx = list(range(start_idx, end_idx))
                    clip_frames = self.preprocess_image(batched_inputs, frame_idx)
                    image_size = clip_frames.tensor.shape[-2:]
                    self.detr.detr.num_frames = len(clip_frames)
                    output = self.detr.inference(clip_frames)
                    if video_output is None:
                        interim_size = (output['pred_masks'].shape[-2],output['pred_masks'].shape[-1])
                        video_output = Videos(
                            self.clip_length, video_length, self.num_classes, interim_size, self.merge_device
                        )
                    _clip_results = self.inference_clip(output,  image_size )
                    clip_results = Clips(frame_idx, _clip_results.to(self.merge_device))
                    video_output.update(clip_results)
                    if is_last_clip:
                        break
                ori_height = batched_inputs[0].get("height", image_size[0])
                ori_width = batched_inputs[0].get("width", image_size[1])
                pred_cls, pred_masks_logits = video_output.get_result() # NxHxW / NxC.
                video_output = self.clip_matching_postprocess(pred_cls, pred_masks_logits, (ori_height, ori_width), image_size)

            return video_output


    def prepare_targets(self, batched_inputs):
        targets_for_clip_prediction = []
        for video in batched_inputs:
            clip_boxes = []
            clip_masks = []
            clip_classes = []

            for frame_target in video["instances"]:  # per frame annotations
                frame_target=frame_target.to(self.device)
                
                h, w = frame_target.image_size
                image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
                
                gt_classes = frame_target.gt_classes
                gt_boxes = frame_target.gt_boxes.tensor / image_size_xyxy
                gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
                gt_masks = frame_target.gt_masks.tensor
                inst_ids = frame_target.gt_ids
                valid_id = inst_ids!=-1  # if an object is now show on this frame, gt_ids = -1

                clip_boxes.append(gt_boxes)
                clip_masks.append(gt_masks)
                clip_classes.append(gt_classes*valid_id)
            
            targets_for_clip_prediction.append({"labels": torch.stack(clip_classes,dim=0).max(0)[0], 
                                "boxes": torch.stack(clip_boxes,dim=1),   # [num_inst,num_frame,4]
                                'masks': torch.stack(clip_masks,dim=1),   # [num_inst,num_frame,H,W]
                                'size': torch.as_tensor([h, w], dtype=torch.long, device=self.device),
                                # 'inst_id':inst_ids, 
                                # 'valid':valid_id
                                })
 
        return targets_for_clip_prediction


    def inference_clip(self, output, image_size):
        mask_cls = output["pred_logits"][0].sigmoid()
        mask_pred = output["pred_masks"][0]

        # For all 300 masks, we select top 10 as valid masks.
        
        
        scores, labels = mask_cls.max(-1)
        topkv, indices10 = torch.topk(mask_cls.max(1)[0],k=10)
        valid = indices10.tolist()
        scores = scores[valid]
        labels = labels[valid]
        mask_cls = mask_cls[valid]
        mask_pred = mask_pred[valid]

        results = Instances(image_size)
        results.scores = scores
        results.pred_classes = labels
        results.cls_probs = mask_cls
        results.pred_masks = mask_pred

        return results

    def clip_matching_postprocess(self, pred_cls, pred_masks, ori_size, image_sizes):
        
        if len(pred_cls) > 0:
            if self.is_multi_cls:
                is_above_thres = torch.where(pred_cls > self.apply_cls_thres)
                scores = pred_cls[is_above_thres]
                labels = is_above_thres[1]
                pred_masks = pred_masks[is_above_thres[0]]
            else:
                scores, labels = pred_cls.max(-1)


            output_h, output_w = pred_masks.shape[-2:]
            pred_masks =F.interpolate(pred_masks,  size=(output_h*self.mask_stride, output_w*self.mask_stride) 
                ,mode="bilinear", align_corners=False).sigmoid()

            pred_masks = pred_masks[:,:,:image_sizes[0],:image_sizes[1]] #crop padding area
            pred_masks = F.interpolate(pred_masks, size=(ori_size[0], ori_size[1]), mode='nearest')

            masks = pred_masks > 0.5

            out_scores = scores.tolist()
            out_labels = labels.tolist()
            out_masks = [m for m in masks.cpu()]
        else:
            out_scores = []
            out_labels = []
            out_masks = []

        video_output = {
            "image_size": ori_size,
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
        }

        return video_output

    def whole_video_inference(self, outputs, ori_size, image_sizes):
        """
        Arguments:
            
        Returns:
            results (List[Instances]): a list of #images elements.
        """
        # results = []

        logits = outputs['pred_logits'][0]
        output_mask = outputs['pred_masks'][0]
        output_boxes = outputs['pred_boxes'][0]
        output_h, output_w = output_mask.shape[-2:]
        topkv, indices10 = torch.topk(logits.sigmoid().max(1)[0],k=10)
        indices10 = indices10.tolist()
        valid_logits = logits[indices10].sigmoid()
        output_mask = output_mask[indices10]

        pred_masks =F.interpolate(output_mask,  size=(output_h*self.mask_stride, output_w*self.mask_stride) ,mode="bilinear", align_corners=False).sigmoid()
        if len(valid_logits) > 0:
            if self.is_multi_cls:
                is_above_thres = torch.where(valid_logits > self.apply_cls_thres)
                scores = valid_logits[is_above_thres]
                labels = is_above_thres[1]
                pred_masks = pred_masks[is_above_thres[0]]
            else:
                scores, labels = valid_logits.max(-1)

            pred_masks = pred_masks[:,:,:image_sizes[0],:image_sizes[1]] 
            pred_masks = F.interpolate(pred_masks, size=(ori_size[0], ori_size[1]), mode='nearest')

            masks = pred_masks > 0.5
            
            out_scores = scores.tolist()
            out_labels = labels.tolist()
            out_masks = [m for m in masks.cpu()]
        else:
            out_scores = []
            out_labels = []
            out_masks = []
        video_output = {
            "image_size": ori_size,
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
        }

        return video_output


    def preprocess_image(self, batched_inputs, clip_idx=None):
        """
        Normalize, pad and batch the input images.
        """
        if clip_idx is None:
            images = []
            for video in batched_inputs:
                for frame in video["image"]:
                    images.append(self.normalizer(frame.to(self.device)))
            images = ImageList.from_tensors(images)
        else:
            images = []
            for video in batched_inputs:
                for idx in clip_idx:
                    images.append(self.normalizer(video["image"][idx].to(self.device)))
            images = ImageList.from_tensors(images)
        return images


