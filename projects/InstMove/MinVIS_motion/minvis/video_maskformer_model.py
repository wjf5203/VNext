# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE

# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
from typing import Tuple
import einops

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks

from mask2former_video.modeling.criterion import VideoSetCriterion
from mask2former_video.modeling.matcher import VideoHungarianMatcher
from mask2former_video.utils.memory import retry_if_cuda_oom

from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


def mask_iou(mask1, mask2):
    mask1 = mask1.char()
    mask2 = mask2.char()

    intersection = (mask1[:,:,:] * mask2[:,:,:]).sum(-1).sum(-1)
    union = (mask1[:,:,:] + mask2[:,:,:] - mask1[:,:,:] * mask2[:,:,:]).sum(-1).sum(-1)

    return (intersection+1e-6) / (union+1e-6)

def padding_resize(GT_masks, img_gt, target_size):  # gt_masks [numisnt,len,h,w]  img_gt [1,3,h,w]

    num_inst,n_frames,H,W = GT_masks.shape
    _,_,imgH,imgW = img_gt.shape
    hw_mask = max(H,W)
    hw_img = max(imgH,imgW)

    resized_GT_mask = torch.zeros([num_inst,n_frames,hw_mask,hw_mask]).to(GT_masks)
    resized_img = torch.zeros([1,3,hw_img,hw_img]).to(img_gt)

    dh = int((hw_mask-H)/2)
    dw = int((hw_mask-W)/2)
    resized_GT_mask[:,:,dh:dh+H,dw:dw+W] = GT_masks

    dh_img = int((hw_img-imgH)/2)
    dw_img = int((hw_img-imgW)/2)
    resized_img[:,:,dh_img:dh_img+imgH,dw_img:dw_img+imgW] = img_gt

    resized_GT_mask = F.interpolate(resized_GT_mask, (target_size,target_size),mode="nearest")
    resized_img = F.interpolate(resized_img, (target_size,target_size),mode="bilinear")
    resized_GT_mask = resized_GT_mask.unsqueeze(2)
    resized_img = resized_img.unsqueeze(1)

    return resized_GT_mask,resized_img

def unpadding_resize(GT_masks, ori_imge_size_):
    H,W = ori_imge_size_
    hw = max(H,W)
    dh = int((hw-H)/2)
    dw = int((hw-W)/2)
    resized_GT_mask = F.interpolate(GT_masks, (hw,hw),mode="bilinear")
    ori_GT_mask = resized_GT_mask[:,:,dh:dh+H,dw:dw+W]

    return ori_GT_mask




@META_ARCH_REGISTRY.register()
class VideoMaskFormer_frame(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # video
        num_frames,
        window_inference,
        use_motion,
        ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.num_frames = num_frames
        self.window_inference = window_inference
        self.use_motion = use_motion

        from motion_models.model_withImgR6 import Predictor
        self.motion_pred_model = Predictor(100).to(self.device)
        ckpt = torch.load('motion_model.pth')
        new_ckpt = {}
        for k,v in ckpt.items():
            newk = k.replace('module.','')
            new_ckpt[newk]=ckpt[k]
        self.motion_pred_model.load_state_dict(new_ckpt)
        self.motion_pred_model.eval()


    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = VideoHungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = VideoSetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": True,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # video
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "window_inference": cfg.MODEL.MASK_FORMER.TEST.WINDOW_INFERENCE,
            "use_motion": cfg.MODEL.USE_MOTION
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        if not self.training and self.window_inference:
            outputs = self.run_window_inference(images.tensor)
        else:
            features = self.backbone(images.tensor)
            outputs = self.sem_seg_head(features)

        if self.training:
            # mask classification target
            targets = self.prepare_targets(batched_inputs, images)

            outputs, targets = self.frame_decoder_loss_reshape(outputs, targets)

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            outputs = self.post_processing(outputs, images)

            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]

            mask_cls_result = mask_cls_results[0]
            mask_pred_result = mask_pred_results[0]
            first_resize_size = (images.tensor.shape[-2], images.tensor.shape[-1])

            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]  # image size without padding after data augmentation

            height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
            width = input_per_image.get("width", image_size[1])

            return retry_if_cuda_oom(self.inference_video)(mask_cls_result, mask_pred_result, image_size, height, width, first_resize_size)

    def frame_decoder_loss_reshape(self, outputs, targets):
        outputs['pred_masks'] = einops.rearrange(outputs['pred_masks'], 'b q t h w -> (b t) q () h w')
        outputs['pred_logits'] = einops.rearrange(outputs['pred_logits'], 'b t q c -> (b t) q c')
        if 'aux_outputs' in outputs:
            for i in range(len(outputs['aux_outputs'])):
                outputs['aux_outputs'][i]['pred_masks'] = einops.rearrange(
                    outputs['aux_outputs'][i]['pred_masks'], 'b q t h w -> (b t) q () h w'
                )
                outputs['aux_outputs'][i]['pred_logits'] = einops.rearrange(
                    outputs['aux_outputs'][i]['pred_logits'], 'b t q c -> (b t) q c'
                )

        gt_instances = []
        for targets_per_video in targets:
            # labels: N (num instances)
            # ids: N, num_labeled_frames
            # masks: N, num_labeled_frames, H, W
            num_labeled_frames = targets_per_video['ids'].shape[1]
            for f in range(num_labeled_frames):
                labels = targets_per_video['labels']
                ids = targets_per_video['ids'][:, [f]]
                masks = targets_per_video['masks'][:, [f], :, :]
                gt_instances.append({"labels": labels, "ids": ids, "masks": masks})

        return outputs, gt_instances

    def match_from_embds(self, tgt_embds, cur_embds, motion_mask, current_mask):

        cur_embds = cur_embds / cur_embds.norm(dim=1)[:, None]
        tgt_embds = tgt_embds / tgt_embds.norm(dim=1)[:, None]
        cos_sim = torch.mm(cur_embds, tgt_embds.transpose(0,1))

        cost_embd = 1 - cos_sim

        if motion_mask is not None:
            motion_score = mask_iou(current_mask.unsqueeze(1)>0 , motion_mask.permute(1,0,2,3) )
            cost_motion = 1 - motion_score
            # C =  1 * cost_motion
            C = 1.0 * cost_embd + 0.5 * cost_motion
        else:
            C = 1.0 * cost_embd
        C = C.cpu()

        indices = linear_sum_assignment(C.transpose(0, 1))  # target x current
        indices = indices[1]  # permutation that makes current aligns to target

        return indices

    def post_processing(self, outputs, ori_images):
        pred_logits, pred_masks, pred_embds = outputs['pred_logits'], outputs['pred_masks'], outputs['pred_embds']

        # pred_logits: 1 t q c
        # pred_masks: 1 q t h w
        pred_logits = pred_logits[0]
        pred_masks = einops.rearrange(pred_masks[0], 'q t h w -> t q h w')
        pred_embds = einops.rearrange(pred_embds[0], 'c t q -> t q c')

        pred_logits = list(torch.unbind(pred_logits))
        pred_masks = list(torch.unbind(pred_masks))
        pred_embds = list(torch.unbind(pred_embds))

        out_logits = []
        out_masks = []
        out_embds = []
        out_logits.append(pred_logits[0])
        out_masks.append(pred_masks[0])
        out_embds.append(pred_embds[0])

        for i in range(1, len(pred_logits)):
            if self.use_motion and i>1: # predict motion mask from the third frame
                if i>3:
                    previous_mask = torch.stack([out_masks[-4], out_masks[-3],out_masks[-2],out_masks[-1]],dim=0) # [t q h w] 
                else:
                    previous_mask = torch.stack([out_masks[-2],out_masks[-1]],dim=0) # [t q h w]  
                previous_mask = previous_mask.permute(1,0,2,3)  # [num_inst len h w]
                ori_img = ori_images.tensor[i].unsqueeze(0)  #[1,3,h,w]
                target_size = 384
                oriH,oriW = previous_mask.shape[-2:]

                ######## select query ids with high cls score as valid id to compute motion
                check_valid_logits = sum(out_logits)/len(out_logits)
                check_scores = F.softmax(check_valid_logits, dim=-1)[:, :-1]
                
                if True: #motion_type == 'topk':
                    scores_per_image, check_valid_id = check_scores.max(-1)[0].topk(20, sorted=False)
                    valid_mun = len(check_valid_id)
                else:
                    check_valid_id =  check_scores.max(-1)[0] > 0.1
                    valid_mun = check_valid_id.sum().item()


                previous_mask = previous_mask[check_valid_id]


                short_data, img_gt = padding_resize(previous_mask, ori_img, target_size)
                short_data = (short_data>0).float() 

                img_gt = img_gt.repeat(valid_mun,1,1,1,1)

                motion_pred = self.motion_pred_model(short_x = short_data, long_x = None, out_len = 1, phase=2, img=img_gt)
                motion_pred = unpadding_resize(motion_pred[:,:,0],(oriH,oriW))
                motion_pred = motion_pred.sigmoid() > 0.5
                
                padding_pred_motion = torch.zeros(100,1,oriH,oriW ).to(motion_pred)>0
                padding_pred_motion[check_valid_id] = motion_pred

                if False:  # if visualize
                    print('current frame:',i)            
                    import cv2
                    ori_img_vis = ori_img[0].permute(1,2,0).detach().cpu().numpy()
                    ori_img_vis = (ori_img_vis-ori_img_vis.min())/(ori_img_vis.max()-ori_img_vis.min())

                    cv2.imwrite('motion_visualize/ori_img.png', ori_img_vis*255)
                    for insti in range(valid_mun):
                        prev1 = previous_mask[insti,0].cpu().detach().numpy()>0
                        prev2 = previous_mask[insti,1].cpu().detach().numpy()>0
                        prev3 = previous_mask[insti,2].cpu().detach().numpy()>0
                        prev4 = previous_mask[insti,3].cpu().detach().numpy()>0
                        test_mask = motion_pred[insti,0].cpu().detach().numpy()
                        cv2.imwrite('motion_visualize/id_{}_previous_1.png'.format(str(insti)),prev1*255)
                        cv2.imwrite('motion_visualize/id_{}_previous_2.png'.format(str(insti)),prev2*255)
                        cv2.imwrite('motion_visualize/id_{}_previous_3.png'.format(str(insti)),prev3*255)
                        cv2.imwrite('motion_visualize/id_{}_previous_4.png'.format(str(insti)),prev4*255)
                        cv2.imwrite('motion_visualize/id_{}_xmotion.png'.format(str(insti)),test_mask*255)        

            else:
                padding_pred_motion = None


            indices = self.match_from_embds(out_embds[-1], pred_embds[i], padding_pred_motion, pred_masks[i] )
            out_logits.append(pred_logits[i][indices, :])
            out_masks.append(pred_masks[i][indices, :, :])
            out_embds.append(pred_embds[i][indices, :])

        out_logits = sum(out_logits)/len(out_logits)
        out_masks = torch.stack(out_masks, dim=1)  # q h w -> q t h w

        out_logits = out_logits.unsqueeze(0)
        out_masks = out_masks.unsqueeze(0)

        outputs['pred_logits'] = out_logits
        outputs['pred_masks'] = out_masks

        return outputs

    def run_window_inference(self, images_tensor, window_size=30):
        iters = len(images_tensor) // window_size
        if len(images_tensor) % window_size != 0:
            iters += 1
        out_list = []
        for i in range(iters):
            start_idx = i * window_size
            end_idx = (i+1) * window_size

            features = self.backbone(images_tensor[start_idx:end_idx])
            out = self.sem_seg_head(features)
            del features['res2'], features['res3'], features['res4'], features['res5']
            for j in range(len(out['aux_outputs'])):
                del out['aux_outputs'][j]['pred_masks'], out['aux_outputs'][j]['pred_logits']
            out_list.append(out)

        # merge outputs
        outputs = {}
        outputs['pred_logits'] = torch.cat([x['pred_logits'] for x in out_list], dim=1).detach()
        outputs['pred_masks'] = torch.cat([x['pred_masks'] for x in out_list], dim=2).detach()
        outputs['pred_embds'] = torch.cat([x['pred_embds'] for x in out_list], dim=2).detach()

        return outputs

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        gt_instances = []
        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [_num_instance, self.num_frames, h_pad, w_pad]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)

            gt_ids_per_video = []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.image_size

                gt_ids_per_video.append(targets_per_frame.gt_ids[:, None])
                gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor

            gt_ids_per_video = torch.cat(gt_ids_per_video, dim=1)
            valid_idx = (gt_ids_per_video != -1).any(dim=-1)

            gt_classes_per_video = targets_per_frame.gt_classes[valid_idx]          # N,
            gt_ids_per_video = gt_ids_per_video[valid_idx]                          # N, num_frames

            gt_instances.append({"labels": gt_classes_per_video, "ids": gt_ids_per_video})
            gt_masks_per_video = gt_masks_per_video[valid_idx].float()          # N, num_frames, H, W
            gt_instances[-1].update({"masks": gt_masks_per_video})

        return gt_instances

    def inference_video(self, pred_cls, pred_masks, img_size, output_height, output_width, first_resize_size):
        if len(pred_cls) > 0:
            scores = F.softmax(pred_cls, dim=-1)[:, :-1]
            labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
            # keep top-10 predictions
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(10, sorted=False)
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // self.sem_seg_head.num_classes
            pred_masks = pred_masks[topk_indices]

            pred_masks = F.interpolate(
                pred_masks, size=first_resize_size, mode="bilinear", align_corners=False
            )

            pred_masks = pred_masks[:, :, : img_size[0], : img_size[1]]
            pred_masks = F.interpolate(
                pred_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
            )

            masks = pred_masks > 0.

            out_scores = scores_per_image.tolist()
            out_labels = labels_per_image.tolist()
            out_masks = [m for m in masks.cpu()]
        else:
            out_scores = []
            out_labels = []
            out_masks = []

        video_output = {
            "image_size": (output_height, output_width),
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
        }

        return video_output
