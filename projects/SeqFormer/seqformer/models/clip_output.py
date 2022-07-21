from typing import List
import torch
from scipy.optimize import linear_sum_assignment

from detectron2.structures import Instances
from detectron2.utils.memory import retry_if_cuda_oom

from ..util.misc import interpolate


class Videos:
    """
    This structure is to support Section 3.3: Clip-level instance tracking.
    NOTE most errors occuring in this structure is due to
    the number of predictions exceeding num_max_inst.
    TODO make further GPU-memory friendly while maintaining speed,
    and max number of instances be dynamically changed.
    """
    def __init__(self, num_frames, video_length, num_classes, image_size, device):
        self.num_frames = num_frames
        self.video_length = video_length
        self.device = device

        num_max_inst = 120
        self.match_threshold = 0.01

        self.num_inst = 0
        self.num_clip = 0
        self.saved_idx_set = set()

        self.saved_logits = torch.zeros((video_length, num_max_inst, self.video_length, *image_size), dtype=torch.float, device=device)
        self.saved_masks  = torch.zeros((video_length, num_max_inst, self.video_length, *image_size), dtype=torch.float, device=device)
        self.saved_valid  = torch.zeros((video_length, num_max_inst, self.video_length), dtype=torch.bool, device=device)
        self.saved_cls    = torch.zeros((video_length, num_max_inst, num_classes), dtype=torch.float, device=device)

    def get_siou(self, input_masks, saved_masks, saved_valid):
        # input_masks : N_i, T, H, W
        # saved_masks : C, N_s, T, H, W
        # saved_valid : C, N_s, T

        input_masks = input_masks.flatten(-2)   #    N_i, T, HW
        saved_masks = saved_masks.flatten(-2)   # C, N_s, T, HW

        input_masks = input_masks[None, None]   # 1, 1, N_i, T, HW
        saved_masks = saved_masks.unsqueeze(2)  # C, N_s, 1, T, HW
        saved_valid = saved_valid[:, :, None, :, None]  # C, N_s, 1, T, 1

        # C, N_s, N_i, T, HW
        numerator = saved_masks * input_masks
        denominator = saved_masks + input_masks - saved_masks * input_masks

        numerator = (numerator * saved_valid).sum(dim=(-1, -2))
        denominator = (denominator * saved_valid).sum(dim=(-1, -2))

        siou = numerator / (denominator + 1e-6) # C, N_s, N_i

        # To divide only the frames that are being compared
        num_valid_clip = (saved_valid.flatten(2).sum(dim=2) > 0).sum(dim=0) # N_s,

        siou = siou.sum(dim=0) / (num_valid_clip[..., None] + 1e-6)

        return siou

    def update(self, input_clip):
        # gather intersection
        inter_input_idx, inter_saved_idx = [], []
        for o_i, f_i in enumerate(input_clip.frame_idx):
            if f_i in self.saved_idx_set:
                inter_input_idx.append(o_i)
                inter_saved_idx.append(f_i)

        # compute sIoU
        i_masks = input_clip.mask_probs[:, inter_input_idx]
        s_masks = self.saved_masks[
            max(self.num_clip-len(input_clip.frame_idx), 0) : self.num_clip, : self.num_inst, inter_saved_idx
        ]
        s_valid = self.saved_valid[
            max(self.num_clip-len(input_clip.frame_idx), 0) : self.num_clip, : self.num_inst, inter_saved_idx
        ]

        scores = self.get_siou(i_masks, s_masks, s_valid)   # N_s, N_i

        # bipartite match
        above_thres = scores > self.match_threshold
        scores = scores * above_thres.float()

        row_idx, col_idx = linear_sum_assignment(scores.cpu(), maximize=True)

        existed_idx = []
        for is_above, r, c in zip(above_thres[row_idx, col_idx], row_idx, col_idx):
            if not is_above:
                continue

            self.saved_logits[self.num_clip, r, input_clip.frame_idx] = input_clip.mask_logits[c]
            self.saved_masks[self.num_clip, r, input_clip.frame_idx] = input_clip.mask_probs[c]
            self.saved_valid[self.num_clip, r, input_clip.frame_idx] = True
            self.saved_cls[self.num_clip, r] = input_clip.cls_probs[c]
            existed_idx.append(c)

        left_idx = [i for i in range(input_clip.num_instance) if i not in existed_idx]
        try:
            self.saved_logits[self.num_clip,
                            self.num_inst:self.num_inst+len(left_idx),
                            input_clip.frame_idx] = input_clip.mask_logits[left_idx]
        except:
            print('shape mismatch error!')
        self.saved_masks[self.num_clip,
                         self.num_inst:self.num_inst+len(left_idx),
                         input_clip.frame_idx] = input_clip.mask_probs[left_idx]
        self.saved_valid[self.num_clip,
                         self.num_inst:self.num_inst+len(left_idx),
                         input_clip.frame_idx] = True
        self.saved_cls[self.num_clip, self.num_inst:self.num_inst+len(left_idx)] = input_clip.cls_probs[left_idx]

        # Update status
        self.saved_idx_set.update(input_clip.frame_set)
        self.num_clip += 1
        self.num_inst += len(left_idx)

    def get_result(self,):
        _mask_logits = self.saved_logits[:self.num_clip, :self.num_inst]
        _valid = self.saved_valid[:self.num_clip, :self.num_inst]
        _cls = self.saved_cls[:self.num_clip, :self.num_inst]

        _mask_logits = _mask_logits.sum(dim=0) / _valid.sum(dim=0)[..., None, None]
        
        out_cls = _cls.sum(dim=0) / (_valid.sum(dim=2) > 0).sum(dim=0)[..., None]
        # out_masks = retry_if_cuda_oom(lambda x: x > 0.0)(_mask_logits)

        return out_cls, _mask_logits


class Clips:
    def __init__(self, frame_idx: List[int], results: List[Instances]):
        self.frame_idx = frame_idx
        self.frame_set = set(frame_idx)

        self.classes = results.pred_classes
        self.scores = results.scores
        self.cls_probs = results.cls_probs
        self.mask_logits = results.pred_masks
        self.mask_probs = results.pred_masks.sigmoid()

        self.num_instance = len(self.scores)
