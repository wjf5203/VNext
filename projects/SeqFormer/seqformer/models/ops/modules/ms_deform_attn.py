# ------------------------------------------------------------------------
# SeqFormer Deformable Attention.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, mode='encode'):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64
        self.mode = mode
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.output_proj_box = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)

        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, query_box, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):

        if self.mode == 'encode':
            return self.encode_forward(query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask)
        elif self.mode == 'decode':
            return self.decode_forward(query, query_box, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask)

    def encode_forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        N, nf, Len_q, _ = query.shape
        N, nf, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        value = value.view(N, nf, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, nf, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, nf, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, nf, Len_q, self.n_heads, self.n_levels, self.n_points)

        value_list = []
        result_list =[]
        for i in range(nf):
            value_list.append(value[:,i].contiguous())
        for idx_f in range(nf):
            sampling_offsets_i = sampling_offsets[:,idx_f]  
            if reference_points.shape[-1] == 2:
                offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
                sampling_locations_i = reference_points[:, :, None, :, None, :] \
                                    + sampling_offsets_i / offset_normalizer[None, None, None, :, None, :]
            else:
                raise ValueError(
                    'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
            attention_weights_i = attention_weights[:,idx_f].contiguous()

            output_samp_i = MSDeformAttnFunction.apply(
                    value_list[idx_f], input_spatial_shapes, input_level_start_index, sampling_locations_i, attention_weights_i, self.im2col_step)
            result_list.append(output_samp_i.unsqueeze(1))
        result_list = torch.cat(result_list, dim=1)
        output = self.output_proj(result_list)

        return output #, sampling_locations, attention_weights



    def decode_forward(self, query, query_box, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):

        if len(query_box.shape)==3: #first layer
            N, Len_q, _ = query.shape  # decoder: [bz,300,256]
            N,nf, Len_in, _ = input_flatten.shape
            assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in
            value = self.value_proj(input_flatten)
            if input_padding_mask is not None:
                value = value.masked_fill(input_padding_mask[..., None], float(0))
            value = value.view(N, nf, Len_in, self.n_heads, self.d_model // self.n_heads)
            sampling_offsets = self.sampling_offsets(query_box).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
            attention_weights = self.attention_weights(query_box).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
            attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
            # sampling_offsets [bz, 300, 8, 4, 4, 2]]
            #attention_weights [bz, 300, 8, 4, 4]
            
            value_list = []
            point_list = []
            result_list =[]
            for i in range(nf):
                value_list.append(value[:,i].contiguous())
                point_list.append(reference_points[:,i].contiguous() )

            result_idx_f = []
            
            for samp_i in range(nf): # perform deformable attention per frame
                
                reference_points_i = point_list[samp_i]
                if reference_points_i.shape[-1] == 2:
                    offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
                    sampling_locations = reference_points_i[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
                elif reference_points_i.shape[-1] == 4:  
                    sampling_locations = reference_points_i[:, :, None, :, None, :2] \
                                        + sampling_offsets / self.n_points * reference_points_i[:, :, None, :, None, 2:] * 0.5
                else:
                    raise ValueError(
                        'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
                output_samp_i = MSDeformAttnFunction.apply(
                    value_list[samp_i], input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
                result_idx_f.append(output_samp_i.unsqueeze(1))
            result_idx_f = torch.cat(result_idx_f,dim=1)
            result_sum = result_idx_f
            output = self.output_proj(result_sum)
            output_box = self.output_proj_box(result_idx_f)

            return output, output_box, sampling_locations, attention_weights
        else:
            assert len(query_box.shape) == 4 #[bz,nf,300,C]
            N,nf, Len_q, _ = query_box.shape  # decoder: [bz,nf,300,256]
            N,nf, Len_in, _ = input_flatten.shape
            assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in
            value = self.value_proj(input_flatten)
            if input_padding_mask is not None:
                value = value.masked_fill(input_padding_mask[..., None], float(0))
            value = value.view(N, nf, Len_in, self.n_heads, self.d_model // self.n_heads)
            sampling_offsets = self.sampling_offsets(query_box).view(N, nf,Len_q, self.n_heads, self.n_levels, self.n_points, 2)
            attention_weights = self.attention_weights(query_box).view(N, nf, Len_q, self.n_heads, self.n_levels * self.n_points)
            attention_weights = F.softmax(attention_weights, -1).view(N, nf, Len_q, self.n_heads, self.n_levels, self.n_points)
   
            value_list = []
            point_list = []
            sampling_offsets_list = []
            attention_weights_list = []
            for i in range(nf):
                value_list.append(value[:,i].contiguous())
                point_list.append(reference_points[:,i].contiguous() )
                sampling_offsets_list.append(sampling_offsets[:,i].contiguous() )
                attention_weights_list.append(attention_weights[:,i].contiguous() )
            result_idx_f = []

            for samp_i in range(nf): # perform deformable attention per frame
                reference_points_i = point_list[samp_i]
                if reference_points_i.shape[-1] == 2:
                    offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
                    sampling_locations = reference_points_i[:, :, None, :, None, :] + sampling_offsets_list[samp_i] / offset_normalizer[None, None, None, :, None, :]
                elif reference_points_i.shape[-1] == 4: 
                    sampling_locations = reference_points_i[:, :, None, :, None, :2] \
                                        + sampling_offsets_list[samp_i] / self.n_points * reference_points_i[:, :, None, :, None, 2:] * 0.5
                else:
                    raise ValueError(
                        'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
                output_samp_i = MSDeformAttnFunction.apply(
                    value_list[samp_i], input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights_list[samp_i], self.im2col_step)
                result_idx_f.append(output_samp_i.unsqueeze(1))
            result_idx_f = torch.cat(result_idx_f,dim=1) 
            result_sum = result_idx_f
            output = self.output_proj(result_sum)
            output_box = self.output_proj_box(result_idx_f)

            return output, output_box, sampling_locations, attention_weights


