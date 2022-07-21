# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from ..util.misc import inverse_sigmoid
from .ops.modules import MSDeformAttn


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_frames=1, num_feature_levels=4, 
                 dec_n_points=4,  enc_n_points=4,
                 ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_feature_levels = num_feature_levels
        

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels , 
                                                          nhead, enc_n_points)

        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels , 
                                                          nhead, dec_n_points)

        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        assert  query_embed is not None
        # srcs: 4(N, C, Hi, Wi)
        # query_embed: [300, C] 
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, nf, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(3).transpose(2, 3)   # src: [N, nf, Hi*Wi, C]
            mask = mask.flatten(2)   # mask: [N, nf, Hi*Wi]
            pos_embed = pos_embed.flatten(3).transpose(2, 3)  # pos_embed: [N, nf, Hp*Wp, C]
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, 1, -1)
            # TODO: add temporal embed for different frames' feature. Since frames is not fixed, it should be hard encoded 

            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        # src_flatten: [\sigma(N*Hi*Wi), C]
        src_flatten = torch.cat(src_flatten, 2)
        mask_flatten = torch.cat(mask_flatten, 2)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 2)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m[:,0]) for m in masks], 1)  

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        # src_flatten,lvl_pos_embed_flatten shape= [bz, nf, 4lvl*wi*hi, C]    mask_flatten: [bz, nf, 4lvl*wi*hi]  


        # prepare input for decoder
        bs, nf,  _, c = memory.shape
       
            
        query_embed, tgt = torch.split(query_embed, c, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_embed).sigmoid()
        reference_points = reference_points.unsqueeze(1).repeat(1,nf,1,1)     #[bz,nf,300,2]
        init_reference_out = reference_points

        # decoder
        hs, hs_box, inter_references, inter_samples = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)       
   
        return hs, hs_box, memory, init_reference_out, inter_references, inter_samples, None, valid_ratios


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, 'encode')
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), None, reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # ffn
        src = self.forward_ffn(src)
        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, 'decode')
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1_box = nn.Dropout(dropout)
        self.norm1_box = nn.LayerNorm(d_model)


        # self attention for mask&class query
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

         # self attention for box query
        self.self_attn_box = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2_box = nn.Dropout(dropout)
        self.norm2_box = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)


        # ffn for box
        self.linear1_box = nn.Linear(d_model, d_ffn)
        self.activation_box = _get_activation_fn(activation)
        self.dropout3_box = nn.Dropout(dropout)
        self.linear2_box = nn.Linear(d_ffn, d_model)
        self.dropout4_box = nn.Dropout(dropout)
        self.norm3_box = nn.LayerNorm(d_model)

        self.time_attention_weights = nn.Linear(d_model, 1)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    @staticmethod
    def with_pos_embed_multf(tensor, pos):  # boardcase pos to every frame features
        return tensor if pos is None else tensor + pos.unsqueeze(1)

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_ffn_box(self, tgt):
        tgt2 = self.linear2_box(self.dropout3_box(self.activation_box(self.linear1_box(tgt))))
        tgt = tgt + self.dropout4_box(tgt2)
        tgt = self.norm3_box(tgt)
        return tgt

    def forward(self, tgt, tgt_box, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # self attention
        
        q1 = k1 = self.with_pos_embed(tgt, query_pos)
        
        tgt2 = self.self_attn(q1.transpose(0, 1), k1.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        

        if len(tgt_box.shape) == 3:  # tgt_box [bz,300,C]. # first layer

            # box-tgt self attention
            q_box = k_box = self.with_pos_embed(tgt_box, query_pos)
            tgt2_box = self.self_attn_box(q_box.transpose(0, 1), k_box.transpose(0, 1), tgt_box.transpose(0, 1))[0].transpose(0, 1)
            tgt_box = tgt_box + self.dropout2_box(tgt2_box)
            tgt_box = self.norm2_box(tgt_box)
            # cross attention
            tgt2, tgt2_box, sampling_locations, attention_weights = self.cross_attn(self.with_pos_embed(tgt, query_pos), 
                                self.with_pos_embed(tgt_box, query_pos),
                                reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask)

        else: #tgt_box [bz,nf, 300,C] 
            assert len(tgt_box.shape) == 4 
            N, nf, num_q,C = tgt_box.shape
            # self attention
            tgt_list = []
            for i_f in range(nf):  
                tgt_box_i =  tgt_box[:,i_f]
                q_box = k_box = self.with_pos_embed(tgt_box_i, query_pos)
                tgt2_box_i = self.self_attn_box(q_box.transpose(0, 1), k_box.transpose(0, 1), tgt_box_i.transpose(0, 1))[0].transpose(0, 1)
                tgt_box_i = tgt_box_i + self.dropout2_box(tgt2_box_i)
                tgt_box_i = self.norm2_box(tgt_box_i)
                tgt_list.append(tgt_box_i.unsqueeze(1))
            tgt_box = torch.cat(tgt_list,dim=1)
            
            # cross attention
            tgt2, tgt2_box, sampling_locations, attention_weights = self.cross_attn(self.with_pos_embed(tgt, query_pos), 
                                self.with_pos_embed_multf(tgt_box, query_pos),
                                reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask)
        
        
        if len(tgt_box.shape) == 3: 
            tgt_box = tgt_box.unsqueeze(1) + self.dropout1_box(tgt2_box)
        else:
            tgt_box = tgt_box + self.dropout1_box(tgt2_box)
        tgt_box = self.norm1_box(tgt_box)       
        # ffn box
        tgt_box = self.forward_ffn_box(tgt_box)

        time_weight = self.time_attention_weights(tgt_box)
        time_weight = F.softmax(time_weight, 1)
        tgt2 = (tgt2*time_weight).sum(1)
        
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt, tgt_box, sampling_locations, attention_weights


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt
        intermediate = [] # save mask&class query in each decoder layers
        intermediate_box = []  # save box query 
        intermediate_reference_points = []
        intermediate_samples = []

        # reference_points： [bz, nf, 300, 2]
        # src: [2, nf, len_q, 256] encoder output

        output_box = tgt   # box and mask&class share the same initial tgt, but perform deformable attention across frames independently,
        # before first decoder layer, output_box is  [bz,300,C]
        # after the first deformable attention, output_box becomes [bz, nf, 300, C] and keep shape between each decoder layers    
        

        for lid, layer in enumerate(self.layers):
            
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None, None] 
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, :, None] * src_valid_ratios[:,None, None] 
                # reference_points_input [bz, nf, 300, 4, 2] 
            
       
            output, output_box, sampling_locations, attention_weights = \
                 layer(output, output_box, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)
       
           
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output_box)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_box.append(output_box)
                intermediate_reference_points.append(reference_points)
                # intermediate_samples.append(samples_keep)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_box), torch.stack(intermediate_reference_points), None 

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_frames=args.num_frames,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,)

