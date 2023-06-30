# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,
    SoftPhongShader,
    SoftSilhouetteShader,
    HardPhongShader,
    TexturesVertex
)
from pytorch3d.renderer.mesh.shader import SoftDepthShader
from rm_config import rm_cfg
from nets.layer import make_linear_layers, make_linear_layers_bn, make_linear_layers_tanh, make_linear_layers_tanh_bn, make_conv_layers,\
    make_linear_layers_ln, make_deconv_layers, make_upsample_layers, make_conv1d_layers, HGFilter, ConvPool1D
from nets.resnet import ResNetBackbone
import math
import os
import numpy as np
import smplx
import copy
from utils.preprocessing import estimate_focal_length
from nets.cliff_models.cliff_intaghand.encoder import ResNetSimple, resnet_mid
from nets.cliff_models.cliff_intaghand.decoder import get_decoder
from nets.cliff_models.backbones.hrnet.cls_hrnet import HighResolutionNet
from nets.cliff_models.backbones.hrnet.hrnet_config import cfg as hr_cfg
from nets.cliff_models.backbones.hrnet.hrnet_config import update_config


class Render_Mask(nn.Module):
    def __init__(self):
        super(Render_Mask, self).__init__()
        

    def forward(self, img_size, verts, faces, focal_length, princpt):
        # verts:[bs, 778*2, 3]
        # faces:[bs, nf*2, 3]
        # focal_length:[bs, 2]
        # princpt:[bs, 2]
        
        device = verts.device
        bs, nv = verts.shape[:2]
        meshes = Meshes(verts=verts, faces=faces)
        color = torch.ones(bs, nv, 3, device=device)
        color = color * 255.
        meshes.textures = TexturesVertex(verts_features=color)
        
        img_height, img_weight = img_size
        fcl, prp = focal_length.clone(), princpt.clone()
        fcl[:, 0] = fcl[:, 0] * 2 / min(img_height, img_weight)
        fcl[:, 1] = fcl[:, 1] * 2 / min(img_height, img_weight)
        prp[:, 0] = - (prp[:, 0] - img_weight // 2) * 2 / img_weight
        prp[:, 1] = - (prp[:, 1] - img_height // 2) * 2 / img_height

        cameras = PerspectiveCameras(focal_length=-fcl, principal_point=prp, device=device)
        sigma = 1e-4
        raster_settings = RasterizationSettings(
            image_size=(img_height, img_weight), 
            blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
            faces_per_pixel=50, 
        )
        
        # renderer
        shader_sil = SoftSilhouetteShader()
        renderer_sil = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=raster_settings,
            ),
            shader=shader_sil
        )
        
        # render
        rendered_sil = renderer_sil(meshes, cameras=cameras, eps=1e-8)          # [bs, H, W, 4]
        valid_mask = rendered_sil[..., -1].unsqueeze(1)   # [bs, 1, H, W]
        # valid_mask = (valid_mask > 0.5).float()
        return valid_mask   


class IMG_Encoder_Intaghand(nn.Module):
    def __init__(self, pretrain, pret_model_path=''):
        super(IMG_Encoder_Intaghand, self).__init__()
        resnet_type = rm_cfg.resnet_type
        resnet_type = 'resnet{}'.format(resnet_type)
        self.encoder = ResNetSimple(model_type=resnet_type,
                                pretrained=True,
                                fmapDim=[128, 128, 128, 128],
                                handNum=2,
                                heatmapDim=21)
        self.mid_model = resnet_mid(model_type=resnet_type,
                            in_fmapDim=[128, 128, 128, 128],
                            out_fmapDim=[256, 256, 256, 256])
        
        self.gf_dim = 2048
        if pretrain:
            print("load pretrained intag encoder")
            cpkt = torch.load(pret_model_path)['network']
            estat = {}
            mstat = {}
            for k, v in cpkt.items():
                if k.startswith('module.cliff_model.encoder'):
                    estat[k[7+12+8:]] = v
                elif k.startswith('module.cliff_model.mid_model'):
                    mstat[k[7+12+10:]] = v
            self.encoder.load_state_dict(estat)
            self.mid_model.load_state_dict(mstat)

    def forward(self, img):
        # img:[bs, 3, H, W]
        hms, mask, dp, img_fmaps, hms_fmaps, dp_fmaps = self.encoder(img)
        # hms:[bs, 42, 64, 64]
        # mask:[bs, 2, 64, 64]
        # dp:[bs, 6, 64, 64]
        # img_fmaps:[bs, 2048, 8, 8]
        #           [bs, 1024, 16, 16]
        #           [bs, 512, 32, 32]
        #           [bs, 256, 64, 64]
        # hms_maps/dp_fmaps:[bs, 256, 8, 8]
        #                   [bs, 256, 16, 16]
        #                   [bs, 256, 32, 32]
        #                   [bs, 256, 64, 64]
        xf, fmaps = self.mid_model(img_fmaps, hms_fmaps, dp_fmaps)  # xf:[bs, 2048]
        # fmaps: torch.Size([1, 256, 8, 8])
        #        torch.Size([1, 256, 16, 16])
        #        torch.Size([1, 256, 32, 32])
        #        torch.Size([1, 256, 64, 64])
        return xf, fmaps


class IMG_Encoder_Hr48(nn.Module):
    def __init__(self, pretrain, pret_model_path=''):
        super(IMG_Encoder_Hr48, self).__init__()
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(curr_dir, "../cliff_models/backbones/hrnet/models/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml")
        update_config(hr_cfg, config_file)
        self.encoder = HighResolutionNet(hr_cfg)
        
        if pretrain:
            print("load pretrained intag encoder")
            cpkt = torch.load(pret_model_path)['network']
            estat = {}
            for k, v in cpkt.items():
                if k.startswith('module.cliff_model.encoder'):
                    estat[k[7+12+8:]] = v
            self.encoder.load_state_dict(estat)

    def forward(self, img):
        # img:[bs, 3, H, W]
        xf = self.encoder(img)   # [bs, 2048]
        return xf


class IMG_Encoder_HGF(nn.Module):
    def __init__(self, pretrain, pret_model_path=''):
        super(IMG_Encoder_HGF, self).__init__()
        self.encoder = HGFilter(rm_cfg.hgf_opt)
        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
        )
        self.gf_dim = 256

    def forward(self, img):
        # img:[bs, 3, H, W]
        im_feat_list, _, _ = self.encoder(img)
        global_feat = self.output_layer(im_feat_list[-1])  # [bs, 256]
        # im_feat_list:[bs, 256, 64, 64]
        #              [bs, 256, 64, 64]
        #              [bs, 256, 64, 64]
        #              [bs, 256, 64, 64]
        return global_feat, im_feat_list
    

class ManoParam_Encoder(nn.Module):
    def __init__(self, gf_dim=2048):
        super(ManoParam_Encoder, self).__init__()
        self.npose = 16 * 2 * 3
        self.nshape = 10 * 2
        self.final_dim = gf_dim
        self.lsc = make_linear_layers([self.nshape // 2, 512, 1024, self.final_dim], relu_final=False)
        self.lpc = make_linear_layers([self.npose // 2, 512, 1024, self.final_dim], relu_final=False)
        self.rsc = make_linear_layers([self.nshape // 2, 512, 1024, self.final_dim], relu_final=False)
        self.rpc = make_linear_layers([self.npose // 2, 512, 1024, self.final_dim], relu_final=False)
        # self.lsc = make_linear_layers_bn([self.nshape // 2, 512, 1024, self.final_dim], relu_final=False)
        # self.lpc = make_linear_layers_bn([self.npose // 2, 512, 1024, self.final_dim], relu_final=False)
        # self.rsc = make_linear_layers_bn([self.nshape // 2, 512, 1024, self.final_dim], relu_final=False)
        # self.rpc = make_linear_layers_bn([self.npose // 2, 512, 1024, self.final_dim], relu_final=False)
        
        # if pretrain:
        #     print("load pretrained intag encoder")
        #     cpkt = torch.load(pre_model_path)['network']
        #     estat = {}
        #     for k, v in cpkt.items():
        #         if k.startswith('module.cliff_model.encoder'):
        #             estat[k[7+12+8:]] = v
        #     self.encoder.load_state_dict(estat)

    def forward(self, shape, pose):
        # shape:[bs, 2, 10]
        # pose:[bs, 2, 48]
        
        lsf = self.lsc(shape[:, 1, :])    # [bs, final_dim]
        lpf = self.lpc(pose[:, 1, :])     # [bs, final_dim]
        rsf = self.lsc(shape[:, 0, :])    # [bs, final_dim]
        rpf = self.lpc(pose[:, 0, :])     # [bs, final_dim]
        sf = torch.stack([rsf, lsf], 1)   # [bs, 2, final_dim]
        pf = torch.stack([rpf, lpf], 1)   # [bs, 2, final_dim]
        return sf, pf


class Joint_Encoder(nn.Module):
    def __init__(self):
        super(Joint_Encoder, self).__init__()
        # self.filters = []
        # self.filter_channels = [2048, 1024, 512, 256]
        # for l in range(0, len(self.filter_channels) - 1):
        #     self.filters.append(nn.Conv1d(
        #         self.filter_channels[l],
        #         self.filter_channels[l + 1],
        #         1))
        #     self.add_module("linear%d" % l, self.filters[l])

    def forward(self, joint_xy, img_fmaps):
        # joint_xy:[bs, 42, 2], -1~1
        # img_feat:[bs, 2048, 8, 8]
        #          [bs, 1024, 16, 16]
        #          [bs, 512, 32, 32]
        #          [bs, 256, 64, 64]

        uv = joint_xy.unsqueeze(2)  # [B, 2*(778+21), 1, 2]

        j_f = torch.nn.functional.grid_sample(img_fmaps[-1], uv, padding_mode='border', 
                                                    align_corners=True, mode='bilinear')[:, :, :, 0]  # [bs, 256, 2*(778+21)]
        j_feat = torch.cat([j_f, joint_xy.transpose(1, 2)], dim=1)   # [bs, 256+2, 2*(778+21)]
        
        return j_feat
    
    
# class Joint_Encoder(nn.Module):
#     def __init__(self):
#         super(Joint_Encoder, self).__init__()
#         self.joint_deconv_1 = make_deconv_layers([256, 256, 256, 256])
#         self.joint_deconv_2 = make_deconv_layers([256, 256, 256, 256])
#         self.vert_num = 778
#         self.joint_num = 21

#     def forward(self, joint_xy, img_fmaps):
#         # joint_xy:[bs, 42, 2], -1~1
#         # img_fmaps:[bs, 256, 8, 8]
#         #          [bs, 256, 16, 16]
#         #          [bs, 256, 32, 32]
#         #          [bs, 256, 64, 64]
#         rimg_feat = self.joint_deconv_1(img_fmaps[0])   # [bs, 256, 64, 64]
#         limg_feat = self.joint_deconv_2(img_fmaps[0])   # [bs, 256, 64, 64]
        
#         ruv = joint_xy[:, :self.vert_num+self.joint_num, :].unsqueeze(2)  # [B, 778+21, 1, 2]
#         luv = joint_xy[:, self.vert_num+self.joint_num:, :].unsqueeze(2)  # [B, 778+21, 1, 2]

#         rj_f = torch.nn.functional.grid_sample(rimg_feat, ruv, padding_mode='border', 
#                                                     align_corners=True, mode='bilinear')[:, :, :, 0]  # [bs, 256, 778+21]
#         rj_feat = torch.cat([rj_f, ruv.squeeze(2).transpose(1, 2)], dim=1)   # [bs, 256+2, 778+21]
        
#         lj_f = torch.nn.functional.grid_sample(limg_feat, luv, padding_mode='border', 
#                                                     align_corners=True, mode='bilinear')[:, :, :, 0]  # [bs, 256, 778+21]
#         lj_feat = torch.cat([lj_f, luv.squeeze(2).transpose(1, 2)], dim=1)   # [bs, 256+2, 778+21]
        
#         j_feat = torch.cat([rj_feat, lj_feat], -1)      # [bs, 256+2, 2*(778+21)]
#         return j_feat


class Mask_Encoder(nn.Module):
    def __init__(self):
        super(Mask_Encoder, self).__init__()
        self.final_dim = 256
        self.filters = make_conv_layers([1, 128, 256, self.final_dim], kernel=3, stride=1, padding=1, bnrelu_final=True)
        self.output_layer = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(start_dim=1),
                )
        
    def forward(self, avg_mask, voxel_kp_pred):
        # avg_mask:[bs, 1, 32, 32]
        # voxel_kp_pred:[bs, 1, 32, 32]
        
        sub_mask = self.filters(avg_mask - voxel_kp_pred)   # [bs, 256, 32, 32]
        global_feat = self.output_layer(sub_mask)           # [bs, 256]
        
        return global_feat
    

# class ManoLoss_Net(nn.Module):
#     def __init__(self, mf_dim=2048):
#         super(ManoLoss_Net, self).__init__()
#         self.npose = 16 * 2 * 3
#         self.nshape = 10 * 2
#         self.lslmlp = make_linear_layers([mf_dim, 512, 256, 128, self.nshape // 2], relu_final=False)
#         self.lplmlp = make_linear_layers([mf_dim, 512, 256, 128, self.npose // 2], relu_final=False)
#         self.rslmlp = make_linear_layers([mf_dim, 512, 256, 128, self.nshape // 2], relu_final=False)
#         self.rplmlp = make_linear_layers([mf_dim, 512, 256, 128, self.npose // 2], relu_final=False)
        
#         # self.lslmlp = make_linear_layers_bn([mf_dim, 512, 256, 128, self.nshape // 2], relu_final=False)
#         # self.lplmlp = make_linear_layers_bn([mf_dim, 512, 256, 128, self.npose // 2], relu_final=False)
#         # self.rslmlp = make_linear_layers_bn([mf_dim, 512, 256, 128, self.nshape // 2], relu_final=False)
#         # self.rplmlp = make_linear_layers_bn([mf_dim, 512, 256, 128, self.npose // 2], relu_final=False)
#         # self.scale_shape_mlp = make_linear_layers([self.nshape * 2, 64, self.nshape], relu_final=False)
#         # self.scale_pose_mlp = make_linear_layers([self.npose * 2, 128, self.npose], relu_final=False)
#         self.final_dim = 256
#         self.lmlp = make_linear_layers([self.nshape + self.npose, 128, self.final_dim], relu_final=False)

#     def forward(self, global_feat, shape_feat, pose_feat, pred_hand_type=None, sub_shape=None, sub_pose=None):
#         # global_feat:[bs, 2048]
#         # shape_feat:[bs, 2, 2048]
#         # pose:[bs, 2, 2048]
#         # pred_hand_type:[bs, 2]
#         # sub_shape:[bs, 2*10]
#         # sub_pose:[bs, 2*48]
        
#         lsl_feat = self.lslmlp(global_feat - shape_feat[:, 1, :])   # [bs, 10]
#         lpl_feat = self.lplmlp(global_feat - pose_feat[:, 1, :])    # [bs, 48]
#         rsl_feat = self.rslmlp(global_feat - shape_feat[:, 0, :])   # [bs, 10]
#         rpl_feat = self.rplmlp(global_feat - pose_feat[:, 0, :])    # [bs, 48]
#         sl_feat = torch.cat([rsl_feat, lsl_feat], 1)  # [bs, 2*10]
#         # sl_feat = self.scale_shape_mlp(torch.cat([sl_feat, sub_shape], -1))  # [bs, 2*10]
#         if pred_hand_type is not None:
#             sl_half_dim = sl_feat.shape[1] // 2
#             sl_valid = pred_hand_type.unsqueeze(-1).repeat(1, 1, sl_half_dim).view(-1, 2*sl_half_dim)  # [bs, 2*10]
#             sl_feat = sl_feat * sl_valid
#         pl_feat = torch.cat([rpl_feat, lpl_feat], 1)  # [bs, 2*48]
#         # pl_feat = self.scale_pose_mlp(torch.cat([pl_feat, sub_pose], -1))  # [bs, 2*48]
#         if pred_hand_type is not None:
#             pl_half_dim = pl_feat.shape[1] // 2
#             pl_valid = pred_hand_type.unsqueeze(-1).repeat(1, 1, pl_half_dim).view(-1, 2*pl_half_dim)  # [bs, 2*48]
#             pl_feat = pl_feat * pl_valid
#         l_feat = self.lmlp(torch.cat([sl_feat, pl_feat], 1))  # [bs, final_dim]
#         return l_feat, sl_feat.view(-1, 2, self.nshape // 2), pl_feat.view(-1, 2, self.npose // 2)


class ManoLoss_Net(nn.Module):
    def __init__(self, mf_dim=2048):
        super(ManoLoss_Net, self).__init__()
        self.npose = 16 * 2 * 3
        self.nshape = 10 * 2
        
        self.rglob_mlp = make_linear_layers([mf_dim, mf_dim, mf_dim], relu_final=False)
        self.lglob_mlp = make_linear_layers([mf_dim, mf_dim, mf_dim], relu_final=False)
        
        self.lslmlp = make_linear_layers([mf_dim, 512, 256, 128, self.nshape // 2], relu_final=False)
        self.lplmlp = make_linear_layers([mf_dim, 512, 256, 128, self.npose // 2], relu_final=False)
        self.rslmlp = make_linear_layers([mf_dim, 512, 256, 128, self.nshape // 2], relu_final=False)
        self.rplmlp = make_linear_layers([mf_dim, 512, 256, 128, self.npose // 2], relu_final=False)
        
        # self.lslmlp = make_linear_layers_bn([mf_dim, 512, 256, 128, self.nshape // 2], relu_final=False)
        # self.lplmlp = make_linear_layers_bn([mf_dim, 512, 256, 128, self.npose // 2], relu_final=False)
        # self.rslmlp = make_linear_layers_bn([mf_dim, 512, 256, 128, self.nshape // 2], relu_final=False)
        # self.rplmlp = make_linear_layers_bn([mf_dim, 512, 256, 128, self.npose // 2], relu_final=False)
        # self.scale_shape_mlp = make_linear_layers([self.nshape * 2, 64, self.nshape], relu_final=False)
        # self.scale_pose_mlp = make_linear_layers([self.npose * 2, 128, self.npose], relu_final=False)
        self.final_dim = 256
        self.lmlp = make_linear_layers([self.nshape + self.npose, 128, self.final_dim], relu_final=False)

    def forward(self, global_feat, shape_feat, pose_feat, pred_hand_type=None, sub_shape=None, sub_pose=None):
        # global_feat:[bs, 2048]
        # shape_feat:[bs, 2, 2048]
        # pose:[bs, 2, 2048]
        # pred_hand_type:[bs, 2]
        # sub_shape:[bs, 2*10]
        # sub_pose:[bs, 2*48]
        
        rglob_feat = self.rglob_mlp(global_feat)     # [bs, 2048]
        lglob_feat = self.lglob_mlp(global_feat)     # [bs, 2048]
        
        lsl_feat = self.lslmlp(lglob_feat - shape_feat[:, 1, :])   # [bs, 10]
        lpl_feat = self.lplmlp(lglob_feat - pose_feat[:, 1, :])    # [bs, 48]
        rsl_feat = self.rslmlp(rglob_feat - shape_feat[:, 0, :])   # [bs, 10]
        rpl_feat = self.rplmlp(rglob_feat - pose_feat[:, 0, :])    # [bs, 48]
        sl_feat = torch.cat([rsl_feat, lsl_feat], 1)  # [bs, 2*10]
        # sl_feat = self.scale_shape_mlp(torch.cat([sl_feat, sub_shape], -1))  # [bs, 2*10]
        if sub_shape is not None:
            sl_feat = sl_feat * sub_shape
        if pred_hand_type is not None:
            sl_half_dim = sl_feat.shape[1] // 2
            sl_valid = pred_hand_type.unsqueeze(-1).repeat(1, 1, sl_half_dim).view(-1, 2*sl_half_dim)  # [bs, 2*10]
            sl_feat = sl_feat * sl_valid
        pl_feat = torch.cat([rpl_feat, lpl_feat], 1)  # [bs, 2*48]
        # pl_feat = self.scale_pose_mlp(torch.cat([pl_feat, sub_pose], -1))  # [bs, 2*48]
        if sub_pose is not None:
            pl_feat = pl_feat * sub_pose 
        if pred_hand_type is not None:
            pl_half_dim = pl_feat.shape[1] // 2
            pl_valid = pred_hand_type.unsqueeze(-1).repeat(1, 1, pl_half_dim).view(-1, 2*pl_half_dim)  # [bs, 2*48]
            pl_feat = pl_feat * pl_valid
        l_feat = self.lmlp(torch.cat([sl_feat, pl_feat], 1))  # [bs, final_dim]
        return l_feat, sl_feat.view(-1, 2, self.nshape // 2), pl_feat.view(-1, 2, self.npose // 2)


class ManoLoss_Net_IntagHand(nn.Module):
    def __init__(self, encoder_info):
        super(ManoLoss_Net_IntagHand, self).__init__()
        self.decoder = get_decoder(encoder_info)
        
        reduced_pnum = 10
        pf_ch=64
        if_ch=256
        gNNum=252
        self.lPool = ConvPool1D(gNNum, reduced_pnum)
        self.rPool = ConvPool1D(gNNum, reduced_pnum)
        self.pose_head = nn.Sequential(
            nn.Linear(2 * pf_ch * reduced_pnum + if_ch, pf_ch * reduced_pnum),
            nn.LayerNorm(pf_ch * reduced_pnum, 1e-6),
            nn.ReLU(True),
            nn.Linear(pf_ch * reduced_pnum, 2048),    # original number of joints in mano minus root
        )
        self.shape_head = nn.Sequential(
            nn.Linear(2 * pf_ch * reduced_pnum + if_ch, pf_ch * reduced_pnum),
            nn.LayerNorm(pf_ch * reduced_pnum, 1e-6),
            nn.ReLU(True),
            nn.Linear(pf_ch * reduced_pnum, 2048),
        )
        self.fconv = nn.Sequential(
            nn.Conv2d(if_ch, if_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(if_ch),
            nn.ReLU(True),
            nn.Conv2d(if_ch, if_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(if_ch),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.npose = 16 * 2 * 3
        self.nshape = 10 * 2
        self.slmlp = make_linear_layers([2048, 512, 256, 128, self.nshape], relu_final=False)
        self.plmlp = make_linear_layers([2048, 512, 256, 128, self.npose], relu_final=False)
        
    def forward(self, global_feat, shape_feat, pose_feat, fmaps):
        # global_feat:[bs, 2048]
        # shape_feat:[bs, 2048]
        # pose_feat:[bs, 2048]
        # fmaps: torch.Size([1, 256, 8, 8])
        #        torch.Size([1, 256, 16, 16])
        #        torch.Size([1, 256, 32, 32])
        #        torch.Size([1, 256, 64, 64])
        
        lfs, rfs = self.decoder(global_feat, fmaps)      # lfs/rfs:[bs, 252, 64]
        B = lfs.shape[0]
        lfs = self.lPool(lfs).view(B, -1)   # [bs, 10*D]
        rfs = self.rPool(rfs).view(B, -1)   # [bs, 10*D]
        gf = self.fconv(fmaps[-1]).squeeze(3).squeeze(2)  # [bs, 256]
        fs = torch.cat([rfs, lfs, gf], dim=1)              # [bs, 2*10*D+256]
        pose = self.pose_head(fs)               # [bs, 2048]
        shape = self.shape_head(fs)             # [bs, 2048]
        pl = self.plmlp(pose - pose_feat)
        sl = self.slmlp(shape - shape_feat)
        return sl, pl
      

# class JointLoss_Net(nn.Module):
#     def __init__(self, filter_channels, no_residual=True, last_op=None):
#         super(JointLoss_Net, self).__init__()
#         self.filters = []
#         self.no_residual = no_residual
#         self.last_op = last_op

#         if self.no_residual:
#             for l in range(0, len(filter_channels) - 1):
#                 self.filters.append(nn.Conv1d(
#                     filter_channels[l],
#                     filter_channels[l + 1],
#                     1))
#                 self.add_module("conv%d" % l, self.filters[l])
#         else:
#             for l in range(0, len(filter_channels) - 1):
#                 if 0 != l:
#                     self.filters.append(
#                         nn.Conv1d(
#                             filter_channels[l] + filter_channels[0],
#                             filter_channels[l + 1],
#                             1))
#                 else:
#                     self.filters.append(nn.Conv1d(
#                         filter_channels[l],
#                         filter_channels[l + 1],
#                         1))

#                 self.add_module("conv%d" % l, self.filters[l])
                
#         self.joint_num = 21
#         self.vert_num = 778
#         self.final_dim = 256

#         self.lmlp = make_linear_layers([self.vert_num * 2, 512, self.final_dim], relu_final=False)
#         # self.scale_jproj_mlp = make_linear_layers([self.vert_num * 2 * 2, 2048, self.vert_num * 2], relu_final=False)
#         # self.lmlp = make_linear_layers([self.joint_num * 2, 64, 128, self.final_dim], relu_final=False)
#         # if rm_cfg.w_ht:
#         #     self.htmlp = make_linear_layers([2+2048, 512, self.vert_num * 2], relu_final=True)
        
#     def forward(self, j_feat, pred_hand_type=None):
#         # j_feat:[bs, 256+2, (778+21)*2]
#         bs = j_feat.shape[0]
#         y = j_feat     # y:[bs, 256+2, (778+21)*2]
#         tmpy = j_feat  # temp:[bs, 256+2, (778+21)*2]
#         for i, f in enumerate(self.filters):
#             if self.no_residual:
#                 y = self._modules['conv' + str(i)](y)
#             else:
#                 y = self._modules['conv' + str(i)](
#                     y if i == 0
#                     else torch.cat([y, tmpy], 1)
#                 )
#             if i != len(self.filters) - 1:
#                 y = F.leaky_relu(y)

#         if self.last_op:
#             y = self.last_op(y)
#         # y:[bs, 1, (778+21)*2]
        
#         y = y.transpose(1, 2).contiguous().view(bs, -1)    # [bs, (778+21)*2]
#         if pred_hand_type is not None:
#             jl_half_dim = y.shape[1] // 2
#             jl_valid = pred_hand_type.unsqueeze(-1).repeat(1, 1, jl_half_dim).view(-1, 2*jl_half_dim)  # [bs, (778+21)*2]
#             # jl_valid = self.htmlp(torch.cat([pred_hand_type, global_feat], dim=-1))
#             y = y * jl_valid

#         rv, lv = y[:, :self.vert_num], y[:, self.vert_num+self.joint_num:self.vert_num*2+self.joint_num]
#         l_feat = self.lmlp(torch.cat([rv, lv], -1))  # [bs, final_dim]
#         return l_feat, y  # [bs, final_dim], [bs, nv+nj]
    

class JointLoss_Net(nn.Module):
    def __init__(self, filter_channels, no_residual=True, last_op=None):
        super(JointLoss_Net, self).__init__()
        self.rfilters = []
        self.lfilters = []
        self.no_residual = no_residual
        self.last_op = last_op

        if self.no_residual:
            for l in range(0, len(filter_channels) - 1):
                self.rfilters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    1))
                self.lfilters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    1))
                self.add_module("rconv%d" % l, self.rfilters[l])
                self.add_module("lconv%d" % l, self.lfilters[l])
        else:
            for l in range(0, len(filter_channels) - 1):
                if 0 != l:
                    self.rfilters.append(
                        nn.Conv1d(
                            filter_channels[l] + filter_channels[0],
                            filter_channels[l + 1],
                            1))
                    self.lfilters.append(
                        nn.Conv1d(
                            filter_channels[l] + filter_channels[0],
                            filter_channels[l + 1],
                            1))
                else:
                    self.rfilters.append(nn.Conv1d(
                        filter_channels[l],
                        filter_channels[l + 1],
                        1))
                    self.lfilters.append(nn.Conv1d(
                        filter_channels[l],
                        filter_channels[l + 1],
                        1))

                self.add_module("rconv%d" % l, self.rfilters[l])
                self.add_module("lconv%d" % l, self.lfilters[l])
                
        self.joint_num = 21
        self.vert_num = 778
        self.final_dim = 256

        self.lmlp = make_linear_layers([self.vert_num * 2, 512, self.final_dim], relu_final=False)
        # self.scale_jproj_mlp = make_linear_layers([self.vert_num * 2 * 2, 2048, self.vert_num * 2], relu_final=False)
        # self.lmlp = make_linear_layers([self.joint_num * 2, 64, 128, self.final_dim], relu_final=False)
        # if rm_cfg.w_ht:
        #     self.htmlp = make_linear_layers([2+2048, 512, self.vert_num * 2], relu_final=True)
        
    def forward(self, j_feat, pred_hand_type=None):
        # j_feat:[bs, 256+2, (778+21)*2]
        bs = j_feat.shape[0]
        ry = j_feat[:, :, :self.vert_num+self.joint_num]     # ry:[bs, 256+2, 778+21]
        rtmpy = j_feat[:, :, :self.vert_num+self.joint_num]  # rtemp:[bs, 256+2, 778+21]
        ly = j_feat[:, :, self.vert_num+self.joint_num:]     # ly:[bs, 256+2, 778+21]
        ltmpy = j_feat[:, :, self.vert_num+self.joint_num:]  # ltemp:[bs, 256+2, 778+21]
        for i, f in enumerate(self.rfilters):
            if self.no_residual:
                ry = self._modules['rconv' + str(i)](ry)
                ly = self._modules['lconv' + str(i)](ly)
            else:
                ry = self._modules['rconv' + str(i)](
                    ry if i == 0
                    else torch.cat([ry, rtmpy], 1)
                )
                ly = self._modules['lconv' + str(i)](
                    ly if i == 0
                    else torch.cat([ly, ltmpy], 1)
                )
            if i != len(self.rfilters) - 1:
                ry = F.leaky_relu(ry)
                ly = F.leaky_relu(ly)

        if self.last_op:
            ry = self.last_op(ry)
            ly = self.last_op(ly)
        y = torch.cat([ry, ly], -1)
        # y:[bs, 1, (778+21)*2]
        
        y = y.transpose(1, 2).contiguous().view(bs, -1)    # [bs, (778+21)*2]
        if pred_hand_type is not None:
            jl_half_dim = y.shape[1] // 2
            jl_valid = pred_hand_type.unsqueeze(-1).repeat(1, 1, jl_half_dim).view(-1, 2*jl_half_dim)  # [bs, (778+21)*2]
            # jl_valid = self.htmlp(torch.cat([pred_hand_type, global_feat], dim=-1))
            y = y * jl_valid

        rv, lv = y[:, :self.vert_num], y[:, self.vert_num+self.joint_num:self.vert_num*2+self.joint_num]
        l_feat = self.lmlp(torch.cat([rv, lv], -1))  # [bs, final_dim]
        return l_feat, y  # [bs, final_dim], [bs, nv+nj]


class MaskLoss_Net(nn.Module):
    def __init__(self):
        super(MaskLoss_Net, self).__init__()
                
        self.joint_num = 21
        self.vert_num = 778
        self.final_dim = 256

        self.lmlp = make_linear_layers([self.vert_num * 2, 512, self.final_dim], relu_final=False)
        
    def forward(self, mk_feat, pred_hand_type=None):
        # mk_feat:[bs, 1, (778+21)*2]
        bs = mk_feat.shape[0]
        y = mk_feat.clone()     # y:[bs, 1, (778+21)*2]
        
        y = y.transpose(1, 2).contiguous().view(bs, -1)    # [bs, (778+21)*2]
        if pred_hand_type is not None:
            jl_half_dim = y.shape[1] // 2
            jl_valid = pred_hand_type.unsqueeze(-1).repeat(1, 1, jl_half_dim).view(-1, 2*jl_half_dim)  # [bs, (778+21)*2]
            y = y * jl_valid

        rv, lv = y[:, :self.vert_num], y[:, self.vert_num+self.joint_num:self.vert_num*2+self.joint_num]
        l_feat = self.lmlp(torch.cat([rv, lv], -1))  # [bs, final_dim]
        return l_feat  # [bs, final_dim]


class AugMano_Net(nn.Module):
    def __init__(self):
        super(AugMano_Net, self).__init__()

        self.aug_rsf_mlp = make_linear_layers([2048 * 2, 2048 * 2, 2048 * 2], relu_final=False)
        self.aug_lsf_mlp = make_linear_layers([2048 * 2, 2048 * 2, 2048 * 2], relu_final=False)
        self.aug_rpf_mlp = make_linear_layers([2048 * 2, 2048 * 2, 2048 * 2], relu_final=False)
        self.aug_lpf_mlp = make_linear_layers([2048 * 2, 2048 * 2, 2048 * 2], relu_final=False)
        
    def forward(self, sf_ls, pf_ls):
        # sf_ls:[bs, com_num, 2, 2048]
        # pf_ls:[bs, com_num, 2, 2048]
        bs, com_num, _, _ = sf_ls.shape
        
        aug_rsf = self.aug_rsf_mlp(sf_ls[:, :, 0, :].reshape(bs, -1)).reshape(bs, com_num, -1)  # [bs, com_num, 2048]
        aug_lsf = self.aug_lsf_mlp(sf_ls[:, :, 1, :].reshape(bs, -1)).reshape(bs, com_num, -1)  # [bs, com_num, 2048]
        aug_rpf = self.aug_rpf_mlp(pf_ls[:, :, 0, :].reshape(bs, -1)).reshape(bs, com_num, -1)  # [bs, com_num, 2048]
        aug_lpf = self.aug_lpf_mlp(pf_ls[:, :, 1, :].reshape(bs, -1)).reshape(bs, com_num, -1)  # [bs, com_num, 2048]
        
        aug_sf = torch.stack([aug_rsf, aug_lsf], 2)    # [bs, com_num, 2, 2048]
        aug_pf = torch.stack([aug_rpf, aug_lpf], 2)    # [bs, com_num, 2, 2048]
        return aug_sf, aug_pf


class AugJoint_Net(nn.Module):
    def __init__(self):
        super(AugJoint_Net, self).__init__()

        self.aug_jf_mlp = make_conv1d_layers([(256+2) * 2, (256+2) * 2, (256+2) * 2], kernel=1, padding=0, stride=1)
        
    def forward(self, jf_ls):
        # jf_ls:[bs, com_num, 256+2, (778+21)*2]
        bs, com_num, dim, nv = jf_ls.shape
        aug_jf = self.aug_jf_mlp(jf_ls.reshape(bs, com_num*dim, nv)).reshape(bs, com_num, dim, nv)
        
        return aug_jf  # [bs, com_num, 256+2, (778+21)*2]


class AugMask_Net(nn.Module):
    def __init__(self):
        super(AugMask_Net, self).__init__()

        self.aug_mkf_mlp = make_linear_layers([256 * 2, 256 * 2, 256 * 2], relu_final=False)
        
    def forward(self, mkf_ls):
        # mkf_ls:[bs, com_num, 256]
        bs, com_num, _ = mkf_ls.shape
        
        aug_mkf = self.aug_mkf_mlp(mkf_ls.reshape(bs, -1)).reshape(bs, com_num, -1)  # [bs, com_num, 256]

        return aug_mkf


class Prob_Net(nn.Module):
    def __init__(self, ml_fin_dim=256):
        super(Prob_Net, self).__init__()

        self.ml_fin_dim = ml_fin_dim
        # if 'mask' in rm_cfg.recons_ls:
        #     self.ml_fin_dim += 1
        if rm_cfg.mlp_r_act == 'relu':
            self.prob_mlp = make_linear_layers([self.ml_fin_dim, 64, 1], relu_final=False)
        elif rm_cfg.mlp_r_act == 'tanh':
            self.prob_mlp = make_linear_layers_tanh([self.ml_fin_dim, 64, 1], tanh_final=False)
        elif rm_cfg.mlp_r_act == 'cat_relu':
            self.prob_mlp = make_linear_layers([self.ml_fin_dim*2, 128, 1], relu_final=False)
        elif rm_cfg.mlp_r_act == 'cat_tanh':
            self.prob_mlp = make_linear_layers_tanh([self.ml_fin_dim*2, 128, 1], tanh_final=False)
        elif rm_cfg.mlp_r_act == 'relu_bn':
            self.prob_mlp = make_linear_layers_bn([self.ml_fin_dim, 128, 1], relu_final=False)
        elif rm_cfg.mlp_r_act == 'tanh_bn':
            self.prob_mlp = make_linear_layers_tanh_bn([self.ml_fin_dim, 128, 1], tanh_final=False)
        elif rm_cfg.mlp_r_act == 'cat_relu_bn':
            self.prob_mlp = make_linear_layers_bn([self.ml_fin_dim*2, 128, 1], relu_final=False)
        elif rm_cfg.mlp_r_act == 'cat_tanh_bn':
            self.prob_mlp = make_linear_layers_tanh_bn([self.ml_fin_dim*2, 128, 1], tanh_final=False)
        elif rm_cfg.mlp_r_act == 'relu_ln':
            self.prob_mlp = make_linear_layers_ln([self.ml_fin_dim, 128, 1], relu_final=False)
        self.out = nn.Sigmoid()

    def forward(self, loss_ls, prob_from_mask=None):
        # loss_ls: compare_num*[bs, ml_fin_dim]
        # prob_from_mask:[bs,]
        
        new_loss_ls = loss_ls
        if 'cat' in rm_cfg.mlp_r_act:
            mlp_r_in = torch.cat([new_loss_ls[0], new_loss_ls[1]], -1)
        else:
            mlp_r_in = new_loss_ls[0] - new_loss_ls[1]
            
        if prob_from_mask is not None:
            p = prob_from_mask.clone()
            k = prob_from_mask.unsqueeze(-1)
            mlp_r_in_i = torch.cat([mlp_r_in, k], dim=-1)
        else:
            mlp_r_in_i = mlp_r_in
            
        mlp_r_out = self.prob_mlp(mlp_r_in_i)        # [bs, 1]
        prob = self.out(mlp_r_out).squeeze(-1)
        
        if prob_from_mask is not None:
            out = (prob + p) / 2
        else:
            out = prob
            
        return out
    
    
# class Prob_Net(nn.Module):
#     def __init__(self):
#         super(Prob_Net, self).__init__()
#         self.ml_fin_dim = 0
#         if 'mano' in rm_cfg.recons_ls:
#             self.ml_fin_dim += 2*(10+48)
#             # self.sub_sl_mlp = make_linear_layers_bn([2*10, 2*10, 1], relu_final=False)
#             # self.sub_pl_mlp = make_linear_layers_bn([2*48, 2*10, 1], relu_final=False)
#         if 'jproj' in rm_cfg.recons_ls:
#             self.ml_fin_dim += 2*(778+21)
#             # self.sub_pointl_mlp = make_linear_layers_bn([2*(778+21), 2*(778+21), 1], relu_final=False)
#         if 'mask' in rm_cfg.recons_ls:
#             self.ml_fin_dim += 2*(778+21)
#         #     # self.sub_maskl_mlp = make_linear_layers_bn([2*(778+21), 2*(778+21), 1], relu_final=False)
        
#         if rm_cfg.mlp_r_act == 'relu':
#             self.prob_mlp = make_linear_layers([self.ml_fin_dim, self.ml_fin_dim, 1], relu_final=False)
#         elif rm_cfg.mlp_r_act == 'tanh':
#             self.prob_mlp = make_linear_layers_tanh([self.ml_fin_dim, self.ml_fin_dim, 1], tanh_final=False)
#         elif rm_cfg.mlp_r_act == 'relu_bn':
#             self.prob_mlp = make_linear_layers_bn([self.ml_fin_dim, self.ml_fin_dim, 1], relu_final=False)
#         elif rm_cfg.mlp_r_act == 'tanh_bn':
#             self.prob_mlp = make_linear_layers_tanh_bn([self.ml_fin_dim, self.ml_fin_dim, 1], tanh_final=False)
        
#         self.out = nn.Sigmoid()

#     def forward(self, sl_ls, pl_ls, pointl_ls, maskl_ls, sub_dict=None):
#         # sl_ls: compare_num*[bs, 2, 10]
#         # pl_ls: compare_num*[bs, 2, 48]
#         # pointl_ls: compare_num*[bs, 2*(778+21)]
#         # maskl_ls: compare_num*[bs, 2*(778+21)]
#         mlp_r_in_ls = []
#         if 'mano' in rm_cfg.recons_ls:
#             bs = sl_ls[0].shape[0]
#             sub_shape = sub_dict['shape']    # [bs, 2, 10]
#             sub_pose = sub_dict['pose']      # [bs, 2, 48]
#             # sub_sl = ((sl_ls[0] - sl_ls[1]) * sub_shape).view(bs, -1)                      # [bs, 2*10]
#             # sub_pl = ((pl_ls[0] - pl_ls[1]) * sub_pose).view(bs, -1)                       # [bs, 2*48]
#             sub_sl = (sl_ls[0] - sl_ls[1]).view(bs, -1)                      # [bs, 2*10]
#             sub_pl = (pl_ls[0] - pl_ls[1]).view(bs, -1)                       # [bs, 2*48]
#             # sub_sl = self.sub_sl_mlp(sub_sl)                  # [bs, 1]
#             # sub_pl = self.sub_pl_mlp(sub_pl)                  # [bs, 1]
#             mlp_r_in_ls.append(sub_sl)
#             mlp_r_in_ls.append(sub_pl)
        
#         if 'jproj' in rm_cfg.recons_ls:
#             sub_jproj = sub_dict['jproj']    # [bs, 2*(778+21)]
#             # sub_pointl = (pointl_ls[0] - pointl_ls[1]) * sub_jproj         # [bs, 2*(778+21)]
#             sub_pointl = pointl_ls[0] - pointl_ls[1]                       # [bs, 2*(778+21)]
#             # sub_pointl = self.sub_pointl_mlp(sub_pointl)      # [bs, 1]
#             # if 'mask' in rm_cfg.recons_ls:
#             #     sub_pointl = maskl_ls[0] * pointl_ls[0] - maskl_ls[1] * pointl_ls[1]
#             mlp_r_in_ls.append(sub_pointl)
        
#         if 'mask' in rm_cfg.recons_ls:
#             sub_maskl = maskl_ls[0] - maskl_ls[1]             # [bs, 2*(778+21)]
#             # sub_maskl = self.sub_maskl_mlp(sub_maskl)         # [bs, 1]
#             mlp_r_in_ls.append(sub_maskl)
        
#         mlp_r_in = torch.cat(mlp_r_in_ls, -1)    # [bs, 2*(10+48+(778+21)*2)]
#         mlp_r_out = self.prob_mlp(mlp_r_in)        # [bs, 1]
#         prob = self.out(mlp_r_out).squeeze(-1)
        
#         return prob