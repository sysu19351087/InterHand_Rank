# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
from torch.nn import functional as F
from config import cfg
from nets.layer import make_linear_layers, make_conv_layers, make_deconv_layers, make_upsample_layers
from nets.resnet import ResNetBackbone
import math
import os
import numpy as np
import smplx
from utils.preprocessing import estimate_focal_length


class BackboneNet(nn.Module):
    def __init__(self):
        super(BackboneNet, self).__init__()
        self.resnet = ResNetBackbone(cfg.resnet_type)

    def init_weights(self):
        self.resnet.init_weights()

    def forward(self, img):
        #print('img:',img.shape)
        img_feat = self.resnet(img)
        #print('img_feat:',img_feat.shape)
        return img_feat


class PoseNet(nn.Module):
    def __init__(self, joint_num):
        super(PoseNet, self).__init__()
        self.joint_num = joint_num  # single hand, 21
        '''
        self.joint_deconv_1 = make_deconv_layers([2048, 256, 256, 256])
        self.joint_conv_1 = make_conv_layers([256, self.joint_num * cfg.output_hm_shape[0]], kernel=1, stride=1,
                                             padding=0, bnrelu_final=False)
        self.joint_deconv_2 = make_deconv_layers([2048, 256, 256, 256])
        self.joint_conv_2 = make_conv_layers([256, self.joint_num * cfg.output_hm_shape[0]], kernel=1, stride=1,
                                             padding=0, bnrelu_final=False)
        '''
        self.root_fc = make_linear_layers([2048, 512, cfg.output_root_hm_shape], relu_final=False)
        self.hand_fc = make_linear_layers([2048, 512, 2], relu_final=False)

    def soft_argmax_1d(self, heatmap1d):
        # heatmap1d:[bs, output_root_hm_shape]
        heatmap1d = F.softmax(heatmap1d, 1)
        accu = heatmap1d * torch.arange(cfg.output_root_hm_shape).float().cuda()[None, :]
        coord = accu.sum(dim=1)
        return coord  # [bs,]

    def forward(self, img_feat):
        # img_feat:[bs, 2048]
        
        ib_feat = img_feat.clone()   # [bs, 2048]
        '''
        joint_img_feat_1 = self.joint_deconv_1(ib_feat)
        joint_heatmap3d_1 = self.joint_conv_1(joint_img_feat_1).view(-1, self.joint_num, cfg.output_hm_shape[0],
                                                                     cfg.output_hm_shape[1], cfg.output_hm_shape[2])
        joint_img_feat_2 = self.joint_deconv_2(ib_feat)
        joint_heatmap3d_2 = self.joint_conv_2(joint_img_feat_2).view(-1, self.joint_num, cfg.output_hm_shape[0],
                                                                     cfg.output_hm_shape[1], cfg.output_hm_shape[2])
        joint_heatmap3d = torch.cat((joint_heatmap3d_1, joint_heatmap3d_2), 1)   # [bs, 42, 64, 64, 64]
        '''
        #img_feat_gap = F.avg_pool2d(ib_feat, (img_feat.shape[2], img_feat.shape[3])).view(bz, -1)  # [bs, 2048]
        img_feat_gap = ib_feat
        root_heatmap1d = self.root_fc(img_feat_gap)    # [bs, output_root_hm_shape]
        root_depth = self.soft_argmax_1d(root_heatmap1d).view(-1, 1)   # find depth, [bs, 1]
        hand_type = torch.sigmoid(self.hand_fc(img_feat_gap))          # [bs, 2]

        #return joint_heatmap3d, root_depth, hand_type
        return root_depth, hand_type

class ManoParamHead(nn.Module):
    def __init__(self):
        super(ManoParamHead, self).__init__()
        current_path = os.path.abspath(__file__)
        joint_regressor = np.load(os.path.abspath(current_path + '/../../../smplx/models/mano/J_regressor_mano_ih26m.npy'))
        self.register_buffer('joint_regressor', torch.tensor(joint_regressor).unsqueeze(0))
        #self.joint_regressor = torch.Tensor(joint_regressor).unsqueeze(0).cuda()
        smplx_path = os.path.abspath(current_path + '/../../../smplx/models')
        self.rmano_layer = smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=True, create_transl=False)
        self.lmano_layer = smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=False, create_transl=False)
        # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
        if torch.sum(torch.abs(
                self.lmano_layer.shapedirs[:, 0, :] - self.rmano_layer.shapedirs[:, 0, :])) < 1:
            self.lmano_layer.shapedirs[:, 0, :] *= -1

    def get_coord(self, output, cam_trans, focal_length, img_shape):
        # mesh_cam = output.vertices * 1000
        #print('xrange:', np.max(output.vertices.detach().cpu().numpy()[:, :, 0], -1) - np.min(output.vertices.detach().cpu().numpy()[:, :, 0], -1))
        #print('yrange:', np.max(output.vertices.detach().cpu().numpy()[:, :, 1], -1) - np.min(output.vertices.detach().cpu().numpy()[:, :, 1], -1))
        mesh_cam = output.vertices
        joint_cam_m = torch.matmul(self.joint_regressor, mesh_cam.to(torch.float32))
        root_cam_m = joint_cam_m[:, 20, None, :].clone()
        joint_cam_m = joint_cam_m - root_cam_m
        joint_cam_mm = (joint_cam_m.clone() + cam_trans[:, None, :]) * 1000
        
        #print(joint_cam.shape)
        #print(cam_trans.shape)
        """ x = (joint_cam[:, :, 0].clone() + cam_trans[:, None, 0]) / (joint_cam[:, :, 2].clone() + cam_trans[:, None, 2] + 1e-8) * \
            focal_length[:, 0].unsqueeze(-1) + img_shape[:, 0].unsqueeze(-1) // 2
        y = (joint_cam[:, :, 1].clone() + cam_trans[:, None, 1]) / (joint_cam[:, :, 2].clone() + cam_trans[:, None, 2] + 1e-8) * \
            focal_length[:, 1].unsqueeze(-1) + img_shape[:, 1].unsqueeze(-1) // 2 """
        #print(img_shape[:, 0], img_shape[:, 1])
        x = joint_cam_mm[:, :, 0].clone() / (joint_cam_mm[:, :, 2].clone() + 1e-8) * \
            focal_length[:, 0].unsqueeze(-1) + (img_shape[:, 1].clone().unsqueeze(-1) / 2)
        y = joint_cam_mm[:, :, 1].clone() / (joint_cam_mm[:, :, 2].clone() + 1e-8) * \
            focal_length[:, 1].unsqueeze(-1) + (img_shape[:, 0].clone().unsqueeze(-1) / 2)
        if not cfg.crop2full:
            x = x / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
            y = y / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
        joint_proj = torch.stack((x, y), 2)

        # joint_cam_m:[bs, 21, 3], cam space, base root
        # joint_proj:[bs, 21, 3], hm space, no base root
        return joint_cam_m * 1000, joint_proj, (mesh_cam - root_cam_m + cam_trans[:, None, :]) * 1000
    
    def get_camtrans(self, pred_cam, center, b_scale, focal_length, img_shape):
        if cfg.crop2full:
            img_h, img_w = img_shape[:, 0].clone(), img_shape[:, 1].clone()
            cx, cy, b = center[:, 0], center[:, 1], b_scale
            w_2, h_2 = img_w / 2., img_h / 2.
            bs = b * pred_cam[:, 0] + 1e-9
            
            tz = cfg.mano_size * focal_length[:, 0].clone() / bs
            tx = (cfg.mano_size * (cx - w_2) / bs) + pred_cam[:, 1]
            ty = (cfg.mano_size * (cy - h_2) / bs) + pred_cam[:, 2]
            full_cam = torch.stack([tx, ty, tz], dim=-1)
            #print(focal_length[:,0])
            #print('pred_cam:', pred_cam[:, :])
            #print('pred_cam_reans:', full_cam[:, :])
        else:
            img_h, img_w = img_shape[:, 0], img_shape[:, 1]
            
            bs = estimate_focal_length(img_h, img_w) * pred_cam[:, 0] + 1e-9
            tz = cfg.mano_size * focal_length[:, 0] / bs
            tx = pred_cam[:, 1]
            ty = pred_cam[:, 2]
            full_cam = torch.stack([tx, ty, tz], dim=-1)   
        return full_cam   # [bs, 3]

    def forward(self, poses, betas, pred_cams, center, b_scale, focal_length, img_shape):
        # poses:[bs, 32*3], betas:[bs, 20], cam_trans:[bs, 6]
        # focal_length:[bs, 2], princpt:[bs, 2]
        rroot_pose, rhand_pose, rshape, rcam_trans = poses[:, 15*3:15*3+3], poses[:, :15*3], betas[:, :10], self.get_camtrans(pred_cams[:, :3], center, b_scale, focal_length, img_shape)
        lroot_pose, lhand_pose, lshape, lcam_trans = poses[:, 31*3:31*3+3], poses[:, 15*3+3:31*3], betas[:, 10:], self.get_camtrans(pred_cams[:, 3:], center, b_scale, focal_length, img_shape)
        
        rout = self.rmano_layer(global_orient=rroot_pose, hand_pose=rhand_pose, betas=rshape)
        #rout = self.rmano_layer(global_orient=rroot_pose, hand_pose=rhand_pose, betas=rshape, transl=rcam_trans)
        rjoint_cam, rjoint_proj, rmesh_cam = self.get_coord(rout, rcam_trans, focal_length, img_shape)
        lout = self.lmano_layer(global_orient=lroot_pose, hand_pose=lhand_pose, betas=lshape)
        #lout = self.lmano_layer(global_orient=lroot_pose, hand_pose=lhand_pose, betas=lshape, transl=lcam_trans)
        ljoint_cam, ljoint_proj, lmesh_cam = self.get_coord(lout, lcam_trans, focal_length, img_shape)
        mano_pose = torch.cat([torch.cat([rroot_pose, rhand_pose], dim=1).unsqueeze(1), torch.cat([lroot_pose, lhand_pose], dim=1).unsqueeze(1)], dim=1)
        mano_shape = torch.cat([rshape.unsqueeze(1), lshape.unsqueeze(1)], dim=1)
        cam_transl = torch.cat([rcam_trans.unsqueeze(1), lcam_trans.unsqueeze(1)], dim=1)
        mano_joint_cam = torch.cat([rjoint_cam, ljoint_cam], dim=1)
        mano_joint_proj = torch.cat([rjoint_proj, ljoint_proj], dim=1)
        mano_mesh_cam = torch.cat([rmesh_cam, lmesh_cam], dim=1)
        return mano_pose, mano_shape, cam_transl, mano_joint_cam, mano_joint_proj, mano_mesh_cam
        #return mano_joint_cam, mano_joint_proj