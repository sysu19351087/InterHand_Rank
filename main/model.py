# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from nets.module import BackboneNet, PoseNet, ManoParamHead
from nets.loss import JointHeatmapLoss, HandTypeLoss, RelRootDepthLoss, VertLoss, ParamL1Loss
from nets.cliff_models.cliff_hr48.cliff import CLIFF as cliff_hr48
from nets.cliff_models.cliff_res50.cliff import CLIFF as cliff_res50
from nets.cliff_models.cliff_intaghand.cliff import CLIFF as cliff_intaghand

import math


class Model(nn.Module):
    def __init__(self, cliff_net, pose_net):
        super(Model, self).__init__()

        # modules
        # Create the model instance
        self.cliff_model = cliff_net
        self.pose_net = pose_net

        # loss functions
        self.joint_heatmap_loss = JointHeatmapLoss()
        self.rel_root_depth_loss = RelRootDepthLoss()
        self.hand_type_loss = HandTypeLoss()
        print("predict mano vertices")
        self.L1loss = VertLoss(weight=1, is2D=False)
        self.L1loss_2d = VertLoss(weight=1, is2D=True)
        self.mano_param = cfg.mano_param
        if self.mano_param:
            self.mano_phead = ManoParamHead()
            self.paramloss = ParamL1Loss(weight=1)

    def render_gaussian_heatmap(self, joint_coord):
        x = torch.arange(cfg.output_hm_shape[2])
        y = torch.arange(cfg.output_hm_shape[1])
        z = torch.arange(cfg.output_hm_shape[0])
        zz, yy, xx = torch.meshgrid(z, y, x)
        xx = xx[None, None, :, :, :].cuda().float()
        yy = yy[None, None, :, :, :].cuda().float()
        zz = zz[None, None, :, :, :].cuda().float()
        # xx/yy/zz:[1, 1, 64, 64, 64]

        x = joint_coord[:, :, 0, None, None, None]
        y = joint_coord[:, :, 1, None, None, None]
        z = joint_coord[:, :, 2, None, None, None]
        # x/y/z:[bs, 42, 1, 1, 1]
        heatmap = torch.exp(
            -(((xx - x) / cfg.sigma) ** 2) / 2 - (((yy - y) / cfg.sigma) ** 2) / 2 - (((zz - z) / cfg.sigma) ** 2) / 2)
        heatmap = heatmap * 255
        return heatmap    # [bs, 42, 64, 64, 64]

    def forward(self, inputs, targets, meta_info, mode):
        input_img = inputs['img']    # [bs, 3, 256, 256]
        batch_size = input_img.shape[0]
        
        center, b = inputs['center'], inputs['b_scale']
        img_shape = inputs['img_shape']
        focal_length = inputs['focal_length']   # [bs, 2]
        cx, cy = center[:, 0], center[:, 1]
        if cfg.crop2full:
            bbox_info = torch.stack([cx - img_shape[:, 1] / 2., cy - img_shape[:, 0] / 2., b], dim=-1)  # [bs, 3]
            bbox_info[:, :3] = bbox_info[:, :3] / focal_length[:, 0, None]  # [-1, 1]
            #bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)  # [-1, 1]
        else:
            bbox_info = torch.stack([cx, cy, b], dim=-1)
        
        img_feat, pred_axis_angle, pred_betas, pred_cam_crop = self.cliff_model(input_img.float(), bbox_info.float(), init_zeros=cfg.init_zeros)
        # img_feat:[bs, 2048]
        # jpred_rotmat:[bs, 16*2*3]
        # pred_betas:[bs, 10*2]
        # pred_cam_crop:[bs, 3*2]
        if self.mano_param:
            pred_mano_pose, pred_mano_shape, pred_cam_transl, pred_mano_joint_cam, pred_mano_joint_proj, pred_mano_mesh_cam = self.mano_phead.forward(pred_axis_angle, pred_betas, pred_cam_crop, center, b, focal_length, img_shape)
            # pred_mano_pose:[bs, 2, 16*3]
            # pred_mano_shape:[bs, 2, 10]
            # pred_mano_joint_cam:[bs, 42, 3], 
            # pred_mano_joint_proj:[bs, 42, 3]
        rel_root_depth_out, hand_type = self.pose_net(img_feat)

        if mode == 'train':
            #target_joint_heatmap = self.render_gaussian_heatmap(targets['joint_coord'])
            target_joint_imgs = targets['joint_coord_full'] if cfg.crop2full else targets['joint_coord']
            #target_mano_joint_imgs = targets['mano_joint_imgs']
            joint_valid_uns = meta_info['joint_valid'].unsqueeze(2)

            loss = {}
            #loss['joint_heatmap'] = self.joint_heatmap_loss(joint_heatmap_out, target_joint_heatmap,
            #                                                meta_info['joint_valid'])
            loss['rel_root_depth'] = self.rel_root_depth_loss(rel_root_depth_out, targets['rel_root_depth'],
                                                              meta_info['root_valid'])
            loss['hand_type'] = self.hand_type_loss(hand_type, targets['hand_type'], meta_info['hand_type_valid'])
            
            "joint point related"
            #loss['joint_from_heatmap'] = self.L1loss(joint_coord, target_joint_imgs, joint_valid_uns)
            # loss['joint_from_heatmap_manoGT'] = self.L1loss_2d(joint_coord, target_mano_joint_imgs, meta_info['mano_joint_valids'])

            if self.mano_param:
                #print(targets["mano_shapes"].shape)
                loss['mano_shape'] = self.paramloss(pred_mano_shape, targets["mano_shapes"], targets['hand_type'])
                loss['mano_pose'] = self.paramloss(pred_mano_pose, targets["mano_poses"], targets['hand_type'])
                #loss['cam_param'] = self.paramloss(pred_cam_crop.reshape(-1, 2, 3), targets["cam_param"], targets['hand_type'])
                #loss['cam_transl'] = self.paramloss(pred_cam_transl, targets["cam_transl"], targets['hand_type'])
                loss['joint_proj'] = self.L1loss_2d(pred_mano_joint_proj[:, :, :2], target_joint_imgs[:, :, :2], joint_valid_uns)
                #loss['joint_cam'] = self.L1loss(pred_mano_joint_cam, targets["joint_cam"], joint_valid_uns)
                loss['joint_cam'] = self.L1loss(pred_mano_joint_cam, targets["mano_joint_cams"], joint_valid_uns)
                # loss['mano_joint_cam'] = self.L1loss_2d(pred_mano_joint_cam, targets["mano_joint_cams"], meta_info['mano_joint_valids'])
            return loss
        elif mode == 'test':
            out = {}
            '''
            val_z, idx_z = torch.max(joint_heatmap_out, 2)
            val_zy, idx_zy = torch.max(val_z, 2)
            val_zyx, joint_x = torch.max(val_zy, 2)
            joint_x = joint_x[:, :, None]
            joint_y = torch.gather(idx_zy, 2, joint_x)
            joint_z = torch.gather(idx_z, 2, joint_y[:, :, :, None].repeat(1, 1, 1, cfg.output_hm_shape[1]))[:, :, 0, :]
            joint_z = torch.gather(joint_z, 2, joint_x)
            joint_coord_out = torch.cat((joint_x, joint_y, joint_z), 2).float()   # [bs, 42, 3]
            '''
            
            out['joint_coord'] = pred_mano_joint_proj                         # [bs, 42, 2]
            out['rel_root_depth'] = rel_root_depth_out                            # [bs, 1]
            out['hand_type'] = hand_type                                          # [bs, 2]
            if 'inv_trans' in meta_info:
                out['inv_trans'] = meta_info['inv_trans']
            if 'joint_coord' in targets:
                out['target_joint'] = targets['joint_coord']
            if 'joint_valid' in meta_info:
                out['joint_valid'] = meta_info['joint_valid']
            if 'hand_type_valid' in meta_info:
                out['hand_type_valid'] = meta_info['hand_type_valid']
            if self.mano_param:
                out["mano_shape"] = pred_mano_shape
                out["mano_pose"] = pred_mano_pose
                out["mano_joint_cam"] = pred_mano_joint_cam     # [bs, 42, 3], base root
                out["mano_mesh_cam"] = pred_mano_mesh_cam       # [bs, 2*Nv, 3], no base root
                # x = pred_mano_mesh_cam[:, :, 0].clone() / (pred_mano_mesh_cam[:, :, 2].clone() + 1e-8) * \
                #     focal_length[:, 0].unsqueeze(-1) + (img_shape[:, 1].clone().unsqueeze(-1) / 2)
                # y = pred_mano_mesh_cam[:, :, 1].clone() / (pred_mano_mesh_cam[:, :, 2].clone() + 1e-8) * \
                #     focal_length[:, 1].unsqueeze(-1) + (img_shape[:, 0].clone().unsqueeze(-1) / 2)
                # mano_mesh_proj = torch.stack((x, y), 2)         # [bs, 2*Nv, 2]
                # out['mano_mesh_proj'] = mano_mesh_proj
                out['pred_cam_param'] = pred_cam_crop.view(-1, 2, 3) # [bs, 2, 3]
                out['cam_transl'] = pred_cam_transl               # [bs, 2, 3]
            return out


def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        nn.init.constant_(m.bias, 0)


def get_model(mode, joint_num):
    cliff = eval("cliff_" + cfg.backbone)
    cliff_net = cliff(cfg.SMPL_MEAN_PARAMS)
    pose_net = PoseNet(joint_num)

    if mode == 'train':
        #cliff_net.init_weights()
        pose_net.apply(init_weights)

    model = Model(cliff_net, pose_net)
    return model
