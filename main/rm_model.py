# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
# from scipy.spatial import cKDTree as KDTree
from nets.layer import make_linear_layers, make_linear_layers_bn, hgf_init_weights, create_grid_points_from_bounds
from nets.rm_module import IMG_Encoder_Intaghand as ie_intaghand
from nets.rm_module import IMG_Encoder_Hr48 as ie_hr48
from nets.rm_module import IMG_Encoder_HGF as ie_hgf
from nets.rm_module import ManoParam_Encoder, Joint_Encoder, Mask_Encoder, ManoLoss_Net, \
    Render_Mask, ManoLoss_Net_IntagHand, JointLoss_Net, MaskLoss_Net, Prob_Net, AugMano_Net, AugJoint_Net, AugMask_Net
from nets.rm_loss import RM_JointLoss, RM_ParamL1Loss, MeanLoss
from nets.cliff_models.cliff_hr48.cliff import CLIFF as cliff_hr48
from nets.cliff_models.cliff_res50.cliff import CLIFF as cliff_res50
from nets.cliff_models.cliff_intaghand.cliff import CLIFF as cliff_intaghand
from utils.preprocessing import trans_point2d
from utils.vis import vis_keypoints, vis_meshverts_together_with_sampleresults, \
    vis_meshverts_together_with_mask, vis_3d_keypoints, render_mesh, vis_meshverts_together
from rm_config import rm_cfg
import math
import numpy as np
import PIL.Image
import os


def get_gt_prob(gt_loss, bs, com_num, device, mask_losses=None):
    target_mean_loss = torch.zeros(bs, com_num).to(device)
    if 'mano' in rm_cfg.recons_ls:
        target_mean_sl = gt_loss['mean_loss_mano_shape']    # [compare_num,]
        target_mean_pl = gt_loss['mean_loss_mano_pose']     # [compare_num,]
        #print('mano:', target_mean_sl+target_mean_pl)
        target_mean_loss = target_mean_loss + target_mean_sl + target_mean_pl       # [compare_num,]
    if 'jproj' in rm_cfg.recons_ls:
        target_mean_jl = gt_loss['mean_loss_joint_proj']    # [compare_num,]
        #print('cam:', target_mean_jl)
        target_mean_loss = target_mean_loss + target_mean_jl
        # print(target_mean_jl)
    if 'mask' in rm_cfg.recons_ls and mask_losses is not None:
        # print(mask_losses)
        target_mean_loss = target_mean_loss + mask_losses.to(device)
        
    sub_target_mean_loss = target_mean_loss[:, 0] - target_mean_loss[:, 1] 
    prob_one = (sub_target_mean_loss >= 0.0).float()    # [bs,]
    target_prob = prob_one
    
    return target_prob


class RM_Model(nn.Module):
    def __init__(self, img_encoder, mano_encoder, joint_encoder, mask_encoder, manol_net, jointl_net, maskl_net, prob_net):
        super(RM_Model, self).__init__()

        # modules
        # Create the model instance
        self.img_encoder = img_encoder
        self.final_dim_dict = {}
        if 'mano' in rm_cfg.recons_ls:
            self.mano_encoder = mano_encoder
            self.manol_net = manol_net
            self.final_dim_dict['mano'] = self.manol_net.final_dim
            if rm_cfg.aug_feat:
                self.augmano_net = AugMano_Net()
        if 'jproj' in rm_cfg.recons_ls:
            self.joint_encoder = joint_encoder
            self.jointl_net = jointl_net
            self.final_dim_dict['jproj'] = self.jointl_net.final_dim
            if rm_cfg.aug_feat:
                self.augjoint_net = AugJoint_Net()
        if 'mask' in rm_cfg.recons_ls:
            self.reso_grid = rm_cfg.reso_grid
            self.avgpool = nn.AdaptiveAvgPool2d((self.reso_grid, self.reso_grid))
            # self.maxpool = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
            # self.grid_points = create_grid_points_from_bounds(-1.0, 1.0, self.reso_grid)   # [32*32, 2]
            # self.kdtree = KDTree(self.grid_points)
            self.mask_encoder = mask_encoder
            #self.maskl_net = maskl_net
            self.render_net = Render_Mask()
            self.final_dim_dict['mask'] = self.mask_encoder.final_dim
            if rm_cfg.aug_feat:
                self.augmask_net = AugMask_Net()
        
        self.prob_net = prob_net
        
        if len(self.final_dim_dict) > 1:
            self.final_dim = list(self.final_dim_dict.values())[0]
            for v in self.final_dim_dict.values():
                assert v == self.final_dim
            self.all_lmlp = make_linear_layers([len(self.final_dim_dict) * self.final_dim, self.final_dim], relu_final=False)
        
        self.gt_prob_net = nn.Sigmoid()
        
        self.npose = mano_encoder.npose
        self.nshape = mano_encoder.nshape
        self.joint_num = 21
        self.vert_num = 778

        # loss functions
        self.prob_loss = nn.BCELoss()
        self.prob_reverse_loss = MeanLoss(weight=0.1)   
        self.mano_loss = RM_ParamL1Loss(weight=1)
        self.joint_loss = RM_JointLoss(weight=1)


    def forward(self, inputs, preds, gt_loss, mode):
        input_img = inputs['img']    # [bs, 3, 256, 256]
        if 'mask' in rm_cfg.recons_ls:
            input_mask = inputs['mask'].unsqueeze(1)  # [bs, 1, 256, 256]
            input_img = input_img * input_mask
        bs, _, ori_H, ori_W = input_img.shape
        
        img_shape = inputs['img_shape']               # [bs, 2]
        sub_dict = {}
       
        if 'mano' in rm_cfg.recons_ls:
            pred_shape = preds['pred_mano_shape']  # [bs, compare_num, 2, 10]
            pred_pose = preds['pred_mano_pose']    # [bs, compare_num, 2, 48]
            com_num = pred_pose.shape[1]
            sub_shape = torch.abs(pred_shape[:, 0] - pred_shape[:, 1]) * 100 # [bs, 2, 10]
            sub_pose = torch.abs(pred_pose[:, 0] - pred_pose[:, 1]) * 100   # [bs, 2, 48]
            # print('sub_s:', sub_shape)
            # print('sub_p:', sub_pose)
            # sub_shape = F.softmax(sub_shape, dim=-1)                # [bs, 2, 10]
            # sub_pose = F.softmax(sub_pose, dim=-1)                  # [bs, 2, 48]
            sub_dict['shape'] = sub_shape
            sub_dict['pose'] = sub_pose
            # print('sub_shape:',sub_shape)
            # print('sub_pose:',sub_pose)
            
        if 'jproj' in rm_cfg.recons_ls or 'mask' in rm_cfg.recons_ls:
            rp_vert_xy, lp_vert_xy = preds['pred_mano_mesh_crop_proj'][:, :, :self.vert_num, :], preds['pred_mano_mesh_crop_proj'][:, :, self.vert_num:, :]   # [bs, compare_num, 2*778, 2]
            rp_joint_xy, lp_joint_xy = preds['pred_joint_crop_coord'][:, :, :self.joint_num, :], preds['pred_joint_crop_coord'][:, :, self.joint_num:, :]     # [bs, compare_num, 2*21, 2]
            p_xy = torch.cat([rp_vert_xy, rp_joint_xy, lp_vert_xy, lp_joint_xy], -2)   # [bs, compare_num, 2*(778+21), 2]
            com_num = p_xy.shape[1] 
            joint_xy = torch.zeros_like(p_xy).to(p_xy.device)
            joint_xy[:, :, :, 0] = p_xy[:, :, :, 0] / int(rm_cfg.input_img_shape[1]) * 2 - 1
            joint_xy[:, :, :, 1] = p_xy[:, :, :, 1] / int(rm_cfg.input_img_shape[0]) * 2 - 1
            joint_xy = torch.clamp(joint_xy, max=1.0, min=-1.0)      # [bs, compare_num, 2*(778+21), 2]
            sub_jproj = torch.mean(torch.abs(p_xy[:, 0] - p_xy[:, 1]), dim=-1)       # [bs, 2*(778+21)]
            #print('sub_j:', sub_jproj.view(bs, 2, -1))
            # sub_jproj = F.softmax(sub_jproj.view(bs, 2, -1), dim=-1).view(bs, -1)             # [bs, 2*(778+21)]
            sub_dict['jproj'] = sub_jproj
            #print('sub_jproj:',sub_jproj.view(bs, 2, -1))
            
        if 'mask' in rm_cfg.recons_ls:
            verts = preds['pred_mesh_verts']   # [bs, compare_num, 778*2, 3]
            faces = preds['pred_mesh_faces']   # [bs, compare_num, nf*2, 3]
            focal_length = inputs['render_focal_length']   # [bs, 2]
            princpt = inputs['render_princpt']             # [bs, 2]
            avg_mask = self.avgpool(input_mask)         # [bs, 1, 32, 32]
            _, _, avg_H, avg_W = avg_mask.shape
            voxel_kp_pred = torch.zeros([bs, com_num, 1, avg_H, avg_W]).to(input_img.device)   # [bs, compare_num, 1, 32, 32]
            pred_ori_mask = torch.zeros([bs, com_num, 1, ori_H, ori_W]).to(input_img.device)   # [bs, compare_num, 1, 256, 256]
            for cid in range(com_num):
                render_mask = self.render_net((ori_H, ori_W), verts[:, cid], faces[:, cid], focal_length, princpt)
                pred_ori_mask[:, cid] = render_mask.clone()
                voxel_kp_pred[:, cid] = self.avgpool(render_mask)
            
            # occupancies = torch.zeros(bs, com_num, len(self.grid_points)).to(input_img.device)   # [bs, compare_num, 32*32]

            # for cid in range(com_num):
            #     pred_ht = preds['pred_hand_type'][:, cid] if rm_cfg.w_ht else None   # [bs, 2]
            #     valid_ht = torch.sum((pred_ht > 0.5).float(), dim=-1, keepdim=True)  # [bs, 1]
            #     pred_ht = (valid_ht == 0).float() * 1. + pred_ht                    # [bs, 2]
            #     all_kp = joint_xy[:, cid].clone().detach().cpu().numpy()                      # [bs, 2*(778+21), 2]
            #     for b in range(bs):
            #         valid_kp = []
            #         ht = pred_ht[b]      # [2,]
            #         if ht[0] > 0.5:
            #             valid_kp.append(all_kp[b, :self.vert_num+self.joint_num])         # [778+21, 2]
            #         if ht[1] > 0.5:
            #             valid_kp.append(all_kp[b, self.vert_num+self.joint_num:])         # [778+21, 2]
            #         valid_kp = np.concatenate(valid_kp, axis=0)       # [2*(778+21), 2]
                    
            #         _, idx = self.kdtree.query(valid_kp)
            #         occupancies[b, cid, idx] = 1
            
            # voxel_kp_pred = occupancies.view(bs, com_num, 1, self.reso_grid, self.reso_grid)     # [bs, compare_num, 1, 32, 32]
            # voxel_kp_pred = voxel_kp_pred.transpose(-1, -2)
            
            # for cid in range(com_num):
            #     voxel_kp_pred[:, cid] = self.maxpool(voxel_kp_pred[:, cid])
            
            # if mode == 'test':
            #     img2save0 = voxel_kp_pred[0, 0].permute(1, 2, 0).detach().cpu().numpy()  # [32, 32, 1]
            #     img2save0 = np.uint8(img2save0 * 255.).repeat(3, axis=-1)                 # [32, 32, 3]
            #     img2save1 = voxel_kp_pred[0, 1].permute(1, 2, 0).detach().cpu().numpy()  # [32, 32, 1]
            #     img2save1 = np.uint8(img2save1 * 255.).repeat(3, axis=-1)                 # [32, 32, 3]
            #     avgmask2save = avg_mask[0].permute(1, 2, 0).detach().cpu().numpy()      # [32, 32, 1]
            #     avgmask2save = np.uint8(avgmask2save * 255.).repeat(3, axis=-1)         # [32, 32, 3]
            #     img2save = np.concatenate([avgmask2save, img2save0, img2save1], 1)
            #     msimg2save = input_img[0].permute(1, 2, 0).detach().cpu().numpy()  # [256, 256, 3]
            #     msimg2save = np.uint8(msimg2save * 255.)      # [32, 32, 3]
            #     PIL.Image.fromarray(img2save).save(os.path.join('.', 'test_occupy.png'))
            #     PIL.Image.fromarray(msimg2save).save(os.path.join('.', 'test_mkimg.png'))
            #     vis_meshverts_together_with_mask(input_img[0].permute(1, 2, 0).detach().cpu().numpy(), input_mask[0, 0].detach().cpu().numpy(), \
            #         preds['pred_mano_mesh_crop_proj'][0], preds['pred_hand_type'][0, 0], 'test_proj.png', '.')
            
        if rm_cfg.recons_ls != 'mask':
            global_feat, img_fmaps = self.img_encoder(input_img)
        # img_fmaps: torch.Size([1, 256, 8, 8])
        #            torch.Size([1, 256, 16, 16])
        #            torch.Size([1, 256, 32, 32])
        #            torch.Size([1, 256, 64, 64])
        # global_feat:[bs, 2048]
        
        sf_ls = []
        pf_ls = []
        mkf_ls = []
        jf_ls = []
        for pid in range(com_num):
            if 'mano' in rm_cfg.recons_ls:
                sf, pf = self.mano_encoder(pred_shape[:, pid], pred_pose[:, pid])  # sf/pf:[bs, 2, 2048]
                sf_ls.append(sf.clone())
                pf_ls.append(pf.clone())
            
            if 'mask' in rm_cfg.recons_ls:
                mkf = self.mask_encoder(avg_mask, voxel_kp_pred[:, pid])  # mkf:[bs, 256]
                mkf_ls.append(mkf.clone())
            
            if 'jproj' in rm_cfg.recons_ls:
                jf = self.joint_encoder(joint_xy[:, pid].clone(), img_fmaps)  # jf:[bs, 256+2, (778+21)*2]
                jf_ls.append(jf.clone())
        sf_ls = torch.stack(sf_ls, 1) if len(sf_ls) != 0 else []       # [bs, com_num, 2, 2048]
        pf_ls = torch.stack(pf_ls, 1) if len(pf_ls) != 0 else []       # [bs, com_num, 2, 2048]
        mkf_ls = torch.stack(mkf_ls, 1) if len(mkf_ls) != 0 else []    # [bs, com_num, 256]
        jf_ls = torch.stack(jf_ls, 1) if len(jf_ls) != 0 else []       # [bs, com_num, 256+2, (778+21)*2]
        
        if rm_cfg.aug_feat:
            if 'mano' in rm_cfg.recons_ls:
                sf_ls, pf_ls = self.augmano_net(sf_ls, pf_ls)
            if 'mask' in rm_cfg.recons_ls:
                mkf_ls = self.augmask_net(mkf_ls)
            if 'jproj' in rm_cfg.recons_ls:
                jf_ls = self.augjoint_net(jf_ls)
                
        # mask_losses = None
        # prob_from_mask = None
        # if 'mask' in rm_cfg.recons_ls:
        #     mask_loss_ls = []
        #     for cid in range(com_num):
        #         pred_mk = pred_ori_mask[:, cid].clone()  # [bs, 1, 256, 256]
        #         mask_loss = torch.mean((input_mask - pred_mk) ** 2, dim=(-1, -2, -3)) * 30  # [bs,]
        #         # mask_loss = torch.mean(F.binary_cross_entropy(pred_mk, input_mask, reduction='none'), dim=(-1, -2, -3))
        #         mask_loss_ls.append(mask_loss)
        #     mask_losses = torch.stack(mask_loss_ls, -1)  # [bs, compare_num]
        #     prob_from_mask = torch.sigmoid(mask_losses[:, 0] - mask_losses[:, 1])  # [bs,]
            
        loss_ls = []
        sl_ls = []
        pl_ls = []
        pointl_ls = []
        maskl_ls = []
        for pid in range(com_num):
            l_ls = []
            pred_ht = preds['pred_hand_type'][:, pid] if rm_cfg.w_ht else None   # [bs, 2]
            valid_ht = torch.sum((pred_ht > 0.5).float(), dim=-1, keepdim=True)  # [bs, 1]
            pred_ht = (valid_ht == 0).float() * 1. + pred_ht                     # [bs, 2]
            if 'mano' in rm_cfg.recons_ls:
                sub_s = sub_dict['shape'].view(bs, -1)
                sub_p = sub_dict['pose'].view(bs, -1)
                mano_lf, sl, pl = self.manol_net(global_feat, sf_ls[:, pid], pf_ls[:, pid], pred_hand_type=pred_ht, sub_shape=None, sub_pose=None)
                # mano_lf:[bs, 256], sl:[bs, 2, 10], pl:[bs, 2, 48]
                l_ls.append(mano_lf.clone())
                sl_ls.append(sl.clone())        # [bs, 2*10]
                pl_ls.append(pl.clone())        # [bs, 2*48]
            
            if 'mask' in rm_cfg.recons_ls:
                mk_lf = mkf_ls[:, pid]    # mk_lf:[bs, 256]
                l_ls.append(mk_lf.clone())
            
            if 'jproj' in rm_cfg.recons_ls:
                j_lf, pointl = self.jointl_net(jf_ls[:, pid], pred_hand_type=pred_ht.clone())
                # j_lf:[bs, 256], pointl:[bs, (778+21)*2]
                l_ls.append(j_lf.clone())
                pointl_ls.append(pointl.clone())    # [bs, (778+21)*2]
                
            l = l_ls[0] if len(l_ls) == 1 else self.all_lmlp(torch.cat(l_ls, -1))
            
            loss_ls.append(l)
        
        prob = self.prob_net(loss_ls, prob_from_mask=None)   # [bs,]
        # prob = self.prob_net(sl_ls, pl_ls, pointl_ls, maskl_ls, sub_dict=sub_dict)   # [bs,]
        
        if 'mano' in rm_cfg.recons_ls:
            sls = torch.stack(sl_ls, 1)                      # [bs, compare_num, 2, 10]
            pls = torch.stack(pl_ls, 1)                      # [bs, compare_num, 2, 48]
        if 'jproj' in rm_cfg.recons_ls:
            pointls = torch.stack(pointl_ls, 1)              # [bs, compare_num, (778+21)*2]
        if rm_cfg.prob_rever_loss:
            loss_ls_reverse = []
            for pid in range(com_num-1, -1, -1):
                loss_ls_reverse.append(loss_ls[pid].clone())
            prob_reverse = self.prob_net(torch.stack(loss_ls_reverse, 0)).squeeze(-1)   # [bs,]
        
        mask_losses = None
        if 'mask' in rm_cfg.recons_ls:
            mask_loss_ls = []
            for cid in range(com_num):
                pred_mk = pred_ori_mask[:, cid].clone()  # [bs, 1, 256, 256]
                mask_loss = torch.mean((input_mask - pred_mk) ** 2, dim=(-1, -2, -3)) * 50  # [bs,]
                # mask_loss = torch.mean(F.binary_cross_entropy(pred_mk, input_mask, reduction='none'), dim=(-1, -2, -3))
                mask_loss_ls.append(mask_loss)
            mask_losses = torch.stack(mask_loss_ls, -1)  # [bs, compare_num]
            prob_from_mask = torch.sigmoid(mask_losses[:, 0] - mask_losses[:, 1])  # [bs,]
            # prob = (prob + prob_from_mask) / 2.0
            # prob = prob_from_mask
        
        if mode == 'train':
            target_prob = get_gt_prob(gt_loss, bs, com_num, input_img.device, mask_losses=mask_losses)
             
            loss = {}
            # print('gt:', target_prob)
            # print('pred:', prob)
            loss['prob_loss'] = self.prob_loss(prob, target_prob)
            if 'mano' in rm_cfg.recons_ls:
                target_sl = gt_loss['loss_mano_shape']    # [bs, compare_num, 2, 10]
                target_pl = gt_loss['loss_mano_pose']     # [bs, compare_num, 2, 48]
                loss['sl_loss'] = self.mano_loss(sls, target_sl)
                loss['pl_loss'] = self.mano_loss(pls, target_pl)
            if 'jproj' in rm_cfg.recons_ls:
                target_jl = gt_loss['loss_joint_proj'].mean(dim=-1)     # [bs, compare_num, 42]
                rjls, ljls = pointls[:, :, self.vert_num:self.vert_num+self.joint_num], pointls[:, :, self.vert_num*2+self.joint_num:]  # [bs, compare_num, 21]
                jls = torch.cat([rjls, ljls], -1)          # [bs, compare_num, 21*2]
                loss['jl_loss'] = self.joint_loss(jls, target_jl)
            if rm_cfg.prob_rever_loss:
                loss['prob_reverse_loss'] = self.prob_loss(prob_reverse, 1-target_prob)
            return loss
        elif mode == 'test':
            out = {}
            
            prob = prob - 0.5                   # [bs,], -0.5~0.5
            # prob_one = (prob >= 0.03).float()    # [bs,]
            # prob_half = (torch.abs(prob) < 0.03).float()  # [bs,]
            # pred_prob = prob_one + 0.5 * prob_half         # [bs,]
            prob_one = (prob >= 0.0).float()    # [bs,]
            pred_prob = prob_one                # [bs,]
            out['prob'] = prob                  # [bs,]
            out['label'] = pred_prob            # [bs,]
            # pred_ori_mask.retain_grad()
            # mask_losses.retain_grad()
            # prob.retain_grad()
            out['pred_ori_mask'] = pred_ori_mask if 'mask' in rm_cfg.recons_ls else None
            out['mask_losses'] = mask_losses if 'mask' in rm_cfg.recons_ls else None
            # out['mask_losses'] = mask_loss_ls   # [bs, compare_num]
            # print('prob:', prob)
            # print('pred_prob:', pred_prob)
            return out


def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.Conv1d:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        nn.init.constant_(m.bias, 0)


def get_model(mode, pretrain=True, pret_model_path=''):
    ie = eval("ie_" + rm_cfg.backbone)
    img_encoder = ie(pretrain, pret_model_path)
    mano_encoder = ManoParam_Encoder(img_encoder.gf_dim)
    joint_encoder = Joint_Encoder()
    mask_encoder = Mask_Encoder()
    manol_net = ManoLoss_Net(mano_encoder.final_dim)
    # manol_net = ManoLoss_Net_IntagHand(img_encoder.mid_model.get_info())
    jointl_net = JointLoss_Net(rm_cfg.mlp_dim, no_residual=True, last_op=None)
    maskl_net = MaskLoss_Net()
    prob_net = Prob_Net(manol_net.final_dim)
    # prob_net = Prob_Net()


    if mode == 'train':
        #img_encoder.apply(init_weights)
        hgf_init_weights(img_encoder, init_type='normal', init_gain=0.02)
        mano_encoder.apply(init_weights)
        joint_encoder.apply(init_weights)
        mask_encoder.apply(init_weights)
        # manol_net.apply(init_weights)
        jointl_net.apply(init_weights)
        maskl_net.apply(init_weights)
        prob_net.apply(init_weights)

    model = RM_Model(img_encoder, mano_encoder, joint_encoder, mask_encoder, manol_net, jointl_net, maskl_net, prob_net)
    # print("load pretrained rm18 encoder")
    # cpkt = torch.load('/data1/linxiaojian/InterHand2.6M-main/rm18_cliff10_InterHand2.6M_0.1/model_dump/snapshot_19.pth.tar')['network']
    # iestat = {}
    # mestat = {}
    # jestat = {}
    # mlstat = {}
    # jlstat = {}
    # alstat = {}
    # for k, v in cpkt.items():
    #     if k.startswith('module.img_encoder'):
    #         iestat[k[7+11+1:]] = v
    #     elif k.startswith('module.mano_encoder'):
    #         mestat[k[7+12+1:]] = v
    #     elif k.startswith('module.joint_encoder'):
    #         jestat[k[7+13+1:]] = v
    #     elif k.startswith('module.manol_net'):
    #         mlstat[k[7+9+1:]] = v
    #     elif k.startswith('module.jointl_net'):
    #         jlstat[k[7+10+1:]] = v
    #     elif k.startswith('module.all_lmlp'):
    #         alstat[k[7+8+1:]] = v

    # model.img_encoder.load_state_dict(iestat)
    # model.mano_encoder.load_state_dict(mestat)
    # model.joint_encoder.load_state_dict(jestat)
    # model.manol_net.load_state_dict(mlstat)
    # model.jointl_net.load_state_dict(jlstat)
    # model.all_lmlp.load_state_dict(alstat)
    
    return model
