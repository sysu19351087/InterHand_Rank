# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import sys
sys.path.append('../../main')
sys.path.append('../../common')
import numpy as np
import PIL.Image
import os
import smplx
from tqdm import tqdm
import copy
import trimesh
import torch
import torch.utils.data
import cv2
from glob import glob
import os.path as osp
from rm_config import rm_cfg
from utils.preprocessing import load_img, load_skeleton, get_bbox, process_bbox, augmentation, \
    transform_input_to_output_space, trans_point2d, trans_poinst2d, estimate_focal_length, gen_ori2rotori_trans, estimate_translation, gen_trans_from_patch_cv
from utils.transforms import world2cam, cam2pixel, pixel2cam, perspective
from utils.vis import vis_keypoints, vis_meshverts_together_with_sampleresults, vis_meshverts_together_with_mask, vis_3d_keypoints, render_mesh, vis_meshverts_together
from nets.loss import JointHeatmapLoss, HandTypeLoss, RelRootDepthLoss, VertLoss, ParamL1Loss
from PIL import Image, ImageDraw
import random
import json
import math
from pycocotools.coco import COCO
import scipy.io as sio
#import open3d

class RM_Dataset(torch.utils.data.Dataset):
    def __init__(self, transform, datapath, mode, debug=False):
        self.mode = mode  # train, test, val
        self.transform = transform
        self.debug = debug
        self.debug_dir = './rm_debug_dataset'
        if self.debug:
            os.makedirs(self.debug_dir, exist_ok=True)
        self.joint_num = 21
        self.vert_num = 778
        self.compare_num = 2
        self.easy_pair_num = 0 if not self.debug else 0
        self.hard_pair_num = 20
        self.all_pair_num = self.easy_pair_num + self.hard_pair_num
        current_path = os.path.abspath(__file__)
        self.img_path = os.path.abspath(current_path + '/../images')
        self.mask_path = os.path.abspath(current_path + '/../masks')
        self.joint_regressor = torch.tensor(np.load(os.path.abspath(current_path + '/../../../smplx/models/mano/J_regressor_mano_ih26m.npy'))).float()
        smplx_path = os.path.abspath(current_path + '/../../../smplx/models')
        self.rmano_layer = smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=True, create_transl=False)
        self.lmano_layer = smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=False, create_transl=False)
        # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
        if torch.sum(torch.abs(
                self.lmano_layer.shapedirs[:, 0, :] - self.rmano_layer.shapedirs[:, 0, :])) < 1:
            self.lmano_layer.shapedirs[:, 0, :] *= -1
        
        self.pret_epoch_ls = [folder for folder in os.listdir(datapath) if folder.startswith('epoch_')]
        ins_path_ls = [os.path.join(datapath, epoch, self.mode, 'ins.pth') for epoch in self.pret_epoch_ls]
        preds_path_ls = [os.path.join(datapath, epoch, self.mode, 'preds.pth') for epoch in self.pret_epoch_ls]
        gts_path_ls = [os.path.join(datapath, epoch, self.mode, 'gts.pth') for epoch in self.pret_epoch_ls]
        if self.debug:
            print(self.pret_epoch_ls)
        self.ins_ls = [torch.load(ins_path) for ins_path in ins_path_ls]
        #print('Load input data from:', ins_path)
        self.preds_ls = [torch.load(preds_path) for preds_path in preds_path_ls]
        #print('Load predicted data from:', preds_path)
        self.gts_ls = [torch.load(gts_path) for gts_path in gts_path_ls]
        #print('Load gt loss data from:', gt_loss_path)

        self.easy_sample_ls = list(range(len(self.pret_epoch_ls)))
        epoch4hard_sample = [5, 10, 15, 19]
        self.hard_sample_ls = [eid for eid in range(len(self.pret_epoch_ls)) if int(self.pret_epoch_ls[eid][6:]) in epoch4hard_sample]
        if len(self.hard_sample_ls) < 2:
            self.hard_sample_ls *= 2
        
        if self.compare_num > len(self.ins_ls):
            self.compare_num = len(self.ins_ls)
        
        assert self.compare_num >= 2, 'Compare num must >= 2'
        
        # sam_ls = np.load(os.path.abspath(current_path + '/../../../sam_res.npy'))
        self.data_len = min([len(ins['img_path']) for ins in self.ins_ls])
        self.valid_idx_ls = []
        ins = self.ins_ls[0]
        for idx in range(self.data_len):
            if self.mode == 'test' and idx % 100 != 0:
                continue
            if self.debug and idx % 500 != 0:
                continue
            joint_valid = ins['joint_valid'][idx]
            if ('jproj' in rm_cfg.recons_ls or 'mask' in rm_cfg.recons_ls) and np.sum(joint_valid) == 0:
                continue
            
            img_path = ins['img_path'][idx]
            img_path_spl = img_path.split('/')
            img_path_spl = img_path_spl[img_path_spl.index('images')+1:]
            mask_path = '/'.join([self.mask_path]+img_path_spl)
            if not os.path.exists(mask_path):
                continue
            
            self.valid_idx_ls.append(idx)
            
        print('Number of RMDataset:' + str(len(self.valid_idx_ls) * self.all_pair_num))
        
        # loss functions
        self.rel_root_depth_loss = RelRootDepthLoss()
        self.hand_type_loss = HandTypeLoss()
        # print("predict mano vertices")
        self.L1loss = VertLoss(weight=1, is2D=False)
        self.L1loss_2d = VertLoss(weight=1, is2D=True)
        self.paramloss = ParamL1Loss(weight=1)
        
        self.valid = 0
        
    def get_coord(self, output, face, cam_trans, focal_length, img_shape):
        mesh_cam_m = output.vertices
        
        joint_cam_m = torch.matmul(self.joint_regressor, mesh_cam_m.to(torch.float32))
        root_cam_m = joint_cam_m[:, 20, None, :].clone()
        # mesh_cam_mm = (mesh_cam_m - root_cam_m + cam_trans[:, None, :]) * 1000
        
        # mesh = open3d.geometry.TriangleMesh()

        # mesh.triangles = open3d.utility.Vector3iVector(face)
        # mesh.vertices = open3d.utility.Vector3dVector(mesh_cam_m.cpu().numpy().reshape(-1, 3))
        
        # fine_mesh = mesh.subdivide_loop(number_of_iterations=2)
        # fine_mesh_cam_m = torch.tensor(np.array(fine_mesh.vertices)).float()
        fine_mesh_cam_m = mesh_cam_m
        fine_mesh_cam_mm = (fine_mesh_cam_m - root_cam_m + cam_trans[:, None, :]) * 1000
        
        mesh_x = fine_mesh_cam_mm[:, :, 0].clone() / (fine_mesh_cam_mm[:, :, 2].clone() + 1e-8) * \
            focal_length[:, 0].unsqueeze(-1) + (img_shape[:, 1].clone().unsqueeze(-1) / 2)
        mesh_y = fine_mesh_cam_mm[:, :, 1].clone() / (fine_mesh_cam_mm[:, :, 2].clone() + 1e-8) * \
            focal_length[:, 1].unsqueeze(-1) + (img_shape[:, 0].clone().unsqueeze(-1) / 2)
        mesh_proj = torch.stack((mesh_x, mesh_y), 2)
        
        joint_cam_m = joint_cam_m - root_cam_m
        joint_cam_mm = (joint_cam_m.clone() + cam_trans[:, None, :]) * 1000
        x = joint_cam_mm[:, :, 0].clone() / (joint_cam_mm[:, :, 2].clone() + 1e-8) * \
            focal_length[:, 0].unsqueeze(-1) + (img_shape[:, 1].clone().unsqueeze(-1) / 2)
        y = joint_cam_mm[:, :, 1].clone() / (joint_cam_mm[:, :, 2].clone() + 1e-8) * \
            focal_length[:, 1].unsqueeze(-1) + (img_shape[:, 0].clone().unsqueeze(-1) / 2)
        joint_proj = torch.stack((x, y), 2)
        return mesh_proj, joint_proj, fine_mesh_cam_mm, joint_cam_m * 1000
        
    def get_mano_mesh_proj(self, betas, poses, cam_transls, focal_length, img_shape):
        
        shp = betas if torch.is_tensor(betas) else torch.tensor(betas).float()
        ps = poses if torch.is_tensor(poses) else torch.tensor(poses).float()
        ct = cam_transls if torch.is_tensor(cam_transls) else torch.tensor(cam_transls).float()
        fl = focal_length if torch.is_tensor(focal_length) else torch.tensor(focal_length).float()
        imsh = img_shape if torch.is_tensor(img_shape) else torch.tensor(img_shape).float()
            
        rroot_pose, rhand_pose, rshape, rcam_trans = ps[None, 0, :3], ps[None, 0, 3:], shp[None, 0, :], ct[None, 0, :]
        lroot_pose, lhand_pose, lshape, lcam_trans = ps[None, 1, :3], ps[None, 1, 3:], shp[None, 1, :], ct[None, 1, :]
        
        rout = self.rmano_layer(global_orient=rroot_pose, hand_pose=rhand_pose, betas=rshape)
        #rout = self.rmano_layer(global_orient=rroot_pose, hand_pose=rhand_pose, betas=rshape, transl=rcam_trans)
        rmesh_proj, rjoint_proj, rmesh_cam, rjoint_cam = self.get_coord(rout, self.rmano_layer.faces, rcam_trans, fl.unsqueeze(0), imsh.unsqueeze(0))    # [nv, 2]
        lout = self.lmano_layer(global_orient=lroot_pose, hand_pose=lhand_pose, betas=lshape)
        #lout = self.lmano_layer(global_orient=lroot_pose, hand_pose=lhand_pose, betas=lshape, transl=lcam_trans)
        lmesh_proj, ljoint_proj, lmesh_cam, ljoint_cam = self.get_coord(lout, self.lmano_layer.faces, lcam_trans, fl.unsqueeze(0), imsh.unsqueeze(0))    # [nv, 2]
        
        mesh_proj = torch.cat([rmesh_proj[0], lmesh_proj[0]], 0)       # [nv*2, 2]
        joint_proj = torch.cat([rjoint_proj[0], ljoint_proj[0]], 0)    # [21*2, 2]
        mesh_cam = torch.cat([rmesh_cam[0], lmesh_cam[0]], 0)          # [nv*2, 3]
        joint_cam = torch.cat([rjoint_cam[0], ljoint_cam[0]], 0)       # [21*2, 3]
        return mesh_proj, joint_proj, mesh_cam, joint_cam
        
    def get_gt_loss(self, img_idx, in_data, pred_data, gt_data):
        
        # (1) meta info
        img_path = in_data['img_path']
        bbox = in_data['bbox']           # [4,]
        do_flip = in_data['do_flip']
        scale = in_data['scale']
        rot = in_data['rot']
        color_scale = in_data['color_scale']
        trans_full2crop = in_data['trans']
        
        img_path_spl = img_path.split('/')
        img_path_spl = img_path_spl[img_path_spl.index('images')+1:]
        img_path = '/'.join([self.img_path]+img_path_spl)
        img = load_img(img_path)           # [H_ori, W_ori, 3]
        H_ori, W_ori = img.shape[:2]
        if do_flip == 1:
            img = img[:, ::-1, :]
        original_img = img.copy()          # [H_ori, W_ori, 3]
        img = cv2.warpAffine(img, trans_full2crop, (int(rm_cfg.input_img_shape[1]), int(rm_cfg.input_img_shape[0])), flags=cv2.INTER_LINEAR)
        img = np.clip(img * color_scale[None, None, :], 0, 255)
        save_img = img.copy()
        img = self.transform(img.astype(np.float32)) / 255.    # [3, H, W], 0~1
        
        mask_path = '/'.join([self.mask_path]+img_path_spl)
        mask = cv2.imread(mask_path)      # [H_ori, W_ori, 3]
        if do_flip == 1:
            mask = mask[:, ::-1, :]
        mask = cv2.warpAffine(mask, trans_full2crop, (int(rm_cfg.input_img_shape[1]), int(rm_cfg.input_img_shape[0])), flags=cv2.INTER_LINEAR)
        mask = self.transform(mask.astype(np.float32)) / 255.  # [3, H, W], 0 or 1
        # save_img = save_img * mask.permute(1,2,0).numpy()
        mask = torch.mean(mask, dim=0)                         # [H, W], 0 or 1
        mask = (mask > 0.5).float()
        
        center = torch.tensor(in_data['center']).float()
        b_scale = torch.tensor(in_data['b_scale']).float()
        focal_length = torch.tensor(in_data['focal_length']).float()
        img_shape = torch.tensor(in_data['img_shape']).float()
        trans_full2crop = torch.tensor(in_data['trans']).float()
        
        trans_rot2crop = gen_trans_from_patch_cv(center[0], center[1], bbox[2], bbox[3], int(rm_cfg.input_img_shape[1]), int(rm_cfg.input_img_shape[0]), scale, 0)
        trans_full2rot = gen_ori2rotori_trans(center[0], center[1], bbox[2], bbox[3], scale, rot, inv=False)
        trans_full2rot = torch.tensor(trans_full2rot).float()
        
        if 'mask' in rm_cfg.recons_ls:
            princpt = torch.tensor([img_shape[1] // 2, img_shape[0] // 2]).float()
            render_princpt = trans_point2d(princpt, trans_rot2crop)
            render_focal_length = [
                focal_length[0] * rm_cfg.input_img_shape[1] / (scale * bbox[2]),
                focal_length[1] * rm_cfg.input_img_shape[0] / (scale * bbox[3]),
            ]
            render_focal_length = torch.tensor(render_focal_length).float()
        
        root_valid = torch.tensor(in_data['root_valid'])     # [1,]
        hand_type_valid = torch.tensor(in_data['hand_type_valid'])  # [1,]
        joint_valid = torch.tensor(in_data['joint_valid'])   # [42,]
        joint_valid_uns = joint_valid.unsqueeze(1)
        
        # (2) gt data
        gt_joint_coord = torch.tensor(gt_data['gt_joint_coord']).float()           # [42, 2]
        gt_rel_root_depth = torch.tensor(gt_data['gt_rel_root_depth']).float()     # [1,]
        gt_hand_type = torch.tensor(gt_data['gt_hand_type']).float()               # [2,]
        gt_mano_shape = torch.tensor(gt_data['gt_mano_shape']).float()             # [2, 10]
        gt_mano_pose = torch.tensor(gt_data['gt_mano_pose']).float()               # [2, 48]
        gt_mano_joint_cam = torch.tensor(gt_data['gt_mano_joint_cam']).float()     # [42, 3]
        gt_cam_transl = torch.tensor(gt_data['gt_cam_transl']).float()             # [2, 3]
        if self.debug:
            # gt_mano_mesh_proj = torch.tensor(gt_data['gt_mano_mesh_proj']).float()     # [778*2, 2]
            gt_proj = self.get_mano_mesh_proj(gt_mano_shape, gt_mano_pose, gt_cam_transl, focal_length, img_shape)   # [778*2, 2]
            gt_mano_mesh_proj = gt_proj[0]
            gt_joint_crop_coord = trans_poinst2d(gt_joint_coord[:, :2].clone().cpu().numpy(), trans_rot2crop)
            gt_mano_mesh_crop_proj = trans_poinst2d(gt_mano_mesh_proj[:, :2].clone().cpu().numpy(), trans_rot2crop)
            gt_mesh_cam = gt_proj[2]     # [nv*2, 3]
        if self.debug:
            print(img_idx, do_flip, scale, rot, gt_hand_type)
        
        # (3) pred data
        pred_joint_coord = torch.tensor(pred_data['pred_joint_coord']).float()         # [compare_num, 42, 2]
        pred_rel_root_depth = torch.tensor(pred_data['pred_rel_root_depth']).float()   # [compare_num, 1]
        pred_hand_type = torch.tensor(pred_data['pred_hand_type'] >= 0.5).float()      # [compare_num, 2]
        valid_ht = torch.sum(pred_hand_type, dim=-1, keepdim=True)                     # [compare_num, 1]
        pred_hand_type = (valid_ht == 0).float() * 1. + pred_hand_type                 # [compare_num, 2]
        pred_mano_shape = torch.tensor(pred_data['pred_mano_shape']).float()           # [compare_num, 2, 10]
        pred_mano_pose = torch.tensor(pred_data['pred_mano_pose']).float()             # [compare_num, 2, 48]
        pred_mano_joint_cam = torch.tensor(pred_data['pred_mano_joint_cam']).float()   # [compare_num, 42, 3]
        pred_cam_transl = torch.tensor(pred_data['pred_cam_transl']).float()           # [compare_num, 2, 3]
        # pred_mano_mesh_proj = torch.tensor(pred_data['pred_mano_mesh_proj']).float()   # [compare_num, 778*2, 2]
        pred_joint_crop_coord = pred_joint_coord.clone().cpu().numpy()
        if 'jproj' in rm_cfg.recons_ls or 'mask' in rm_cfg.recons_ls:
            
            pred_mano_mesh_proj_ls = []
            pred_mask_ls = []
            if self.debug:
                pred_mano_render_ls = []
            cam_param = {'focal': focal_length,
                         'princpt': torch.tensor([img_shape[1] // 2, img_shape[0] // 2]).float()}
            pred_mesh_verts = []
            pred_mesh_faces = []
            for i in range(self.compare_num):
                pred_proj = self.get_mano_mesh_proj(pred_mano_shape[i], pred_mano_pose[i], pred_cam_transl[i], focal_length, img_shape)   # [778*2, 2]
                pred_mano_mesh_proj_ls.append(pred_proj[0])
                
                if 'mask' in rm_cfg.recons_ls:
                    mesh = pred_proj[2] / 1000  # milimeter to meter      # [778*2, 3]
                    rfaces = torch.tensor(self.rmano_layer.faces.copy().astype(np.float32)).float()
                    lfaces = torch.tensor(self.lmano_layer.faces.copy().astype(np.float32)).float()
                    rverts = mesh[:len(mesh) // 2]
                    lverts = mesh[len(mesh) // 2:]
                    half_nv, half_nf = len(rverts), len(rfaces)
                    verts_ls = []
                    faces_ls = []
                    if pred_hand_type[i, 0] > 0.5:
                        faces_ls.append(rfaces)
                        verts_ls.append(rverts)
                    if pred_hand_type[i, 1] > 0.5:
                        faces_ls.append(lfaces + len(verts_ls) * half_nv)
                        verts_ls.append(lverts)
                    
                    l = len(verts_ls)
                    for _ in range(2 - l):
                        faces_ls.append(-1*torch.ones(half_nf, 3))
                        verts_ls.append(torch.zeros(half_nv, 3))
                        
                    verts_ts = torch.cat(verts_ls, 0).float()       # [778*2, 3]
                    faces_ts = torch.cat(faces_ls, 0).float()       # [nf*2, 3]
                    pred_mesh_verts.append(verts_ts)
                    pred_mesh_faces.append(faces_ts)
                    
                    if self.debug:
                        pred_mano_render, pred_original_mask = render_mesh(original_img.copy(), verts_ts, faces_ts, cam_param)     # [H_ori, W_ori], 0 or 1
                        pred_mask = cv2.warpAffine(pred_original_mask.astype(np.float32), trans_rot2crop, \
                            (int(rm_cfg.input_img_shape[1]), int(rm_cfg.input_img_shape[0])), flags=cv2.INTER_LINEAR)  # [H, W]
                        pred_mano_render = cv2.warpAffine(pred_mano_render.astype(np.float32), trans_rot2crop, \
                            (int(rm_cfg.input_img_shape[1]), int(rm_cfg.input_img_shape[0])), flags=cv2.INTER_LINEAR)  # [H, W, 3]
                        pred_mano_render_ls.append(pred_mano_render)
                        pred_mask_ls.append(torch.tensor(pred_mask).float())
                
                if in_data['pid'] >= self.easy_pair_num and i in in_data['rand_cid']:
                    pred_joint_coord[i] = pred_proj[1]
            pred_mano_mesh_proj = torch.stack(pred_mano_mesh_proj_ls, 0)    # [compare_num, 778*2, 2]
            if 'mask' in rm_cfg.recons_ls and self.debug:
                pred_mask = torch.stack(pred_mask_ls, 0)     # [compare_num, H, W], 0 or 1
            pred_mano_mesh_crop_proj = pred_mano_mesh_proj.clone().cpu().numpy()
        
        # (4) get gt loss
        loss_rel_root_depth = []
        loss_hand_type = []
        loss_mano_shape = []
        loss_mano_pose = []
        loss_joint_proj = []
        loss_joint_cam = []
        loss_mask = []
        mean_loss_rel_root_depth = []
        mean_loss_hand_type = []
        mean_loss_mano_shape = []
        mean_loss_mano_pose = []
        mean_loss_joint_proj = []
        mean_loss_joint_cam = []
        mean_loss_mask = []
        for cid in range(self.compare_num):    
            loss_rel_root_depth.append(self.rel_root_depth_loss(pred_rel_root_depth[cid].unsqueeze(0), gt_rel_root_depth.unsqueeze(0), root_valid.unsqueeze(0))[0])   # [1,]
            loss_hand_type.append(self.hand_type_loss(pred_hand_type[cid].unsqueeze(0), gt_hand_type.unsqueeze(0), hand_type_valid.unsqueeze(0))[0])                  # [1,]
            loss_mano_shape.append(self.paramloss(pred_mano_shape[cid].unsqueeze(0), gt_mano_shape.unsqueeze(0), gt_hand_type.unsqueeze(0))[0])                       # [2, 10]
            loss_mano_pose.append(self.paramloss(pred_mano_pose[cid].unsqueeze(0), gt_mano_pose.unsqueeze(0), gt_hand_type.unsqueeze(0))[0])                          # [2, 48]
            loss_joint_proj.append(self.L1loss_2d(pred_joint_coord[cid].unsqueeze(0), gt_joint_coord.unsqueeze(0), joint_valid_uns.unsqueeze(0))[0])                  # [42, 2]
            loss_joint_cam.append(self.L1loss(pred_mano_joint_cam[cid].unsqueeze(0), gt_mano_joint_cam.unsqueeze(0), joint_valid_uns.unsqueeze(0))[0])                # [42, 3]
            if 'mask' in rm_cfg.recons_ls and self.debug:
                loss_mask.append(torch.abs(pred_mask[cid] - mask))        # [H, W]
            
            mean_loss_rel_root_depth.append(torch.mean(loss_rel_root_depth[cid], -1))
            mean_loss_hand_type.append(torch.mean(loss_hand_type[cid], -1))
            mean_loss_mano_shape.append(torch.mean(loss_mano_shape[cid], (-2, -1)))
            mean_loss_mano_pose.append(torch.mean(loss_mano_pose[cid], (-2, -1)))
            mean_loss_joint_proj.append(torch.mean(loss_joint_proj[cid], (-2, -1)))
            mean_loss_joint_cam.append(torch.mean(loss_joint_cam[cid], (-2, -1)))
            if 'mask' in rm_cfg.recons_ls and self.debug:
                mean_loss_mask.append(torch.mean(loss_mask[cid], (-2, -1)))
            if 'jproj' in rm_cfg.recons_ls or 'mask' in rm_cfg.recons_ls:
                pred_joint_crop_coord[cid] = trans_poinst2d(pred_joint_coord[cid, :, :2].clone().cpu().numpy(), trans_rot2crop)
                pred_mano_mesh_crop_proj[cid] = trans_poinst2d(pred_mano_mesh_proj[cid, :, :2].clone().cpu().numpy(), trans_rot2crop)
            # print('pred_joint_coord_max:', torch.max(pred_joint_coord))
            # print('pred_joint_coord_min:', torch.min(pred_joint_coord))
            # print('pred_mano_mesh_crop_proj_max:', np.max(pred_mano_mesh_crop_proj))
            # print('pred_mano_mesh_crop_proj_min:', np.min(pred_mano_mesh_crop_proj))
            
        inputs = {
            'img': img,       # [3, H, W]
            'mask': mask,     # [H, W]
            'center': center, 
            'b_scale': b_scale, 
            'focal_length': focal_length, 
            'img_shape': img_shape, 
            'trans': trans_full2crop, 
            'joint_valid': joint_valid,
            'trans_rot2crop': torch.tensor(trans_rot2crop).float(),
            'trans_full2rot': trans_full2rot,
            'gt_cam_transl': gt_cam_transl,
            #'original_img': self.transform(original_img.astype(np.float32))
        }
        preds = {
            'pred_joint_coord': pred_joint_coord,
            'pred_rel_root_depth': pred_rel_root_depth,
            'pred_hand_type': pred_hand_type,
            'pred_mano_shape': pred_mano_shape,
            'pred_mano_pose': pred_mano_pose,
            'pred_mano_joint_cam': pred_mano_joint_cam,
        }
        if 'jproj' in rm_cfg.recons_ls or 'mask' in rm_cfg.recons_ls:
            preds.update({
                'pred_joint_crop_coord': torch.tensor(pred_joint_crop_coord).float(),
                'pred_mano_mesh_crop_proj': torch.tensor(pred_mano_mesh_crop_proj).float(),
            })
        if 'mask' in rm_cfg.recons_ls:
            preds.update({
                'pred_mesh_verts': torch.stack(pred_mesh_verts, 0),
                'pred_mesh_faces': torch.stack(pred_mesh_faces, 0),
            })
            inputs.update({
                'render_focal_length': render_focal_length,
                'render_princpt': torch.tensor(render_princpt).float(),
            })
            
        loss= {
            'loss_rel_root_depth': torch.stack(loss_rel_root_depth, 0),   # [compare_num, 1]
            'loss_hand_type': torch.stack(loss_hand_type, 0),             # [compare_num, 1]
            'loss_mano_shape': torch.stack(loss_mano_shape, 0),           # [compare_num, 2, 10]
            'loss_mano_pose': torch.stack(loss_mano_pose, 0),             # [compare_num, 2, 48]
            'loss_joint_proj': torch.stack(loss_joint_proj, 0),           # [compare_num, 42, 2]
            'loss_joint_cam': torch.stack(loss_joint_cam, 0),             # [compare_num, 42, 3]
            'mean_loss_rel_root_depth': torch.stack(mean_loss_rel_root_depth, 0),   # [compare_num,]
            'mean_loss_hand_type': torch.stack(mean_loss_hand_type, 0),             # [compare_num,]
            'mean_loss_mano_shape': torch.stack(mean_loss_mano_shape, 0),           # [compare_num,]
            'mean_loss_mano_pose': torch.stack(mean_loss_mano_pose, 0),             # [compare_num,]
            'mean_loss_joint_proj': torch.stack(mean_loss_joint_proj, 0),           # [compare_num,]
            'mean_loss_joint_cam': torch.stack(mean_loss_joint_cam, 0),             # [compare_num,]
            'gt_joint_crop_coord': torch.tensor(trans_poinst2d(gt_joint_coord[:, :2].clone().cpu().numpy(), trans_rot2crop)).float()
        }
        if 'mask' in rm_cfg.recons_ls and self.debug:
            loss.update({
                'loss_mask': torch.stack(loss_mask, 0),                       # [compare_num, H, W]
                'mean_loss_mask': torch.stack(mean_loss_mask, 0),             # [compare_num,]
            })
        
        if self.debug:
            # for k, v in inputs.items():
            #     print(k, v.shape)
            # print('.......')
            # for k, v in preds.items():
            #     print(k, v.shape)
            # print('.......')
            # for k, v in loss.items():
            #     print(k, v.shape)
            # print(pred_joint_coord)
            # print('pred_shape:', torch.abs(pred_mano_shape[in_data['rand_cid']]))
            # print('ls_shape:', loss['loss_mano_shape'][in_data['rand_cid']])
            # print('loss_shape:', loss['loss_mano_shape'][in_data['rand_cid']] / torch.abs(pred_mano_shape[in_data['rand_cid']]))
            # print('loss_pose:', loss['loss_mano_pose'][in_data['rand_cid']] / torch.abs(pred_mano_pose[in_data['rand_cid']]))
            rp_vert_xy, lp_vert_xy = preds['pred_mano_mesh_crop_proj'][:, :self.vert_num, :], preds['pred_mano_mesh_crop_proj'][:, self.vert_num:, :]   # [compare_num, 2*778, 2]
            rp_joint_xy, lp_joint_xy = preds['pred_joint_crop_coord'][:, :self.joint_num, :], preds['pred_joint_crop_coord'][:, self.joint_num:, :]     # [compare_num, 2*21, 2]
            p_xy = torch.cat([rp_vert_xy, rp_joint_xy, lp_vert_xy, lp_joint_xy], -2)   # [compare_num, 2*(778+21), 2]
            joint_xy = torch.zeros_like(p_xy).to(p_xy.device)
            joint_xy[:, :, 0] = p_xy[:, :, 0] / int(rm_cfg.input_img_shape[1]) * 2 - 1
            joint_xy[:, :, 1] = p_xy[:, :, 1] / int(rm_cfg.input_img_shape[0]) * 2 - 1
            joint_xy = torch.clamp(joint_xy, max=1.0, min=-1.0)      # [compare_num, 2*(778+21), 2]
            mean_maskl_ls = []
            v_ls = []
            for cid in range(self.compare_num):
                
                uv = joint_xy[cid].unsqueeze(0).unsqueeze(2)  # [bs, 2*(778+21), 1, 2]

                j_f = torch.nn.functional.grid_sample(1.0-mask.unsqueeze(0).unsqueeze(1), uv, padding_mode='border', 
                                                            align_corners=True, mode='bilinear')[:, :, :, 0]  # [bs, 1, 2*(778+21)]
                j_feat = j_f.squeeze(0).squeeze(0)      # [2*(778+21),]
                jl_half_dim = j_feat.shape[0] // 2
                jl_valid = gt_hand_type.unsqueeze(-1).repeat(1, jl_half_dim).view(2*jl_half_dim)  # [(778+21)*2]
                # jl_valid = self.htmlp(torch.cat([pred_hand_type, global_feat], dim=-1))
                j_feat = j_feat * jl_valid
                
                rv, lv = j_feat[:self.vert_num], j_feat[self.vert_num+self.joint_num:self.vert_num*2+self.joint_num]
                v = torch.cat([rv, lv], -1)
                v_ls.append(v)
                mean_maskl_ls.append(torch.mean(v))
            
            cam_prob = (mean_maskl_ls[0] > mean_maskl_ls[1]).float()
            #cam_prob = loss['target_prob']
            mask_prob = (mean_loss_mask[0] > mean_loss_mask[1]).float()
            
            print('cam_prob:',cam_prob)
            print('mask_prob:',mask_prob)
            
            proj2show = preds['pred_mano_mesh_crop_proj']
            # print(proj2show.shape)
            vis_meshverts_together_with_mask(save_img, mask.numpy(), proj2show, gt_hand_type, 'data_debug{}.png'.format(img_idx), self.debug_dir)
            vis_meshverts_together_with_sampleresults(save_img, mask.numpy(), proj2show, v_ls, gt_hand_type, 'data_debug{}_sr.png'.format(img_idx), self.debug_dir)
            
            mesh = gt_mesh_cam / 1000  # milimeter to meter
            rfaces = self.rmano_layer.faces.copy()
            lfaces = self.lmano_layer.faces.copy()
            rverts = mesh[:len(mesh) // 2].cpu().numpy()
            lverts = mesh[len(mesh) // 2:].cpu().numpy()
            verts = []
            faces = []

            if gt_hand_type[0] > 0.5:
                verts.append(rverts)
                faces.append(rfaces)
            if gt_hand_type[1] > 0.5:
                verts.append(lverts)
                if gt_hand_type[0] > 0.5:
                    lfaces += len(mesh) // 2
                faces.append(lfaces)
            verts = np.concatenate(verts,0)
            faces = np.concatenate(faces,0)
            mesh = trimesh.Trimesh(verts, faces)
            mesh.export(os.path.join(self.debug_dir, 'data_debug{}.obj'.format(img_idx)))
            
            # vis pred mask
            save_img = save_img*mask[:, :, None].numpy()
            all_img2save = []
            font = cv2.FONT_HERSHEY_TRIPLEX
            for i in range(len(pred_mask)):
                pm = pred_mask[i].numpy()   # [H, W]
                img2save = (1-pm[:, :, None]) * save_img + pm[:, :, None]*pred_mano_render_ls[i]
                if cam_prob == i:
                    cv2.putText(img2save, 'Better_Cam' , (10, 20), font, 0.5, (255, 0, 0), 1)
                if mask_prob == i:
                    cv2.putText(img2save, 'Better_Mask' , (10, 35), font, 0.5, (0, 255, 0), 1)
                all_img2save.append(np.uint8(img2save))
            all_img2save.append(np.uint8(save_img))
            all_img2save = np.concatenate(all_img2save, 1)
            PIL.Image.fromarray(all_img2save).save(os.path.join(self.debug_dir, 'data_debug{}_mask.png'.format(img_idx)))
            #PIL.Image.fromarray(np.uint8((1-mask[:, :, None]) * save_img + 255*mask[:, :, None])).save(os.path.join(self.debug_dir, 'data_debug{}_mask_gt.png'.format(img_idx)))
            
            if cam_prob == mask_prob:
                self.valid += 1
        
        return inputs, preds, loss

    def __len__(self):
        return len(self.valid_idx_ls) * self.all_pair_num

    def __getitem__(self, index):
        idx = self.valid_idx_ls[index // self.all_pair_num]
        pid = index % self.all_pair_num

        if pid >= self.easy_pair_num or self.debug:
            ckid_ls = random.sample(self.hard_sample_ls, self.compare_num)
        else:
            ckid_ls = random.sample(self.easy_sample_ls, self.compare_num)
        # if self.mode == 'test' and (idx * self.pair_num + pid) % 50 != 0:
        #     continue
        # if self.mode == 'test':
        #     ckid_ls = sam_ls[idx * self.pair_num + pid].tolist()
        #     ckid_ls.reverse()
        # if pret_epoch_ls.index('epoch_-1') not in ckid_ls:
        #     if random.random() <= 0.5:
        #         ckid_ls[0] = pret_epoch_ls.index('epoch_-1')
        #     else:
        #         ckid_ls[1] = pret_epoch_ls.index('epoch_-1')
        # if self.debug:
        #     # ckid_ls = sam_ls[idx * self.pair_num + pid].tolist()
        #     ckid_ls[0] = self.pret_epoch_ls.index('epoch_19')
            
        ins_gts_id = ckid_ls[0]
        ins = self.ins_ls[ins_gts_id]
        gts = self.gts_ls[ins_gts_id]
        
        img_path = ins['img_path'][idx]
        center = ins['center'][idx]       # [2,]
        b_scale = ins['b_scale'][idx]
        focal_length = ins['focal_length'][idx]  # [2,]
        img_shape = ins['img_shape'][idx]        # [2,]
        bbox = ins['bbox'][idx]                  # [4,]
        do_flip = ins['do_flip'][idx] 
        scale = ins['scale'][idx]
        rot = ins['rot'][idx]
        color_scale = ins['color_scale'][idx]    # [3,]
        trans = ins['trans'][idx]                # [2, 3]
        root_valid = ins['root_valid'][idx]      # [1,]
        hand_type_valid = ins['hand_type_valid'][idx]  # [1,]
        joint_valid = ins['joint_valid'][idx]          # [42,]
        
        gt_joint_coord = gts['joint_coord'][idx]        # [42, 2]
        gt_rel_root_depth = gts['rel_root_depth'][idx]  # [1,]
        gt_hand_type = gts['hand_type'][idx]            # [2,]
        gt_mano_shape = gts['mano_shape'][idx]          # [2, 10]
        gt_mano_pose = gts['mano_pose'][idx]            # [2, 48]
        gt_mano_joint_cam = gts['mano_joint_cam'][idx]  # [42, 3]
        gt_cam_transl = gts['cam_transl'][idx]          # [2, 3]
        # gt_mano_mesh_proj = self.get_mano_mesh_proj(gt_mano_shape, gt_mano_pose, gt_cam_transl, focal_length, img_shape).cpu().numpy()   # [778*2, 2]

        pred_joint_coord = []        
        pred_rel_root_depth = []
        pred_hand_type = []
        pred_mano_shape = []
        pred_mano_pose = []
        pred_mano_joint_cam = []
        pred_cam_transl = []
        # pred_mano_mesh_proj = []
        #gt_cid_ls = [0 if random.random() <= 0.5 else 1]
        gt_cid_ls = []
        for i, ckid in enumerate(ckid_ls):
            preds = self.preds_ls[ckid]
            if i in gt_cid_ls:
                pred_joint_coord.append(gt_joint_coord.copy())
                pred_rel_root_depth.append(gt_rel_root_depth.copy())
                pred_hand_type.append(gt_hand_type.copy())
                pred_mano_shape.append(gt_mano_shape.copy())
                pred_mano_pose.append(gt_mano_pose.copy())
                pred_mano_joint_cam.append(gt_mano_joint_cam.copy())
                pred_cam_transl.append(gt_cam_transl.copy())
                # pred_mano_mesh_proj.append(gt_mano_mesh_proj.copy())
            else:
                pred_joint_coord.append(preds['joint_coord'][idx])        # [42, 2]
                pred_rel_root_depth.append(preds['rel_root_depth'][idx])  # [1,]
                pred_hand_type.append(preds['hand_type'][idx])            # [2,]
                pred_mano_shape.append(preds['mano_shape'][idx])          # [2, 10]
                pred_mano_pose.append(preds['mano_pose'][idx])            # [2, 48]
                pred_mano_joint_cam.append(preds['mano_joint_cam'][idx])  # [42, 3]
                pred_cam_transl.append(preds['cam_transl'][idx])          # [2, 3]
                # mano_mesh_proj= self.get_mano_mesh_proj(preds['mano_shape'][idx], preds['mano_pose'][idx], preds['cam_transl'][idx], focal_length, img_shape).cpu().numpy()   # [778*2, 2]
                # pred_mano_mesh_proj.append(mano_mesh_proj)  # [778*2, 2]
                
        # if self.debug:
        #     print('-----------')    
    
        in_data = {'img_path': img_path, 'center': center, 'b_scale': b_scale, 'focal_length': focal_length, 'img_shape': img_shape, \
            'bbox': bbox, 'do_flip': do_flip, 'scale': scale, 'rot': rot, 'color_scale': color_scale, 'trans': trans,\
            'root_valid': root_valid, 'hand_type_valid': hand_type_valid, 'joint_valid': joint_valid, 'pid': pid}
        pred_data = {
            'pred_joint_coord': np.stack(pred_joint_coord, axis=0),             # [compare_num, 42, 2]
            'pred_rel_root_depth': np.stack(pred_rel_root_depth, axis=0),       # [compare_num, 1]
            'pred_hand_type': np.stack(pred_hand_type, axis=0),                 # [compare_num, 2]
            'pred_mano_shape': np.stack(pred_mano_shape, axis=0),               # [compare_num, 2, 10]
            'pred_mano_pose': np.stack(pred_mano_pose, axis=0),                 # [compare_num, 2, 48]
            'pred_mano_joint_cam': np.stack(pred_mano_joint_cam, axis=0),       # [compare_num, 42, 3]
            'pred_cam_transl': np.stack(pred_cam_transl, axis=0),               # [compare_num, 2, 3]
            # 'pred_mano_mesh_proj': np.stack(pred_mano_mesh_proj, axis=0)        # [compare_num, 778*2, 2]
        }
        gt_data = {
            'gt_joint_coord': gt_joint_coord,
            'gt_rel_root_depth': gt_rel_root_depth,
            'gt_hand_type': gt_hand_type,
            'gt_mano_shape': gt_mano_shape,
            'gt_mano_pose': gt_mano_pose,
            'gt_mano_joint_cam': gt_mano_joint_cam,
            'gt_cam_transl': gt_cam_transl,
            # 'gt_mano_mesh_proj': gt_mano_mesh_proj
        }
        
        if pid >= self.easy_pair_num or self.debug:
            #rand_cid_ls = [0 if random.random() <= 0.5 else 1]
            rand_cid_ls = list(range(self.compare_num))
            in_data['rand_cid'] = rand_cid_ls
            for k, v in pred_data.items():
                if k in ['pred_mano_shape', 'pred_mano_pose', 'pred_cam_transl']:
                    for rand_cid in rand_cid_ls:
                        ori_v = v[rand_cid].copy()
                        gt_v = gt_data['gt_'+k[5:]].copy()
                        #pred_data[k][rand_cid] = ori_v * ((np.random.random(ori_v.shape) * 2 - 1) * 0.1 + 1)
                        pred_data[k][rand_cid] = ori_v + (gt_v - ori_v) * np.random.random(ori_v.shape)
            # in_data['rand_cid'] = self.compare_num - 1 - cid
        
        inputs, preds, loss = self.get_gt_loss(idx, in_data, pred_data, gt_data)
        return inputs, preds, loss

    def eval_prob(self, pred_probs, gt_loss, logger):
        eval_type = "Rank Accuracy"
        one_err = []
        zero_err = []
        half_err = []
        tot_err = []
        sample_num = len(pred_probs)
        for n in tqdm(range(sample_num)):
            
            target_prob = gt_loss['target_prob'][n]
            pred_prob = pred_probs[n]
            # pred_prob = random.randint(0, 2) * 0.5
            tot_err.append((target_prob == pred_prob).astype(float))
            if target_prob == 1.0:
                one_err.append((target_prob == pred_prob).astype(float))
            elif target_prob == 0.0:
                zero_err.append((target_prob == pred_prob).astype(float))
            # elif target_prob == 0.5:
            #     half_err.append((target_prob == pred_prob).astype(float))

        print(eval_type)
        print('%s for all labels: %d / %d = %.2f' % (eval_type, np.sum(tot_err), len(tot_err), np.mean(tot_err)))
        print('%s for one labels: %d / %d = %.2f' % (eval_type, np.sum(one_err), len(one_err), np.mean(one_err)))
        print('%s for zero labels: %d / %d = %.2f' % (eval_type, np.sum(zero_err), len(zero_err), np.mean(zero_err)))
        # print('%s for half labels: %d / %d = %.2f' % (eval_type, np.sum(half_err), len(half_err), np.mean(half_err)))
        logger.info(eval_type)
        logger.info('%s for all labels: %d / %d = %.2f' % (eval_type, np.sum(tot_err), len(tot_err), np.mean(tot_err)))
        logger.info('%s for one labels: %d / %d = %.2f' % (eval_type, np.sum(one_err), len(one_err), np.mean(one_err)))
        logger.info('%s for zero labels: %d / %d = %.2f' % (eval_type, np.sum(zero_err), len(zero_err), np.mean(zero_err)))
        # logger.info('%s for half labels: %d / %d = %.2f' % (eval_type, np.sum(half_err), len(half_err), np.mean(half_err)))

        # eval_summary = '{} MPJPE for each joint: \n'.format(eval_type)
        # for j in range(self.joint_num * 2):
        #     mpjpe_sh[j] = np.mean(np.stack(mpjpe_sh[j]))
        #     joint_name = self.skeleton[j]['name']
        #     eval_summary += (joint_name + ': %.2f, ' % mpjpe_sh[j])
        # print(eval_summary)
        # print('%s MPJPE for single hand sequences: %.2f' % (eval_type, np.mean(mpjpe_sh)))
        # print()
        # logger.info(eval_summary)
        # logger.info('%s MPJPE for single hand sequences: %.2f' % (eval_type, np.mean(mpjpe_sh)))

        # eval_summary = '{} MPJPE for each joint: \n'.format(eval_type)
        # for j in range(self.joint_num * 2):
        #     mpjpe_ih[j] = np.mean(np.stack(mpjpe_ih[j]))
        #     joint_name = self.skeleton[j]['name']
        #     eval_summary += (joint_name + ': %.2f, ' % mpjpe_ih[j])
        # print(eval_summary)
        # print('%s MPJPE for interacting hand sequences: %.2f' % (eval_type, np.mean(mpjpe_ih)))
        # logger.info(eval_summary)
        # logger.info('%s MPJPE for interacting hand sequences: %.2f' % (eval_type, np.mean(mpjpe_ih)))

def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0] + 1) + '/' + str(f[i][0] + 1) + ' ' + str(f[i][1] + 1) + '/' + str(
            f[i][1] + 1) + ' ' + str(f[i][2] + 1) + '/' + str(f[i][2] + 1) + '\n')
    obj_file.close()
    
if __name__ == "__main__":
    import torchvision.transforms as transforms

    datapath = '/data1/linxiaojian/InterHand2.6M-main/cliff10_InterHand2.6M_0.1/result'
    testset_loader = RM_Dataset(transforms.ToTensor(), datapath, 'train', debug=True)
    count = 0
    for idx in range(0, len(testset_loader), testset_loader.all_pair_num):
        item = testset_loader.__getitem__(idx)
        count += 1
    print('%d / %d : %f'%(testset_loader.valid, count,\
        testset_loader.valid / count))