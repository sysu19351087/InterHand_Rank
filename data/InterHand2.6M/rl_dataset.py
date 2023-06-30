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
import torch
import torch.utils.data
import cv2
from glob import glob
import os.path as osp
from config import cfg
from utils.preprocessing import load_img, load_skeleton, get_bbox, process_bbox, augmentation, \
    transform_input_to_output_space, trans_point2d, estimate_focal_length, gen_ori2rotori_trans, estimate_translation
from utils.transforms import world2cam, cam2pixel, pixel2cam, perspective
from utils.vis import vis_keypoints, vis_3d_keypoints
from PIL import Image, ImageDraw
import random
import json
import math
from pycocotools.coco import COCO
import scipy.io as sio

class RL_Dataset(torch.utils.data.Dataset):
    def __init__(self, transform, mode, debug=False):
        self.mode = mode  # train, test, val
        current_path = os.path.abspath(__file__)
        self.img_path = os.path.abspath(current_path + '/../images')
        self.annot_path = os.path.abspath(current_path + '/../annotations')
        if self.mode == 'val':
            self.rootnet_output_path = os.path.abspath(current_path + '/../rootnet_output/rootnet_interhand2.6m_output_val.json')
        else:
            self.rootnet_output_path = os.path.abspath(current_path + '/../rootnet_output/rootnet_interhand2.6m_output_test.json')
        self.transform = transform
        self.joint_num = 21  # single hand
        self.vert_num = 778
        self.root_joint_idx = {'right': 20, 'left': 41}
        #print(os.path.abspath(current_path + '/../../..'))
        self.joint_regressor = np.load(os.path.abspath(current_path + '/../../../smplx/models/mano/J_regressor_mano_ih26m.npy'))
        self.joint_type = {'right': np.arange(0, self.joint_num), 'left': np.arange(self.joint_num, self.joint_num * 2)}
        self.skeleton = load_skeleton(osp.join(self.annot_path, 'skeleton.txt'), self.joint_num * 2)

        self.datalist = []
        self.datalist_sh = []
        self.datalist_ih = []
        self.sequence_names = []
        
        self.debug = debug
        if self.debug:
            self.debug_dir = "debug_dataset"
            os.makedirs(self.debug_dir, exist_ok=True)

        # mano layer
        smplx_path = os.path.abspath(current_path + '/../../../smplx/models')
        self.mano_layer = {'right': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=True, create_transl=False),
                           'left': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=False, create_transl=False)}
        # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
        if torch.sum(torch.abs(
                self.mano_layer['left'].shapedirs[:, 0, :] - self.mano_layer['right'].shapedirs[:, 0, :])) < 1:
            self.mano_layer['left'].shapedirs[:, 0, :] *= -1
        
        # load annotation
        print("Load annotation from  " + osp.join(self.annot_path, self.mode))
        db = COCO(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_data.json'))
        print("Number of images {}".format(len(db.anns.keys())))
        
        with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_camera.json')) as f:
            cameras = json.load(f)
        with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_joint_3d.json')) as f:
            joints = json.load(f)
        with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_MANO_NeuralAnnot.json')) as f:
            mano_params = json.load(f)

        if (self.mode == 'val' or self.mode == 'test') and cfg.trans_test == 'rootnet':
            print("Get bbox and root depth from " + self.rootnet_output_path)
            rootnet_result = {}
            with open(self.rootnet_output_path) as f:
                annot = json.load(f)
            for i in range(len(annot)):
                rootnet_result[str(annot[i]['annot_id'])] = annot[i]
        else:
            print("Get bbox and root depth from groundtruth annotation")
              
        self.train_ratio = cfg.train_ratio
        train_select_step = int(1 / self.train_ratio) if not self.debug else 1000
        count = -1
        debug_type = "interacting"
        #debug_type = 'right'
        for aid in db.anns.keys():
            if self.debug and debug_type is not None:
                hand_type = db.anns[aid]['hand_type'].lower()
                if hand_type != debug_type:
                    continue
            count += 1
            if self.mode == 'train' and (count + 5) % train_select_step != 0:
                continue
            
            ann = db.anns[aid]
            if not ann['hand_type_valid']:
                continue
            
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_width, img_height = img['width'], img['height']

            capture_id = img['capture']
            seq_name = img['seq_name']
            cam = img['camera']
            frame_idx = img['frame_idx']
            img_path = osp.join(self.img_path, self.mode, img['file_name'])

            campos, camrot = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32), np.array(
                cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
            focal, princpt = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32), np.array(
                cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)
            joint_world = np.array(joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32)
            joint_cam = world2cam(joint_world.transpose(1, 0), camrot, campos.reshape(3, 1)).transpose(1, 0)  # [J, 3]
            joint_img = cam2pixel(joint_cam, focal, princpt)[:, :2]   # [J, 2]

            joint_valid = np.array(ann['joint_valid'], dtype=np.float32).reshape(self.joint_num * 2)
            # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
            joint_valid[self.joint_type['right']] *= joint_valid[self.root_joint_idx['right']]
            joint_valid[self.joint_type['left']] *= joint_valid[self.root_joint_idx['left']]
            hand_type = ann['hand_type']
            hand_type_valid = np.array((ann['hand_type_valid']), dtype=np.float32)

            if (self.mode == 'val' or self.mode == 'test') and cfg.trans_test == 'rootnet':
                bbox = np.array(rootnet_result[str(aid)]['bbox'], dtype=np.float32)
                abs_depth = {'right': rootnet_result[str(aid)]['abs_depth'][0],
                             'left': rootnet_result[str(aid)]['abs_depth'][1]}
            else:
                img_width, img_height = img['width'], img['height']
                bbox = np.array(ann['bbox'], dtype=np.float32)  # x,y,w,h
                bbox = process_bbox(bbox, (img_height, img_width))
                abs_depth = {'right': joint_cam[self.root_joint_idx['right'], 2],
                             'left': joint_cam[self.root_joint_idx['left'], 2]}

            cam_param = {'focal': focal, 'princpt': princpt, 'campos': campos, 'camrot': camrot}
            joint = {'cam_coord': joint_cam, 'img_coord': joint_img, 'valid': joint_valid}
            
            try:
                mano_param = mano_params[str(capture_id)][str(frame_idx)]
            except:
                # print("missing mano {} {}".format(str(capture_id), str(frame_idx)))
                continue
            
            if hand_type == 'right' and mano_param['right'] is None:
                continue
            elif hand_type == 'left' and mano_param['left'] is None:
                continue
            elif hand_type == 'interacting' and (mano_param['right'] is None or mano_param['left'] is None): 
                continue
            
            data = {'img_path': img_path, 'seq_name': seq_name, 'cam_param': cam_param, 'mano_param': mano_param, 
                    'bbox': bbox, 'joint': joint, 'img_shape': [img_height, img_width],
                    'hand_type': hand_type, 'hand_type_valid': hand_type_valid, 'abs_depth': abs_depth,
                    'file_name': img['file_name'], 'capture': capture_id, 'cam': cam, 'frame': frame_idx}
            if hand_type == 'right' or hand_type == 'left':
                self.datalist_sh.append(data)
            else:
                self.datalist_ih.append(data)
            if seq_name not in self.sequence_names:
                self.sequence_names.append(seq_name)
            if self.debug:
                if len(self.datalist_sh) + len(self.datalist_ih) >= 10:
                    break

        self.datalist = self.datalist_sh + self.datalist_ih
        print('Number of annotations in single hand sequences: ' + str(len(self.datalist_sh)))
        print('Number of annotations in interacting hand sequences: ' + str(len(self.datalist_ih)))

    def handtype_str2array(self, hand_type):
        if hand_type == 'right':
            return np.array([1, 0], dtype=np.float32)
        elif hand_type == 'left':
            return np.array([0, 1], dtype=np.float32)
        elif hand_type == 'interacting':
            return np.array([1, 1], dtype=np.float32)
        else:
            assert 0, print('Not supported hand type: ' + hand_type)

    def __len__(self):
        return len(self.datalist)
        
    def get_mano_coord(self, mano_param, cam_param, do_flip, hand_type, merge_cam_rot=True, root_trans=None):

        mano_pose = torch.FloatTensor(mano_param['pose']).view(-1, 3)
        R, t = np.array(cam_param['camrot'], dtype=np.float32).reshape(3, 3), np.array(cam_param['campos'],
                                                                                       dtype=np.float32).reshape(
            3)  # camera rotation and translation
        t = -np.dot(R, t.reshape(3, 1)).reshape(3)  # -Rt -> t
        if merge_cam_rot:
            # merge root pose and camera rotation
            root_pose = mano_pose[0].view(1, 3).numpy()
            root_pose, _ = cv2.Rodrigues(root_pose)
            root_pose, _ = cv2.Rodrigues(np.dot(R, root_pose))
            mano_pose[0] = torch.from_numpy(root_pose).view(3)

        # get mesh and joint coordinates
        root_pose = mano_pose[0].view(1, 3)
        hand_pose = mano_pose[1:, :].view(1, -1)
        mano_shape = torch.FloatTensor(mano_param['shape']).view(1, -1)

        if root_trans is None:   # without augmentation
            trans = torch.FloatTensor(mano_param['trans']).view(1, -1)
            output = self.mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=mano_shape,
                                                transl=trans)
            #print('xrange:', max(output.vertices[0].numpy()[:, 0]) - min(output.vertices[0].numpy()[:, 0]))
            #print('yrange:', max(output.vertices[0].numpy()[:, 1]) - min(output.vertices[0].numpy()[:, 1]))
            mesh_cam = output.vertices[0].numpy() * 1000
            joint_cam = np.dot(self.joint_regressor, mesh_cam)
            root_joint = joint_cam[20, None, :]
            rootj_CR = np.dot(R, root_joint.reshape(3, 1)).reshape(1, 3)
            mesh_cam = mesh_cam - root_joint + rootj_CR + t
            joint_cam = joint_cam - root_joint + rootj_CR + t
            if merge_cam_rot and do_flip:
                #print('????')
                mesh_cam[:, 0] = -mesh_cam[:, 0]
                joint_cam[:, 0] = -joint_cam[:, 0]


        else:
            output = self.mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=mano_shape)
            mesh_cam = output.vertices[0].numpy() * 1000
            joint_cam = np.dot(self.joint_regressor, mesh_cam)
            root_joint = joint_cam[20, None, :]
            mesh_cam = mesh_cam - root_joint + root_trans
            joint_cam = joint_cam - root_joint + root_trans

        mano_pose = mano_pose.view(1, -1)
        return mesh_cam, joint_cam, mano_pose, mano_shape

    def __getitem__(self, idx):
        data = self.datalist[idx]
        img_path, bbox, img_shape, joint, hand_type, hand_type_valid = data['img_path'], data['bbox'], data['img_shape'], data['joint'], data[
            'hand_type'], data['hand_type_valid']
        joint_cam = joint['cam_coord'].copy()     # [J, 3]
        joint_img = joint['img_coord'].copy()     # [J, 2]
        joint_valid = joint['valid'].copy()       # [J,]
        hand_type = self.handtype_str2array(hand_type)
        joint_coord = np.concatenate((joint_img, joint_cam[:, 2, None].copy()), 1)    # [J, 3]
        joint_coord_full = joint_coord.copy()

        # image load
        img = load_img(img_path)
        img_ = img.copy()
        # augmentation
        img, joint_coord, joint_valid, hand_type, trans, inv_trans, center, b_scale, rot, do_flip, scale, color_scale = augmentation(img, bbox, joint_coord, joint_valid,
                                                                           hand_type, self.mode, self.joint_type)
        
        if do_flip:
            img_ = img_[:, ::-1, :]
            joint_coord_full[:, 0] = img_shape[1] - joint_coord_full[:, 0] - 1
            joint_coord_full[self.joint_type['right']], joint_coord_full[self.joint_type['left']] = joint_coord_full[self.joint_type['left']].copy(), \
                                                                                joint_coord_full[self.joint_type['right']].copy()
        
        # rot orig_img and joint_coord_full
        trans_ = gen_ori2rotori_trans(center[0], center[1], bbox[2], bbox[3], scale, rot, inv=False)
        img_ = cv2.warpAffine(img_, trans_, (int(img_shape[1]), int(img_shape[0])), flags=cv2.INTER_LINEAR)
        for i in range(len(joint_coord_full)):
            joint_coord_full[i, :2] = trans_point2d(joint_coord_full[i, :2], trans_)
                                                                           
        if self.debug:
            print('rot:', rot, 'scale:', scale)
            save_img = img.copy()
            save_img = img_.copy()
        #print(img_path)
        
        "augment joint_cams"
        joint_cams = joint['cam_coord'].copy()
        joint_cams[:self.joint_num, :] = joint_cams[:self.joint_num, :] - joint_cams[self.joint_num -1:self.joint_num, :]
        joint_cams[self.joint_num:, :] = joint_cams[self.joint_num:, :] - joint_cams[-1:, :]
        if do_flip:
            joint_cams[:, 0] = -joint_cams[:, 0]
            joint_cams[self.joint_type['right']], joint_cams[self.joint_type['left']] = joint_cams[self.joint_type['left']].copy(), \
                                                                                joint_cams[self.joint_type['right']].copy()

        # 3D data rotation augmentation
        rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                                [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                                [0, 0, 1]], dtype=np.float32)

        joint_cams = np.dot(rot_aug_mat, joint_cams.transpose(1, 0)).transpose(1, 0)
        joint_cams[:self.joint_num, :] = joint_cams[:self.joint_num, :] - joint_cams[self.joint_num -1:self.joint_num, :]
        joint_cams[self.joint_num:, :] = joint_cams[self.joint_num:, :] - joint_cams[-1:, :]

        # get mano param
        mano_params = data['mano_param']
        mano_mesh_cams = []
        mano_mesh_imgs = []
        mano_mesh_valids = []
        mano_joint_cams = []
        mano_joint_imgs = []
        mano_joint_valids = []
        mano_poses = []
        mano_shapes = []
        if self.debug:
            mano_trans = []
            mano_mesh_reprojs = []
            mano_recat_imgs = []
            
        # get camera parameters
        cam_param = data["cam_param"]
        focal, princpt = cam_param['focal'], cam_param['princpt']
        if do_flip:
            princpt[0] = img_shape[1] - princpt[0] - 1
        princpt_aug = trans_point2d(princpt, trans)
        focal_aug = [
            focal[0] * cfg.input_img_shape[1] / (scale * bbox[2]),
            focal[1] * cfg.input_img_shape[0] / (scale * bbox[3]),
        ]
        calib = np.eye(4)
        calib[0, 0] = focal_aug[0]
        calib[1, 1] = focal_aug[1]
        calib[:2, 2:3] = princpt_aug[:, None]
        mano_calib = torch.from_numpy(calib).float()

        for ht in ["right", "left"]:
            mano_param = mano_params[ht]
            if mano_param is None:
                mano_mesh_cams.append(torch.zeros((self.vert_num, 3)))
                mano_mesh_imgs.append(torch.zeros((self.vert_num, 3)))
                mano_mesh_valids.append(torch.zeros((self.vert_num, 1)))
                mano_joint_cams.append(torch.zeros((self.joint_num, 3)))
                mano_joint_imgs.append(torch.zeros((self.joint_num, 3)))
                mano_joint_valids.append(torch.zeros((self.joint_num, 1)))
                mano_poses.append(torch.zeros((1, 48)))
                mano_shapes.append(torch.zeros((1, 10)))
                if self.debug:
                    mano_trans.append(torch.zeros((1, 3)))
                    mano_mesh_reprojs.append(torch.zeros((self.vert_num, 3)))
                    mano_recat_imgs.append(torch.zeros((self.joint_num, 3)))
                continue

            with torch.no_grad():
                mesh_cam, mano_joint_cam, mano_pose, mano_shape = self.get_mano_coord(mano_param, cam_param, do_flip, ht)

            "augmentation"
            cat_cam = np.concatenate([mesh_cam, mano_joint_cam], axis=0)


            # coordinate rotation
            cat_cam = np.dot(rot_aug_mat, cat_cam.transpose(1, 0)).transpose(1, 0)

            # rotate root pose
            mano_pose = mano_pose.view(-1, 3)
            if do_flip:
                mano_pose[:, 1:3] *= -1     # from pose2pose, but why y and z?
            root_pose = mano_pose[0].numpy().reshape(1, 3)
            root_pose, _ = cv2.Rodrigues(root_pose)
            root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat, root_pose))
            mano_pose[0, :] = torch.from_numpy(root_pose).reshape(3)
            mano_pose = mano_pose.view(1, -1)

            # perspective projection
            cat_img = torch.from_numpy(cat_cam.copy()).float().permute(1, 0).unsqueeze(0)
            cat_img = perspective(cat_img, mano_calib.unsqueeze(0))
            cat_img = cat_img.permute(0, 2, 1).squeeze(0).numpy()
            cat_img[:, 2] = ((cat_cam[:, 2].copy() - cat_cam[-1, 2].copy()) / (cfg.bbox_3d_size / 2) + 1) / 2. * cfg.output_hm_shape[
                0]  # hand depth discretize following the transform_input_to_output_space on interhand2.6m
            cat_img[:, 0] = cat_img[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
            cat_img[:, 1] = cat_img[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]

            # check truncation
            cat_valid = ((cat_img[:, 0] >= 0) * (cat_img[:, 0] < cfg.output_hm_shape[2]) * \
                         (cat_img[:, 1] >= 0) * (cat_img[:, 1] < cfg.output_hm_shape[1]) * \
                         (cat_img[:, 2] >= 0) * (cat_img[:, 2] < cfg.output_hm_shape[0])).reshape(-1, 1).astype(
                np.float32)

            # gather outputs
            mano_mesh_cams.append(torch.from_numpy(cat_cam[:self.vert_num]))
            mano_mesh_imgs.append(torch.from_numpy(cat_img[:self.vert_num]))
            mano_mesh_valids.append(torch.from_numpy(cat_valid[:self.vert_num]))
            mano_joint_cams.append(torch.from_numpy(cat_cam[self.vert_num:]))
            mano_joint_imgs.append(torch.from_numpy(cat_img[self.vert_num:]))
            mano_joint_valids.append(torch.from_numpy(cat_valid[self.vert_num:]))
            mano_poses.append(mano_pose)
            mano_shapes.append(mano_shape)

            with torch.no_grad():
                aug_mano_param = {
                    "pose": mano_pose,
                    "shape": mano_param["shape"],
                    "trans": mano_param["trans"]
                }
                if do_flip:
                    if ht == "right":
                        aug_ht = "left"
                    else:
                        aug_ht = "right"
                else:
                    aug_ht = ht

                if self.debug:
                    #print('!!!!')
                    mano_transl = cat_cam[-1].reshape(1, 3)
                    re_cam, re_joint_cam, _, _ = self.get_mano_coord(aug_mano_param, cam_param, do_flip, aug_ht,
                                                          merge_cam_rot=False, root_trans=mano_transl)
                    save_obj(re_cam, self.mano_layer[aug_ht].faces,
                                 osp.join(self.debug_dir, 'hand_{}_{}_mano'.format(idx, aug_ht) + '.obj'))
                    save_obj(cat_cam[:self.vert_num], self.mano_layer[aug_ht].faces,
                                 osp.join(self.debug_dir, 'hand_{}_{}_cam'.format(idx, aug_ht) + '.obj'))
                    mano_mesh_reprojs.append(torch.from_numpy(re_cam))
                    mano_trans.append(torch.from_numpy(mano_transl))
                    print(re_joint_cam[0])
                    print(cat_cam[self.vert_num:][0])
                    joint_c = joint_cams[self.joint_type[aug_ht]].copy()
                    joint_c -= joint_c[-1:, :]
                    joint_c += mano_transl
                    print(joint_c[0])
                    recat_img = torch.from_numpy(re_joint_cam - re_joint_cam[-1, None, :]).float()
                    # calib_ = np.eye(4)
                    # calib_[0, 0] = focal[0]
                    # calib_[1, 1] = focal[1]
                    # princpt_aug_ = trans_point2d(princpt, trans_)
                    # calib_[:2, 2:3] = princpt_aug_[:, None]
                    princpt_ = np.array([img_shape[1] / 2, img_shape[0] / 2]).astype(float)
                    focal_ = estimate_focal_length(img_shape[0], img_shape[1])
                    focal_ = np.array([focal_, focal_]).astype(float)
                    joint_coord_full_ = joint_coord_full[None, self.joint_type[aug_ht], :2].copy()
                    gt_keypoints_2d_orig = torch.cat([torch.from_numpy(joint_coord_full_), torch.ones(1, re_joint_cam.shape[0], 1)], -1) # [1, 21, 3]
                    gt_cam_t = estimate_translation(recat_img.unsqueeze(0).clone(), gt_keypoints_2d_orig, focal_length=focal_,
                                        img_size=princpt_)  # [1, 3]
                    #print('gt_cam_t:', gt_cam_t)
                    recat_img += gt_cam_t
                    recat_img = recat_img.permute(1, 0).unsqueeze(0)
                    calib_ = np.eye(4)
                    calib_[0, 0] = focal_[0]
                    calib_[1, 1] = focal_[1]
                    calib_[:2, 2:3] = princpt_[:, None]
                    mano_calib_ = torch.from_numpy(calib_).float()
                    recat_img = perspective(recat_img, mano_calib_.unsqueeze(0))
                    recat_img = recat_img.permute(0, 2, 1).squeeze(0).numpy()
                    mano_recat_imgs.append(torch.from_numpy(recat_img))


        if do_flip:
            lidx, ridx = 0, 1
        else:
            lidx, ridx = 1, 0

        mano_mesh_cams = torch.cat([mano_mesh_cams[ridx].clone(), mano_mesh_cams[lidx].clone()], dim=0).float()
        mano_mesh_imgs = torch.cat([mano_mesh_imgs[ridx].clone(), mano_mesh_imgs[lidx].clone()], dim=0).float()
        mano_mesh_valids = torch.cat([mano_mesh_valids[ridx].clone(), mano_mesh_valids[lidx].clone()], dim=0).float()
        mano_joint_cams = torch.cat([mano_joint_cams[ridx].clone(), mano_joint_cams[lidx].clone()], dim=0).float()
        mano_joint_imgs = torch.cat([mano_joint_imgs[ridx].clone(), mano_joint_imgs[lidx].clone()], dim=0).float()
        mano_joint_valids = torch.cat([mano_joint_valids[ridx].clone(), mano_joint_valids[lidx].clone()], dim=0).float()
        mano_poses = torch.cat([mano_poses[ridx].clone(), mano_poses[lidx].clone()], dim=0).float()
        mano_shapes = torch.cat([mano_shapes[ridx].clone(), mano_shapes[lidx].clone()], dim=0).float()
        if self.debug:
            mano_trans = torch.cat([mano_trans[ridx].clone(), mano_trans[lidx].clone()], dim=0).float()
            mano_mesh_reprojs = torch.cat([mano_mesh_reprojs[ridx].clone(), mano_mesh_reprojs[lidx].clone()], dim=0).float()
            mano_recat_img = torch.cat([mano_recat_imgs[ridx].clone(), mano_recat_imgs[lidx].clone()], dim=0).float()
        
        rel_root_depth = np.array(
            [joint_coord[self.root_joint_idx['left'], 2] - joint_coord[self.root_joint_idx['right'], 2]],
            dtype=np.float32).reshape(1)
        root_valid = np.array([joint_valid[self.root_joint_idx['right']] * joint_valid[self.root_joint_idx['left']]],
                              dtype=np.float32).reshape(1) if hand_type[0] * hand_type[1] == 1 else np.zeros((1),
                                                                                                             dtype=np.float32)
        # transform to output heatmap space
        
        joint_coord, joint_valid, rel_root_depth, root_valid = transform_input_to_output_space(joint_coord, joint_valid,
                                                                                               rel_root_depth,
                                                                                               root_valid,
                                                                                               self.root_joint_idx,
                                                                                               self.joint_type)
        
        img = self.transform(img.astype(np.float32)) / 255.
        joint_coord = torch.from_numpy(joint_coord).float()
        joint_cams = torch.from_numpy(joint_cams).float()
        
        if not cfg.crop2full:
            img_shape = cfg.input_img_shape
        img_shape = np.array(img_shape).astype(float)
        if cfg.crop2full:    
            focal_length = estimate_focal_length(img_shape[0], img_shape[1])
            focal_length = np.array([focal_length, focal_length]).astype(float)
        else:
            focal_length = np.array([cfg.focal_length, cfg.focal_length]).astype(float)

        mano_joint_cams[:self.joint_num, :] = mano_joint_cams[:self.joint_num, :] - mano_joint_cams[self.joint_num -1:self.joint_num, :]
        mano_joint_cams[self.joint_num:, :] = mano_joint_cams[self.joint_num:, :] - mano_joint_cams[-1:, :]

        vir_princpt = np.array([img_shape[1] / 2, img_shape[0] / 2]).astype(float)  # [2,]
        vir_focal = focal_length.copy()  # [2,]
        vir_calib = np.eye(4)
        vir_calib[0, 0] = vir_focal[0]
        vir_calib[1, 1] = vir_focal[1]
        vir_calib[:2, 2:3] = vir_princpt[:, None]
        vir_calib = torch.from_numpy(vir_calib).float()
        gt_cam_transl = torch.zeros([2, 3])
        gt_cam_param = torch.zeros([2, 3])
        if self.debug:
            test_joint_proj = np.zeros([self.joint_num*2, 2])
        for i, t in enumerate(["right", "left"]):
            if hand_type[i] == 0:
                continue
            j_2d_full = joint_coord_full[None, self.joint_type[t], :2].copy()   # [1, 21, 2]
            gt_keypoints_2d_orig = torch.cat([torch.from_numpy(j_2d_full), torch.ones(1, self.joint_num, 1)], -1) # [1, 21, 3]
            j_3d = joint_cams[None, self.joint_type[t], :].clone()   # [1, 21, 3]
            #print(j_3d)
            gt_cam_t = estimate_translation(j_3d, gt_keypoints_2d_orig, focal_length=vir_focal, img_size=vir_princpt)  # [1, 3]
            gt_cam_transl[i] = gt_cam_t[0].clone()
            gt_pred_cam_s = cfg.mano_size * focal_length[0] /  ((gt_cam_t[0, 2] / 1000) * b_scale + 1e-9)
            gt_pred_cam_x = gt_cam_t[0, 0] / 1000 - (cfg.mano_size * (center[0] - img_shape[1] / 2) / (gt_pred_cam_s * b_scale))
            gt_pred_cam_y = gt_cam_t[0, 1] / 1000 - (cfg.mano_size * (center[1] - img_shape[0] / 2) / (gt_pred_cam_s * b_scale))
            gt_cam_param[i] = torch.tensor([gt_pred_cam_s, gt_pred_cam_x, gt_pred_cam_y]).float()
            if self.debug:
                j_3d = mano_joint_cams[self.joint_type[t], :].clone() + gt_cam_t   # [21, 3]
                # j_2d_full = j_3d.permute(1, 0).unsqueeze(0)
                # j_2d_full = perspective(j_2d_full, vir_calib.unsqueeze(0))
                # j_2d_full = j_2d_full.permute(0, 2, 1).squeeze(0).numpy()[:, :2]
                x = j_3d[:, 0].clone() / (j_3d[:, 2].clone() + 1e-8) * \
                    torch.from_numpy(vir_focal[None, 0]) + torch.from_numpy(vir_princpt[None, 0])
                y = j_3d[:, 1].clone() / (j_3d[:, 2].clone() + 1e-8) * \
                    torch.from_numpy(vir_focal[None, 1]) + torch.from_numpy(vir_princpt[None, 1])
                j_2d_full = torch.stack((x, y), 1)
                
                test_joint_proj[self.joint_type[t], :] = j_2d_full.numpy()
                print('gt_camt_x:', gt_cam_t[0, 0]/1000)
                print('gt_camt_y:', gt_cam_t[0, 1]/1000)
                print('gt_camt_z:', gt_cam_t[0, 2]/1000)
                #gt_pred_cam_s = cfg.mano_size * focal_length[0] /  ((gt_cam_t[0, 2] / 1000) * b_scale + 1e-9)
                #gt_pred_cam_x = gt_cam_t[0, 0] / 1000 - (cfg.mano_size * (center[0] - img_shape[1] / 2) / (gt_pred_cam_s * b_scale))
                #gt_pred_cam_y = gt_cam_t[0, 1] / 1000 - (cfg.mano_size * (center[1] - img_shape[0] / 2) / (gt_pred_cam_s * b_scale))
                print('gt_pred_cam_s:', gt_pred_cam_s)
                print('gt_pred_cam_x:', gt_pred_cam_x)
                print('gt_pred_cam_y:', gt_pred_cam_y)
                
        gt_cam_transl /= 1000  # [2, 3], mm -> m
        #print('gt_cam_transl:', gt_cam_transl)
        if cfg.crop2full:
            for i in range(self.joint_num*2):
                joint_valid[i] = joint['valid'].copy()[i] * (joint_coord_full[i, 0] >= 0) * (joint_coord_full[i, 0] < img_shape[1]) * (
                            joint_coord_full[i, 1] >= 0) * (joint_coord_full[i, 1] < img_shape[0])
            
        inputs = {'img': img, 'center': center, 'b_scale': b_scale, 'focal_length': focal_length, 'img_shape': img_shape}
        targets = {'joint_coord': joint_coord,   # hm space, [42, 3], no base root
                   'joint_coord_full': joint_coord_full,   # full_img space, [42, 3], no base root
                   'joint_cam': joint_cams,      # cam space, [42, 3], base root
                   'rel_root_depth': rel_root_depth, 'hand_type': hand_type,
                   'mano_mesh_cams': mano_mesh_cams, 'mano_mesh_imgs': mano_mesh_imgs,
                   'mano_joint_cams': mano_joint_cams, 'mano_joint_imgs': mano_joint_imgs,
                   'mano_poses': mano_poses, 'mano_shapes': mano_shapes, 'cam_transl': gt_cam_transl,
                   'cam_param': gt_cam_param}
        meta_info = {'joint_valid': joint_valid, 'root_valid': root_valid, 'hand_type_valid': hand_type_valid,
                     'trans': trans, 'inv_trans': inv_trans, 'capture': int(data['capture']), 'cam': int(data['cam']),
                     'frame': int(data['frame']), 'do_flip': int(do_flip), 'color_scale': color_scale, 'idx': idx}
        if self.debug:
            print("{} flip {}".format(idx, do_flip))
            targets['mano_mesh_reprojs'] = mano_mesh_reprojs
            targets['mano_trans'] = mano_trans
            torch.save([inputs, targets, meta_info], os.path.join(self.debug_dir, 'data_{}.pth'.format(idx)))
            for jidx, [jv, jproj] in enumerate(zip(mano_joint_valids, test_joint_proj)):
                # if jv == 1:
                #print(jproj)
                #print(mano_joint_imgs[jidx])
                #show_scaley, show_scalex = cfg.input_img_shape[0] / cfg.output_hm_shape[1], cfg.input_img_shape[1] / cfg.output_hm_shape[2]
                #jproj[0] *= show_scalex
                #jproj[1] *= show_scaley
                #jproj[:2] = torch.tensor(trans_point2d(jproj[:2], inv_trans)).float()
                #jproj[:2] = torch.tensor(trans_point2d(jproj[:2], inv_trans)).float()
                jx, jy = int(jproj[0]), int(jproj[1])
                if jx < 0:
                    jx = 0
                if jx >= save_img.shape[1]:
                    jx = save_img.shape[1] - 1
                if jy < 0:
                    jy = 0
                if jy >= save_img.shape[0]:
                    jy = save_img.shape[0] - 1
                if jidx < self.joint_num:
                    save_img[jy, jx, 0] = 0
                    save_img[jy, jx, 1] = 0
                    save_img[jy, jx, 2] = 255
                else:
                    save_img[jy, jx, 0] = 0
                    save_img[jy, jx, 1] = 255
                    save_img[jy, jx, 2] = 0
            PIL.Image.fromarray(np.uint8(save_img)).save(os.path.join(self.debug_dir, 'data_debug{}.png'.format(idx)))
        return inputs, targets, meta_info
        
    def eval_cam_mpjpe(self, pred_joint_cams, logger):
        eval_type = "vert from shape and pose"
        gts = self.datalist
        mpjpe_sh = [[] for _ in range(self.joint_num * 2)]
        mpjpe_ih = [[] for _ in range(self.joint_num * 2)]
        tot_err = []
        sample_num = len(gts)
        for n in range(sample_num):
            data = gts[n]
            joint, gt_hand_type, hand_type_valid = data['joint'], data['hand_type'], data['hand_type_valid']
            gt_joint_coord = joint['cam_coord']
            joint_valid = joint['valid']
            joint_cams = pred_joint_cams[n]
            for hindex, ht in enumerate(["right", "left"]):
                gt_joint_coord[self.joint_type[ht]] = gt_joint_coord[self.joint_type[ht]] - gt_joint_coord[self.root_joint_idx[ht], None, :]

            # mpjpe
            for j in range(self.joint_num * 2):
                if joint_valid[j]:
                    err = np.sqrt(np.sum((joint_cams[j] - gt_joint_coord[j]) ** 2))
                    if gt_hand_type == 'right' or gt_hand_type == 'left':
                        mpjpe_sh[j].append(err)
                    else:
                        mpjpe_ih[j].append(err)


        eval_summary = '{} MPJPE for each joint: \n'.format(eval_type)
        for j in range(self.joint_num * 2):
            tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_sh[j]), np.stack(mpjpe_ih[j]))))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % tot_err_j)
            tot_err.append(tot_err_j)
        print(eval_summary)
        print('%s MPJPE for all hand sequences: %.2f' % (eval_type, np.mean(tot_err)))
        print()
        logger.info(eval_summary)
        logger.info('%s MPJPE for all hand sequences: %.2f' % (eval_type, np.mean(tot_err)))

        eval_summary = '{} MPJPE for each joint: \n'.format(eval_type)
        for j in range(self.joint_num * 2):
            mpjpe_sh[j] = np.mean(np.stack(mpjpe_sh[j]))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % mpjpe_sh[j])
        print(eval_summary)
        print('%s MPJPE for single hand sequences: %.2f' % (eval_type, np.mean(mpjpe_sh)))
        print()
        logger.info(eval_summary)
        logger.info('%s MPJPE for single hand sequences: %.2f' % (eval_type, np.mean(mpjpe_sh)))

        eval_summary = '{} MPJPE for each joint: \n'.format(eval_type)
        for j in range(self.joint_num * 2):
            mpjpe_ih[j] = np.mean(np.stack(mpjpe_ih[j]))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % mpjpe_ih[j])
        print(eval_summary)
        print('%s MPJPE for interacting hand sequences: %.2f' % (eval_type, np.mean(mpjpe_ih)))
        logger.info(eval_summary)
        logger.info('%s MPJPE for interacting hand sequences: %.2f' % (eval_type, np.mean(mpjpe_ih)))

    def evaluate(self, preds):

        print()
        print('Evaluation start...')

        gts = self.datalist
        preds_joint_coord, preds_rel_root_depth, preds_hand_type, inv_trans = preds['joint_coord'], preds[
            'rel_root_depth'], preds['hand_type'], preds['inv_trans']
        assert len(gts) == len(preds_joint_coord)
        sample_num = len(gts)

        mpjpe_sh = [[] for _ in range(self.joint_num * 2)]
        mpjpe_ih = [[] for _ in range(self.joint_num * 2)]
        mrrpe = []
        acc_hand_cls = 0;
        hand_cls_cnt = 0;
        for n in range(sample_num):
            data = gts[n]
            bbox, cam_param, joint, gt_hand_type, hand_type_valid = data['bbox'], data['cam_param'], data['joint'], \
                                                                    data['hand_type'], data['hand_type_valid']
            focal = cam_param['focal']
            princpt = cam_param['princpt']
            gt_joint_coord = joint['cam_coord']
            joint_valid = joint['valid']

            # restore xy coordinates to original image space
            pred_joint_coord_img = preds_joint_coord[n].copy()
            pred_joint_coord_img[:, 0] = pred_joint_coord_img[:, 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
            pred_joint_coord_img[:, 1] = pred_joint_coord_img[:, 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
            for j in range(self.joint_num * 2):
                pred_joint_coord_img[j, :2] = trans_point2d(pred_joint_coord_img[j, :2], inv_trans[n])
            # restore depth to original camera space
            pred_joint_coord_img[:, 2] = (pred_joint_coord_img[:, 2] / cfg.output_hm_shape[0] * 2 - 1) * (
                        cfg.bbox_3d_size / 2)

            # mrrpe
            if gt_hand_type == 'interacting' and joint_valid[self.root_joint_idx['left']] and joint_valid[
                self.root_joint_idx['right']]:
                pred_rel_root_depth = (preds_rel_root_depth[n] / cfg.output_root_hm_shape * 2 - 1) * (
                            cfg.bbox_3d_size_root / 2)

                pred_left_root_img = pred_joint_coord_img[self.root_joint_idx['left']].copy()
                pred_left_root_img[2] += data['abs_depth']['right'] + pred_rel_root_depth
                pred_left_root_cam = pixel2cam(pred_left_root_img[None, :], focal, princpt)[0]

                pred_right_root_img = pred_joint_coord_img[self.root_joint_idx['right']].copy()
                pred_right_root_img[2] += data['abs_depth']['right']
                pred_right_root_cam = pixel2cam(pred_right_root_img[None, :], focal, princpt)[0]

                pred_rel_root = pred_left_root_cam - pred_right_root_cam
                gt_rel_root = gt_joint_coord[self.root_joint_idx['left']] - gt_joint_coord[self.root_joint_idx['right']]
                mrrpe.append(float(np.sqrt(np.sum((pred_rel_root - gt_rel_root) ** 2))))

            # add root joint depth
            pred_joint_coord_img[self.joint_type['right'], 2] += data['abs_depth']['right']
            pred_joint_coord_img[self.joint_type['left'], 2] += data['abs_depth']['left']

            # back project to camera coordinate system
            pred_joint_coord_cam = pixel2cam(pred_joint_coord_img, focal, princpt)

            # root joint alignment
            for h in ('right', 'left'):
                pred_joint_coord_cam[self.joint_type[h]] = pred_joint_coord_cam[
                                                               self.joint_type[h]] - pred_joint_coord_cam[
                                                                                     self.root_joint_idx[h], None, :]
                gt_joint_coord[self.joint_type[h]] = gt_joint_coord[self.joint_type[h]] - gt_joint_coord[
                                                                                          self.root_joint_idx[h], None,
                                                                                          :]

            # mpjpe
            for j in range(self.joint_num * 2):
                if joint_valid[j]:
                    if gt_hand_type == 'right' or gt_hand_type == 'left':
                        mpjpe_sh[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j]) ** 2)))
                    else:
                        mpjpe_ih[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j]) ** 2)))

            # handedness accuray
            if hand_type_valid:
                if gt_hand_type == 'right' and preds_hand_type[n][0] > 0.5 and preds_hand_type[n][1] < 0.5:
                    acc_hand_cls += 1
                elif gt_hand_type == 'left' and preds_hand_type[n][0] < 0.5 and preds_hand_type[n][1] > 0.5:
                    acc_hand_cls += 1
                elif gt_hand_type == 'interacting' and preds_hand_type[n][0] > 0.5 and preds_hand_type[n][1] > 0.5:
                    acc_hand_cls += 1
                hand_cls_cnt += 1

            vis = False
            if vis:
                img_path = data['img_path']
                cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                _img = cvimg[:, :, ::-1].transpose(2, 0, 1)
                vis_kps = pred_joint_coord_img.copy()
                vis_valid = joint_valid.copy()
                capture = str(data['capture'])
                cam = str(data['cam'])
                frame = str(data['frame'])
                filename = 'out_' + str(n) + '_' + gt_hand_type + '.jpg'
                vis_keypoints(_img, vis_kps, vis_valid, self.skeleton, filename)

            vis = False
            if vis:
                filename = 'out_' + str(n) + '_3d.jpg'
                vis_3d_keypoints(pred_joint_coord_cam, joint_valid, self.skeleton, filename)

        if hand_cls_cnt > 0: print('Handedness accuracy: ' + str(acc_hand_cls / hand_cls_cnt))
        if len(mrrpe) > 0: print('MRRPE: ' + str(sum(mrrpe) / len(mrrpe)))
        print()

        tot_err = []
        eval_summary = 'MPJPE for each joint: \n'
        for j in range(self.joint_num * 2):
            tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_sh[j]), np.stack(mpjpe_ih[j]))))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % tot_err_j)
            tot_err.append(tot_err_j)
        print(eval_summary)
        print('MPJPE for all hand sequences: %.2f' % (np.mean(tot_err)))
        print()

        eval_summary = 'MPJPE for each joint: \n'
        for j in range(self.joint_num * 2):
            mpjpe_sh[j] = np.mean(np.stack(mpjpe_sh[j]))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % mpjpe_sh[j])
        print(eval_summary)
        print('MPJPE for single hand sequences: %.2f' % (np.mean(mpjpe_sh)))
        print()

        eval_summary = 'MPJPE for each joint: \n'
        for j in range(self.joint_num * 2):
            mpjpe_ih[j] = np.mean(np.stack(mpjpe_ih[j]))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % mpjpe_ih[j])
        print(eval_summary)
        print('MPJPE for interacting hand sequences: %.2f' % (np.mean(mpjpe_ih)))

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

    testset_loader = Dataset(transforms.ToTensor(), 'train', debug=True)
    for idx in range(len(testset_loader)):
        item = testset_loader.__getitem__(idx)