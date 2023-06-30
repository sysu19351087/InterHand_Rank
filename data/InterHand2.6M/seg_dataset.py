# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import sys
import os
current_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(current_path + '/../../../common'))
sys.path.append(os.path.abspath(current_path + '/../../../main'))

import numpy as np
import PIL.Image
import smplx
import torch
import torch.utils.data
import cv2
from glob import glob
import os.path as osp
from PIL import Image, ImageDraw
import random
import json
import math
import copy
from pycocotools.coco import COCO
import scipy.io as sio
from seg_config import seg_cfg
from segment_anything.utils.transforms import ResizeLongestSide
from utils.preprocessing import load_img, load_skeleton, get_bbox, process_bbox, augmentation, \
    transform_input_to_output_space, trans_point2d, estimate_focal_length, gen_ori2rotori_trans, gen_trans_from_patch_cv, estimate_translation
from utils.transforms import world2cam, cam2pixel, pixel2cam, perspective
from utils.vis import render_mesh

class Seg_Dataset(torch.utils.data.Dataset):
    def __init__(self, mode, debug=False):
        self.mode = mode  # train, test, val
        current_path = os.path.abspath(__file__)
        self.img_path = os.path.abspath(current_path + '/../images')
        self.annot_path = os.path.abspath(current_path + '/../annotations')
        if self.mode == 'val':
            self.rootnet_output_path = os.path.abspath(current_path + '/../rootnet_output/rootnet_interhand2.6m_output_val.json')
        else:
            self.rootnet_output_path = os.path.abspath(current_path + '/../rootnet_output/rootnet_interhand2.6m_output_test.json')

        self.joint_num = 21  # single hand
        self.vert_num = 778
        self.root_joint_idx = {'right': 20, 'left': 41}
        #print(os.path.abspath(current_path + '/../../..'))
        self.joint_regressor = np.load(os.path.abspath(current_path + '/../../../smplx/models/mano/J_regressor_mano_ih26m.npy'))
        self.joint_type = {'right': np.arange(0, self.joint_num), 'left': np.arange(self.joint_num, self.joint_num * 2)}
        self.skeleton = load_skeleton(osp.join(self.annot_path, 'skeleton.txt'), self.joint_num * 2)
        self.resize_transform = ResizeLongestSide(seg_cfg.encoder_img_size)
        
        self.datalist = []
        self.datalist_sh = []
        self.datalist_ih = []
        self.sequence_names = []
        
        self.debug = debug
        if self.debug:
            self.debug_dir = "debug_gen_dataset"
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

        if (self.mode == 'val' or self.mode == 'test') and seg_cfg.trans_test == 'rootnet':
            print("Get bbox and root depth from " + self.rootnet_output_path)
            rootnet_result = {}
            with open(self.rootnet_output_path) as f:
                annot = json.load(f)
            for i in range(len(annot)):
                rootnet_result[str(annot[i]['annot_id'])] = annot[i]
        else:
            print("Get bbox and root depth from groundtruth annotation")
              
        self.train_ratio = 1
        train_select_step = int(1 / self.train_ratio) if not self.debug else 1000
        count = -1
        # debug_type = "interacting"
        debug_type = 'right'
        for aid in db.anns.keys():
            if self.debug and debug_type is not None:
                hand_type = db.anns[aid]['hand_type'].lower()
                if hand_type != debug_type:
                    continue
            count += 1
            if self.mode == 'train' and count % train_select_step != 0:
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
            mask_path = osp.join('/data1/linxiaojian/Datasets/interhand/masks', self.mode, img['file_name'])
            # if os.path.exists(mask_path):
            #     continue
            # if img_height == 512:
            #     continue
            
            campos, camrot = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32), np.array(
                cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
            focal, princpt = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32), np.array(
                cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)
            # joint_world = np.array(joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32)
            # joint_cam = world2cam(joint_world.transpose(1, 0), camrot, campos.reshape(3, 1)).transpose(1, 0)  # [J, 3]
            # joint_img = cam2pixel(joint_cam, focal, princpt)[:, :2]   # [J, 2]

            # joint_valid = np.array(ann['joint_valid'], dtype=np.float32).reshape(self.joint_num * 2)
            # # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
            # joint_valid[self.joint_type['right']] *= joint_valid[self.root_joint_idx['right']]
            # joint_valid[self.joint_type['left']] *= joint_valid[self.root_joint_idx['left']]
            hand_type = ann['hand_type']
            # hand_type_valid = np.array((ann['hand_type_valid']), dtype=np.float32)

            if (self.mode == 'val' or self.mode == 'test') and seg_cfg.trans_test == 'rootnet':
                bbox = np.array(rootnet_result[str(aid)]['bbox'], dtype=np.float32)
            else:
                img_width, img_height = img['width'], img['height']
                bbox = np.array(ann['bbox'], dtype=np.float32)  # x,y,w,h
                bbox = process_bbox(bbox, (img_height, img_width))

            cam_param = {'focal': focal, 'princpt': princpt, 'campos': campos, 'camrot': camrot}
            # joint = {'cam_coord': joint_cam, 'img_coord': joint_img, 'valid': joint_valid}
            
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
            
            data = {'img_path': img_path, 
                    'seq_name': seq_name,
                    'cam_param': cam_param, 
                    'mano_param': mano_param, 
                    'bbox': bbox, 
                    # 'joint': joint, 
                    'img_shape': [img_height, img_width],
                    'hand_type': hand_type, 
                    # 'hand_type_valid': hand_type_valid, 
                    'file_name': img['file_name'], 
                    'capture': capture_id, 
                    'cam': cam, 
                    'frame': frame_idx}
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
    
    def prepare_image(self, image):
        img = self.resize_transform.apply_image(image)
        img = torch.as_tensor(img) 
        return img.permute(2, 0, 1).contiguous()
    
    def handtype_str2array(self, hand_type):
        if hand_type == 'right':
            return np.array([1, 0], dtype=np.float32)
        elif hand_type == 'left':
            return np.array([0, 1], dtype=np.float32)
        elif hand_type == 'interacting':
            return np.array([1, 1], dtype=np.float32)
        else:
            assert 0, print('Not supported hand type: ' + hand_type)
    
    def get_mano_coord(self, mano_param, cam_param, hand_type, merge_cam_rot=True, root_trans=None):

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


        else:
            output = self.mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=mano_shape)
            mesh_cam = output.vertices[0].numpy() * 1000
            joint_cam = np.dot(self.joint_regressor, mesh_cam)
            root_joint = joint_cam[20, None, :]
            mesh_cam = mesh_cam - root_joint + root_trans
            joint_cam = joint_cam - root_joint + root_trans

        mano_pose = mano_pose.view(1, -1)
        return mesh_cam, joint_cam, mano_pose, mano_shape
    
    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        data = self.datalist[idx]
        img_path, bbox, hand_type = data['img_path'], data['bbox'], data['hand_type']   # bbox:[4,], xywh
        mano_params = data['mano_param']
        cam_param = data["cam_param"]
        focal, princpt = cam_param['focal'].copy(), cam_param['princpt'].copy()
        hand_type = self.handtype_str2array(hand_type)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # [H, W, 3]
        ori_image = img.copy()
        ori_H, ori_W, _ = ori_image.shape
        
        need_rot = 0. 
        if ori_H < ori_W:
            need_rot = 1.
            # focal[0], focal[1] = focal[1], focal[0]
            princpt[0], princpt[1] = ori_H - princpt[1], princpt[0]
            rot = -90
            rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                                    [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                                    [0, 0, 1]], dtype=np.float32)
        
        # get valid verts and faces          
        verts = []
        faces = []
        for ht in ["right", "left"]:
            mano_param = mano_params[ht]
            if mano_param is None:
                continue
            with torch.no_grad():
                mc, _, _, _ = self.get_mano_coord(mano_param, cam_param, ht)   # [778, 3]
            if ori_H < ori_W and not self.debug:
                mc = np.dot(rot_aug_mat, mc.transpose(1, 0)).transpose(1, 0)
            faces.append(self.mano_layer[ht].faces.copy() + self.vert_num * len(verts))
            verts.append(mc)
        
        half_nf = faces[0].shape[0]
        half_nv = verts[0].shape[0]
        l = len(verts)
        for _ in range(2 - l):
            faces.append(-1*np.ones((half_nf, 3)))
            verts.append(np.zeros((half_nv, 3)))
        verts = np.concatenate(verts,0)
        faces = np.concatenate(faces,0)
        
        # render......
        if self.debug:
            rig = ori_image.copy()
            # if ori_H < ori_W:
            #     rig = cv2.rotate(rig, cv2.ROTATE_90_CLOCKWISE)
            cp = cam_param
            # cp = {'focal':focal, 'princpt': princpt}
            render_img, render_mask = render_mesh(rig, verts, faces, cp)    # [H, W]
            print(idx, img_path)
            PIL.Image.fromarray(np.uint8(render_img)).save(f'{self.debug_dir}/{self.mode}_{idx}.png')
        
        # x = verts[:, 0].copy() / (verts[:, 2].copy() + 1e-8) * \
        #     focal[0] + princpt[0]
        # y = verts[:, 1].copy() / (verts[:, 2].copy() + 1e-8) * \
        #     focal[1] + princpt[1]
        # verts_proj = np.stack((x, y), 1)   # [nv, 2]
        
        # bbox[0], bbox[1] = min(np.where(render_mask)[1]), min(np.where(render_mask)[0])
        # bbox[2], bbox[3] = max(np.where(render_mask)[1]) - bbox[0], max(np.where(render_mask)[0]) - bbox[1]
        
        # bbox[2], bbox[3] = bbox[0] + bbox[2], bbox[1] + bbox[3]
        # ori_box = bbox.copy().astype(int)
        # boxes = torch.tensor(bbox).unsqueeze(0)  # [1, 4]
        
        batch = {
         'image': self.prepare_image(img),   # [3, H, W]
         'original_size': img.shape[:2],
         'ori_image': ori_image,
         'idx': idx,
         'valid_verts': torch.tensor(verts.astype(float)).float(),
         'valid_faces': torch.tensor(faces.astype(float)).float(),
         'render_focal': torch.tensor(focal.astype(float)).float(),
         'render_princpt': torch.tensor(princpt.astype(float)).float(),
         'need_rot': need_rot,
        }
        return batch
    
if __name__ == "__main__":
    # datapath = '/data1/linxiaojian/InterHand2.6M-main/cliff10_InterHand2.6M_0.1/result'
    testset_loader = Seg_Dataset('test', debug=True)
    for idx in range(len(testset_loader)):
        item = testset_loader.__getitem__(idx)  
