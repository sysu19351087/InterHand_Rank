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
from pycocotools.coco import COCO
import scipy.io as sio
from config import cfg
from optimize import get_valid_verts_faces
from utils.preprocessing import load_img, load_skeleton, get_bbox, process_bbox, augmentation, \
    transform_input_to_output_space, trans_point2d, estimate_focal_length, gen_ori2rotori_trans, gen_trans_from_patch_cv, estimate_translation
from utils.transforms import world2cam, cam2pixel, pixel2cam, perspective
from utils.vis import vis_keypoints, vis_3d_keypoints, render_mesh


class Detgt_Dataset(torch.utils.data.Dataset):
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
            self.debug_dir = "debug_betgt_dataset"
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
        
        
        debug_type = "interacting"
        #debug_type = 'right'
        for aid in db.anns.keys():
            if self.debug and debug_type is not None:
                hand_type = db.anns[aid]['hand_type'].lower()
                if hand_type != debug_type:
                    continue
            
            # if self.mode == 'test' and count % 10 != 0:
            #     continue
            
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
            # joint_world = np.array(joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32)
            # joint_cam = world2cam(joint_world.transpose(1, 0), camrot, campos.reshape(3, 1)).transpose(1, 0)  # [J, 3]
            # joint_img = cam2pixel(joint_cam, focal, princpt)[:, :2]   # [J, 2]

            # joint_valid = np.array(ann['joint_valid'], dtype=np.float32).reshape(self.joint_num * 2)
            # # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
            # joint_valid[self.joint_type['right']] *= joint_valid[self.root_joint_idx['right']]
            # joint_valid[self.joint_type['left']] *= joint_valid[self.root_joint_idx['left']]
            hand_type = ann['hand_type']
            hand_type_valid = np.array((ann['hand_type_valid']), dtype=np.float32)

            if (self.mode == 'val' or self.mode == 'test') and cfg.trans_test == 'rootnet':
                bbox = np.array(rootnet_result[str(aid)]['bbox'], dtype=np.float32)
                # abs_depth = {'right': rootnet_result[str(aid)]['abs_depth'][0],
                #              'left': rootnet_result[str(aid)]['abs_depth'][1]}
            else:
                img_width, img_height = img['width'], img['height']
                bbox = np.array(ann['bbox'], dtype=np.float32)  # x,y,w,h
                bbox = process_bbox(bbox, (img_height, img_width))
                # abs_depth = {'right': joint_cam[self.root_joint_idx['right'], 2],
                #              'left': joint_cam[self.root_joint_idx['left'], 2]}

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
                    'hand_type_valid': hand_type_valid, 
                    # 'abs_depth': abs_depth,
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

        self.datalist = self.datalist_sh + self.datalist_ih
        print('Number of annotations in single hand sequences: ' + str(len(self.datalist_sh)))
        print('Number of annotations in interacting hand sequences: ' + str(len(self.datalist_ih)))
        
        self.frame_cam_dict = {}
        for indx in range(len(self.datalist)):
            
            data = self.datalist[indx]
            capture_id, seq_name, frame_idx = data['capture'], data['seq_name'], data['frame']

            frame = '/'.join([str(capture_id), seq_name, str(frame_idx)])
            
            if frame not in self.frame_cam_dict.keys():
                if self.debug and len(self.frame_cam_dict.keys()) > 10:
                    break
                self.frame_cam_dict[frame] = [indx]
            else:
                self.frame_cam_dict[frame].append(indx)
        for k, v in self.frame_cam_dict.items():
            redundant = len(v) % cfg.num_gpus
            if redundant != 0:
                need2del = random.sample(v, redundant)
                new_v = []
                for idx in v:
                    if idx not in need2del:
                        new_v.append(idx)
                self.frame_cam_dict[k] = new_v

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
        return len(self.frame_cam_dict.keys())
        
    # def get_mano_coord(self, mano_param, cam_param, hand_type, root_trans=None):

    #     mano_pose = torch.FloatTensor(mano_param['pose']).view(-1, 3)
    #     R, t = np.array(cam_param['camrot'], dtype=np.float32).reshape(3, 3), np.array(cam_param['campos'],
    #                                                                                    dtype=np.float32).reshape(
    #         3)  # camera rotation and translation
    #     t = -np.dot(R, t.reshape(3, 1)).reshape(3)  # -Rt -> t

    #     root_pose = mano_pose[0].view(1, 3).numpy()
    #     root_pose, _ = cv2.Rodrigues(root_pose)
    #     root_pose, _ = cv2.Rodrigues(np.dot(R, root_pose))
    #     mano_pose[0] = torch.from_numpy(root_pose).view(3)

    #     # get mesh and joint coordinates
    #     root_pose = mano_pose[0].view(1, 3)
    #     hand_pose = mano_pose[1:, :].view(1, -1)
    #     mano_shape = torch.FloatTensor(mano_param['shape']).view(1, -1)

    #     if root_trans is None:   # without augmentation
    #         trans = torch.FloatTensor(mano_param['trans']).view(1, -1)
    #         output = self.mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=mano_shape,
    #                                             transl=trans)
    #         #print('xrange:', max(output.vertices[0].numpy()[:, 0]) - min(output.vertices[0].numpy()[:, 0]))
    #         #print('yrange:', max(output.vertices[0].numpy()[:, 1]) - min(output.vertices[0].numpy()[:, 1]))
    #         mesh_cam = output.vertices[0].numpy() * 1000
    #         joint_cam = np.dot(self.joint_regressor, mesh_cam)
    #         root_joint = joint_cam[20, None, :]
    #         # root_joint = trans.numpy() * 1000
    #         rootj_CR = np.dot(R, root_joint.reshape(3, 1)).reshape(1, 3)
    #         mesh_cam = mesh_cam - root_joint + rootj_CR + t
    #         joint_cam = joint_cam - root_joint + rootj_CR + t


    #     else:
    #         output = self.mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=mano_shape)
    #         mesh_cam = output.vertices[0].numpy() * 1000
    #         joint_cam = np.dot(self.joint_regressor, mesh_cam)
    #         root_joint = joint_cam[20, None, :]
    #         mesh_cam = mesh_cam - root_joint + root_trans
    #         joint_cam = joint_cam - root_joint + root_trans

    #     mano_pose = mano_pose.view(1, -1)
    #     return mesh_cam, joint_cam, mano_pose, mano_shape
    
    def get_mano_coord(self, mano_param, cam_param, hand_type):

        mano_pose = torch.FloatTensor(mano_param['pose']).view(-1, 3)
        R, t = np.array(cam_param['camrot'], dtype=np.float32).reshape(3, 3), np.array(cam_param['campos'],
                                                                                       dtype=np.float32).reshape(
            3)  # camera rotation and translation
        t = -np.dot(R, t.reshape(3, 1)).reshape(3, 1)  # -Rt -> t

        # get mesh and joint coordinates
        root_pose = mano_pose[0].view(1, 3)
        hand_pose = mano_pose[1:, :].view(1, -1)
        mano_shape = torch.FloatTensor(mano_param['shape']).view(1, -1)

        trans = torch.FloatTensor(mano_param['trans']).view(1, -1)
        output = self.mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=mano_shape,
                                            transl=trans)
        #print('xrange:', max(output.vertices[0].numpy()[:, 0]) - min(output.vertices[0].numpy()[:, 0]))
        #print('yrange:', max(output.vertices[0].numpy()[:, 1]) - min(output.vertices[0].numpy()[:, 1]))
        mesh_cam = output.vertices[0].numpy() * 1000
        mesh_cam = (np.dot(R, mesh_cam.T) + t).T  # [778, 3]
        
        return mesh_cam
    
    def __getitem__(self, idx):
        frame = list(self.frame_cam_dict.keys())[idx]
        dataidx_list = self.frame_cam_dict[frame]
        
        outputs = {
            'img': [],
            'gt_masks': [],
            'shapes': [],
            'poses': [],
            'transls': [],
            'focal_length': [],
            'princpt': [],
            'hand_type': [],
            'R': [],
            't': []
        }
        if self.debug:
            sn_list = []
            img_list = []
            outputs['mesh_verts'] = []
            outputs['mesh_faces'] = []
            
        for idx in dataidx_list:
            data = self.datalist[idx]
            img_path, bbox, img_shape, hand_type, hand_type_valid = data['img_path'], data['bbox'], data['img_shape'], data[
                'hand_type'], data['hand_type_valid']
            
            hand_type = self.handtype_str2array(hand_type)   # [2,]
            
            cx, cy = bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2
            trans4crop = gen_trans_from_patch_cv(cx, cy, bbox[2], bbox[3], \
                int(cfg.input_img_shape[1]), int(cfg.input_img_shape[0]), 1., 0)    # [2, 3]
            
            # image load
            img = load_img(img_path)           # [H_ori, W_ori, 3]
            original_img = img.copy()          # [H_ori, W_ori, 3]
            img = cv2.warpAffine(img, trans4crop, (int(cfg.input_img_shape[1]), int(cfg.input_img_shape[0])), flags=cv2.INTER_LINEAR)
            save_img = img.copy()
            img = self.transform(img.astype(np.float32)) / 255.    # [3, H, W], 0~1

            # get camera parameters
            cam_param = data["cam_param"]
            focal, princpt = cam_param['focal'], cam_param['princpt']
            princpt_aug = trans_point2d(princpt, trans4crop)
            focal_aug = [
                focal[0] * cfg.input_img_shape[1] / bbox[2],
                focal[1] * cfg.input_img_shape[0] / bbox[3],
            ]
            R = np.array(cam_param['camrot'], dtype=np.float32).reshape(3, 3).copy()
            t = np.array(cam_param['campos'], dtype=np.float32).reshape(3).copy()  # camera rotation and translation
            t = -np.dot(R, t.reshape(3, 1)).reshape(3, 1)  # -Rt -> t
            
            # get mano param
            mano_params = data['mano_param']
            mano_poses = []
            mano_shapes = []
            mano_transes = []
            mesh_verts = []
            mesh_faces = []
            for ht in ["right", "left"]:
                mano_param = mano_params[ht]
                if mano_param is None:
                    mano_shapes.append(torch.zeros((1, 10)))
                    mano_poses.append(torch.zeros((1, 48)))
                    mano_transes.append(torch.zeros((1, 3)))
                    continue

                with torch.no_grad():
                    mesh_cam = self.get_mano_coord(mano_param, cam_param, ht)

                # gather outputs
                mano_shapes.append(torch.FloatTensor(mano_param['shape']).view(1, 10))
                mano_poses.append(torch.FloatTensor(mano_param['pose']).view(1, 48))
                mano_transes.append(torch.FloatTensor(mano_param['trans']).view(1, 3))
                mesh_faces.append(self.mano_layer[ht].faces.copy() + self.vert_num * len(mesh_verts))
                mesh_verts.append(mesh_cam)

            lidx, ridx = 1, 0

            mano_poses = torch.cat([mano_poses[ridx].clone(), mano_poses[lidx].clone()], dim=0).float()
            mano_shapes = torch.cat([mano_shapes[ridx].clone(), mano_shapes[lidx].clone()], dim=0).float()
            mano_transes = torch.cat([mano_transes[ridx].clone(), mano_transes[lidx].clone()], dim=0).float()
            
            # get mask
            mask_path = img_path.replace('images', 'masks', 1)
            mask = cv2.imread(mask_path)      # [H_ori, W_ori, 3]
            mask = cv2.warpAffine(mask, trans4crop, (int(cfg.input_img_shape[1]), int(cfg.input_img_shape[0])), flags=cv2.INTER_LINEAR)
            mask = self.transform(mask.astype(np.float32)) / 255.  # [3, H, W], 0 or 1
            mask = torch.mean(mask, dim=0)                         # [H, W], 0 or 1
            mask = (mask > 0.5).float()
            
            outputs['img'].append(img)                                          # [nc, 3, H, W]
            outputs['gt_masks'].append(mask)                                    # [nc, H, W]
            outputs['shapes'].append(mano_shapes)                               # [nc, 2, 10], same
            outputs['poses'].append(mano_poses)                                 # [nc, 2, 48], same
            outputs['transls'].append(mano_transes)                             # [nc, 2, 3], same
            outputs['focal_length'].append(torch.FloatTensor(focal_aug))        # [nc, 2]
            outputs['princpt'].append(torch.FloatTensor(princpt_aug))           # [nc, 2]
            outputs['hand_type'].append(torch.FloatTensor(hand_type))           # [nc, 2]
            outputs['R'].append(torch.FloatTensor(R))                           # [nc, 3, 3]
            outputs['t'].append(torch.FloatTensor(t))                           # [nc, 3, 1]
            
            if self.debug:
                capture_name = data['capture']
                seq_name = data['seq_name']
                frame = data['frame']
                cam_name = data['cam']
                sn_list.append(f'{capture_name}_{seq_name}_{frame}_{cam_name}')
                img_list.append(img)
                
                half_nf = mesh_faces[0].shape[0]
                half_nv = mesh_verts[0].shape[0]
                l = len(mesh_verts)
                for _ in range(2 - l):
                    mesh_faces.append(-1*np.ones((half_nf, 3)))
                    mesh_verts.append(np.zeros((half_nv, 3)))
                mesh_verts = np.concatenate(mesh_verts,0)
                mesh_faces = np.concatenate(mesh_faces,0)
                outputs['mesh_verts'].append(torch.FloatTensor(mesh_verts.astype(np.float32)).clone())
                outputs['mesh_faces'].append(torch.FloatTensor(mesh_faces.astype(np.float32)).clone())
        
        for k, v in outputs.items():
            outputs[k] = torch.stack(v, dim=0)
            
        if self.debug:
            # print(len(outputs['gt_masks']))
            for i, m in enumerate(outputs['gt_masks']):
                # m:[H, W]
                img = img_list[i].permute(1, 2, 0).cpu().numpy()
                m2s = m.unsqueeze(-1).cpu().numpy()
                i2s = np.uint8(m2s * img * 255)
                sn = sn_list[i]
                
                cp = {'focal': outputs['focal_length'][i], 'princpt': outputs['princpt'][i]}
                render_img, render_mask = render_mesh(img*255., outputs['mesh_verts'][i], outputs['mesh_faces'][i], cp)    # [H, W]
                
                i2s = np.concatenate([i2s, np.uint8(render_img)], 1)
                PIL.Image.fromarray(i2s).save(os.path.join(self.debug_dir, f'{sn}.jpg'))
                print(sn)
        return outputs
    
if __name__ == "__main__":
    import torchvision.transforms as transforms

    testset_loader = Detgt_Dataset(transforms.ToTensor(), 'train', debug=True)
    for idx in range(len(testset_loader)):
        item = testset_loader.__getitem__(idx)