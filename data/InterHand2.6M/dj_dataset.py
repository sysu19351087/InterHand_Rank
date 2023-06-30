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
import trimesh
from pycocotools.coco import COCO
import scipy.io as sio
from config import cfg
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
    TexturesVertex
)
from pytorch3d.renderer.mesh.shader import SoftDepthShader
from utils.preprocessing import load_img, load_skeleton, get_bbox, process_bbox, augmentation, \
    transform_input_to_output_space, trans_point2d, estimate_focal_length, gen_ori2rotori_trans, gen_trans_from_patch_cv, estimate_translation
from utils.transforms import world2cam, cam2pixel, pixel2cam, perspective
from utils.vis import vis_keypoints, vis_3d_keypoints, vis_keypoints_together, vis_meshverts_together

class Dj_Dataset(torch.utils.data.Dataset):
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
            self.debug_dir = os.path.abspath(current_path + '/../dj_debug_dataset')
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
        
        self.all_action_cam = {}
        debug_type = "right"
        #debug_type = 'right'
        for aid in db.anns.keys():
            if self.debug and debug_type is not None:
                hand_type = db.anns[aid]['hand_type'].lower()
                if hand_type != debug_type:
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
            
            pic_name = img['file_name'].split('/')
            pic_name.pop(-2)
            pic_name = '/'.join(pic_name)
            filename_spl = '/'.join(img['file_name'].split('/')[:-2])
            current_cam = img['file_name'].split('/')[-2]
            if pic_name not in self.all_action_cam.keys():
                cam_ls = os.listdir(osp.join(self.img_path, self.mode, filename_spl))
                
                select_cam = random.sample(cam_ls, 1)
                self.all_action_cam[pic_name] = select_cam[0]

            if current_cam != self.all_action_cam[pic_name]:
                continue
            
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

            # if (self.mode == 'val' or self.mode == 'test') and cfg.trans_test == 'rootnet':
            #     bbox = np.array(rootnet_result[str(aid)]['bbox'], dtype=np.float32)
            #     abs_depth = {'right': rootnet_result[str(aid)]['abs_depth'][0],
            #                  'left': rootnet_result[str(aid)]['abs_depth'][1]}
            # else:
            #     img_width, img_height = img['width'], img['height']
            #     bbox = np.array(ann['bbox'], dtype=np.float32)  # x,y,w,h
            #     bbox = process_bbox(bbox, (img_height, img_width))
            #     abs_depth = {'right': joint_cam[self.root_joint_idx['right'], 2],
            #                  'left': joint_cam[self.root_joint_idx['left'], 2]}

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
                    'joint': joint, 'img_shape': [img_height, img_width],
                    'hand_type': hand_type, 'hand_type_valid': hand_type_valid,
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

        mano_pose = torch.FloatTensor(mano_param['pose']).view(-1, 3).clone()
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
        mano_shape = torch.FloatTensor(mano_param['shape']).view(1, -1).clone()

        if root_trans is None:   # without augmentation
            trans = torch.FloatTensor(mano_param['trans']).view(1, -1).clone()
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

    def render_mesh(self, img, verts, faces, focal_length, princpt):
        # img:[bs, H, W, 3]
        # verts:[bs, nv, 3]
        # face:[bs, nf, 3]
        # focal_length:[bs, 2]
        # princpt:[bs, 2]
        
        # mesh
        meshes = Meshes(verts=verts, faces=faces)
        device = meshes.device
        bs, nv = len(meshes.verts_list()), len(meshes.verts_list()[0])
        color = torch.ones(bs, nv, 3, device=device)
        color = color * 255.
        meshes.textures = TexturesVertex(verts_features=color)
        
        _, img_height, img_weight, _ = img.shape
        fcl, prp = focal_length.clone(), princpt.clone()
        fcl[:, 0] = fcl[:, 0] * 2 / min(img_height, img_weight)
        fcl[:, 1] = fcl[:, 1] * 2 / min(img_height, img_weight)
        prp[:, 0] = - (prp[:, 0] - img_weight // 2) * 2 / img_weight
        prp[:, 1] = - (prp[:, 1] - img_height // 2) * 2 / img_height

        cameras = PerspectiveCameras(focal_length=-fcl, principal_point=prp, device=device)
        raster_settings = RasterizationSettings(
            image_size=(img_height, img_weight), 
            blur_radius=0.0, 
            faces_per_pixel=50, 
        )
        lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

        shader_depth=SoftDepthShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )

        # renderer
        renderer_depth = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=shader_depth
        )

        # render
        znear = 0.0
        zfar = 1000.0
        rendered_depth = renderer_depth(meshes, znear=znear, zfar=zfar)    # [bs, H, W, 1]
        # valid_mask = (rendered_depth < zfar).float()      # [bs, H, W, 1]
        valid_mask = (zfar - rendered_depth) / zfar
        return valid_mask      # [bs, H, W, 1]

    def __getitem__(self, idx):
        data = self.datalist[idx]
        img_path, img_shape, joint, hand_type, hand_type_valid = data['img_path'], data['img_shape'], data['joint'], data[
            'hand_type'], data['hand_type_valid']
        joint_cam = joint['cam_coord'].copy()     # [J, 3]
        joint_img = joint['img_coord'].copy()     # [J, 2]
        hand_type = self.handtype_str2array(hand_type)
        # joint_coord = np.concatenate((joint_img, joint_cam[:, 2, None].copy()), 1)    # [J, 3]
         # get camera parameters
        cam_param = data["cam_param"]
        # focal_length, princpt = cam_param['focal'], cam_param['princpt']
        
        # image load
        img = load_img(img_path)   # [H, W, 3]
        # ori_H, ori_W = img.shape[:2]
        # need_flip = int(ori_H < ori_W)
        
        # src = np.zeros((3, 2), dtype=np.float32)
        # src[0, :] = [ori_W//2, ori_H//2]
        # src[1, :] = [0, ori_H//2]
        # src[2, :] = [ori_W//2, 0]
        # dst = np.zeros((3, 2), dtype=np.float32)
        # dst[0, :] = [ori_H//2, ori_W//2]
        # dst[1, :] = [ori_H//2, 0]
        # dst[2, :] = [ori_H, ori_W//2]
        # trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        # inv_trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        # if need_flip == 1:
        #     img = cv2.warpAffine(img, trans, (int(ori_H), int(ori_W)), flags=cv2.INTER_LINEAR)
        #     render_princpt = torch.tensor(trans_point2d(princpt, trans)).float()
        #     render_focal_length = [
        #         focal_length[1],
        #         focal_length[0],
        #     ]
        #     render_focal_length = torch.tensor(render_focal_length).float()
        # else:
        #     render_princpt = torch.tensor(princpt).float()
        #     render_focal_length = torch.tensor(focal_length).float()
        img = self.transform(img.astype(np.float32)) # [3, H, W]
        
        # get mano param
        mano_params = data['mano_param']
        mano_mesh_cams = []
        mano_shapes = []
        mano_poses = []
        mano_transes = []
            
        for ht in ["right", "left"]:
            mano_param = mano_params[ht]
            if mano_param is None:
                mano_mesh_cams.append(torch.zeros((self.vert_num, 3)))
                mano_shapes.append(torch.zeros((1, 10)))
                mano_poses.append(torch.zeros((1, 48)))
                mano_transes.append(torch.zeros((1, 3)))
                continue

            with torch.no_grad():
                mesh_cam, mano_joint_cam, _, _ = self.get_mano_coord(mano_param, cam_param, False, ht)
            "augmentation"
            cat_cam = np.concatenate([mesh_cam, mano_joint_cam], axis=0)

            # gather outputs
            mano_mesh_cams.append(torch.from_numpy(cat_cam[:self.vert_num]))
            mano_shapes.append(torch.FloatTensor(mano_param['shape']).view(1, 10))
            mano_poses.append(torch.FloatTensor(mano_param['pose']).view(1, 48))
            mano_transes.append(torch.FloatTensor(mano_param['trans']).view(1, 3))
            
        lidx, ridx = 1, 0

        mano_mesh_cams = torch.cat([mano_mesh_cams[ridx].clone(), mano_mesh_cams[lidx].clone()], dim=0).float()
        mano_shapes = torch.cat([mano_shapes[ridx].clone(), mano_shapes[lidx].clone()], dim=0).float()
        mano_poses = torch.cat([mano_poses[ridx].clone(), mano_poses[lidx].clone()], dim=0).float()
        mano_transes = torch.cat([mano_transes[ridx].clone(), mano_transes[lidx].clone()], dim=0).float()
        # joint_coord = torch.from_numpy(joint_coord).float()
        
        mesh = mano_mesh_cams / 1000   # [778*2, 3]
        rfaces = torch.tensor(self.mano_layer['right'].faces.copy().astype(np.float32)).float()
        lfaces = torch.tensor(self.mano_layer['left'].faces.copy().astype(np.float32)).float()
        rverts = mesh[:len(mesh) // 2]
        lverts = mesh[len(mesh) // 2:]
        half_nv, half_nf = len(rverts), len(rfaces)
        verts_ls = []
        faces_ls = []
        if hand_type[0] > 0.5:
            faces_ls.append(rfaces)
            verts_ls.append(rverts)
        if hand_type[1] > 0.5:
            faces_ls.append(lfaces + len(verts_ls) * half_nv)
            verts_ls.append(lverts)
            
        # for _ in range(2 - len(verts_ls)):
        #     faces_ls.append(-1*torch.ones(half_nf, 3))
        #     verts_ls.append(torch.zeros(half_nv, 3))
            
        verts_ts = torch.cat(verts_ls, 0).float()       # [778*2, 3]
        faces_ts = torch.cat(faces_ls, 0).float()       # [nf*2, 3]
                
        outputs = {'img': img,               # [3, H, W]
                #    'joint_coord': joint_coord[:, :2],
                #    'need_flip': torch.tensor(need_flip).float(),
                #    'inv_trans': torch.tensor(inv_trans).float(),
                   'verts': verts_ts,
                   'faces': faces_ts,
                #    'focal_length': render_focal_length,
                #    'princpt': render_princpt,
                   'hand_type': torch.tensor(hand_type).float(),
                   'shape': mano_shapes,
                   'pose': mano_poses,
                   'trans': mano_transes
                }
        
        return outputs
        
    
if __name__ == "__main__":
    import torchvision.transforms as transforms

    testset_loader = Dj_Dataset(transforms.ToTensor(), 'train', debug=True)
    for idx in range(len(testset_loader)):
        item = testset_loader.__getitem__(idx)
        for k, v in item.items():
            item[k] = v.unsqueeze(0).cuda()
        
        img = item['img'].permute(0, 2, 3, 1)
        verts = item['verts']
        faces = item['faces']
        focal_length = item['focal_length']
        princpt = item['princpt']
        render_depth = testset_loader.render_mesh(img, verts, faces, focal_length, princpt)
        img2save = render_depth.cpu().numpy()[0].repeat(3, axis=-1)
        if item['need_flip'][0] == 1.0:
            inv_trans = item['inv_trans'].cpu().numpy()[0]
            print(inv_trans.shape)
        print(img2save.shape)
        print(np.max(img2save))
        PIL.Image.fromarray((img2save*255.).astype('uint8')).save(os.path.join(testset_loader.debug_dir, 'data_debug{}_mask.png'.format(idx)))
        
        joint_coord = item['joint_coord'].cpu().numpy()[0]  # [J, 2]
        hand_type = item['hand_type'].cpu().numpy()[0]
        joint_valid = hand_type[:, None].repeat(21, axis=-1)  # [2, 21]
        joint_valid = joint_valid.reshape(-1)
        
        original_img = img.cpu().numpy()[0]  # [H, W, 3]
        all_jkpts = np.stack([joint_coord], 0)
        vis_keypoints_together(original_img.copy().transpose(2,0,1), all_jkpts, joint_valid, testset_loader.skeleton, \
            f'j_proj_together_{idx}.jpg', save_path=testset_loader.debug_dir)

        mano_mesh_cam = verts.cpu().numpy()[0]
        x = mano_mesh_cam[:, 0].copy() / (mano_mesh_cam[:, 2].copy() + 1e-8) * \
            focal_length.cpu().numpy()[0, 0] + (princpt.cpu().numpy()[0, 0])
        y = mano_mesh_cam[:, 1].copy() / (mano_mesh_cam[:, 2].copy() + 1e-8) * \
            focal_length.cpu().numpy()[0, 1] + (princpt.cpu().numpy()[0, 1])
        mano_mesh_proj = np.stack((x, y), 1)   # [778*2, 3]
        
        vis_meshverts_together(original_img, mano_mesh_proj[None, ...], hand_type, \
            'meshverts_data_debug{}.png'.format(idx), testset_loader.debug_dir)
        
        mesh = trimesh.Trimesh(verts.cpu().numpy()[0], faces.cpu().numpy()[0])
        mesh.export(os.path.join(testset_loader.debug_dir, f'mesh_{idx}.obj'))