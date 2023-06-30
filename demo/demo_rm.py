# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import torch
import trimesh
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
import PIL.Image
sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data/InterHand2.6M'))
sys.path.insert(0, osp.join('..', 'common'))
from rm_config import rm_cfg
from rm_model import get_model
from utils.preprocessing import load_img, load_skeleton, process_bbox, generate_patch_image, transform_input_to_output_space, trans_point2d, \
      estimate_focal_length
from utils.vis import vis_keypoints_together, vis_3d_keypoints
from rm_dataset import RM_Dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--rm_run_name', type=str)
    parser.add_argument('--test_epoch', type=int, dest='test_epoch')
    parser.add_argument('--test_img_id', type=int)
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    assert args.test_epoch >= 0, 'Test epoch is required.'
    return args

# argument parsing
args = parse_args()
rm_model_path = f'../{args.rm_run_name}_cliff10_InterHand2.6M_0.1/model_dump/snapshot_{args.test_epoch}.pth.tar'
rm_ckpt = torch.load(rm_model_path)
print('Load checkpoint from {}'.format(rm_model_path))
rm_cfg_dict = rm_ckpt['cfg_dict_to_save']
rm_cfg.set_args(args.gpu_ids, **rm_cfg_dict)

cudnn.benchmark = True

# joint set information is in annotations/skeleton.txt
joint_num = 21 # single hand
root_joint_idx = {'right': 20, 'left': 41}
joint_type = {'right': np.arange(0,joint_num), 'left': np.arange(joint_num,joint_num*2)}
skeleton = load_skeleton(osp.join('../data/InterHand2.6M/annotations/skeleton.txt'), joint_num*2)

# snapshot load

model = get_model('test', pretrain=False)
model = DataParallel(model).cuda()
model.load_state_dict(rm_ckpt['network'], strict=False)
model.eval()

datapath = osp.join(rm_cfg.root_dir, '{}_{}_{}'.format(rm_cfg.run_name, rm_cfg.dataset, rm_cfg.train_ratio), 'result')
testset_loader = RM_Dataset(transforms.ToTensor(), datapath, 'train')
# for idx in range(len(testset_loader)):
#     item = testset_loader.__getitem__(idx)

# forward
inputs, preds, gt_loss = testset_loader.__getitem__(args.test_img_id)

gt_joint_coord = gt_loss['gt_joint_crop_coord']
idx = testset_loader.valid_idx_ls[args.test_img_id // testset_loader.all_pair_num]
img_path = testset_loader.ins_ls[0]['img_path'][idx]
bbox = testset_loader.ins_ls[0]['bbox'][idx]
print('img_path:', img_path)
print('bbox:', [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
        
inputs = {k:v.unsqueeze(0) for k, v in inputs.items()}
preds = {k:v.unsqueeze(0) for k, v in preds.items()}
# for k in preds.keys():
#     preds[k][:, 1] = preds[k][:, 0]
gt_loss = {k:v.unsqueeze(0) for k, v in gt_loss.items()}
with torch.no_grad():
    out = model(inputs, preds, gt_loss, 'test')
    
img = inputs['img'][0].cpu().numpy() * 255# 3, H, W
joint_valid = inputs['joint_valid'][0].cpu().numpy()
pred_joint_crop_coord = preds['pred_joint_crop_coord'][0].cpu().numpy() # [2, 42, 2]
pred_joint_crop_coord = np.concatenate([pred_joint_crop_coord, gt_joint_coord[None]], 0)
img2save = vis_keypoints_together(img, pred_joint_crop_coord, joint_valid, skeleton, 'j_crop_proj_together.jpg', save_path='.')

gt_label = gt_loss['target_prob']
label = out['label'][0].cpu().numpy()
prob = out['prob'][0].cpu().numpy()
print('gt_label:',gt_label)
print('pred_label:',label)
print('pred_prob:',prob+0.5)

font = cv2.FONT_HERSHEY_TRIPLEX
cv2.putText(img2save, '%.4f'%(prob+0.5) , (10, 30), font, 1, (255, 0, 0), 1)
PIL.Image.fromarray(img2save.astype('uint8')).save('j_crop_proj_together.jpg')

rever_preds = {}
for k, v in preds.items():
    rever_preds[k] = v.clone()
    rever_preds[k][:, 0] = v[:, 1].clone()
    rever_preds[k][:, 1] = v[:, 0].clone()
with torch.no_grad():
    rever_out = model(inputs, rever_preds, gt_loss, 'test')
pred_joint_crop_coord = rever_preds['pred_joint_crop_coord'][0].cpu().numpy() # [2, 42, 2]
pred_joint_crop_coord = np.concatenate([pred_joint_crop_coord, gt_joint_coord[None]], 0)
vis_keypoints_together(img, pred_joint_crop_coord, joint_valid, skeleton, 'j_crop_proj_together_rever.jpg', save_path='.')

rever_label = rever_out['label'][0].cpu().numpy()
rever_prob = rever_out['prob'][0].cpu().numpy()
print('rever_pred_label:',rever_label)
print('rever_pred_prob:',rever_prob+0.5)
print('sum:', prob+0.5+rever_prob+0.5)
# visualize mano mesh
# mesh = mano_mesh_cam / 1000  # milimeter to meter
# rfaces = model.module.mano_phead.rmano_layer.faces
# lfaces = model.module.mano_phead.lmano_layer.faces
# rverts = mesh[:len(mesh) // 2]
# lverts = mesh[len(mesh) // 2:]
# verts = []
# faces = []

# if right_exist:
#   verts.append(rverts)
#   faces.append(rfaces)
# if left_exist:
#   verts.append(lverts)
#   if right_exist:
#       lfaces += len(mesh) // 2
#   faces.append(lfaces)
# verts = np.concatenate(verts,0)
# faces = np.concatenate(faces,0)
# mesh = trimesh.Trimesh(verts, faces)
# mesh.export('./mesh.obj')

# x = mano_mesh_cam[:, 0].copy() / (mano_mesh_cam[:, 2].copy() + 1e-8) * \
#     focal_length.cpu().numpy()[0, 0] + (img_shape_np[1] / 2)
# y = mano_mesh_cam[:, 1].copy() / (mano_mesh_cam[:, 2].copy() + 1e-8) * \
#     focal_length.cpu().numpy()[0, 1] + (img_shape_np[0] / 2)
# mano_mesh_proj = np.stack((x, y), 1)
# save_img = original_img.copy()
# for jidx, jproj in enumerate(mano_mesh_proj):
#     # if jv == 1:
#     #print(jproj)
#     #print(mano_joint_imgs[jidx])
#     #show_scaley, show_scalex = cfg.input_img_shape[0] / cfg.output_hm_shape[1], cfg.input_img_shape[1] / cfg.output_hm_shape[2]
#     #jproj[0] *= show_scalex
#     #jproj[1] *= show_scaley
#     #jproj[:2] = torch.tensor(trans_point2d(jproj[:2], inv_trans)).float()
#     #jproj[:2] = torch.tensor(trans_point2d(jproj[:2], inv_trans)).float()
#     jx, jy = int(jproj[0]), int(jproj[1])
#     if jx < 0:
#         jx = 0
#     if jx >= save_img.shape[1]:
#         jx = save_img.shape[1] - 1
#     if jy < 0:
#         jy = 0
#     if jy >= save_img.shape[0]:
#         jy = save_img.shape[0] - 1
#     if jidx < 778 and right_exist:
#         save_img[jy, jx, 0] = 0
#         save_img[jy, jx, 1] = 0
#         save_img[jy, jx, 2] = 255
#     elif jidx >= 778 and left_exist:
#         save_img[jy, jx, 0] = 0
#         save_img[jy, jx, 1] = 255
#         save_img[jy, jx, 2] = 0
# PIL.Image.fromarray(np.uint8(save_img)).save(os.path.join('', 'data_debug{}.png'.format(os.path.basename(img_path[:-4]))))

