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
from optimize import get_mano_mesh_proj, trans_bs_poinst2d

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--rm_run_name', type=str)
    parser.add_argument('--test_epoch', type=int, dest='test_epoch')
    parser.add_argument('--opt_result_path', type=str)
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

opt_results = torch.load(args.opt_result_path)
inputs = opt_results['inputs']
preds = opt_results['preds']
gt_loss = {}
meta_info = opt_results['meta_info']
first_pred = opt_results['first_pred']
with torch.no_grad():
    out = model(inputs, preds, gt_loss, 'test')
    
img = inputs['img'][0].cpu().numpy() * 255# 3, H, W

hand_type = preds['pred_hand_type'][:, 0]   # [bs, 2]
bs, com_num = preds['pred_hand_type'].shape[0], preds['pred_hand_type'].shape[1]
joint_valid = np.zeros((joint_num*2), dtype=np.float32)
right_exist = False
if hand_type.cpu().numpy()[0, 0] > 0.5: 
    right_exist = True
    joint_valid[joint_type['right']] = 1
left_exist = False
if hand_type.cpu().numpy()[0, 1] > 0.5:
    left_exist = True
    joint_valid[joint_type['left']] = 1

# preds = {k: torch.cat([first_pred[k].unsqueeze(1), v], 1) for k, v in preds.items()}
# pred_joint_crop_coord = torch.zeros(bs, com_num+1, joint_num*2, 2)
# for cid in range(com_num+1):
#     _, joint_proj, _, _ = get_mano_mesh_proj(preds['pred_mano_shape'][:, cid], preds['pred_mano_pose'][:, cid], preds['pred_cam_param'][:, cid], \
#             meta_info['center'], meta_info['b_scale'], meta_info['focal_length'], meta_info['img_shape'])   # [bs, 778*2, 2]
#     joint_crop_proj = trans_bs_poinst2d(joint_proj, meta_info['trans_rot2crop'])     # [bs, 42, 2]
#     pred_joint_crop_coord[:, cid] = joint_crop_proj

pred_joint_crop_coord = torch.cat([first_pred['pred_joint_crop_coord'].unsqueeze(1), preds['pred_joint_crop_coord']], dim=1)
vis_keypoints_together(img, pred_joint_crop_coord.detach().cpu().numpy()[0], joint_valid, skeleton, f'j_crop_proj_together.jpg', save_path='.')

label = out['label'][0].cpu().numpy()
prob = out['prob'][0].cpu().numpy()
# print('gt_label:',gt_label[0])
print('pred_label:',label)
print('pred_prob:',prob+0.5)