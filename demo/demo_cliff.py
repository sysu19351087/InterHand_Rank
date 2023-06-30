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
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from model import get_model
from utils.preprocessing import load_img, load_skeleton, process_bbox, generate_patch_image, transform_input_to_output_space, trans_point2d, \
      estimate_focal_length
from utils.vis import vis_keypoints, vis_3d_keypoints, render_mesh

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--train_ratio', type=float, default=1)
    parser.add_argument('--img_path', type=str, dest='img_path')
    parser.add_argument('--bbox', type=str, dest='bbox', help='xyxy')
    parser.add_argument('--test_epoch', type=int, dest='test_epoch')
    parser.add_argument('--init_zeros', dest='init_zeros', action='store_true')
    parser.add_argument('--crop', dest='crop', action='store_true')
    parser.add_argument('--backbone', type=str, default='hr48')
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
    assert args.img_path, 'Test image path is required.'
    assert args.bbox, 'Test bbox is required.'
    return args

# argument parsing
args = parse_args()
cfg.set_args(args.gpu_ids, train_ratio=args.train_ratio, run_name=args.run_name, backbone=args.backbone, \
    init_zeros=args.init_zeros, crop=args.crop)
cudnn.benchmark = True

# joint set information is in annotations/skeleton.txt
joint_num = 21 # single hand
root_joint_idx = {'right': 20, 'left': 41}
joint_type = {'right': np.arange(0,joint_num), 'left': np.arange(joint_num,joint_num*2)}
skeleton = load_skeleton(osp.join('../data/InterHand2.6M/annotations/skeleton.txt'), joint_num*2)

# snapshot load
model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth.tar' % args.test_epoch)
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_model('test', joint_num)
model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

# prepare input image
transform = transforms.ToTensor()
img_path = args.img_path
original_img = load_img(img_path)
original_img_height, original_img_width = original_img.shape[:2]

# prepare bbox
#bbox = [69, 137, 165, 153] # xmin, ymin, width, height
#bbox = [50, 130, 230-50, 300-130] # xmin, ymin, width, height
bbox = [int(i) for i in args.bbox.split(',')]
bbox[2] = bbox[2] - bbox[0]
bbox[3] = bbox[3] - bbox[1]
bbox = process_bbox(bbox, (original_img_height, original_img_width))
img, trans, inv_trans = generate_patch_image(original_img, bbox, False, 1.0, 0.0, cfg.input_img_shape)
img = transform(img.astype(np.float32))/255
img = img.cuda()[None,:,:,:]
device = img.device

center = torch.Tensor([bbox[0] + 0.5*bbox[2], bbox[1] + 0.5*bbox[3]]).float().to(device).unsqueeze(0)  # [1, 2]
b_scale = torch.Tensor([estimate_focal_length(bbox[3], bbox[2])]).float().to(device)  # [1,]
img_shape_np = original_img.shape[:2] if cfg.crop2full else cfg.input_img_shape
if cfg.crop2full:    
    focal_length_np = estimate_focal_length(img_shape_np[0], img_shape_np[1])    
else:
    focal_length_np = cfg.focal_length
img_shape = torch.Tensor(img_shape_np).float().to(device).unsqueeze(0)   # [1, 2]
focal_length = torch.Tensor([focal_length_np, focal_length_np]).float().to(device).unsqueeze(0)   # [1, 2]

# forward
inputs = {'img': img, 'center': center, 'b_scale': b_scale, 'focal_length': focal_length, 'img_shape': img_shape}
targets = {}
meta_info = {}
with torch.no_grad():
    out = model(inputs, targets, meta_info, 'test')
img = img[0].cpu().numpy().transpose(1,2,0) # cfg.input_img_shape[1], cfg.input_img_shape[0], 3
mano_joint_proj = out['joint_coord'][0].cpu().numpy() # x,y pixel, [42, 2]
rel_root_depth = out['rel_root_depth'][0].cpu().numpy() # discretized depth
hand_type = out['hand_type'][0].cpu().numpy() # handedness probability
mano_shape = out["mano_shape"][0].cpu().numpy() # [2, 10]
mano_mesh_cam = out["mano_mesh_cam"][0].cpu().numpy() # [2*Nv, 3]
#print(mano_shape[0])
#print(mano_shape[1])

# restore joint coord to original image space and continuous depth space
if not cfg.crop2full:
    mano_joint_proj[:,0] = mano_joint_proj[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
    mano_joint_proj[:,1] = mano_joint_proj[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
    mano_joint_proj[:,:2] = np.dot(inv_trans, np.concatenate((mano_joint_proj[:,:2], np.ones_like(mano_joint_proj[:,:1])),1).transpose(1,0)).transpose(1,0)
#mano_joint_proj[:,2] = (mano_joint_proj[:,2]/cfg.output_hm_shape[0] * 2 - 1) * (cfg.bbox_3d_size/2)

# restore right hand-relative left hand depth to continuous depth space
rel_root_depth = (rel_root_depth/cfg.output_root_hm_shape * 2 - 1) * (cfg.bbox_3d_size_root/2)

# right hand root depth == 0, left hand root depth == rel_root_depth
#joint_coord[joint_type['left'],2] += rel_root_depth

# handedness
joint_valid = np.zeros((joint_num*2), dtype=np.float32)
right_exist = False
if hand_type[0] > 0.5: 
    right_exist = True
    joint_valid[joint_type['right']] = 1
left_exist = False
if hand_type[1] > 0.5:
    left_exist = True
    joint_valid[joint_type['left']] = 1

print('Right hand exist: ' + str(right_exist) + ' Left hand exist: ' + str(left_exist))

# visualize joint coord in 2D space
filename = 'result_2d.jpg'
vis_img = original_img.copy().transpose(2,0,1)
#print(mano_joint_proj[:21])
#print(mano_joint_proj[21:])
vis_img = vis_keypoints(vis_img, mano_joint_proj, joint_valid, skeleton, filename, save_path='.')

# visualize joint coord in 3D space
# The 3D coordinate in here consists of x,y pixel and z root-relative depth.
# To make x,y, and z in real unit (e.g., mm), you need to know camera intrincis and root depth.
# The root depth can be obtained from RootNet (https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE)
#filename = 'result_3d'
#vis_3d_keypoints(joint_coord, joint_valid, skeleton, filename)

# visualize mano mesh
mesh = mano_mesh_cam  # milimeter
rfaces = model.module.mano_phead.rmano_layer.faces
lfaces = model.module.mano_phead.lmano_layer.faces
rverts = mesh[:len(mesh) // 2]
lverts = mesh[len(mesh) // 2:]
verts = []
faces = []

if right_exist:
  verts.append(rverts)
  faces.append(rfaces)
if left_exist:
  verts.append(lverts)
  if right_exist:
      lfaces += len(mesh) // 2
  faces.append(lfaces)
verts = np.concatenate(verts,0)
faces = np.concatenate(faces,0)
mesh = trimesh.Trimesh(verts, faces)
mesh.export('./mesh.obj')

cam_param = {'focal': focal_length[0],
             'princpt': torch.tensor([img_shape[0, 1] // 2, img_shape[0, 0] // 2]).float().to(device)}
verts_ts = torch.from_numpy(verts).to(device)
faces_ts = torch.from_numpy(faces.astype(np.float32)).to(device)
nv = len(rverts) * 2
nf = len(rfaces) * 2
if len(verts_ts) < nv:
    verts_ts = torch.cat([verts_ts, torch.zeros(nv//2, 3).to(device)], 0)
if len(faces_ts) < nf:
    faces_ts = torch.cat([faces_ts, -1*torch.ones(nf//2, 3).to(device)], 0)
save_img, _ = render_mesh(original_img.copy(), verts_ts, faces_ts, cam_param)
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
PIL.Image.fromarray(np.uint8(save_img)).save(os.path.join('', 'data_debug{}.png'.format(os.path.basename(img_path[:-4]))))

