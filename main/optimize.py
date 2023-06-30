import argparse
import torch
import os
import sys
import os.path as osp
import numpy as np
import cv2
import trimesh
from PIL import Image
import smplx
from tqdm import tqdm
import torchvision.transforms as transforms
current_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(current_path + '/../../common'))
sys.path.append(os.path.abspath(current_path + '/../../data/InterHand2.6M'))
import torch.backends.cudnn as cudnn
from torch.nn.parallel.data_parallel import DataParallel

from config import cfg
from rm_config import rm_cfg
from model import get_model as get_cliff_model
from rm_model import get_model as get_rm_model
from timer import Timer
from logger import colorlogger
from utils.preprocessing import load_img, load_skeleton, process_bbox, generate_patch_image, transform_input_to_output_space, trans_point2d, \
    estimate_focal_length, trans_poinst2d, gen_trans_from_patch_cv
from utils.vis import vis_keypoints_together, vis_meshverts_together, render_mesh

joint_regressor = torch.tensor(np.load(os.path.abspath(current_path + '/../../smplx/models/mano/J_regressor_mano_ih26m.npy'))).float()
#self.joint_regressor = torch.Tensor(joint_regressor).unsqueeze(0).cuda()
smplx_path = os.path.abspath(current_path + '/../../smplx/models')
rmano_layer = smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=True, create_transl=False)
lmano_layer = smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=False, create_transl=False)
# fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
if torch.sum(torch.abs(
    lmano_layer.shapedirs[:, 0, :] - rmano_layer.shapedirs[:, 0, :])) < 1:
    lmano_layer.shapedirs[:, 0, :] *= -1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--bbox', type=str, dest='bbox', help='xyxy')
    
    parser.add_argument('--rm_name', type=str)
    parser.add_argument('--rm_epoch', type=int, default=19)
    parser.add_argument('--cliff_name', type=str)
    parser.add_argument('--cliff_epoch', type=int, default=19)
    # parser.add_argument('--run_name', type=str)
    # parser.add_argument('--train_ratio', type=float, default=1)
    # parser.add_argument('--cliff_backbone', type=str, default='hr48')
    # parser.add_argument('--init_zeros', dest='init_zeros', action='store_true')
    # parser.add_argument('--crop', dest='crop', action='store_true')
    
    parser.add_argument('--end_step', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-5)
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args


def trans_bs_poinst2d(pt_2d, trans):
    # pt_2d:[bs, nv, 2]
    # trans:[bs, 2, 3]
    bs, nv, _ = pt_2d.shape
    ones = torch.ones([bs, nv, 1]).to(pt_2d.device)
    src_pt = torch.cat([pt_2d, ones], -1).transpose(1, 2)   # [bs, 3, nv]
    dst_pt = torch.matmul(trans, src_pt)                    # [bs, 2, nv] 
    
    return dst_pt.transpose(1, 2)     # [bs, nv, 2]

def get_coord(mano_cam, cam_trans, focal_length, img_shape):
    mesh_cam_m = mano_cam.clone()
    joint_cam_m = torch.matmul(joint_regressor.to(cam_trans.device), mesh_cam_m)
    root_cam_m = joint_cam_m[:, 20, None, :].clone()
    mesh_cam_mm = (mesh_cam_m - root_cam_m + cam_trans[:, None, :]) * 1000
    
    mesh_x = mesh_cam_mm[:, :, 0].clone() / (mesh_cam_mm[:, :, 2].clone() + 1e-8) * \
        focal_length[:, 0].unsqueeze(-1) + (img_shape[:, 1].clone().unsqueeze(-1) / 2)
    mesh_y = mesh_cam_mm[:, :, 1].clone() / (mesh_cam_mm[:, :, 2].clone() + 1e-8) * \
        focal_length[:, 1].unsqueeze(-1) + (img_shape[:, 0].clone().unsqueeze(-1) / 2)
    mesh_proj = torch.stack((mesh_x, mesh_y), 2)
    
    joint_cam_m = joint_cam_m - root_cam_m
    joint_cam_mm = (joint_cam_m.clone() + cam_trans[:, None, :]) * 1000
    x = joint_cam_mm[:, :, 0].clone() / (joint_cam_mm[:, :, 2].clone() + 1e-8) * \
        focal_length[:, 0].unsqueeze(-1) + (img_shape[:, 1].clone().unsqueeze(-1) / 2)
    y = joint_cam_mm[:, :, 1].clone() / (joint_cam_mm[:, :, 2].clone() + 1e-8) * \
        focal_length[:, 1].unsqueeze(-1) + (img_shape[:, 0].clone().unsqueeze(-1) / 2)
    joint_proj = torch.stack((x, y), 2)
    return mesh_proj, joint_proj, mesh_cam_mm, joint_cam_m * 1000

def get_camtrans(pred_cam, center, b_scale, focal_length, img_shape):
    if cfg.crop2full:
        img_h, img_w = img_shape[:, 0].clone(), img_shape[:, 1].clone()
        cx, cy, b = center[:, 0], center[:, 1], b_scale
        w_2, h_2 = img_w / 2., img_h / 2.
        bs = b * pred_cam[:, 0] + 1e-9
        
        tz = cfg.mano_size * focal_length[:, 0].clone() / bs
        tx = (cfg.mano_size * (cx - w_2) / bs) + pred_cam[:, 1]
        ty = (cfg.mano_size * (cy - h_2) / bs) + pred_cam[:, 2]
        full_cam = torch.stack([tx, ty, tz], dim=-1)
        #print(focal_length[:,0])
        #print('pred_cam:', pred_cam[:, :])
        #print('pred_cam_reans:', full_cam[:, :])
    else:
        img_h, img_w = img_shape[:, 0], img_shape[:, 1]
        
        bs = estimate_focal_length(img_h, img_w) * pred_cam[:, 0] + 1e-9
        tz = cfg.mano_size * focal_length[:, 0] / bs
        tx = pred_cam[:, 1]
        ty = pred_cam[:, 2]
        full_cam = torch.stack([tx, ty, tz], dim=-1)   
    return full_cam   # [bs, 3]

def get_mano_mesh_proj(betas, poses, cam_trans, focal_length, img_shape):
    shp = betas if torch.is_tensor(betas) else torch.tensor(betas).float()
    ps = poses if torch.is_tensor(poses) else torch.tensor(poses).float()
    ct = cam_trans if torch.is_tensor(cam_trans) else torch.tensor(cam_trans).float()
    # cp = cam_param if torch.is_tensor(cam_param) else torch.tensor(cam_param).float()
    # ce = center if torch.is_tensor(center) else torch.tensor(center).float()
    # bs = b_scale if torch.is_tensor(b_scale) else torch.tensor(b_scale).float()
    fl = focal_length if torch.is_tensor(focal_length) else torch.tensor(focal_length).float()
    imsh = img_shape if torch.is_tensor(img_shape) else torch.tensor(img_shape).float()
        
    rroot_pose, rhand_pose, rshape, rcam_trans = ps[:, 0, :3], ps[:, 0, 3:], shp[:, 0, :], ct[:, 0, :]
    lroot_pose, lhand_pose, lshape, lcam_trans = ps[:, 1, :3], ps[:, 1, 3:], shp[:, 1, :], ct[:, 1, :]
    rout = rmano_layer.to(betas.device)(global_orient=rroot_pose, hand_pose=rhand_pose, betas=rshape)
    #rout = rmano_layer(global_orient=rroot_pose, hand_pose=rhand_pose, betas=rshape, transl=rcam_trans)
    rmesh_proj, rjoint_proj, rmesh_cam, rjoint_cam = get_coord(rout.vertices.to(torch.float32), rcam_trans, fl, imsh)    # [bs, 778, 2], [bs, 21, 2]
    lout = lmano_layer.to(betas.device)(global_orient=lroot_pose, hand_pose=lhand_pose, betas=lshape)
    #lout = lmano_layer(global_orient=lroot_pose, hand_pose=lhand_pose, betas=lshape, transl=lcam_trans)
    lmesh_proj, ljoint_proj, lmesh_cam, ljoint_cam = get_coord(lout.vertices.to(torch.float32), lcam_trans, fl, imsh)    # [bs, 778, 2], [bs, 21, 2]
    
    mesh_proj = torch.cat([rmesh_proj, lmesh_proj], 1)
    joint_proj = torch.cat([rjoint_proj, ljoint_proj], 1)
    mesh_cam = torch.cat([rmesh_cam, lmesh_cam], 1)
    joint_cam = torch.cat([rjoint_cam, ljoint_cam], 1)

    return mesh_proj, joint_proj, mesh_cam, joint_cam   # [bs, 778*2, 2], [bs, 42, 2], [bs, 778*2, 3], [bs, 42, 3]

def get_valid_verts_faces(verts, faces, hand_type):
    # verts:[bs, 778*2, 3]
    # faces:[bs, nf*2, 3]
    # hand_type:[bs, 2]
    right_set = ((hand_type[:, 0] == 1.0) & (hand_type[:, 1] == 0.0)).unsqueeze(-1).unsqueeze(-1)
    left_set = ((hand_type[:, 0] == 0.0) & (hand_type[:, 1] == 1.0)).unsqueeze(-1).unsqueeze(-1)
    interaction_set = ((hand_type[:, 0] == 1.0) & (hand_type[:, 1] == 1.0)).unsqueeze(-1).unsqueeze(-1)
    # print(interaction_set.shape)
    
    half_nv = verts.shape[1] // 2
    half_nf = faces.shape[1] // 2
    rverts = verts[:, :half_nv, :]
    lverts = verts[:, half_nv:, :]
    rfaces = faces[:, :half_nf, :]
    lfaces = faces[:, half_nf:, :]
    
    device = verts.device
    right_verts = torch.cat([rverts.clone(), torch.zeros_like(rverts).to(device)], 1)
    left_verts = torch.cat([lverts.clone(), torch.zeros_like(lverts).to(device)], 1)
    interaction_verts = torch.cat([rverts.clone(), lverts.clone()], 1)
    right_faces = torch.cat([rfaces.clone(), -1*torch.ones_like(rfaces).to(device)], 1)
    left_faces = torch.cat([lfaces.clone(), -1*torch.ones_like(lfaces).to(device)], 1)
    interaction_faces = torch.cat([rfaces.clone(), lfaces.clone() + half_nv], 1)
    # print(interaction_verts.shape)
    valid_verts = right_set * right_verts + left_set * left_verts + interaction_set * interaction_verts
    valid_faces = right_set * right_faces + left_set * left_faces + interaction_set * interaction_faces
    
    return valid_verts, valid_faces

def get_rm_preds(first_pred, better_pred, param_to_train, com_num, focal_length, img_shape, trans_rot2crop, faces):
    rm_preds = {}
    for k, v in first_pred.items():
        one_pred = v.clone()
        rm_preds[k] = one_pred.unsqueeze(1).repeat_interleave(com_num, dim=1)
    
    for k, v in first_pred.items():
        rm_preds[k][:, 0] = v.clone()
    
    for k, v in param_to_train.items():
        rm_k = 'pred_' + k[6:]
        # rm_preds[rm_k][:, 1] = first_pred[rm_k].clone()*(1 + (v * 2 - 1) * 0.8)
        rm_preds[rm_k][:, 1] = first_pred[rm_k].clone() + v
    
    if 'jproj' in rm_cfg.recons_ls or 'mask' in rm_cfg.recons_ls:
        if 'param_mano_cam' not in param_to_train.keys():
            pred_mano_mesh_proj, pred_joint_proj, pred_mano_cam, _ = get_mano_mesh_proj(rm_preds['pred_mano_shape'][:, 1], rm_preds['pred_mano_pose'][:, 1], rm_preds['pred_cam_transl'][:, 1], \
                focal_length, img_shape)
            pred_mano_mesh_crop_proj = trans_bs_poinst2d(pred_mano_mesh_proj, trans_rot2crop)     # [1, 778*2, 2]
            pred_joint_crop_proj = trans_bs_poinst2d(pred_joint_proj, trans_rot2crop)             # [1, 21*2, 2]
            rm_preds['pred_mano_mesh_crop_proj'][:, 1] = pred_mano_mesh_crop_proj.to(torch.float32)
            rm_preds['pred_joint_crop_coord'][:, 1] = pred_joint_crop_proj.to(torch.float32)
            rm_preds['pred_mano_cam'][:, 1] = pred_mano_cam.to(torch.float32)
        else:
            half_nv = rm_preds['pred_mano_cam'].shape[2] // 2
            rmano_cam = rm_preds['pred_mano_cam'][:, 1, :half_nv, :].clone() / 1000  # [bs, 778, 3], mm->m
            lmano_cam = rm_preds['pred_mano_cam'][:, 1, half_nv:, :].clone() / 1000  # [bs, 778, 3], mm->m
            rcam_trans = rm_preds['pred_cam_transl'][:, 1, 0, :].clone()        # [bs, 3]
            lcam_trans = rm_preds['pred_cam_transl'][:, 1, 1, :].clone()        # [bs, 3]
            rmesh_proj, rjoint_proj, rmesh_cam, _ = get_coord(rmano_cam, rcam_trans, focal_length.clone(), img_shape.clone())
            lmesh_proj, ljoint_proj, lmesh_cam, _ = get_coord(lmano_cam, lcam_trans, focal_length.clone(), img_shape.clone())
            mesh_proj = torch.cat([rmesh_proj, lmesh_proj], 1)      # [bs, 778*2, 2]
            joint_proj = torch.cat([rjoint_proj, ljoint_proj], 1)   # [bs, 21*2, 2]
            rm_preds['pred_mano_mesh_crop_proj'][:, 1] = trans_bs_poinst2d(mesh_proj, trans_rot2crop)
            rm_preds['pred_joint_crop_coord'][:, 1] = trans_bs_poinst2d(joint_proj, trans_rot2crop)
            pred_mano_cam = torch.cat([rmesh_cam, lmesh_cam], 1)
    if 'mask' in rm_cfg.recons_ls:
        hand_type = first_pred['pred_hand_type']  # [bs, 2]
        mesh_verts, mesh_faces = get_valid_verts_faces(pred_mano_cam.clone(), faces.clone(), hand_type.clone())
        rm_preds['pred_mesh_verts'][:, 1] = mesh_verts
        rm_preds['pred_mesh_faces'][:, 1] = mesh_faces
    
    return rm_preds

def optimize_by_rank(cliff_inputs, cliff_targets, cliff_meta_info, cliff_model, rm_model, end_step, lr, log_path=None):

    with torch.no_grad():
        out = cliff_model(cliff_inputs, cliff_targets, cliff_meta_info, 'test')
    
    hand_type = (out['hand_type'].detach() >= 0.5).float()             # [1, 2]
    mano_shape = out["mano_shape"].detach().to(torch.float32)          # [1, 2, 10]
    mano_pose = out['mano_pose'].detach().to(torch.float32)            # [1, 2, 48]
    cam_transl = out['cam_transl'].detach().to(torch.float32)          # [1, 2, 3]
    # cam_param = out['pred_cam_param'].detach()                         # [1, 2, 3]
    #cam_transl = cliff_meta_info['gt_cam_transl']                      # [1, 2, 3]

    device = hand_type.device
    trans_rot2crop = cliff_meta_info['trans_rot2crop'].to(device)
    center = cliff_inputs['center'].to(device)           # [bs, 2]
    b_scale = cliff_inputs['b_scale'].to(device)         # [bs,]
    focal_length = cliff_inputs['focal_length'].to(device)
    img_shape = cliff_inputs['img_shape'].to(device)     # [bs, 2]
    
    rm_inputs = {'img': cliff_inputs['img'], 'img_shape': cliff_inputs['img_shape']}
    if 'mask' in rm_cfg.recons_ls:
        rm_inputs['mask'] = cliff_inputs['mask']
        princpt = torch.zeros_like(img_shape).to(device)
        princpt[:, 0] = img_shape[:, 1] // 2
        princpt[:, 1] = img_shape[:, 0] // 2
        render_princpt = trans_bs_poinst2d(princpt.unsqueeze(1), trans_rot2crop).squeeze(1)
        render_focal_length = torch.zeros_like(focal_length).to(device)
        scale = cliff_meta_info['scale'].to(device)   # [bs,]
        bbox = cliff_meta_info['bbox'].to(device)     # [bs, 4]
        render_focal_length[:, 0] = focal_length[:, 0] * rm_cfg.input_img_shape[1] / (scale * bbox[:, 2])
        render_focal_length[:, 1] = focal_length[:, 1] * rm_cfg.input_img_shape[0] / (scale * bbox[:, 3])
        rm_inputs['render_focal_length'] = render_focal_length.to(torch.float32)
        rm_inputs['render_princpt'] = render_princpt.to(torch.float32)
        
    rm_gt_loss = {}
    
    bs = hand_type.shape[0]
    com_num = 2
    better_pred_mano_shape = mano_shape.clone()    # [1, 2, 10]
    better_pred_mano_pose = mano_pose.clone()      # [1, 2, 48]
    better_pred_cam_transl = cam_transl.clone()    # [1, 2, 3]
    # better_pred_cam_param = cam_param.clone()      # [1, 2, 3]
    # better_pred_cam_transl = torch.zeros_like(better_pred_cam_param).to(device)    # [1, 2, 3]
    # for i in range(2):
    #     better_pred_cam_transl[:, i] = get_camtrans(better_pred_cam_param[:, i], center.clone(), b_scale.clone(), focal_length.clone(), img_shape.clone())
    first_mano_mesh_proj, first_joint_proj, first_mano_cam, first_joint_cam = get_mano_mesh_proj(better_pred_mano_shape, better_pred_mano_pose, better_pred_cam_transl, \
        focal_length, img_shape)   # [1, 778*2, 2]
    if 'jproj' in rm_cfg.recons_ls:
        first_mano_mesh_crop_proj = trans_bs_poinst2d(first_mano_mesh_proj, trans_rot2crop)     # [1, 778*2, 2]
        better_pred_mano_mesh_crop_proj = first_mano_mesh_crop_proj.clone()
        first_joint_crop_proj = trans_bs_poinst2d(first_joint_proj, trans_rot2crop)             # [1, 21*2, 2]
        better_pred_joint_crop_proj = first_joint_crop_proj.clone()
    rfaces = torch.tensor(rmano_layer.faces.copy().astype(np.float32)).float().to(device)
    lfaces = torch.tensor(lmano_layer.faces.copy().astype(np.float32)).float().to(device)
    faces = torch.cat([rfaces, lfaces], 0).unsqueeze(0).repeat(bs, 1, 1)  # [bs, nf*2, 3]
    
    # param_to_train = {
    #         "param_mano_shape": torch.nn.Parameter(torch.rand(mano_shape.shape, out=None).to(device), requires_grad=True),
    #         'param_mano_pose': torch.nn.Parameter(torch.rand(mano_pose.shape, out=None).to(device), requires_grad=True),
    #         'param_cam_transl': torch.nn.Parameter(torch.rand(cam_transl.shape, out=None).to(device), requires_grad=True),
    #     }
    param_to_train = {
            "param_mano_shape": torch.nn.Parameter(torch.zeros(mano_shape.shape).to(device), requires_grad=True),
            'param_mano_pose': torch.nn.Parameter(torch.zeros(mano_pose.shape).to(device), requires_grad=True),
            'param_cam_transl': torch.nn.Parameter(torch.zeros(cam_transl.shape).to(device), requires_grad=True),
        }
    # if 'mano' in rm_cfg.recons_ls:
    #     # param_to_train = {
    #     #     "param_mano_shape": torch.nn.Parameter(torch.rand(mano_shape.shape, out=None).to(device), requires_grad=True),
    #     #     'param_mano_pose': torch.nn.Parameter(torch.rand(mano_pose.shape, out=None).to(device), requires_grad=True),
    #     #     #'param_cam_transl': torch.nn.Parameter(torch.rand(cam_transl.shape, out=None).to(device), requires_grad=True),
    #     #     'param_cam_param': torch.nn.Parameter(torch.rand(cam_param.shape, out=None).to(device), requires_grad=True)
    #     # }
    #     param_to_train = {
    #         "param_mano_shape": torch.nn.Parameter(torch.ones(mano_shape.shape).to(device) * 0.5, requires_grad=True),
    #         'param_mano_pose': torch.nn.Parameter(torch.ones(mano_pose.shape).to(device) * 0.5, requires_grad=True),
    #         'param_cam_transl': torch.nn.Parameter(torch.ones(cam_transl.shape).to(device) * 0.5, requires_grad=True),
    #     }
    # else:
    #     param_to_train = {
    #         "param_mano_cam": torch.nn.Parameter(torch.ones(first_mano_cam.shape).to(device) * 0.5, requires_grad=True),
    #     }
    
    better_pred = {}
    better_pred.update({
        'pred_mano_shape': better_pred_mano_shape, 
        'pred_mano_pose': better_pred_mano_pose,
        'pred_cam_transl': better_pred_cam_transl,
    })
    if 'jproj' in rm_cfg.recons_ls or 'mask' in rm_cfg.recons_ls:
        if 'param_mano_cam' not in param_to_train.keys():
            pred_mano_cam = first_mano_cam
        else:
            half_nv = first_mano_cam.shape[1] // 2
            rpred_mano_cam = first_mano_cam[:, :half_nv, :]
            rpred_mano_cam = rpred_mano_cam - rpred_mano_cam[:, 0, None, :]
            lpred_mano_cam = first_mano_cam[:, half_nv:, :]
            lpred_mano_cam = lpred_mano_cam - lpred_mano_cam[:, 0, None, :]
            pred_mano_cam = torch.cat([rpred_mano_cam, lpred_mano_cam], 1)  # mm
        better_pred.update({
            'pred_mano_mesh_crop_proj': better_pred_mano_mesh_crop_proj.to(torch.float32),
            'pred_joint_crop_coord': better_pred_joint_crop_proj.to(torch.float32),
            'pred_mano_cam': pred_mano_cam.to(torch.float32)
        })
    if 'mask' in rm_cfg.recons_ls:
        mesh_verts, mesh_faces = get_valid_verts_faces(first_mano_cam.clone(), faces.clone(), hand_type.clone())
        better_pred.update({
            'pred_mesh_verts': mesh_verts,
            'pred_mesh_faces': mesh_faces,
        })
    
    first_pred = {'pred_hand_type': hand_type.clone()}
    for k, v in better_pred.items():
        first_pred[k] = v.clone()
        
    opt_param = [{'params': v} for k, v in param_to_train.items()]
    optimizer = torch.optim.Adam(opt_param, lr=lr)
    
    if log_path is not None:
        tot_timer = Timer()
        gpu_timer = Timer()
        read_timer = Timer()
        
        # logger
        logger = colorlogger(log_path)
        
        tot_timer.tic()
        read_timer.tic()
        
        better_pred_joint_crop_proj_ls = [better_pred_joint_crop_proj]
    for step in range(end_step):
        if log_path is not None:
            read_timer.toc()
            gpu_timer.tic()

        # if cur_loss <= 0.5 and not dec_lr:
        #     now_lr = now_lr / 10
        #     dec_lr = True
        #     for g in optimizer.param_groups:
        #         g['lr'] = now_lr
        optimizer.zero_grad()
        
        rm_preds = get_rm_preds(first_pred, better_pred, param_to_train, com_num, \
            focal_length.clone(), img_shape.clone(), trans_rot2crop.clone(), faces.clone())
        
        rm_out = rm_model(rm_inputs, rm_preds, rm_gt_loss, 'test')

        prob = (1- (rm_out['prob'] + 0.5))     # [1,]
        # mask_losses = rm_out['mask_losses']    # [1, com_num]
        # prob_from_mask = torch.nn.functional.sigmoid(mask_losses[:, 0] - mask_losses[:, 1])  # [1,]
        # prob_from_mask = 1 - prob_from_mask    # [1,]
        sum(prob).backward()
        # print(rm_out['pred_ori_mask'].grad[0, :, 0, 100:200, 100:200])
        # print(rm_out['mask_losses'].grad)
        # print(rm_out['prob'].grad.sum())
        # print(param_to_train['param_mano_shape'].grad)
        # print(param_to_train['param_mano_pose'].grad)
        optimizer.step()
        if log_path is not None:
            gpu_timer.toc()
        
        bs_id=torch.arange(0, bs).to(device)
        better_id = torch.LongTensor(rm_out['label'].detach().cpu().numpy())  # [1,]
        for k in better_pred.keys():
            better_pred[k] = rm_preds[k].detach()[bs_id, better_id].clone()
            
        if log_path is not None:
            screen = [
                'Step %d/%d:' % (step, end_step),
                'lr: %g' % (lr),
                'speed: %.2f(%.2fs r%.2f)s/itr' % (
                    tot_timer.average_time, gpu_timer.average_time, read_timer.average_time),
                '%.2fh/Optimize' % (tot_timer.average_time / 3600. * end_step),
            ]
            screen += ['prob: %.4f' % ((prob).detach().cpu().numpy()[0])]
            logger.info(' '.join(screen))
            tot_timer.toc()
            tot_timer.tic()
            read_timer.tic()
            better_pred_joint_crop_proj_ls.append(better_pred['pred_joint_crop_coord'])
            
        # cur_loss = prob.detach().cpu().numpy()[0]
        
        # if better_id[0] == 1:
        #     break
    # for k, v in param_to_train.items():
    #     print(k, v)
    if 'param_mano_cam' not in param_to_train.keys():
        final_mano_mesh_proj, final_joint_proj, final_mesh_cam, final_joint_cam = get_mano_mesh_proj(better_pred['pred_mano_shape'], better_pred['pred_mano_pose'], better_pred['pred_cam_transl'], \
            focal_length, img_shape)   # [1, 778*2, 2],[1, 42, 2]
        final_mano_mesh_crop_proj = trans_bs_poinst2d(final_mano_mesh_proj, trans_rot2crop)
    else:
        half_nv = better_pred['pred_mano_cam'].shape[1] // 2
        rmano_cam = better_pred['pred_mano_cam'][:, :half_nv, :].clone() / 1000   # [bs, 778, 3]
        lmano_cam = better_pred['pred_mano_cam'][:, half_nv:, :].clone() / 1000   # [bs, 778, 3]
        rcam_trans = better_pred['pred_cam_transl'][:, 0, :].clone()         # [bs, 3]
        lcam_trans = better_pred['pred_cam_transl'][:, 1, :].clone()         # [bs, 3]
        rmesh_proj, rjoint_proj, rmesh_cam, rjoint_cam = get_coord(rmano_cam, rcam_trans, focal_length.clone(), img_shape.clone())
        lmesh_proj, ljoint_proj, lmesh_cam, ljoint_cam = get_coord(lmano_cam, lcam_trans, focal_length.clone(), img_shape.clone())
        mesh_proj = torch.cat([rmesh_proj, lmesh_proj], 1)      # [bs, 778*2, 2]
        final_joint_proj = torch.cat([rjoint_proj, ljoint_proj], 1)   # [bs, 21*2, 2]
        final_mesh_cam = torch.cat([rmesh_cam, lmesh_cam], 1)         # [bs, 778*2, 2]
        final_joint_cam = torch.cat([rjoint_cam, ljoint_cam], 1)      # [bs, 21*2, 2]
        final_mano_mesh_crop_proj = trans_bs_poinst2d(mesh_proj, trans_rot2crop)
    
    out_first_pred = {
        'mano_mesh_crop_proj': first_mano_mesh_crop_proj,
        'mano_cam': first_mano_cam,
        'joint_proj': first_joint_proj,
        'joint_cam': first_joint_cam,
    }
    out_final_pred = {
        'mano_mesh_crop_proj': final_mano_mesh_crop_proj,
        'mano_cam': final_mesh_cam,
        'joint_proj': final_joint_proj,
        'joint_cam': final_joint_cam,
    }
    
    if 'mask' in rm_cfg.recons_ls:
        cam_param = {
            'focal': rm_inputs['render_focal_length'],
            'princpt': rm_inputs['render_princpt']
        }
        _, out_first_pred['pred_mask'] = render_mesh(rm_inputs['img'].permute(0, 2, 3, 1), first_pred['pred_mesh_verts'], first_pred['pred_mesh_faces'], cam_param)
        _, out_final_pred['pred_mask'] = render_mesh(rm_inputs['img'].permute(0, 2, 3, 1), better_pred['pred_mesh_verts'], better_pred['pred_mesh_faces'], cam_param)
        out_first_pred['pred_mask'] = torch.FloatTensor(out_first_pred['pred_mask']).unsqueeze(1).to(device)    # [bs, 1, H, W]
        out_final_pred['pred_mask'] = torch.FloatTensor(out_final_pred['pred_mask']).unsqueeze(1).to(device)    # [bs, 1, H, W]
    
    if log_path is not None:
        result2save = {
            'inputs': rm_inputs,
            'preds': rm_preds,
            'first_pred':{
                'pred_hand_type': hand_type, 
                'pred_mano_mesh_crop_proj': first_mano_mesh_crop_proj,
                'pred_joint_crop_coord': first_joint_crop_proj,
                'pred_mano_shape': mano_shape, 
                'pred_mano_pose': mano_pose,
                'pred_cam_transl': cam_transl,
            },
            'meta_info': {
                'focal_length': focal_length,
                'img_shape': img_shape,
                'trans_rot2crop': trans_rot2crop,
                'center': center,
                'b_scale': b_scale
            }
        }
        torch.save(result2save, os.path.join(log_path, "opt_results.pth"))
        
        return out_first_pred, out_final_pred, hand_type, better_pred_joint_crop_proj_ls
    
    return out_first_pred, out_final_pred, hand_type
    

def main():
    # argument parse and create log
    args = parse_args()
    rm_model_path = os.path.abspath(current_path + f'/../../{args.rm_name}_cliff10_InterHand2.6M_0.1/model_dump/snapshot_{args.rm_epoch}.pth.tar')
    rm_cfg.set_args(args.gpu_ids)
    rm_ckpt = torch.load(rm_model_path)
    rm_cfg_dict = rm_ckpt['cfg_dict_to_save']
    rm_cfg.set_args(args.gpu_ids, **rm_cfg_dict)
    
    rm_model = get_rm_model('test', rm_cfg.pretrain, rm_cfg.pret_model_path)
    rm_model = DataParallel(rm_model).cuda()
    rm_model.load_state_dict(rm_ckpt['network'])
    print('Load rm checkpoint from {}'.format(rm_model_path))
    for name, param in rm_model.named_parameters():
        param.requires_grad = False
    rm_model.eval()
    
    cliff_model_path = os.path.abspath(current_path + f'/../../{args.cliff_name}_InterHand2.6M_0.1/model_dump/snapshot_{args.cliff_epoch}.pth.tar')
    cliff_ckpt = torch.load(cliff_model_path)
    cliff_cfg_dict = cliff_ckpt['cfg_dict_to_save']
    cfg.set_args(args.gpu_ids, **cliff_cfg_dict)
    joint_num = 21
    cliff_model = get_cliff_model('test', joint_num)
    cliff_model = DataParallel(cliff_model).cuda()
    cliff_model.load_state_dict(cliff_ckpt['network'])
    print('Load cliff checkpoint from {}'.format(cliff_model_path))
    cliff_model.eval()
    cudnn.benchmark = True
    
    joint_type = {'right': np.arange(0,joint_num), 'left': np.arange(joint_num,joint_num*2)}
    skeleton = load_skeleton(os.path.abspath(current_path + '/../../data/InterHand2.6M/annotations/skeleton.txt'), joint_num*2)
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
    save_img_ = img.copy()
    img = transform(img.astype(np.float32))/255
    img = img.cuda()[None,:,:,:]
    
    center = torch.Tensor([bbox[0] + 0.5*bbox[2], bbox[1] + 0.5*bbox[3]]).float().to(img.device).unsqueeze(0)  # [1, 2]
    b_scale = torch.Tensor([estimate_focal_length(bbox[3], bbox[2])]).float().to(img.device)  # [1,]
    img_shape_np = original_img.shape[:2] if cfg.crop2full else cfg.input_img_shape
    if cfg.crop2full:    
        focal_length_np = estimate_focal_length(img_shape_np[0], img_shape_np[1])    
    else:
        focal_length_np = cfg.focal_length
    img_shape = torch.Tensor(img_shape_np).float().to(img.device).unsqueeze(0)   # [1, 2]
    focal_length = torch.Tensor([focal_length_np, focal_length_np]).float().to(img.device).unsqueeze(0)   # [1, 2]
    trans_rot2crop_np = gen_trans_from_patch_cv(bbox[0] + 0.5*bbox[2], bbox[1] + 0.5*bbox[3], bbox[2], bbox[3], int(rm_cfg.input_img_shape[1]), int(rm_cfg.input_img_shape[0]), 1.0, 0)   # [2, 3]
    trans_rot2crop_ts = torch.tensor(trans_rot2crop_np).float().unsqueeze(0).to(img.device)     # [1, 2, 3]
    
    # forward
    inputs = {'img': img, 'center': center, 'b_scale': b_scale, 'focal_length': focal_length, 'img_shape': img_shape}
    targets = {}
    meta_info = {'trans_rot2crop': trans_rot2crop_ts, 
                 'scale':torch.ones(1,).to(img.device), 
                 'bbox':torch.tensor(bbox).float().unsqueeze(0).to(img.device)}
    if 'mask' in rm_cfg.recons_ls:
        mask_path = img_path[:-4] + '_mask.jpg'
        mask = cv2.imread(mask_path)      # [H_ori, W_ori, 3]
        mask = cv2.warpAffine(mask, trans, (int(rm_cfg.input_img_shape[1]), int(rm_cfg.input_img_shape[0])), flags=cv2.INTER_LINEAR)
        mask = transform(mask.astype(np.float32)) / 255.  # [3, H, W], 0 or 1
        # save_img = save_img * mask.permute(1,2,0).numpy()
        mask = torch.mean(mask, dim=0)                         # [H, W], 0 or 1
        mask = (mask > 0.5).float()
        inputs['mask'] = mask.unsqueeze(0)
    
    # optimize
    first_pred, final_pred, hand_type, better_pred_joint_crop_proj_ls = optimize_by_rank(inputs, targets, meta_info, cliff_model, rm_model, args.end_step, args.lr, log_path='./opt')

    # vis results
    joint_valid = np.zeros((joint_num*2), dtype=np.float32)
    right_exist = False
    if hand_type.cpu().numpy()[0, 0] > 0.5: 
        right_exist = True
        joint_valid[joint_type['right']] = 1
    left_exist = False
    if hand_type.cpu().numpy()[0, 1] > 0.5:
        left_exist = True
        joint_valid[joint_type['left']] = 1
    
    first_joint_proj = first_pred['joint_proj']
    final_joint_proj = final_pred['joint_proj']
    first_mano_mesh_crop_proj = first_pred['mano_mesh_crop_proj']
    final_mano_mesh_crop_proj = final_pred['mano_mesh_crop_proj']
    first_mano_cam = first_pred['mano_cam']
    final_mesh_cam = final_pred['mano_cam']
    
    # vis joint proj in original image
    all_jkpts = np.stack([first_joint_proj[0].cpu().numpy(), final_joint_proj[0].detach().cpu().numpy()], 0)
    vis_keypoints_together(original_img.copy().transpose(2,0,1), all_jkpts, joint_valid, skeleton, 'j_proj_together.jpg', save_path='./opt/')
    # # vis mesh_verts proj in crop image 
    # all_vkpts = np.stack([first_mano_mesh_crop_proj[0].cpu().numpy(), final_mano_mesh_crop_proj[0].detach().cpu().numpy()], 0)
    # vis_meshverts_together(save_img_, all_vkpts, hand_type[0].cpu().numpy(), 'mverts_proj_together.jpg', save_path='./opt/')
    # vis mesh
    mano_mesh_cam = final_mesh_cam[0]
    mesh = mano_mesh_cam / 1000  # milimeter to meter
    rfaces = rmano_layer.faces
    lfaces = lmano_layer.faces
    rverts = mesh[:len(mesh) // 2].detach().cpu().numpy()
    lverts = mesh[len(mesh) // 2:].detach().cpu().numpy()
    verts1 = []
    faces1 = []

    if right_exist:
        verts1.append(rverts)
        faces1.append(rfaces)
    if left_exist:
        verts1.append(lverts)
        if right_exist:
            lfaces += len(mesh) // 2
        faces1.append(lfaces)
    verts1 = np.concatenate(verts1, 0)
    faces1 = np.concatenate(faces1, 0)
    verts1_ts = torch.tensor(verts1).float().to(img.device)
    faces1_ts = torch.tensor(faces1.astype(np.float32)).float().to(img.device)
    mesh1 = trimesh.Trimesh(verts1, faces1)
    mesh1.export('./opt/mesh1.obj')
    
    mesh = first_mano_cam[0] / 1000  # milimeter to meter
    rverts = mesh[:len(mesh) // 2].cpu().numpy()
    lverts = mesh[len(mesh) // 2:].cpu().numpy()
    verts0 = []
    faces0 = faces1
    faces0_ts = faces1_ts.clone()
    if right_exist:
        verts0.append(rverts)
    if left_exist:
        verts0.append(lverts)
    verts0 = np.concatenate(verts0, 0)
    verts0_ts = torch.tensor(verts0).float().to(img.device)
    mesh0 = trimesh.Trimesh(verts0, faces0)
    mesh0.export('./opt/mesh0.obj')
    
    # vis mesh_verts proj in original image 
    cam_param = {'focal': focal_length[0],
             'princpt': torch.tensor([img_shape_np[1] // 2, img_shape_np[0] // 2]).float().to(img.device)}
    save_img0, _ = render_mesh(original_img.copy(), verts0_ts, faces0_ts, cam_param)    # [H, W, 3]
    save_img1, _ = render_mesh(original_img.copy(), verts1_ts, faces1_ts, cam_param)    # [H, W, 3]
    save_img = np.concatenate([save_img0, save_img1], axis=1)    # [H, 2*W, 3]
    save_img = Image.fromarray(save_img.astype('uint8')) 
    save_img.save(osp.join('./opt/', 'mverts_proj_together.jpg'))
    
    # vis opt process
    opt_process = []
    for i, bj in enumerate(better_pred_joint_crop_proj_ls):
        render_img = vis_keypoints_together(img[0].detach().cpu().numpy()[::-1, :, :] * 255., bj.detach().cpu().numpy(),\
        joint_valid, skeleton, None)
        opt_process.append(render_img.astype('uint8'))
    
    fps = 10
    size = opt_process[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter("./opt/opt_process.mp4",  fourcc, fps, size, True)
    for i in range(len(opt_process)):
        video.write(opt_process[i])
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()