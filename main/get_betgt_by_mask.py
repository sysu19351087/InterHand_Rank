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
from torch.utils.data import DataLoader
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
from utils.vis import vis_keypoints_together, vis_meshverts_together, render_mesh, vis_mask_overlap
from betgt_dataset import Detgt_Dataset
from nets.rm_module import Render_Mask
from optimize import optimize_by_rank, rmano_layer, lmano_layer, get_mano_mesh_proj, get_valid_verts_faces

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--set', type=str, dest='set')
    
    parser.add_argument('--end_step', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--test_id', type=int, default=-1)
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

def get_mano_coord(shapes, poses, transls, R, t):
    # R:[bs, 3, 3]
    # t:[bs, 3, 1]
    shp = shapes if torch.is_tensor(shapes) else torch.tensor(shapes).float()
    ps = poses if torch.is_tensor(poses) else torch.tensor(poses).float()
    ct = transls if torch.is_tensor(transls) else torch.tensor(transls).float()
        
    rroot_pose, rhand_pose, rshape, rcam_trans = ps[:, 0, :3], ps[:, 0, 3:], shp[:, 0, :], ct[:, 0, :]
    lroot_pose, lhand_pose, lshape, lcam_trans = ps[:, 1, :3], ps[:, 1, 3:], shp[:, 1, :], ct[:, 1, :]
    rout = rmano_layer.to(shapes.device)(global_orient=rroot_pose, hand_pose=rhand_pose, betas=rshape, transl=rcam_trans)
    rmesh_cam = rout.vertices.to(torch.float32) * 1000        # [bs, 778, 3]
    lout = lmano_layer.to(shapes.device)(global_orient=lroot_pose, hand_pose=lhand_pose, betas=lshape, transl=lcam_trans)
    lmesh_cam = lout.vertices.to(torch.float32) * 1000        # [bs, 778, 3]
    mesh_cam = torch.cat([rmesh_cam, lmesh_cam], 1)           # [bs, 778*2, 3]

    mesh_cam = (torch.matmul(R, mesh_cam.transpose(1, 2)) + t).transpose(1, 2)  # [bs, 778*2, 3]
    
    return mesh_cam

def main():
    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.gpu_ids)
    
    joint_num = 21
    cudnn.benchmark = True
    
    joint_type = {'right': np.arange(0,joint_num), 'left': np.arange(joint_num,joint_num*2)}
    skeleton = load_skeleton(os.path.abspath(current_path + '/../../data/InterHand2.6M/annotations/skeleton.txt'), joint_num*2)
    render_net = Render_Mask()
    render_net = DataParallel(render_net).cuda()
    
    set = args.set
    testset_loader = Detgt_Dataset(transforms.ToTensor(), set)

    final_param = {
        'mano_shape': [],
        'mano_pose': [],
        'cam_transl': [],
    }
    for idx in tqdm(range(testset_loader.__len__())):
        if args.debug and idx != args.test_id:
            continue
        out = testset_loader.__getitem__(idx)
        for k, v in out.items():
            out[k] = v.cuda()
        
        img = out['img']                  # [nc, 3, H, W]
        gt_masks = out['gt_masks']        # [nc, H, W]
        focal_lengths = out['focal_length']  # [nc, 2]
        princpts = out['princpt']            # [nc, 2]
        
        shape = out['shapes']          # [nc, 2, 10]
        pose = out['poses']            # [nc, 2, 48]
        transl = out['transls']        # [nc, 2, 3]
        hand_type = out['hand_type']   # [nc, 2]
        R = out['R']                   # [nc, 3, 3]
        t = out['t']                   # [nc, 3, 1]
        
        device = gt_masks.device
        param_to_train = {
            "param_mano_shape": torch.nn.Parameter(torch.zeros(2, 10).to(device), requires_grad=True),
            'param_mano_pose': torch.nn.Parameter(torch.zeros(2, 48).to(device), requires_grad=True),
            'param_cam_transl': torch.nn.Parameter(torch.zeros(2, 3).to(device), requires_grad=True),
        }
        
        rfaces = torch.tensor(rmano_layer.faces.copy().astype(np.float32)).float().to(device)
        lfaces = torch.tensor(lmano_layer.faces.copy().astype(np.float32)).float().to(device)
        faces = torch.cat([rfaces, lfaces], 0).unsqueeze(0).repeat(gt_masks.shape[0], 1, 1)  # [nc, nf*2, 3]
        first_pred_mano_cam = get_mano_coord(shape.clone(), pose.clone(), transl.clone(), R, t)   # [nc, 778*2, 3]
        valid_verts, valid_faces = get_valid_verts_faces(first_pred_mano_cam, faces.clone(), hand_type.clone())
        
        better_pred = {}
        better_pred.update({
            'pred_mano_shape': shape.clone(), 
            'pred_mano_pose': pose.clone(),
            'pred_cam_transl': transl.clone(),
            'pred_mesh_verts': valid_verts,
            'pred_mesh_faces': valid_faces
        })
        
        first_pred = {'pred_hand_type': hand_type.clone()}
        for k, v in better_pred.items():
            first_pred[k] = v.clone()
        
        opt_param = [{'params': v} for k, v in param_to_train.items()]
        optimizer = torch.optim.Adam(opt_param, lr=args.lr)
        com_num = 2
        bs = gt_masks.shape[0]
        print(bs)
        for step in range(args.end_step):
            optimizer.zero_grad()
            rm_preds = {}
            for k, v in first_pred.items():
                one_pred = v.clone()
                rm_preds[k] = one_pred.unsqueeze(1).repeat_interleave(com_num, dim=1)
            
            for k, v in param_to_train.items():
                rm_k = 'pred_' + k[6:]
                rm_preds[rm_k][:, 1] = first_pred[rm_k].clone() + v.unsqueeze(0)
            
            pred_mano_cam = get_mano_coord(rm_preds['pred_mano_shape'][:, 1], rm_preds['pred_mano_pose'][:, 1], rm_preds['pred_cam_transl'][:, 1], \
                R, t)
            valid_verts, valid_faces = get_valid_verts_faces(pred_mano_cam, faces.clone(), hand_type.clone())
            rm_preds['pred_mesh_verts'][:, 1] = valid_verts
            rm_preds['pred_mesh_faces'][:, 1] = valid_faces
            
            _, ori_H, ori_W = gt_masks.shape
            pred_ori_mask = torch.zeros([bs, com_num, 1, ori_H, ori_W]).to(device)   # [bs, compare_num, 1, H, W]
            for cid in range(com_num):
                render_mask = render_net((ori_H, ori_W), rm_preds['pred_mesh_verts'][:, cid], rm_preds['pred_mesh_faces'][:, cid], \
                    focal_lengths.clone(), princpts.clone())   # [bs, 1, H, W]
                pred_ori_mask[:, cid] = render_mask.clone()
                
            mask_loss_ls = []
            for cid in range(com_num):
                pred_mk = pred_ori_mask[:, cid].clone()  # [bs, 1, H, W]
                mask_loss = torch.mean((gt_masks.unsqueeze(1) - pred_mk) ** 2, dim=(-1, -2, -3))  # [bs,]
                mask_loss_ls.append(mask_loss)
            mask_losses = torch.stack(mask_loss_ls, -1)  # [bs, compare_num]
            prob_from_mask = torch.sigmoid(mask_losses[:, 0] - mask_losses[:, 1])  # [bs,]
            prob = (1- prob_from_mask)
            sum(prob).backward()
            optimizer.step()
            
            bs_id=torch.arange(0, bs).to(device)
            better_id = torch.LongTensor((prob_from_mask >= 0.5).float().detach().cpu().numpy())  # [1,]
            for k in better_pred.keys():
                better_pred[k] = rm_preds[k].detach()[bs_id, better_id].clone()
        
        for k, v in final_param.items():
            bk = 'pred_' + k
            final_param[k].append(better_pred[bk][0].detach().cpu().numpy())
        
        if args.debug:
            cam_param = {
                'focal': focal_lengths,
                'princpt': princpts
            }
            _, first_mask = render_mesh(img.permute(0, 2, 3, 1), first_pred['pred_mesh_verts'], first_pred['pred_mesh_faces'], cam_param)
            _, final_mask = render_mesh(img.permute(0, 2, 3, 1), better_pred['pred_mesh_verts'], better_pred['pred_mesh_faces'], cam_param)
            # vis results
            file_name = f'{args.set}_{args.test_id}'
            file_dir = f'./betgt/{file_name}'
            os.makedirs(file_dir, exist_ok=True)
            
            # vis mesh
            mesh_verts = better_pred['pred_mesh_verts'][0]      # [nv*2, 3]
            mesh_faces = better_pred['pred_mesh_faces'][0]      # [nf*2, 3]
            if torch.sum(hand_type[0]) != 2:
                mesh_verts = mesh_verts[:len(mesh_verts)//2]
                mesh_faces = mesh_faces[:len(mesh_faces)//2]
            trimesh.Trimesh(mesh_verts, mesh_faces).export(file_dir + '/mesh.obj')
            # mano_mesh_cam = final_mesh_cam[0]
            # mesh = mano_mesh_cam / 1000  # milimeter to meter
            # rfaces = rmano_layer.faces
            # lfaces = lmano_layer.faces
            # rverts = mesh[:len(mesh) // 2].detach().cpu().numpy()
            # lverts = mesh[len(mesh) // 2:].detach().cpu().numpy()
            # verts1 = []
            # faces1 = []

            # if right_exist:
            #     verts1.append(rverts)
            #     faces1.append(rfaces)
            # if left_exist:
            #     verts1.append(lverts)
            #     if right_exist:
            #         lfaces += len(mesh) // 2
            #     faces1.append(lfaces)
            # verts1 = np.concatenate(verts1, 0)
            # faces1 = np.concatenate(faces1, 0)
            # verts1_ts = torch.tensor(verts1).float().to(device)
            # faces1_ts = torch.tensor(faces1.astype(np.float32)).float().to(device)
            # mesh1 = trimesh.Trimesh(verts1, faces1)
            # mesh1.export('./opt/mesh1.obj')
            
            # mesh = first_mano_cam[0] / 1000  # milimeter to meter
            # rverts = mesh[:len(mesh) // 2].cpu().numpy()
            # lverts = mesh[len(mesh) // 2:].cpu().numpy()
            # verts0 = []
            # faces0 = faces1
            # faces0_ts = faces1_ts.clone()
            # if right_exist:
            #     verts0.append(rverts)
            # if left_exist:
            #     verts0.append(lverts)
            # verts0 = np.concatenate(verts0, 0)
            # verts0_ts = torch.tensor(verts0).float().to(device)
            # mesh0 = trimesh.Trimesh(verts0, faces0)
            # mesh0.export('./opt/mesh0.obj')
            
            # vis opt process
            # opt_process = []
            # for i, bj in enumerate(better_pred_joint_crop_proj_ls):
            #     render_img = vis_keypoints_together(inputs['img'][0].detach().cpu().numpy()[::-1, :, :] * 255., bj.detach().cpu().numpy(),\
            #     joint_valid, skeleton, None)
            #     opt_process.append(render_img.astype('uint8'))
            
            # fps = 10
            # size = opt_process[0].shape[:2]
            # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # video = cv2.VideoWriter(f"./opt/opt_process_{file_name}.mp4",  fourcc, fps, size, True)
            # for i in range(len(opt_process)):
            #     video.write(opt_process[i])
            # video.release()
            # cv2.destroyAllWindows()
            
            # gt_ht = hand_type[:, None].repeat(21, axis=-1).reshape(-1)   # [42,] 
            # gt_joint_cam[:21, :] = gt_joint_cam[:21, :] - gt_joint_cam[20, None, :]
            # gt_joint_cam[21:, :] = gt_joint_cam[21:, :] - gt_joint_cam[41, None, :]
            # gt_joint_cam = gt_joint_cam * gt_ht[:, None]
            
            # first_joint_cam[:21, :] = first_joint_cam[:21, :] - first_joint_cam[20, None, :]
            # first_joint_cam[21:, :] = first_joint_cam[21:, :] - first_joint_cam[41, None, :]
            # first_joint_cam = first_joint_cam * gt_ht[:, None]

            # final_joint_cam[:21, :] = final_joint_cam[:21, :] - final_joint_cam[20, None, :]
            # final_joint_cam[21:, :] = final_joint_cam[21:, :] - final_joint_cam[41, None, :]
            # final_joint_cam = final_joint_cam * gt_ht[:, None]
            
            # print('first:', np.mean(np.abs(gt_joint_cam - first_joint_cam)))
            # print('final:', np.mean(np.abs(gt_joint_cam - final_joint_cam)))
            img = img.cpu().numpy().transpose(0, 2, 3, 1)        # [nc, 256, 256, 3]
            gm = (gt_masks.unsqueeze(-1).cpu().numpy().repeat(3, axis=-1) == 1.)         # [nc, 256, 256, 3]
            gt_mask2save = (gm * img * 255.).astype('uint8')

            first_iou_ls = []
            final_iou_ls = []
            for b in range(bs):
                first_mask2save, first_iou = vis_mask_overlap(gt_masks[b].cpu().numpy(), first_mask[b])
                final_mask2save, final_iou = vis_mask_overlap(gt_masks[b].cpu().numpy(), final_mask[b])
                first_iou_ls.append(first_iou)
                final_iou_ls.append(final_iou)
                mask2save = np.concatenate([gt_mask2save[b], first_mask2save, final_mask2save], axis=1)
                Image.fromarray(mask2save).save(f'{file_dir}/{b}.jpg')
            print('mean_first_iou:', sum(first_iou_ls)/len(first_iou_ls))
            print('mean_final_iou:', sum(final_iou_ls)/len(final_iou_ls))
        
    final_param = {k:np.stack(v, 0) for k, v in final_param.items()}
    # for v in final_param.values():
    #     print(v.shape)
    
if __name__ == "__main__":
    main()