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
from utils.vis import vis_keypoints_together, vis_meshverts_together, render_mesh, vis_mask_overlap
from rm_dataset import RM_Dataset
from optimize import optimize_by_rank, rmano_layer, lmano_layer

# smplx_path = os.path.abspath(current_path + '/../../smplx/models')
# rmano_layer = smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=True, create_transl=False)
# lmano_layer = smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=False, create_transl=False)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--test_img_id', type=int)
    parser.add_argument('--set', type=str)
    
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


def get_iou(gt_mask, pred_mask):
    # gt_mask:[bs, 1, 256, 256], pred_mask:[bs, 1, 256, 256]
    
    gm = (gt_mask == 1.0)
    pm = (pred_mask == 1.0)
    
    overlap = (gm & pm)
    combine = (gm | pm)
    iou = np.sum(overlap, axis=(-1, -2, -3)) / np.sum(combine, axis=(-1, -2, -3))
    return iou

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
    
    datapath = osp.join(rm_cfg.root_dir, '{}_{}_{}'.format(rm_cfg.run_name, rm_cfg.dataset, rm_cfg.train_ratio), 'result')
    set = args.set
    testset_loader = RM_Dataset(transforms.ToTensor(), datapath, set)
    
    inputs, preds, gt_loss = testset_loader.__getitem__(args.test_img_id)

    # forward
    inputs = {k:v.cuda().unsqueeze(0) for k, v in inputs.items()}
    # preds = {k:v.cuda().unsqueeze(0) for k, v in preds.items()}
    device = inputs['img'].device
    targets = {}
    idx = testset_loader.valid_idx_ls[args.test_img_id // testset_loader.all_pair_num]
    img_path = testset_loader.ins_ls[0]['img_path'][idx]
    bbox = testset_loader.ins_ls[0]['bbox'][idx]
    scale = testset_loader.ins_ls[0]['scale'][idx]
    do_flip = testset_loader.ins_ls[0]['do_flip'][idx]
    gt_joint_cam = testset_loader.gts_ls[0]['mano_joint_cam'][idx]   # [42, 3]
    gt_hand_type = testset_loader.gts_ls[0]['hand_type'][idx]        # [2,]
    meta_info = {'trans_rot2crop': inputs['trans_rot2crop'], 
                 'scale': scale*torch.ones(1,).to(device), 
                 'bbox': torch.tensor(bbox).float().unsqueeze(0).to(device),
                 'gt_cam_transl': inputs['gt_cam_transl']}
    
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
    first_joint_cam = first_pred['joint_cam'][0].cpu().numpy()   # [42, 3]
    final_joint_cam = final_pred['joint_cam'][0].cpu().numpy()   # [42, 3]
    first_mask = first_pred['pred_mask'].cpu().numpy()        # [bs, 1, 256, 256]
    final_mask = final_pred['pred_mask'].cpu().numpy()        # [bs, 1, 256, 256]
    file_name = f'{set}_{args.test_img_id}'
    
    # vis joint proj in original image
    img_path_spl = img_path.split('/')
    img_path_spl = img_path_spl[img_path_spl.index('images')+1:]
    img_path = '/'.join([testset_loader.img_path]+img_path_spl)
    original_img = load_img(img_path)           # [H_ori, W_ori, 3]
    H_ori, W_ori = original_img.shape[:2]
    if do_flip == 1:
        original_img = original_img[:, ::-1, :]
    trans_full2rot = inputs['trans_full2rot'][0].cpu().numpy()
    original_img = cv2.warpAffine(original_img, trans_full2rot, (int(W_ori), int(H_ori)), flags=cv2.INTER_LINEAR)
    all_jkpts = np.stack([first_joint_proj[0].cpu().numpy(), final_joint_proj[0].detach().cpu().numpy()], 0)
    vis_keypoints_together(original_img.copy().transpose(2,0,1), all_jkpts, joint_valid, skeleton, f'j_proj_together_{file_name}.jpg', save_path='./opt/')
    # # vis mesh_verts proj in crop image 
    # all_vkpts = np.stack([first_mano_mesh_crop_proj[0].cpu().numpy(), final_mano_mesh_crop_proj[0].detach().cpu().numpy()], 0)
    # vis_meshverts_together(save_img_, all_vkpts, hand_type[0].cpu().numpy(), 'mverts_proj_together.jpg', save_path='./opt/')
    # vis mesh
    mano_mesh_cam = final_mesh_cam[0]
    mesh = mano_mesh_cam  # milimeter
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
    verts1_ts = torch.tensor(verts1).float().to(device)
    faces1_ts = torch.tensor(faces1.astype(np.float32)).float().to(device)
    mesh1 = trimesh.Trimesh(verts1, faces1)
    mesh1.export('./opt/mesh1.obj')
    
    mesh = first_mano_cam[0]  # milimeter
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
    verts0_ts = torch.tensor(verts0).float().to(device)
    mesh0 = trimesh.Trimesh(verts0, faces0)
    mesh0.export('./opt/mesh0.obj')
    
    # vis mesh_verts proj in original image 
    cam_param = {'focal': inputs['focal_length'][0],
             'princpt': torch.tensor([inputs['img_shape'][0][1] // 2, inputs['img_shape'][0][0] // 2]).float().to(device)}
    save_img0, _ = render_mesh(original_img.copy(), verts0_ts, faces0_ts, cam_param)    # [H, W, 3]
    save_img1, _ = render_mesh(original_img.copy(), verts1_ts, faces1_ts, cam_param)    # [H, W, 3]
    save_img = np.concatenate([save_img0, save_img1], axis=1)    # [H, 2*W, 3]
    save_img = Image.fromarray(save_img.astype('uint8')) 
    save_img.save(osp.join('./opt/', f'mverts_proj_together_{file_name}.jpg'))
    
    # vis opt process
    opt_process = []
    for i, bj in enumerate(better_pred_joint_crop_proj_ls):
        render_img = vis_keypoints_together(inputs['img'][0].detach().cpu().numpy()[::-1, :, :] * 255., bj.detach().cpu().numpy(),\
        joint_valid, skeleton, None)
        opt_process.append(render_img.astype('uint8'))
    
    fps = 10
    size = opt_process[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f"./opt/opt_process_{file_name}.mp4",  fourcc, fps, size, True)
    for i in range(len(opt_process)):
        video.write(opt_process[i])
    video.release()
    cv2.destroyAllWindows()
    
    gt_ht = gt_hand_type[:, None].repeat(21, axis=-1).reshape(-1)   # [42,] 
    gt_joint_cam[:21, :] = gt_joint_cam[:21, :] - gt_joint_cam[20, None, :]
    gt_joint_cam[21:, :] = gt_joint_cam[21:, :] - gt_joint_cam[41, None, :]
    gt_joint_cam = gt_joint_cam * gt_ht[:, None]
    
    first_joint_cam[:21, :] = first_joint_cam[:21, :] - first_joint_cam[20, None, :]
    first_joint_cam[21:, :] = first_joint_cam[21:, :] - first_joint_cam[41, None, :]
    first_joint_cam = first_joint_cam * gt_ht[:, None]

    final_joint_cam[:21, :] = final_joint_cam[:21, :] - final_joint_cam[20, None, :]
    final_joint_cam[21:, :] = final_joint_cam[21:, :] - final_joint_cam[41, None, :]
    final_joint_cam = final_joint_cam * gt_ht[:, None]
    
    print('first:', np.mean(np.abs(gt_joint_cam - first_joint_cam)))
    print('final:', np.mean(np.abs(gt_joint_cam - final_joint_cam)))
    
    # compute iou:
    gt_mask = inputs['mask'][:, None].cpu().numpy()     # [bs, 1, 256, 256]
    first_iou = get_iou(gt_mask, first_mask)[0]
    final_iou = get_iou(gt_mask, final_mask)[0]
    print('first_iou:', first_iou)
    print('final_iou:', final_iou)
    
    img = inputs['img'][0].cpu().numpy().transpose(1, 2, 0)        # [256, 256, 3]
    gm = (gt_mask[0].transpose(1, 2, 0).repeat(3, axis=-1) == 1.)         # [256, 256, 3]
    gt_mask2save = (gm * img * 255.).astype('uint8')

    first_mask2save, _ = vis_mask_overlap(gt_mask[0, 0], first_mask[0, 0])
    final_mask2save, _ = vis_mask_overlap(gt_mask[0, 0], final_mask[0, 0])
    mask2save = np.concatenate([gt_mask2save, first_mask2save, final_mask2save], axis=1)
    Image.fromarray(mask2save).save(f'./opt/mask_{file_name}.jpg')
    
if __name__ == "__main__":
    main()