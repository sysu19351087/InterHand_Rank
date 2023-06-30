# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os.path

from tqdm import tqdm
import numpy as np
import cv2
from config import cfg
import torch
import glob
import os
import random
import sys
import copy
from torch.utils.data import DataLoader
current_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(current_path + '/../../common'))
sys.path.append(os.path.abspath(current_path + '/../../data/InterHand2.6M'))
from base import Tester
from dataset import Dataset
from utils.vis import vis_keypoints
import torch.backends.cudnn as cudnn
from utils.transforms import flip
from torch.nn.parallel.data_parallel import DataParallel
import torchvision.transforms as transforms
from model import get_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--test_epoch', type=int, dest='test_epoch')
    parser.add_argument('--set', type=str, dest='set')
    parser.add_argument('--train_ratio', type=float, default=1)
    parser.add_argument('--init_zeros', dest='init_zeros', action='store_true')
    parser.add_argument('--crop', dest='crop', action='store_true')
    parser.add_argument('--backbone', type=str, default='hr48')
    #parser.add_argument('--mean_loss', dest='mean_loss', action='store_true')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

def main():

    args = parse_args()
    cfg.set_args(args.gpu_ids, train_ratio=args.train_ratio, run_name=args.run_name, backbone=args.backbone,\
        init_zeros=args.init_zeros, crop=args.crop)
    cudnn.benchmark = True
    
    if cfg.dataset == 'InterHand2.6M':
        assert args.set, 'Test set is required. Select one of test/val'
    else:
        args.test_set = 'test'

    # tester = Tester(epoch)
    # tester._make_batch_generator(args.set)
    #tester._make_model()
    testset_loader = Dataset(transforms.ToTensor(), args.set)
    batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.num_gpus * cfg.test_batch_size,
                                shuffle=False, num_workers=cfg.num_thread, pin_memory=True)


    # prepare network
    model = get_model('test', testset_loader.joint_num)
    model = DataParallel(model).cuda()
    
    #model_file_list = glob.glob(os.path.join(cfg.model_dir, '*.pth.tar'))
    epoch_ls = [0, 5, 10, 15, 19]
    model_file_list = [os.path.join(cfg.model_dir, f'snapshot_{epoch}.pth.tar') for epoch in epoch_ls]
    ckpt_ls = [torch.load(model_path) for model_path in model_file_list]
    
    epoch_ls.append(-1)
    ckpt_ls.append({
        'network': copy.deepcopy(model.state_dict()),
    })
    
    ins = {'img_path': [], 'center': [], 'b_scale': [], 'focal_length': [], 'img_shape': [],\
        'bbox': [], 'do_flip': [], 'scale': [], 'rot': [], 'color_scale': [], 'trans': [],\
        'root_valid': [], 'hand_type_valid': [], 'joint_valid': []}
        
    gts = {'joint_coord': [], 'rel_root_depth': [], 'hand_type': []}
    if cfg.mano_param:
        gts.update({
            "mano_shape": [],
            "mano_pose": [],
            'mano_joint_cam': [],
            'cam_transl': [],
            #'mano_mesh_proj': []
        })
        
    preds_ls = {f'epoch_{epoch}':{} for epoch in epoch_ls}
    for k in preds_ls.keys():
        preds_ls[k].update({'joint_coord': [], 'rel_root_depth': [], 'hand_type': []})
        if cfg.mano_param:
            preds_ls[k].update({
                "mano_shape": [],
                "mano_pose": [],
                'mano_joint_cam': [],
                'cam_transl': [],
                #'mano_mesh_proj': []
            })
    # gt_loss = {'rel_root_depth':[], 'hand_type':[]}
    # if cfg.mano_param:
    #     gt_loss.update({
    #         "mano_shape": [],
    #         "mano_pose": [],
    #         'joint_proj': [],
    #         'joint_cam': []
    #     })
    # mean_gt_loss = {'rel_root_depth':[], 'hand_type':[]}
    # if cfg.mano_param:
    #     mean_gt_loss.update({
    #         "mano_shape": [],
    #         "mano_pose": [],
    #         'joint_proj': [],
    #         'joint_cam': []
    #     })
    with torch.no_grad():
       for itr, (inputs, targets, meta_info) in enumerate(tqdm(batch_generator)):
            img_paths = [testset_loader.datalist[int(i)]['img_path'] for i in meta_info['idx']]
            ins['img_path'].append(np.array(img_paths))
            ins['center'].append(inputs['center'].cpu().numpy().copy())
            ins['b_scale'].append(inputs['b_scale'].cpu().numpy().copy())
            ins['focal_length'].append(inputs['focal_length'].cpu().numpy().copy())
            ins['img_shape'].append(inputs['img_shape'].cpu().numpy().copy())
            ins['bbox'].append(meta_info['bbox'].cpu().numpy().copy())
            ins['do_flip'].append(meta_info['do_flip'].cpu().numpy().copy())
            ins['scale'].append(meta_info['scale'].cpu().numpy().copy())
            ins['rot'].append(meta_info['rot'].cpu().numpy().copy())
            ins['color_scale'].append(meta_info['color_scale'].cpu().numpy().copy())
            ins['trans'].append(meta_info['trans'].cpu().numpy().copy())
            ins['root_valid'].append(meta_info['root_valid'].cpu().numpy().copy())    # [bs, 1]
            #print(meta_info['root_valid'].shape)
            ins['hand_type_valid'].append(meta_info['hand_type_valid'].unsqueeze(-1).cpu().numpy().copy())  # [bs, 1]
            #print(meta_info['hand_type_valid'].shape)
            ins['joint_valid'].append(meta_info['joint_valid'].cpu().numpy().copy())  # [bs, 42]
            #print(meta_info['joint_valid'].shape)
            
            joint_coord_gt = targets['joint_coord_full'] if cfg.crop2full else targets['joint_coord']
            rel_root_depth_gt = targets['rel_root_depth'].cpu().numpy()
            hand_type_gt = targets['hand_type'].cpu().numpy()
            
            gts['joint_coord'].append(joint_coord_gt[:, :, :2])
            gts['rel_root_depth'].append(rel_root_depth_gt)
            gts['hand_type'].append(hand_type_gt)
            "extra prediction"
            #preds['joint_coord_from_heatmap'].append(out["joint_coord_from_heatmap"].cpu().numpy())
            if cfg.mano_param:
                gts['mano_shape'].append(targets["mano_shapes"].cpu().numpy())
                gts['mano_pose'].append(targets["mano_poses"].cpu().numpy())
                gts['mano_joint_cam'].append(targets["joint_cam"].cpu().numpy())
                gts['cam_transl'].append(targets["cam_transl"].cpu().numpy())
                #gts['mano_mesh_proj'].append(targets["mano_mesh_imgs"][:, :, :2].cpu().numpy())
            # for param in ckpt_ls[-1]['network'].keys():
            #     if ckpt_ls[-1]['network'][param].equal(ckpt_ls[-2]['network'][param]):
            #         print(param)   
            # print(ckpt_ls[1]['network']['module.cliff_model.pip.weight']) 
            # print(ckpt_ls[0]['network']['module.cliff_model.pip.weight']) 
            for ckid in range(len(ckpt_ls)):
                model.load_state_dict(ckpt_ls[ckid]['network'])
                model.eval()
                
                # forward
                out = model(inputs, targets, meta_info, 'test')
                
                joint_coord_out = out['joint_coord'].cpu().numpy()
                rel_root_depth_out = out['rel_root_depth'].cpu().numpy()
                hand_type_out = out['hand_type'].cpu().numpy()

                epoch = epoch_ls[ckid]
                k = f'epoch_{epoch}'
                preds_ls[k]['joint_coord'].append(joint_coord_out)
                preds_ls[k]['rel_root_depth'].append(rel_root_depth_out)
                preds_ls[k]['hand_type'].append(hand_type_out)
                "extra prediction"
                #preds['joint_coord_from_heatmap'].append(out["joint_coord_from_heatmap"].cpu().numpy())
                if cfg.mano_param:
                    preds_ls[k]['mano_shape'].append(out["mano_shape"].cpu().numpy())
                    preds_ls[k]['mano_pose'].append(out["mano_pose"].cpu().numpy())
                    preds_ls[k]['mano_joint_cam'].append(out["mano_joint_cam"].cpu().numpy())
                    preds_ls[k]['cam_transl'].append(out["cam_transl"].cpu().numpy())
                    #preds_ls[k]['mano_mesh_proj'].append(out["mano_mesh_proj"].cpu().numpy())
            
            
    # evaluate
    ins = {k: np.concatenate(v) for k,v in ins.items()}
    gts = {k: np.concatenate(v) for k,v in gts.items()}
    for ek in preds_ls.keys():
        preds_ls[ek] = {k: np.concatenate(v) for k,v in preds_ls[ek].items()}
    # gt_loss = {k: np.concatenate(v) for k,v in gt_loss.items()}
    # mean_gt_loss = {k: np.concatenate(v) for k,v in mean_gt_loss.items()}
    for epoch in epoch_ls:
        os.makedirs(os.path.join(cfg.result_dir, f'epoch_{epoch}', args.set), exist_ok=True)
    
        torch.save(preds_ls[f'epoch_{epoch}'], os.path.join(cfg.result_dir, f'epoch_{epoch}', args.set, "preds.pth"))
        print('Save preds to:', os.path.join(cfg.result_dir, f'epoch_{epoch}', args.set, "preds.pth"))
        torch.save(gts, os.path.join(cfg.result_dir, f'epoch_{epoch}', args.set, "gts.pth"))
        print('Save gts to:', os.path.join(cfg.result_dir, f'epoch_{epoch}', args.set, "gts.pth"))
        # torch.save(gt_loss, os.path.join(cfg.result_dir, f'epoch_{epoch}', args.set, "gt_loss.pth"))
        # print('Save gt_loss to:', os.path.join(cfg.result_dir, f'epoch_{epoch}', args.set, "gt_loss.pth"))
        # torch.save(mean_gt_loss, os.path.join(cfg.result_dir, args.set, "mean_gt_loss.pth"))
        # print('Save mean_gt_loss to:', os.path.join(cfg.result_dir, args.set, "mean_gt_loss.pth"))
        torch.save(ins, os.path.join(cfg.result_dir, f'epoch_{epoch}', args.set, "ins.pth"))
        print('Save ins to:', os.path.join(cfg.result_dir, f'epoch_{epoch}', args.set, "ins.pth"))
    
if __name__ == "__main__":
    main()
