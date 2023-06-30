# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os.path
import sys
current_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(current_path + '/../../common'))
sys.path.append(os.path.abspath(current_path + '/../../data/InterHand2.6M'))
from tqdm import tqdm
import numpy as np
import cv2
from rm_config import rm_cfg
from config import cfg
from model import get_model as get_cliff_model
from rm_model import get_model as get_rm_model
import torch
import os
from logger import colorlogger
from base import Tester
from utils.vis import vis_keypoints
import torch.backends.cudnn as cudnn
from torch.nn.parallel.data_parallel import DataParallel
from utils.transforms import flip
from optimize import optimize_by_rank
from optimize_wid import get_iou

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    
    parser.add_argument('--rm_name', type=str)
    parser.add_argument('--rm_epoch', type=int, default=19)
    parser.add_argument('--cliff_name', type=str)
    parser.add_argument('--cliff_epoch', type=int, default=19)
    parser.add_argument('--test_set', type=str, dest='test_set')
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

def main():
    args = parse_args()
    rm_cfg.set_args(args.gpu_ids)
    rm_model_path = os.path.abspath(current_path + f'/../../{args.rm_name}_cliff10_InterHand2.6M_0.1/model_dump/snapshot_{args.rm_epoch}.pth.tar')
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
    tester = Tester(args.cliff_epoch)
    tester.set_logger(f'./opt/{args.cliff_name}_{args.rm_name}', log_name='test_logs.txt')
    # tester.logger = colorlogger(f'./opt/{args.cliff_name}_{args.rm_name}', log_name='test_logs.txt')
    tester._make_batch_generator(args.test_set)
    tester._make_model()
    
    first_preds = {'joint_cam': [], 'iou': []}
    final_preds = {'joint_cam': [], 'iou': []}
    gt_hand_types = []
    
    for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):
        
        first_pred, final_pred, _ = optimize_by_rank(inputs, targets, meta_info, tester.model, rm_model, args.end_step, args.lr)
        first_joint_cam = first_pred['joint_cam'].cpu().numpy()    # [bs, 42, 3]
        first_pred_mask = first_pred['pred_mask'].cpu().numpy()    # [bs, 42, 3]
        final_joint_cam = final_pred['joint_cam'].cpu().numpy()    # [bs, 1, H, W]
        final_pred_mask = final_pred['pred_mask'].cpu().numpy()    # [bs, 1, H, W]
        
        gt_mask = inputs['mask'].unsqueeze(1).cpu().numpy()    # [bs, 1, H, W]
        gt_hand_types.append(targets['hand_type'].cpu().numpy())
        first_iou = get_iou(gt_mask, first_pred_mask)   # [bs,]
        final_iou = get_iou(gt_mask, final_pred_mask)   # [bs,]
        
        first_preds['joint_cam'].append(first_joint_cam)
        final_preds['joint_cam'].append(final_joint_cam) 
        first_preds['iou'].append(first_iou)
        final_preds['iou'].append(final_iou)
            
    # evaluate
    first_preds = {k: np.concatenate(v) for k,v in first_preds.items()}
    final_preds = {k: np.concatenate(v) for k,v in final_preds.items()}
    gt_hand_types = np.concatenate(gt_hand_types)    # [n, 2]
    
    tester.testset.eval_cam_mpjpe(first_preds['joint_cam'], tester.logger)
    tester.testset.eval_cam_mpjpe(final_preds['joint_cam'], tester.logger)
    tester.testset.eval_iou(first_preds['iou'], tester.logger)
    tester.testset.eval_iou(final_preds['iou'], tester.logger)
    # tester._evaluate(preds)

if __name__ == "__main__":
    main()
