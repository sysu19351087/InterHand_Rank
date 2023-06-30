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
from rm_config import rm_cfg
import torch
import os
import sys
current_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(current_path + '/../../common'))
sys.path.append(os.path.abspath(current_path + '/../../data/InterHand2.6M'))
from rm_base import RM_Tester
from rm_model import get_gt_prob
import torch.backends.cudnn as cudnn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--rm_run_name', type=str)
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--train_ratio', type=float, default=1)
    parser.add_argument('--backbone', type=str, default='hr48')
    parser.add_argument('--test_epoch', type=int, default=20)
    parser.add_argument('--test_set', type=str, dest='test_set')
    parser.add_argument('--recons_ls', type=str, default='mano')
    parser.add_argument('--mlp_r_act', type=str, default='relu')
    parser.add_argument('--prob_rever_loss', action='store_true')
    parser.add_argument('--w_ht', action='store_true')
    parser.add_argument('--aug_feat', action='store_true')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    assert args.test_epoch, 'Test epoch is required.'
    return args

def main():

    args = parse_args()
    cfg_dict = {
        'train_ratio': args.train_ratio,
        'rm_run_name': args.rm_run_name,
        'run_name': args.run_name,
        'backbone': args.backbone,
        'recons_ls': args.recons_ls,
        'mlp_r_act': args.mlp_r_act,
        'prob_rever_loss': args.prob_rever_loss,
        'w_ht': args.w_ht,
        'aug_feat': args.aug_feat,
    }
    rm_cfg.set_args(args.gpu_ids, **cfg_dict)

    cudnn.benchmark = True
    
    if rm_cfg.dataset == 'InterHand2.6M':
        assert args.test_set, 'Test set is required. Select one of test/val'
    else:
        args.test_set = 'test'

    tester = RM_Tester(args.test_epoch)
    tester._make_batch_generator(args.test_set)
    tester._make_model()
    
    rm_preds = {'probs': []}
    # rm_gtloss = {'mean_loss_mano_shape': [], 'mean_loss_mano_pose': [], 'mean_loss_joint_proj': []}
    rm_gtloss = {'target_prob': []}

    with torch.no_grad():
       for itr, (inputs, preds, gt_loss) in enumerate(tqdm(tester.batch_generator)):
            
            # forward
            out = tester.model(inputs, preds, gt_loss, 'test')

            prob = out['label'].cpu().numpy()

            rm_preds['probs'].append(prob)
            
            bs, com_num, _ = preds['pred_hand_type'].shape
            device = preds['pred_hand_type'].device
            target_prob = get_gt_prob(gt_loss, bs, com_num, device, mask_losses=out['mask_losses'])
            # rm_gtloss['mean_loss_mano_shape'].append(gt_loss['mean_loss_mano_shape'])
            # rm_gtloss['mean_loss_mano_pose'].append(gt_loss['mean_loss_mano_pose'])
            # rm_gtloss['mean_loss_joint_proj'].append(gt_loss['mean_loss_joint_proj'])
            rm_gtloss['target_prob'].append(target_prob)
            
    # evaluate
    rm_preds = {k: np.concatenate(v) for k,v in rm_preds.items()}
    rm_gtloss = {k: np.concatenate(v) for k,v in rm_gtloss.items()}
    torch.save(rm_preds, os.path.join(rm_cfg.result_dir, "rm_preds.pth"))
    tester.testset.eval_prob(rm_preds['probs'], rm_gtloss, tester.logger)
    # tester._evaluate(preds)

if __name__ == "__main__":
    main()
