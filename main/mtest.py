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
import os
import sys
current_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(current_path + '/../../common'))
sys.path.append(os.path.abspath(current_path + '/../../data/InterHand2.6M'))
from base import Tester
from utils.vis import vis_keypoints
import torch.backends.cudnn as cudnn
from utils.transforms import flip


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--test_epoch', type=int, dest='test_epoch')
    parser.add_argument('--test_set', type=str, dest='test_set')
    parser.add_argument('--train_ratio', type=float, default=1)
    parser.add_argument('--init_zeros', dest='init_zeros', action='store_true')
    parser.add_argument('--crop', dest='crop', action='store_true')
    parser.add_argument('--backbone', type=str, default='hr48')
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
        'run_name': args.run_name,
        'backbone': args.backbone,
        'crop': args.crop,
        'init_zeros': args.init_zeros,
    }
    cfg.set_args(args.gpu_ids, **cfg_dict)
    cudnn.benchmark = True
    
    if cfg.dataset == 'InterHand2.6M':
        assert args.test_set, 'Test set is required. Select one of test/val'
    else:
        args.test_set = 'test'

    tester = Tester(args.test_epoch)
    tester._make_batch_generator(args.test_set)
    tester._make_model()
    
    preds = {'joint_coord': [], 'rel_root_depth': [], 'hand_type': [], 'inv_trans': []}
    if cfg.mano_param:
        preds.update({
            "mano_shape": [],
            "mano_pose": [],
            'mano_joint_cam': [],
        })
    with torch.no_grad():
       for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):
            
            # forward
            out = tester.model(inputs, targets, meta_info, 'test')

            joint_coord_out = out['joint_coord'].cpu().numpy()
            rel_root_depth_out = out['rel_root_depth'].cpu().numpy()
            hand_type_out = out['hand_type'].cpu().numpy()
            inv_trans = out['inv_trans'].cpu().numpy()

            preds['joint_coord'].append(joint_coord_out)
            preds['rel_root_depth'].append(rel_root_depth_out)
            preds['hand_type'].append(hand_type_out)
            preds['inv_trans'].append(inv_trans)
            "extra prediction"
            #preds['joint_coord_from_heatmap'].append(out["joint_coord_from_heatmap"].cpu().numpy())
            if cfg.mano_param:
                preds['mano_shape'].append(out["mano_shape"].cpu().numpy())
                preds['mano_pose'].append(out["mano_pose"].cpu().numpy())
                preds['mano_joint_cam'].append(out["mano_joint_cam"].cpu().numpy())
            
    # evaluate
    preds = {k: np.concatenate(v) for k,v in preds.items()}
    torch.save(preds, os.path.join(cfg.result_dir, "preds.pth"))
    tester.testset.eval_cam_mpjpe(preds['mano_joint_cam'], tester.logger)
    # tester._evaluate(preds)

if __name__ == "__main__":
    main()
