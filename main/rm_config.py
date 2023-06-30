# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import os.path as osp
import sys
import math
import numpy as np
import glob

def get_latest_epoch(model_dir):
    model_file_list = glob.glob(osp.join(model_dir, '*.pth.tar'))
    cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9: file_name.find('.pth.tar')]) for file_name in
                    model_file_list])
    return cur_epoch

class HGFilter_opt:
    num_stack = 5
    norm = 'group'
    hg_down = 'ave_pool'
    num_hourglass = 4
    hourglass_dim = 256
    

class RM_Config:
    ## dataset
    dataset = 'InterHand2.6M'  # InterHand2.6M, RHD, STB
    train_ratio = 0.1   # ratio of examples used for training

    ## for reward data
    input_img_shape = (256, 256)
    
    ## training config
    lr_dec_epoch = [15, 17] if dataset == 'InterHand2.6M' else [45, 47]
    end_epoch = 20 if dataset == 'InterHand2.6M' else 50
    lr = 1e-4
    lr_dec_factor = 10
    train_batch_size = 64

    ## testing config
    test_batch_size = 64

    ## directory
    rm_run_name = 'exp'
    run_name = "cliff10"
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, '{}_{}_{}_{}'.format(rm_run_name, run_name, dataset, train_ratio))
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')

    ## others
    num_thread = 4
    gpu_ids = '0'
    num_gpus = 1
    continue_train = False
    
     ## for reward model
    pretrain = True
    recons_model_dir = osp.join(root_dir, '{}_{}_{}'.format(run_name, dataset, train_ratio), 'model_dump')
    pret_model_path = osp.join(recons_model_dir, 'snapshot_' + str(get_latest_epoch(recons_model_dir)) + '.pth.tar')
    backbone = "hr48"
    resnet_type = 50  # 18, 34, 50, 101, 152
    mlp_dim = [256+2, 1024, 512, 256, 128, 1]
    hgf_opt = HGFilter_opt()
    recons_ls = 'mano_jproj_mask'
    mlp_r_act = 'relu'
    prob_rever_loss = False
    w_ht = False
    reso_grid = 64
    aug_feat = False

    def set_args(self, gpu_ids, **args):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        for k, v in args.items():
            setattr(self, k, v)
            if k == 'train_ratio':
                assert v <= 1.0, 'train_ratio should <= 1.0'
        # self.continue_train = continue_train
        # self.backbone = backbone
        # self.recons_ls = recons_ls
        # self.mlp_r_act = mlp_r_act
        # assert train_ratio <= 1.0, 'train_ratio should <= 1.0'
        # self.train_ratio = train_ratio
        # self.rm_run_name = rm_run_name
        # self.run_name = run_name
        # self.end_epoch = end_epoch
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))
        self.output_dir = osp.join(self.root_dir, '{}_{}_{}_{}'.format(self.rm_run_name, self.run_name, self.dataset, self.train_ratio))
        self.model_dir = osp.join(self.output_dir, 'model_dump')
        self.vis_dir = osp.join(self.output_dir, 'vis')
        self.log_dir = osp.join(self.output_dir, 'log')
        self.result_dir = osp.join(self.output_dir, 'result')
        self.recons_model_dir = osp.join(self.root_dir, '{}_{}_{}'.format(self.run_name, self.dataset, self.train_ratio), 'model_dump')
        self.pret_model_path = osp.join(self.recons_model_dir, 'snapshot_' + str(get_latest_epoch(self.recons_model_dir)) + '.pth.tar')
        sys.path.insert(0, osp.join(rm_cfg.root_dir, 'common'))
        from utils.dir import add_pypath, make_folder
        add_pypath(osp.join(rm_cfg.data_dir))
        add_pypath(osp.join(rm_cfg.data_dir, rm_cfg.dataset))
        make_folder(rm_cfg.model_dir)
        make_folder(rm_cfg.vis_dir)
        make_folder(rm_cfg.log_dir)
        make_folder(rm_cfg.result_dir)
    


rm_cfg = RM_Config()
