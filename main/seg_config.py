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


class Seg_Config:
    ## testing config
    input_img_shape = (256, 256)
    test_batch_size = 4
    trans_test = 'rootnet'  # gt, rootnet

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    encoder_img_size = 10

    ## others
    num_thread = 20
    gpu_ids = '0'
    num_gpus = 1
    have_set_gpu = False

    def set_args(self, gpu_ids, **args):
        for k, v in args.items():
            setattr(self, k, v)
        if not self.have_set_gpu:
            self.have_set_gpu = True
            self.gpu_ids = gpu_ids
            self.num_gpus = len(self.gpu_ids.split(','))
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
            print('>>> Using GPU: {}'.format(self.gpu_ids))


seg_cfg = Seg_Config()
