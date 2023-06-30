# 2022.07.19 - Changed for CLIFF
#              Huawei Technologies Co., Ltd.

# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2019, University of Pennsylvania, Max Planck Institute for Intelligent Systems

# This program is free software; you can redistribute it and/or modify it
# under the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

# This script is borrowed and extended from SPIN

import torch
import torch.nn as nn
import numpy as np
import pickle
import math
import os.path as osp

from utils.transforms import rot6d_to_rotmat, rot6d_to_axis_angle
from nets.cliff_models.cliff_intaghand.encoder import get_encoder
#from nets.cliff_models.backbones.hrnet.cls_hrnet import HighResolutionNet
#from nets.cliff_models.backbones.hrnet.hrnet_config import cfg
#from nets.cliff_models.backbones.hrnet.hrnet_config import update_config


class CLIFF(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone"""

    def __init__(self, smpl_mean_params, img_feat_num=2048):
        super(CLIFF, self).__init__()
        #curr_dir = osp.dirname(osp.abspath(__file__))
        #config_file = osp.join(curr_dir, "../backbones/hrnet/models/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml")
        #update_config(cfg, config_file)
        #self.encoder = HighResolutionNet(cfg)

        self.npose = 16 * 2 * 6
        self.nshape = 10 * 2
        self.ncam = 3 * 2
        self.nbbox = 3

        fc1_feat_num = 1024
        fc2_feat_num = 1024
        final_feat_num = fc2_feat_num
        reg_in_feat_num = img_feat_num + self.nbbox + self.npose + self.nshape + self.ncam
        
        # pred init param
        self.pip = nn.Linear(img_feat_num + self.nbbox, self.npose)
        self.pis = nn.Linear(img_feat_num + self.nbbox, self.nshape)
        self.pic = nn.Linear(img_feat_num + self.nbbox, self.ncam)
        
        # CUDA Error: an illegal memory access was encountered
        # the above error will occur, if use mobilenet v3 with BN, so don't use BN
        self.fc1 = nn.Linear(reg_in_feat_num, fc1_feat_num)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(fc1_feat_num, fc2_feat_num)
        self.drop2 = nn.Dropout()

        self.decpose = nn.Linear(final_feat_num, self.npose)
        self.decshape = nn.Linear(final_feat_num, self.nshape)
        self.deccam = nn.Linear(final_feat_num, self.ncam)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        self.encoder, self.mid_model = get_encoder(pretrain=True)
        
        lmean_params = pickle.load(open(osp.join(smpl_mean_params + 'MANO_LEFT.pkl'), 'rb'), encoding='iso-8859-1')
        rmean_params = pickle.load(open(osp.join(smpl_mean_params + 'MANO_RIGHT.pkl'), 'rb'), encoding='iso-8859-1')
        init_lpose = torch.from_numpy(lmean_params['hands_mean'][:]).unsqueeze(0)  # [1, 45]
        init_lpose = torch.cat([init_lpose, torch.zeros(1, 3)], 1)  # [1, 48]
        init_rpose = torch.from_numpy(rmean_params['hands_mean'][:]).unsqueeze(0)  # [1, 45]
        init_rpose = torch.cat([init_rpose, torch.zeros(1, 3)], 1)  # [1, 48]
        #init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        #init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', torch.cat([init_rpose, init_lpose], 1)) # [1, 16 * 2 * 3]
        #self.register_buffer('init_shape', init_shape)
        #self.register_buffer('init_cam', init_cam)
        
    def forward(self, x, bbox, init_pose=None, init_shape=None, init_cam=None, n_iter=3, init_zeros=False, need_pose6d=False):
        batch_size = x.shape[0]
        hms, mask, dp, img_fmaps, hms_fmaps, dp_fmaps = self.encoder(x)
        xf, fmaps = self.mid_model(img_fmaps, hms_fmaps, dp_fmaps)  # xf:[bs, 2048]
        if init_pose is None:
            if init_zeros:
                init_pose = torch.zeros(batch_size, self.npose).to(x.device).float()
            else:
                init_pose = self.pip(torch.cat([xf, bbox], 1))          # [bs, 32*6]
        if init_shape is None:
            if init_zeros:
                init_shape = torch.zeros(batch_size, self.nshape).to(x.device).float()
            else:
                init_shape = self.pis(torch.cat([xf, bbox], 1))  # [bs, 20]
        if init_cam is None:
            if init_zeros:
                init_cam = torch.zeros(batch_size, self.ncam).to(x.device).float()
            else:
                init_cam = self.pic(torch.cat([xf, bbox], 1))  # [bs, 6]

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([xf, bbox, pred_pose, pred_shape, pred_cam], 1)
            #print(xc.shape)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)

            pred_pose = self.decpose(xc) + pred_pose       # [bs, 32*6]
            pred_shape = self.decshape(xc) + pred_shape    # [bs, 20]
            pred_cam = self.deccam(xc) + pred_cam          # [bs, 6]

        pred_axis_angle = rot6d_to_axis_angle(pred_pose.reshape(-1, 6)).contiguous().view(batch_size, -1)  # [bs, 32*3]
        
        if need_pose6d:
            return xf, pred_axis_angle, pred_shape, pred_cam, pred_pose.clone()

        return xf, pred_axis_angle, pred_shape, pred_cam
