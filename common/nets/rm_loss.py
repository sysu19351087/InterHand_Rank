# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from rm_config import rm_cfg
import math

class RM_JointLoss(nn.Module):
    def __init__(self, weight=1e-3, is2D=True):
        super(RM_JointLoss, self).__init__()
        self.weight = weight
        self.is2D = is2D
        self.joint_num = 21

    def forward(self, pred_verts, gt_verts):
        loss = self.weight * torch.abs(gt_verts - pred_verts)
        if self.is2D:
            return loss[:, :, :2]
        else:
            return loss

class RM_ParamL1Loss(nn.Module):
    def __init__(self, weight=1e-3):
        super(RM_ParamL1Loss, self).__init__()
        self.weight = weight

    def forward(self, preds, gts):
        loss = self.weight * torch.abs(preds - gts)  # [bs, 2, dim]
        return loss

class MeanLoss(nn.Module):
    def __init__(self, weight=1e-3):
        super(MeanLoss, self).__init__()
        self.weight = weight

    def forward(self, preds, gts):
        loss = self.weight * torch.abs(preds - gts)  # [bs,]
        return loss