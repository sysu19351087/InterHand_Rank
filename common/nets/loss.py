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
from config import cfg
import math

class JointHeatmapLoss(nn.Module):
    def __ini__(self):
        super(JointHeatmapLoss, self).__init__()

    def forward(self, joint_out, joint_gt, joint_valid):
        loss = (joint_out - joint_gt)**2 * joint_valid[:,:,None,None,None]
        return loss

class HandTypeLoss(nn.Module):
    def __init__(self):
        super(HandTypeLoss, self).__init__()

    def forward(self, hand_type_out, hand_type_gt, hand_type_valid):
        loss = F.binary_cross_entropy(hand_type_out, hand_type_gt, reduction='none')
        loss = loss.mean(1)
        loss = loss * hand_type_valid

        return loss

class RelRootDepthLoss(nn.Module):
    def __init__(self):
        super(RelRootDepthLoss, self).__init__()

    def forward(self, root_depth_out, root_depth_gt, root_valid):
        loss = torch.abs(root_depth_out - root_depth_gt) * root_valid
        return loss

class VertLoss(nn.Module):
    def __init__(self, weight=1e-3, is2D=True):
        super(VertLoss, self).__init__()
        self.weight = weight
        self.is2D = is2D

    def forward(self, pred_verts, gt_verts, vert_valid):
        loss = self.weight * vert_valid * torch.abs(gt_verts - pred_verts)
        if self.is2D:
            return loss[:, :, :2]
        else:
            return loss

class ParamL1Loss(nn.Module):
    def __init__(self, weight=1e-3):
        super(ParamL1Loss, self).__init__()
        self.weight = weight

    def forward(self, preds, gts, valid):
        loss = self.weight * valid.unsqueeze(-1) * torch.abs(preds - gts)  # [bs, 2, dim]
        return loss