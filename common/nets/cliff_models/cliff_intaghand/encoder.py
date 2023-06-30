import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from config import cfg


def weights_init(layer):
    classname = layer.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(layer.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(layer.weight.data)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0.0)


def build_activate_layer(actType):
    if actType == 'relu':
        return nn.ReLU(inplace=True)
    elif actType == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif actType == 'elu':
        return nn.ELU(inplace=True)
    elif actType == 'sigmoid':
        return nn.Sigmoid()
    elif actType == 'tanh':
        return nn.Tanh()
    else:
        raise RuntimeError('no such activate layer!')


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class unFlatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1, 1, 1)


def conv1x1(in_channels, out_channels, stride=1, bn_init_zero=False, actFun='relu'):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.constant_(bn.weight, 0. if bn_init_zero else 1.)
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
              build_activate_layer(actFun),
              bn]
    return nn.Sequential(*layers)


def conv3x3(in_channels, out_channels, stride=1, bn_init_zero=False, actFun='relu'):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.constant_(bn.weight, 0. if bn_init_zero else 1.)
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
              build_activate_layer(actFun),
              bn]
    return nn.Sequential(*layers)


class ResNetSimple_decoder(nn.Module):
    def __init__(self, expansion=4,
                 fDim=[256, 256, 256, 256], direction=['flat', 'up', 'up', 'up'],
                 out_dim=3):
        super(ResNetSimple_decoder, self).__init__()
        self.models = nn.ModuleList()
        fDim = [512 * expansion] + fDim
        for i in range(len(direction)):
            kernel_size = 1 if direction[i] == 'flat' else 3
            self.models.append(self.make_layer(fDim[i], fDim[i + 1], direction[i], kernel_size=kernel_size))

        self.final_layer = nn.Conv2d(
            in_channels=fDim[-1],
            out_channels=out_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def make_layer(self, in_dim, out_dim,
                   direction, kernel_size=3, relu=True, bn=True):
        assert direction in ['flat', 'up']
        assert kernel_size in [1, 3]
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 1:
            padding = 0

        layers = []
        if direction == 'up':
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=1, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(out_dim))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x:[bs, 2048, 8, 8]
        fmaps = []
        for i in range(len(self.models)):
            x = self.models[i](x)
            fmaps.append(x)
        # fmaps:[bs, 256, 8, 8]
        #       [bs, 256, 16, 16]
        #       [bs, 256, 32, 32]
        #       [bs, 256, 64, 64]
        x = self.final_layer(x)
        return x, fmaps


class ResNetSimple(nn.Module):
    def __init__(self, model_type='resnet50',
                 pretrained=False,
                 fmapDim=[256, 256, 256, 256],
                 handNum=2,
                 heatmapDim=21):
        super(ResNetSimple, self).__init__()
        assert model_type in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        if model_type == 'resnet18':
            self.resnet = resnet18(pretrained=pretrained)
            self.expansion = 1
        elif model_type == 'resnet34':
            self.resnet = resnet34(pretrained=pretrained)
            self.expansion = 1
        elif model_type == 'resnet50':
            self.resnet = resnet50(pretrained=pretrained)
            self.expansion = 4
        elif model_type == 'resnet101':
            self.resnet = resnet101(pretrained=pretrained)
            self.expansion = 4
        elif model_type == 'resnet152':
            self.resnet = resnet152(pretrained=pretrained)
            self.expansion = 4

        self.hms_decoder = ResNetSimple_decoder(expansion=self.expansion,
                                                fDim=fmapDim,
                                                direction=['flat', 'up', 'up', 'up'],
                                                out_dim=heatmapDim * handNum)
        for m in self.hms_decoder.modules():
            weights_init(m)

        self.dp_decoder = ResNetSimple_decoder(expansion=self.expansion,
                                               fDim=fmapDim,
                                               direction=['flat', 'up', 'up', 'up'],
                                               out_dim=handNum + 3 * handNum)
        self.handNum = handNum

        for m in self.dp_decoder.modules():
            weights_init(m)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x4 = self.resnet.layer1(x)     # [bs, 256, 64, 64]
        x3 = self.resnet.layer2(x4)    # [bs, 512, 32, 32]
        x2 = self.resnet.layer3(x3)    # [bs, 1024, 16, 16]
        x1 = self.resnet.layer4(x2)    # [bs, 2048, 8, 8]

        img_fmaps = [x1, x2, x3, x4]

        hms, hms_fmaps = self.hms_decoder(x1)   # heat maps
        out, dp_fmaps = self.dp_decoder(x1)
        mask = out[:, :self.handNum]    # segmetnation mask
        dp = out[:, self.handNum:]      # dense correspondences
        # hms torch.Size([1, 42, 64, 64])
        # mask torch.Size([1, 2, 64, 64])
        # dp torch.Size([1, 6, 64, 64])
        return hms, mask, dp, img_fmaps, hms_fmaps, dp_fmaps


class resnet_mid(nn.Module):
    def __init__(self,
                 model_type='resnet50',
                 in_fmapDim=[256, 256, 256, 256],
                 out_fmapDim=[256, 256, 256, 256]):
        super(resnet_mid, self).__init__()
        assert model_type in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        if model_type == 'resnet18' or model_type == 'resnet34':
            self.expansion = 1
        elif model_type == 'resnet50' or model_type == 'resnet101' or model_type == 'resnet152':
            self.expansion = 4

        self.img_fmaps_dim = [512 * self.expansion, 256 * self.expansion,
                              128 * self.expansion, 64 * self.expansion]
        self.dp_fmaps_dim = in_fmapDim
        self.hms_fmaps_dim = in_fmapDim

        self.convs = nn.ModuleList()
        for i in range(len(out_fmapDim)):
            inDim = self.dp_fmaps_dim[i] + self.hms_fmaps_dim[i]
            if i > 0:
                inDim = inDim + self.img_fmaps_dim[i]
            self.convs.append(conv1x1(inDim, out_fmapDim[i]))

        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
        )

        self.global_feature_dim = 512 * self.expansion
        self.fmaps_dim = out_fmapDim

    def get_info(self):
        return {'global_feature_dim': self.global_feature_dim,
                'fmaps_dim': self.fmaps_dim}

    def forward(self, img_fmaps, hms_fmaps, dp_fmaps):
        # img_fmaps:[bs, 2048, 8, 8]
        #           [bs, 1024, 16, 16]
        #           [bs, 512, 32, 32]
        #           [bs, 256, 64, 64]
        # hms_maps/dp_fmaps:[bs, 256, 8, 8]
        #                   [bs, 256, 16, 16]
        #                   [bs, 256, 32, 32]
        #                   [bs, 256, 64, 64]
        global_feature = self.output_layer(img_fmaps[0])
        fmaps = []
        for i in range(len(self.convs)):
            x = torch.cat((hms_fmaps[i], dp_fmaps[i]), dim=1)
            if i > 0:
                x = torch.cat((x, img_fmaps[i]), dim=1)
            fmaps.append(self.convs[i](x))
        # global_feature torch.Size([1, 2048])
        # fmaps: torch.Size([1, 256, 8, 8])
        #        torch.Size([1, 256, 16, 16])
        #        torch.Size([1, 256, 32, 32])
        #        torch.Size([1, 256, 64, 64])
        return global_feature, fmaps


def get_encoder(pretrain=True):
    resnet_type = cfg.resnet_type
    resnet_type = 'resnet{}'.format(resnet_type)
    encoder = ResNetSimple(model_type=resnet_type,
                           pretrained=pretrain,
                           fmapDim=[128, 128, 128, 128],
                           handNum=2,
                           heatmapDim=21)
    mid_model = resnet_mid(model_type=resnet_type,
                           in_fmapDim=[128, 128, 128, 128],
                           out_fmapDim=[256, 256, 256, 256])
    if pretrain:
        print("load pretrained intag encoder")
        current_path = os.path.abspath(__file__)
        dir_path = os.path.abspath(current_path + '/../../../../..')
        cpkt = torch.load(os.path.join(dir_path, 'pretrained', 'intagh_encoder.pth'))
        estat = {}
        mstat = {}
        for k, v in cpkt.items():
            if k.startswith('encoder'):
                estat[k[8:]] = v
            else:
                mstat[k[10:]] = v
        encoder.load_state_dict(estat)
        mid_model.load_state_dict(mstat)
    return encoder, mid_model


if __name__ == "__main__":
    encoder, mid_model = get_encoder()
    hms, mask, dp, img_fmaps, hms_fmaps, dp_fmaps = encoder(torch.randn(1, 3, 256, 256))
    global_feature, fmaps = mid_model(img_fmaps, hms_fmaps, dp_fmaps)
    print(global_feature.shape)
    for fmap in fmaps:
        print(fmap.shape)
    # print("hms {}".format(hms.shape))
    # print("mask {}".format(mask.shape))
    # print("dp {}".format(dp.shape))