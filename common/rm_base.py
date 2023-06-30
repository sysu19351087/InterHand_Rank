# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import os.path as osp
import math
import time
import glob
import abc
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms

from rm_config import rm_cfg
from rm_dataset import RM_Dataset
from timer import Timer
from logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
from rm_model import get_model


class RM_Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name='logs.txt'):
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(rm_cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return


class RM_Trainer(RM_Base):

    def __init__(self):
        super(RM_Trainer, self).__init__(log_name='train_logs.txt')

    def get_optimizer(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=rm_cfg.lr)
        return optimizer

    def set_lr(self, epoch):
        if len(rm_cfg.lr_dec_epoch) == 0:
            return rm_cfg.lr

        for e in rm_cfg.lr_dec_epoch:
            if epoch < e:
                break
        if epoch < rm_cfg.lr_dec_epoch[-1]:
            idx = rm_cfg.lr_dec_epoch.index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = rm_cfg.lr / (rm_cfg.lr_dec_factor ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = rm_cfg.lr / (rm_cfg.lr_dec_factor ** len(rm_cfg.lr_dec_epoch))

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']

        return cur_lr

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating train dataset...")
        datapath = osp.join(rm_cfg.root_dir, '{}_{}_{}'.format(rm_cfg.run_name, rm_cfg.dataset, rm_cfg.train_ratio), 'result')
        trainset_loader = RM_Dataset(transforms.ToTensor(), datapath, "train")
        batch_generator = DataLoader(dataset=trainset_loader, batch_size=rm_cfg.num_gpus * rm_cfg.train_batch_size,
                                     shuffle=True, num_workers=rm_cfg.num_thread, pin_memory=True)

        #self.joint_num = trainset_loader.joint_num
        self.itr_per_epoch = math.ceil(trainset_loader.__len__() / rm_cfg.num_gpus / rm_cfg.train_batch_size)
        self.batch_generator = batch_generator

    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = get_model('train', rm_cfg.pretrain, rm_cfg.pret_model_path)
        model = DataParallel(model).cuda()
        optimizer = self.get_optimizer(model)
        #for name, parms in model.named_parameters():	
            #print('-->name:', name)
            #print('-->para:', parms)
            #print('-->grad_requirs:',parms.requires_grad)
        if rm_cfg.continue_train:
            start_epoch, model, optimizer = self.load_model(model, optimizer)
        else:
            start_epoch = 0
        model.train()

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer

    def save_model(self, state, epoch):
        file_path = osp.join(rm_cfg.model_dir, 'snapshot_{}.pth.tar'.format(str(epoch)))
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer):
        model_file_list = glob.glob(osp.join(rm_cfg.model_dir, '*.pth.tar'))
        cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9: file_name.find('.pth.tar')]) for file_name in
                         model_file_list])
        model_path = osp.join(rm_cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')
        self.logger.info('Load checkpoint from {}'.format(model_path))
        ckpt = torch.load(model_path)
        start_epoch = ckpt['epoch'] + 1

        model.load_state_dict(ckpt['network'])
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
        except:
            pass

        return start_epoch, model, optimizer


class RM_Tester(RM_Base):

    def __init__(self, test_epoch):
        self.test_epoch = int(test_epoch)
        super(RM_Tester, self).__init__(log_name='test_logs.txt')

    def _make_batch_generator(self, test_set):
        # data load and construct batch generator
        self.logger.info("Creating " + test_set + " dataset...")
        datapath = osp.join(rm_cfg.root_dir, '{}_{}_{}'.format(rm_cfg.run_name, rm_cfg.dataset, rm_cfg.train_ratio), 'result')
        testset_loader = RM_Dataset(transforms.ToTensor(), datapath, test_set)
        batch_generator = DataLoader(dataset=testset_loader, batch_size=rm_cfg.num_gpus * rm_cfg.test_batch_size,
                                     shuffle=False, num_workers=rm_cfg.num_thread, pin_memory=True)

        # self.joint_num = testset_loader.joint_num
        self.batch_generator = batch_generator
        self.testset = testset_loader

    def _make_model(self):
        model_path = os.path.join(rm_cfg.model_dir, 'snapshot_%d.pth.tar' % self.test_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path

        # prepare network
        self.logger.info("Creating graph...")
        model = get_model('test', rm_cfg.pretrain, rm_cfg.pret_model_path)
        model = DataParallel(model).cuda()
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'])
        self.logger.info('Load checkpoint from {}'.format(model_path))
        model.eval()

        self.model = model

    def _evaluate(self, preds):
        self.testset.evaluate(preds)
