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

from rl_config import rl_cfg
from rm_config import rm_cfg
from rl_dataset import RL_Dataset
from timer import Timer
from logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
from rl_model import get_model as get_rl_model
from rm_model import get_model as get_rm_model


class RL_Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name='logs.txt'):
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(rl_cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return


class RL_Trainer(RL_Base):

    def __init__(self):
        super(RL_Trainer, self).__init__(log_name='train_logs.txt')

    def get_optimizer(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=rl_cfg.lr)
        return optimizer

    def set_lr(self, epoch):
        if len(rl_cfg.lr_dec_epoch) == 0:
            return rl_cfg.lr

        for e in rl_cfg.lr_dec_epoch:
            if epoch < e:
                break
        if epoch < rl_cfg.lr_dec_epoch[-1]:
            idx = rl_cfg.lr_dec_epoch.index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = rl_cfg.lr / (rl_cfg.lr_dec_factor ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = rl_cfg.lr / (rl_cfg.lr_dec_factor ** len(rl_cfg.lr_dec_epoch))

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']

        return cur_lr

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating train dataset...")
        trainset_loader = RL_Dataset(transforms.ToTensor(), "train")
        batch_generator = DataLoader(dataset=trainset_loader, batch_size=rl_cfg.num_gpus * rl_cfg.train_batch_size,
                                     shuffle=True, num_workers=rl_cfg.num_thread, pin_memory=True)

        self.joint_num = trainset_loader.joint_num
        self.itr_per_epoch = math.ceil(trainset_loader.__len__() / rl_cfg.num_gpus / rl_cfg.train_batch_size)
        self.batch_generator = batch_generator

    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = get_rl_model('train', self.joint_num)
        model = DataParallel(model).cuda()
        optimizer = self.get_optimizer(model)
        #for name, parms in model.named_parameters():	
            #print('-->name:', name)
            #print('-->para:', parms)
            #print('-->grad_requirs:',parms.requires_grad)
            
        # load first-stage model
        fs_model_dir = osp.join(rl_cfg.root_dir, '{}_{}_{}'.format(rl_cfg.run_name, rl_cfg.dataset, rl_cfg.train_ratio), 'model_dump')
        model_file_list = glob.glob(osp.join(fs_model_dir, '*.pth.tar'))
        cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9: file_name.find('.pth.tar')]) for file_name in
                         model_file_list])
        model_path = osp.join(fs_model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')
        self.logger.info('Load first-stage checkpoint from {}'.format(model_path))
        ckpt = torch.load(model_path)

        model.load_state_dict(ckpt['network'])
        
        # load reward model
        rm_model = get_rm_model('train', rm_cfg.pretrain, rm_cfg.pret_model_path)
        rm_model = DataParallel(rm_model).cuda()
        rm_model_dir = osp.join(rl_cfg.root_dir, 'rm_{}_{}_{}'.format(rl_cfg.run_name, rl_cfg.dataset, rl_cfg.train_ratio), 'model_dump')
        model_file_list = glob.glob(osp.join(rm_model_dir, '*.pth.tar'))
        cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9: file_name.find('.pth.tar')]) for file_name in
                         model_file_list])
        model_path = osp.join(rm_model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')
        self.logger.info('Load reward model checkpoint from {}'.format(model_path))
        rm_ckpt = torch.load(model_path)

        rm_model.load_state_dict(rm_ckpt['network'])
        rm_model.eval()
        
        # whether continue train
        if rl_cfg.continue_train:
            start_epoch, model, optimizer = self.load_model(model, optimizer)
        else:
            start_epoch = 0
        model.train()

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer
        self.rm_model = rm_model

    def save_model(self, state, epoch):
        file_path = osp.join(rl_cfg.model_dir, 'snapshot_{}.pth.tar'.format(str(epoch)))
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer):
        model_file_list = glob.glob(osp.join(rl_cfg.model_dir, '*.pth.tar'))
        cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9: file_name.find('.pth.tar')]) for file_name in
                         model_file_list])
        model_path = osp.join(rl_cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')
        self.logger.info('Load checkpoint from {}'.format(model_path))
        ckpt = torch.load(model_path)
        start_epoch = ckpt['epoch'] + 1

        model.load_state_dict(ckpt['network'])
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
        except:
            pass

        return start_epoch, model, optimizer


class RL_Tester(RL_Base):

    def __init__(self, test_epoch):
        self.test_epoch = int(test_epoch)
        super(RL_Tester, self).__init__(log_name='test_logs.txt')

    def _make_batch_generator(self, test_set):
        # data load and construct batch generator
        self.logger.info("Creating " + test_set + " dataset...")
        testset_loader = RL_Dataset(transforms.ToTensor(), test_set)
        batch_generator = DataLoader(dataset=testset_loader, batch_size=rl_cfg.num_gpus * rl_cfg.test_batch_size,
                                     shuffle=False, num_workers=rl_cfg.num_thread, pin_memory=True)

        self.joint_num = testset_loader.joint_num
        self.batch_generator = batch_generator
        self.testset = testset_loader

    def _make_model(self):
        model_path = os.path.join(rl_cfg.model_dir, 'snapshot_%d.pth.tar' % self.test_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))

        # prepare network
        self.logger.info("Creating graph...")
        model = get_model('test', self.joint_num)
        model = DataParallel(model).cuda()
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'])
        model.eval()

        self.model = model

    def _evaluate(self, preds):
        self.testset.evaluate(preds)
