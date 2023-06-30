# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from rl_config import rl_cfg
from rm_config import rm_cfg
import torch
import os
import sys
current_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(current_path + '/../../common'))
sys.path.append(os.path.abspath(current_path + '/../../data/InterHand2.6M'))
from rl_base import RL_Trainer
import torch.backends.cudnn as cudnn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--train_ratio', type=float, default=1)
    parser.add_argument('--init_zeros', dest='init_zeros', action='store_true')
    parser.add_argument('--crop', dest='crop', action='store_true')
    parser.add_argument('--backbone', type=str, default='hr48')
    parser.add_argument('--end_epoch', type=int, default=20)
    parser.add_argument('--mean_loss', dest='mean_loss', action='store_true')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args


def main():
    # argument parse and create log
    args = parse_args()
    rl_cfg.set_args(args.gpu_ids, continue_train=args.continue_train, train_ratio=args.train_ratio, run_name=args.run_name, \
        backbone=args.backbone, init_zeros=args.init_zeros, crop=args.crop, end_epoch=args.end_epoch)
    rm_cfg.set_args(args.gpu_ids, continue_train=args.continue_train, train_ratio=args.train_ratio, run_name=args.run_name, \
            backbone=args.backbone, end_epoch=args.end_epoch, mean_loss=args.mean_loss)
    cudnn.benchmark = True

    trainer = RL_Trainer()
    trainer._make_batch_generator()
    trainer._make_model()

    # train
    for epoch in range(trainer.start_epoch, rl_cfg.end_epoch):

        trainer.set_lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()
        for itr, (inputs, targets, meta_info) in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            # forward
            trainer.optimizer.zero_grad()
            loss = trainer.model(inputs, targets, meta_info, 'train', trainer.rm_model)
            loss = {k: loss[k].mean() for k in loss}

            # backward
            sum(loss[k] for k in loss).backward()
            trainer.optimizer.step()
            trainer.gpu_timer.toc()
            screen = [
                'Epoch %d/%d itr %d/%d:' % (epoch, rl_cfg.end_epoch, itr, trainer.itr_per_epoch),
                'lr: %g' % (trainer.get_lr()),
                'speed: %.2f(%.2fs r%.2f)s/itr' % (
                    trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
            ]
            screen += ['%s: %.10f' % ('loss_' + k, v.detach()) for k, v in loss.items()]
            trainer.logger.info(' '.join(screen))

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

        # save model
        trainer.save_model({
            'epoch': epoch,
            'network': trainer.model.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
        }, epoch)


if __name__ == "__main__":
    main()
