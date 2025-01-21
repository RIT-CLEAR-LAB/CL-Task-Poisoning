# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer_new import Buffer
import numpy as np


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class Er(ContinualModel):
    NAME = 'er'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Er, self).__init__(backbone, loss, args, transform)
        # self.buffer = Buffer(self.args.buffer_size, self.device, mode=args.buffer_mode)
        self.buffer = Buffer(
            self.net,
            self.args,
            self.args.buffer_size,
            self.device,
            mode=args.buffer_mode,
        )
        self.task = 0
        self.poisoned_flags = torch.tensor([0]).long().to(self.device)

    def observe(self, inputs, labels, not_aug_inputs, poisoned_flags):

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels.long())
        self.opt.zero_grad()
        loss.backward()
        if self.task > 0:
            buf_data = self.buffer.get_data(
                self.args.minibatch_size,
                transform=self.transform,
                mode=self.args.buffer_retrieve_mode,
            )
            buf_inputs, buf_labels = buf_data[0], buf_data[1]
            if poisoned_flags is not None:
                self.poisoned_flags = buf_data[2]
            buf_outputs = self.net(buf_inputs)
            buf_loss = self.loss(buf_outputs, buf_labels.long())
            buf_loss.backward()

        self.opt.step()

        self.buffer.add_data(
            examples=not_aug_inputs,
            labels=labels[:real_batch_size],
            poisoned_flags=poisoned_flags,
        )

        return loss.item()

    def end_task(self, dataset):
        self.task += 1

    def check_buffer_contamination(self):
        poisoned_flags = self.buffer.poisoned_flags.cpu().numpy()
        poisoned_flags = poisoned_flags[poisoned_flags > -1]
        poisoned_buffer_samples = int(poisoned_flags.sum())
        return poisoned_buffer_samples
    
    def check_poisoned_samples(self):
        poisoned_flags = self.poisoned_flags.cpu().numpy()
        poisoned_flags = poisoned_flags[poisoned_flags > -1]
        poisoned_samples = int(poisoned_flags.sum())
        return poisoned_samples