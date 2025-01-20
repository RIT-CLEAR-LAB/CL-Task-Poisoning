# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    return parser


class Derpp(ContinualModel):
    NAME = 'derpp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Derpp, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device, mode=args.buffer_mode)

    def observe(self, inputs, labels, not_aug_inputs, poisoned_flags):

        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            buf_data = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            buf_inputs, buf_logits = buf_data[0], buf_data[2]
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

            buf_data = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            buf_inputs, buf_labels = buf_data[0], buf_data[1]
            buf_outputs = self.net(buf_inputs)
            loss += self.args.beta * self.loss(buf_outputs, buf_labels)

        loss.backward()
        self.opt.step()

        self.buffer.add_data(
            examples=not_aug_inputs,
            labels=labels,
            logits=outputs.data,
            poisoned_flags=poisoned_flags,
        )

        return loss.item()

    def check_buffer_contamination(self):
        poisoned_flags = self.buffer.poisoned_flags.cpu().numpy()
        poisoned_flags = poisoned_flags[poisoned_flags > -1]
        poisoned_buffer_samples = int(poisoned_flags.sum())
        return poisoned_buffer_samples
