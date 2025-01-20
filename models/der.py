# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args, add_rehearsal_args
# from utils.buffer import Buffer
from utils.buffer_new import Buffer



def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    return parser


class Der(ContinualModel):
    NAME = 'der'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Der, self).__init__(backbone, loss, args, transform)
        # self.buffer = Buffer(self.args.buffer_size, self.device, mode=args.buffer_mode)
        self.buffer = Buffer(
            self.net,
            self.args,
            self.args.buffer_size,
            self.device,
            mode=args.buffer_mode,
        )
        self.poisoned_flags = torch.tensor([0]).long().to(self.device)

    def observe(self, inputs, labels, not_aug_inputs, poisoned_flags):
        real_batch_size = inputs.shape[0]
        if self.args.buffer_retrieve_mode == 'mir':
            self.opt.zero_grad()
            # Step 1: Forward pass for `inputs` and calculate loss
            outputs = self.net(inputs)
            loss_inputs = self.loss(outputs, labels)

            # Step 2: Backward for `inputs` (retain graph for later buffer loss)
            self.opt.zero_grad()
            loss_inputs.backward(retain_graph=True)

            # Step 3: Retrieve data from buffer if not empty
            if not self.buffer.is_empty():
                buf_data = self.buffer.get_data(
                    self.args.minibatch_size,
                    transform=self.transform,
                    mode=self.args.buffer_retrieve_mode,
                )
                buf_inputs, buf_logits = buf_data[0], buf_data[1]
                if poisoned_flags is not None:
                    self.poisoned_flags = buf_data[2]

                # Forward pass for buffer data
                buf_outputs = self.net(buf_inputs)
                loss_buf = self.args.alpha * F.cross_entropy(buf_outputs, buf_logits)

                # Combine the losses and perform backward
                loss = loss_inputs + loss_buf
                self.opt.zero_grad()  # Reset gradients before combined backward
                loss.backward()       # Perform backward pass for the combined loss
                self.opt.step()       # Perform optimization step
            
            else:
                loss = loss_inputs
                self.opt.step()

            

            # Step 4: Backpropagation and optimization
        else:
            self.opt.zero_grad()

            outputs = self.net(inputs)
            loss = self.loss(outputs, labels)

            if not self.buffer.is_empty():
                buf_data = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
                buf_inputs, buf_logits = buf_data[0], buf_data[1]

                buf_outputs = self.net(buf_inputs)
                
                loss += self.args.alpha * F.cross_entropy(buf_outputs, buf_logits)

            loss.backward()
            self.opt.step()

        # self.buffer.add_data(examples=not_aug_inputs, logits=outputs.data)
        self.buffer.add_data(
            examples=not_aug_inputs,
            labels=labels[:real_batch_size],
            poisoned_flags=poisoned_flags,
        )

        return loss.item()
    
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
