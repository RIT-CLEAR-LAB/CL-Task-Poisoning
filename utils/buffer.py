# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def icarl_replay(self, dataset, val_set_split=0):
    """
    Merge the replay buffer with the current task data.
    Optionally split the replay buffer into a validation set.

    :param self: the model instance
    :param dataset: the dataset
    :param val_set_split: the fraction of the replay buffer to be used as validation set
    """

    if self.task > 0:
        buff_val_mask = torch.rand(len(self.buffer)) < val_set_split
        val_train_mask = torch.zeros(len(dataset.train_loader.dataset.data)).bool()
        val_train_mask[torch.randperm(len(dataset.train_loader.dataset.data))[:buff_val_mask.sum()]] = True

        if val_set_split > 0:
            self.val_loader = deepcopy(dataset.train_loader)

        data_concatenate = torch.cat if isinstance(dataset.train_loader.dataset.data, torch.Tensor) else np.concatenate
        need_aug = hasattr(dataset.train_loader.dataset, 'not_aug_transform')
        if not need_aug:
            def refold_transform(x): return x.cpu()
        else:
            data_shape = len(dataset.train_loader.dataset.data[0].shape)
            if data_shape == 3:
                def refold_transform(x): return (x.cpu() * 255).permute([0, 2, 3, 1]).numpy().astype(np.uint8)
            elif data_shape == 2:
                def refold_transform(x): return (x.cpu() * 255).squeeze(1).type(torch.uint8)

        # REDUCE AND MERGE TRAINING SET
        dataset.train_loader.dataset.targets = np.concatenate([
            dataset.train_loader.dataset.targets[~val_train_mask],
            self.buffer.labels.cpu().numpy()[:len(self.buffer)][~buff_val_mask]
        ])
        dataset.train_loader.dataset.data = data_concatenate([
            dataset.train_loader.dataset.data[~val_train_mask],
            refold_transform((self.buffer.examples)[:len(self.buffer)][~buff_val_mask])
        ])

        if val_set_split > 0:
            # REDUCE AND MERGE VALIDATION SET
            self.val_loader.dataset.targets = np.concatenate([
                self.val_loader.dataset.targets[val_train_mask],
                self.buffer.labels.cpu().numpy()[:len(self.buffer)][buff_val_mask]
            ])
            self.val_loader.dataset.data = data_concatenate([
                self.val_loader.dataset.data[val_train_mask],
                refold_transform((self.buffer.examples)[:len(self.buffer)][buff_val_mask])
            ])


def reservoir(num_seen_examples: int, buffer_size: int, current_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size or current_size < buffer_size:
        return min(num_seen_examples, current_size)

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


def balanced_reservoir_sampling(num_seen_examples: int, buffer_size: int, current_size: int, labels: torch.Tensor) -> int:
    """
    Balanced reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size or current_size < buffer_size:
        return min(num_seen_examples, current_size)

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        classes, counts = torch.unique(labels, return_counts=True)
        i = torch.argmax(counts).item()
        l = classes[i]
        idx = torch.argwhere(labels == l).flatten()
        rand_idx = np.random.randint(0, len(idx))
        rand = idx[rand_idx]
        return rand
    else:
        return -1


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size


class Buffer:
    """
    The memory buffer of rehearsal method.
    """

    def __init__(self, buffer_size, device, n_tasks=None, mode='reservoir'):
        assert mode in ('ring', 'reservoir', 'balanced', 'reservoir_batch')
        self.mode = mode
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.current_size = 0
        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ['examples', 'labels', 'logits', 'task_labels', 'original_labels']

    def to(self, device):
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self

    def __len__(self):
        return min(self.num_seen_examples, self.current_size, self.buffer_size)

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, task_labels: torch.Tensor, original_labels: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                setattr(self, attr_str, torch.full((self.buffer_size, *attr.shape[1:]), fill_value=-1, dtype=typ, device=self.device))

    def add_data(self, examples, labels=None, logits=None, task_labels=None, original_labels=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels, original_labels)
        if self.mode == 'reservoir_batch':
            self.reservoir_batch(examples, labels, logits, task_labels, original_labels)
        else:
            for i in range(examples.shape[0]):
                if self.mode == 'reservoir' or self.mode == 'ring':
                    index = reservoir(self.num_seen_examples, self.buffer_size, self.current_size)
                elif self.mode == 'balanced':
                    index = balanced_reservoir_sampling(self.num_seen_examples, self.buffer_size, self.current_size, self.labels)
                else:
                    raise ValueError('Invalid mode')
                self.num_seen_examples += 1
                if index >= 0:
                    self.current_size = min(self.current_size + 1, self.buffer_size)
                    self.examples[index] = examples[i].to(self.device)
                    if labels is not None:
                        self.labels[index] = labels[i].to(self.device)
                    if logits is not None:
                        self.logits[index] = logits[i].to(self.device)
                    if task_labels is not None:
                        self.task_labels[index] = task_labels[i].to(self.device)
                    if original_labels is not None:
                        self.original_labels[index] = original_labels[i].to(self.device)

    def get_data(self, size: int, transform: nn.Module = None, return_index=False) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.examples.shape[0], self.current_size):
            size = min(self.num_seen_examples, self.examples.shape[0], self.current_size)

        cur_size = min(self.num_seen_examples, self.examples.shape[0], self.current_size)
        choice = np.random.choice(np.arange(cur_size), size=size, replace=False)
        if transform is None:
            def transform(x): return x
        ret_tuple = (torch.stack([transform(ee.cpu()) for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        if not return_index:
            return ret_tuple
        else:
            return (torch.tensor(choice).to(self.device), ) + ret_tuple
    
    def reservoir_batch(self, examples, labels=None, logits=None, task_labels=None, original_labels=None) -> int:
        """
        Reservoir sampling algorithm.
        :param num_seen_examples: the number of seen examples
        :param buffer_size: the maximum buffer size
        :return: the target index if the current image is sampled, else -1
        """
        batch_size = examples.shape[0]

        # add whatever still fits in the buffer
        place_left = max(0, self.buffer_size - self.current_size)
        if place_left:
            offset = min(place_left, batch_size)
            self.examples[self.current_size: self.current_size + offset].data.copy_(examples[:offset])
            if labels is not None:
                self.labels[self.current_size: self.current_size + offset].data.copy_(labels[:offset])
            if logits is not None:
                self.logits[self.current_size: self.current_size + offset].data.copy_(logits[:offset])
            if task_labels is not None:
                self.task_labels[self.current_size: self.current_size + offset].data.copy_(task_labels[:offset])
            if original_labels is not None:
                self.original_labels[self.current_size: self.current_size + offset].data.copy_(original_labels[:offset])
            
            self.current_size += offset
            self.num_seen_examples += offset
            # everything was added
            # if offset == batch_size:
            #     filled_idx = list(range(self.current_size - offset, self.current_size, ))
            #     return filled_idx


        #TODO: the buffer tracker will have bug when the mem size can't be divided by batch size

        # remove what is already in the buffer
        examples = examples[place_left:]
        if labels is not None:
            labels = labels[place_left:]
        if logits is not None:
            logits = logits[place_left:]
        if task_labels is not None:
            task_labels = task_labels[place_left:]
        if original_labels is not None:
            original_labels = original_labels[place_left:]

        indices = torch.FloatTensor(examples.shape[0]).to(examples.device).uniform_(0, self.num_seen_examples).long()
        valid_indices = (indices < self.buffer_size).long()

        idx_new_data = valid_indices.nonzero().squeeze(-1)
        idx_buffer   = indices[idx_new_data]

        self.num_seen_examples += examples.shape[0]

        if idx_buffer.numel() == 0:
            return []

        assert idx_buffer.max() < self.buffer_size
        # assert idx_buffer.max() < self.buffer_task.size(0)
        assert idx_new_data.max() < examples.shape[0]

        idx_map = {idx_buffer[i].item(): idx_new_data[i].item() for i in range(idx_buffer.size(0))}
        self.examples[list(idx_map.keys())] = examples[list(idx_map.values())]
        if labels is not None:
            self.labels[list(idx_map.keys())] = labels[list(idx_map.values())]
        if logits is not None:
            self.logits[list(idx_map.keys())] = logits[list(idx_map.values())]
        if task_labels is not None:
            self.task_labels[list(idx_map.keys())] = task_labels[list(idx_map.values())]
        if original_labels is not None:
            self.original_labels[list(idx_map.keys())] = original_labels[list(idx_map.values())]


    def get_data_by_index(self, indexes, transform: nn.Module = None) -> Tuple:
        """
        Returns the data by the given index.
        :param index: the index of the item
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if transform is None:
            def transform(x): return x
        ret_tuple = (torch.stack([transform(ee.cpu())
                                  for ee in self.examples[indexes]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str).to(self.device)
                ret_tuple += (attr[indexes],)
        return ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0 or self.current_size == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: nn.Module = None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None:
            def transform(x): return x
        ret_tuple = (torch.stack([transform(ee.cpu())
                                  for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0
        self.current_size = 0

    def get_class_data(self, label: int, labeldrift: bool = False) -> torch.Tensor:
        """
        Return all samples with given label.
        If label not present in the buffer, then raise ValueError exception.
        """
        if labeldrift:
            idx = torch.argwhere(self.original_labels == label).flatten()
        else:
            idx = torch.argwhere(self.labels == label).flatten()
        if len(idx) == 0:
            print(f'{"Metaclass" if labeldrift else "Class"} label {label} not present in the buffer')
            return 0
        class_samples = self.examples[idx]
        return class_samples

    def get_class_sample_count(self, label: int) -> torch.Tensor:
        """
        Return number of samples present in the buffer with given label.
        """
        if hasattr(self, 'labels'):
            idx = torch.argwhere(self.labels == label).flatten()
            return len(idx)
        else:
            return 0

    def flush_class(self, label: int, labeldrift: bool = False) -> None:
        """
        Removes all samples with given label.
        If label not present in the buffer, then raise ValueError exception.
        """
        if labeldrift:
            idx = torch.argwhere(self.original_labels != label).flatten()
            num_removed = len(self.original_labels) - len(idx)
        else:
            idx = torch.argwhere(self.labels != label).flatten()
            num_removed = len(self.labels) - len(idx)
        if num_removed == 0:
            raise ValueError(f'{"Metaclass" if labeldrift else "Class"} label {label} not present in the buffer')
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                tensor = getattr(self, attr_str)
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                padding = torch.full((num_removed, *tensor.shape[1:]), fill_value=-1, dtype=typ, device=self.device)
                new_tensor = torch.cat([tensor[idx], padding], dim=0)
                assert new_tensor.shape == tensor.shape
                setattr(self, attr_str, new_tensor)
        self.current_size = max(self.current_size - num_removed, 0)
        self.num_seen_examples -= num_removed
        print(f'{"Metaclass" if labeldrift else "Class"} {label} samples removed: {num_removed}')
