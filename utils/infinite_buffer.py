import torch
import torch.nn as nn
import numpy as np
import imagehash
import torchvision.transforms.functional as F

from typing import Tuple


class InfiniteBuffer:
    def __init__(self, device, increment_size=5000) -> None:
        self.device = device
        self.increment_size = increment_size
        self.last_idx = 0

        self.attributes = ['examples', 'labels', 'logits', 'task_labels']
        self.image_hashes = set()

    def __len__(self):
        return self.last_idx

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, task_labels: torch.Tensor) -> None:
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
                setattr(self, attr_str, torch.full((self.increment_size, *attr.shape[1:]), fill_value=-1, dtype=typ, device=self.device))

    def add_data(self, examples, labels=None, logits=None, task_labels=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels)

        for i in range(examples.shape[0]):
            if self.last_idx >= len(self.examples):
                self.grow_buffer()
            img = F.to_pil_image(examples[i])
            img_hash = imagehash.phash(img, hash_size=32)
            if img_hash in self.image_hashes:
                continue
            self.examples[self.last_idx] = examples[i].to(self.device)
            self.image_hashes.add(img_hash)
            if labels is not None:
                self.labels[self.last_idx] = labels[i].to(self.device)
            if logits is not None:
                self.logits[self.last_idx] = logits[i].to(self.device)
            if task_labels is not None:
                self.task_labels[self.last_idx] = task_labels[i].to(self.device)
            self.last_idx += 1

    def grow_buffer(self):
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                new_tensor = torch.full((self.increment_size, *attr.shape[1:]), fill_value=-1, dtype=typ, device=self.device)
                setattr(self, attr_str, torch.cat([attr, new_tensor]))

    def get_data(self, size: int, transform: nn.Module = None, return_index=False) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > self.last_idx:
            size = self.last_idx

        choice = np.random.choice(self.last_idx, size=size, replace=False)
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

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if hasattr(self, 'examples'):
            return len(self.examples) == 0
        else:
            return True

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

    def get_class_data(self, label: int) -> torch.Tensor:
        """
        Return all samples with given label.
        If label not present in the buffer, then raise ValueError exception.
        """
        idx = torch.argwhere(self.labels == label).flatten()
        if len(idx) == 0:
            print(f'Class label {label} not present in the buffer')
            return []
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

    def flush_class(self, label: int) -> None:
        """
        Removes all samples with given label.
        If label not present in the buffer, then raise ValueError exception.
        """
        idx = torch.argwhere(self.labels != label).flatten()
        num_removed = len(self.labels) - len(idx)
        if num_removed == 0:
            raise ValueError(f'Class label {label} not present in the buffer')
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                tensor = getattr(self, attr_str)
                setattr(self, attr_str, tensor[idx])
        self.last_idx -= num_removed
        print(f"Class {label} samples removed: {num_removed}")
