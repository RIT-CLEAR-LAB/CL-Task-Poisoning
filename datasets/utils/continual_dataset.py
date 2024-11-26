# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
from typing import Tuple

import numpy as np
import torch.nn as nn
import torch.optim
import torch.utils
from torch.utils.data import DataLoader
import torch.utils.data
from datasets.stream_spec import StreamSpecification
from datasets.mammoth_dataset import MammothDataset


class ContinualDataset:
    """
    Continual learning evaluation setting.
    """

    NAME: str
    SETTING: str
    N_CLASSES_PER_TASK: int
    N_TASKS: int

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loader = None
        self.test_loaders = []
        self.i = 0
        self.args = args
        self.poisoned_classes = []

        if not all((self.NAME, self.SETTING, self.N_CLASSES_PER_TASK, self.N_TASKS)):
            raise NotImplementedError(
                "The dataset must be initialized with all the required fields."
            )

        if args.n_poisonings or args.n_past_poisonings:
            n_tasks = self.N_TASKS
            n_classes = self.N_CLASSES_PER_TASK * self.N_TASKS
            self.stream_spec = StreamSpecification(
                n_tasks,
                n_classes,
                random_seed=args.seed,
                n_poisonings=args.n_poisonings,
                n_past_poisonings=args.n_past_poisonings,
                classes_per_poisoning=args.classes_per_poisoning,
            )
            self.stream_spec_it = iter(self.stream_spec)

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
        """
        new_classes = list(range(self.i, self.i + self.N_CLASSES_PER_TASK))
        train_dataset = self.get_dataset(train=True)
        train_dataset.select_classes(new_classes)
        test_dataset = self.get_dataset(train=False)
        test_dataset.select_classes(new_classes)

        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4)

        self.train_loader = train_loader
        self.test_loaders.append(test_loader)

        self.i += self.N_CLASSES_PER_TASK
        return train_loader, test_loader

    def get_poisoned_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        current_classes, poisoned_classes = next(self.stream_spec_it)
        self.poisoned_classes = poisoned_classes

        print(f"New Classes: {current_classes}")
        print(f"Poisoned Classes: {poisoned_classes}")

        train_dataset = None
        test_dataset = self.get_dataset(train=False)
        test_dataset.select_classes(current_classes + poisoned_classes)

        if len(current_classes) > 0:
            train_dataset = self.get_dataset(train=True)
            train_dataset.select_classes(current_classes)

        if len(poisoned_classes) > 0:
            poisoned_train_dataset = self.get_dataset(train=True)
            poisoned_train_dataset.select_classes(poisoned_classes)

            assert (
                self.args.n_poisonings is not None
                or self.args.n_past_poisonings is not None
            ), "Must specify poisoning mechanism: n_poisonings or n_past_poisonings \n \
                n_poisonings: For poisoning n-th task(s) with image or label flip poisoning \n \
                n_past_poisonings: For poisoning n-th task(s) with label flip poisoning from previous tasks"

            if self.args.n_poisonings is not None:
                poisoned_train_dataset.apply_poisoning(poisoned_classes)                
            elif self.args.n_past_poisonings is not None:
                assert self.args.poisoning_percentage is not None, "Must specify percentage of poisoned data (0 ~ 100)"
                assert len(train_dataset) > 0, "Need new classes in training data for past label flip poisoning"

                poisoned_train_dataset.apply_past_label_flip_poisoning(poisoned_classes, current_classes)

                total_samples = len(train_dataset)
                num_poisoned_samples = int(total_samples * (self.args.poisoning_percentage / 100))
                num_regular_samples = total_samples - num_poisoned_samples

                num_poisoned_samples = min(len(poisoned_train_dataset), num_poisoned_samples)
                num_regular_samples = min(len(train_dataset), num_regular_samples)
                poisoned_train_dataset = torch.utils.data.Subset(poisoned_train_dataset, range(num_poisoned_samples))
                train_dataset = torch.utils.data.Subset(train_dataset, range(num_regular_samples))

                test_dataset = self.get_dataset(train=False)
                test_dataset.select_classes(current_classes)

            if train_dataset is not None:
                train_dataset = torch.utils.data.ConcatDataset([train_dataset, poisoned_train_dataset])
            else:
                train_dataset = poisoned_train_dataset

        if train_dataset is None:
            raise ValueError("No classes to train on in the current task")

        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        self.train_loader = train_loader

        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4)
        self.test_loaders.append(test_loader)

        return train_loader, test_loader

    def get_dataset(self, train=True) -> MammothDataset:
        """returns native version of represented dataset"""
        raise NotImplementedError

    @staticmethod
    def get_backbone() -> nn.Module:
        """
        Returns the backbone to be used for to the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_transform() -> nn.Module:
        """
        Returns the transform to be used for to the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_loss() -> nn.Module:
        """
        Returns the loss to be used for to the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_normalization_transform() -> nn.Module:
        """
        Returns the transform used for normalizing the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_denormalization_transform() -> nn.Module:
        """
        Returns the transform used for denormalizing the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_scheduler(model, args: Namespace) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Returns the scheduler to be used for to the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_epochs():
        raise NotImplementedError

    @staticmethod
    def get_batch_size():
        raise NotImplementedError

    @staticmethod
    def get_minibatch_size():
        raise NotImplementedError
