# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

from backbone.ResNet18 import resnet18
from PIL import Image
from torchvision.datasets import CIFAR10

from utils.conf import base_path_dataset as base_path
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import ContinualDataset
from datasets.transforms.poisoningTransforms import BackdoorAttack
from datasets.mammoth_dataset import MammothDataset


class TrainCIFAR10BackdoorAttack(MammothDataset, CIFAR10):
    def __init__(self, root: str, transform, not_aug_transform, poisoning_transform, target_transform, poisoning_rate) -> None:
        self.root = root    # Workaround to avoid printing the already downloaded messages
        super().__init__(root, train=True, transform=transform, target_transform=target_transform, download=not self._check_integrity())
        self.not_aug_transform = not_aug_transform
        self.poisoning_transform = poisoning_transform
        self.poisoning_rate = poisoning_rate
        self.classes = list(range(10))
        self.poisoned_flags = [0] * len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:

        img, target, is_poisoned = self.data[index], self.targets[index], self.poisoned_flags[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')

        if target in self.poisoned_classes and np.random.rand() < self.poisoning_rate:
            img = self.poisoning_transform(img)
            target = self.target_transform[target]
            is_poisoned = 1

        original_img = img.copy()
        img = self.transform(img)
        not_aug_img = self.not_aug_transform(original_img)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, is_poisoned, self.logits[index]

        return img, target, not_aug_img, is_poisoned

    def select_classes(self, current_classes: list[int]):
        if len(current_classes) == 0:
            self.data = np.array([])
            self.targets = np.array([])
            self.classes = []
            return

        mask = np.zeros_like(np.array(self.targets))
        for label in current_classes:
            mask = np.logical_or(mask, np.array(self.targets) == label)
        self.data = self.data[mask]
        self.targets = np.array(self.targets)[mask]
        self.classes = current_classes

    def apply_poisoning(self, poisoned_classes: list):
        if len(set(self.classes).union(poisoned_classes)) == 0:
            return
        self.poisoned_classes.extend(poisoned_classes)

    def prepare_normal_data(self):
        pass


class TestCIFAR10BackdoorAttack(MammothDataset, CIFAR10):
    def __init__(self, root, transform, poisoning_transform, trigger_rate) -> None:
        self.root = root    # Workaround to avoid printing the already downloaded messages
        super().__init__(root, train=False, transform=transform, target_transform=None, download=not self._check_integrity())
        self.poisoning_transform = poisoning_transform
        self.trigger_rate = trigger_rate
        self.classes = list(range(10))
        self.poisoned_flags = [0] * len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:

        img, target, is_poisoned = self.data[index], self.targets[index], self.poisoned_flags[index]

        # to return a PIL Image
        img = Image.fromarray(img)

        if target in self.poisoned_classes and np.random.rand() < self.trigger_rate:
            img = self.poisoning_transform(img)
            is_poisoned = 1

        img = self.transform(img)

        return img, target, is_poisoned

    def select_classes(self, current_classes: list[int]):
        if len(current_classes) == 0:
            self.data = np.array([])
            self.targets = np.array([])
            self.classes = []
            return

        mask = np.zeros_like(np.array(self.targets))
        for label in current_classes:
            mask = np.logical_or(mask, np.array(self.targets) == label)
        self.data = self.data[mask]
        self.targets = np.array(self.targets)[mask]
        self.classes = current_classes

    def apply_poisoning(self, poisoned_classes: list):
        if len(set(self.classes).union(poisoned_classes)) == 0:
            return
        self.poisoned_classes.extend(poisoned_classes)

    def prepare_normal_data(self):
        pass


class SequentialCIFAR10BackdoorAttack(ContinualDataset):

    NAME = 'seq-cifar10-backdoor-attack'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5

    TRANSFORM = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615))
    ])

    TEST_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615))
    ])

    NO_AUG_TRANSFORM = transforms.Compose([transforms.ToTensor()])

    def get_dataset(self, train=True):
        POISONING = transforms.Compose(
            [
                BackdoorAttack(),
                transforms.ToPILImage(),
            ]
        )
        TARGET_TRANSFORM = {c: 0 for c in range(10)}
        POISONING_RATE = self.args.poisoning_rate / 100
        TRIGGER_RATE = self.args.trigger_rate / 100

        if train:
            return TrainCIFAR10BackdoorAttack(
                base_path() + "CIFAR10",
                transform=self.TRANSFORM,
                not_aug_transform=self.NO_AUG_TRANSFORM,
                poisoning_transform=POISONING,
                target_transform=TARGET_TRANSFORM,
                poisoning_rate=POISONING_RATE,
            )
        else:
            return TestCIFAR10BackdoorAttack(
                base_path() + "CIFAR10",
                transform=self.TEST_TRANSFORM,
                poisoning_transform=POISONING,
                trigger_rate=TRIGGER_RATE,
            )

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(SequentialCIFAR10BackdoorAttack.N_CLASSES_PER_TASK
                        * SequentialCIFAR10BackdoorAttack.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2615))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2615))
        return transform

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialCIFAR10BackdoorAttack.get_batch_size()
