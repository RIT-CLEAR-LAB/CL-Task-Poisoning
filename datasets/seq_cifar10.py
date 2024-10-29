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
from datasets.transforms.poisoningTransforms import DefocusBlur, GaussianNoise, ShotNoise, SpeckleNoise, RandomNoise, PixelPermutation, Identity
from datasets.mammoth_dataset import MammothDataset


class TrainCIFAR10(MammothDataset, CIFAR10):
    def __init__(self, root: str, transform, not_aug_transform, poisoning_transform) -> None:
        self.root = root    # Workaround to avoid printing the already downloaded messages
        super().__init__(root, train=True, transform=transform, target_transform=None, download=not self._check_integrity())
        self.not_aug_transform = not_aug_transform
        self.poisoning_transform = poisoning_transform
        self.classes = list(range(10))

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:

        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')

        if target in self.poisoned_classes:
            img = self.poisoning_transform(img)

        original_img = img.copy()
        img = self.transform(img)
        not_aug_img = self.not_aug_transform(original_img)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img

    def select_classes(self, classes_list: list[int]):
        if len(classes_list) == 0:
            self.data = np.array([])
            self.targets = np.array([])
            self.classes = []
            return

        mask = np.zeros_like(np.array(self.targets))
        for label in classes_list:
            mask = np.logical_or(mask, np.array(self.targets) == label)
        self.data = self.data[mask]
        self.targets = np.array(self.targets)[mask]
        self.classes = classes_list

    def apply_poisoning(self, classes: list):
        if len(set(self.classes).union(classes)) == 0:
            return
        self.poisoned_classes.extend(classes)

    def prepare_normal_data(self):
        pass


class TestCIFAR10(MammothDataset, CIFAR10):
    def __init__(self, root, transform, poisoning_transform) -> None:
        self.root = root    # Workaround to avoid printing the already downloaded messages
        super().__init__(root, train=False, transform=transform, target_transform=None, download=not self._check_integrity())
        self.poisoning_transform = poisoning_transform
        self.classes = list(range(10))

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:

        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img)

        if target in self.poisoned_classes:
            img = self.poisoning_transform(img)

        img = self.transform(img)

        return img, target

    def select_classes(self, classes_list: list[int]):
        if len(classes_list) == 0:
            self.data = np.array([])
            self.targets = np.array([])
            self.classes = []
            return

        mask = np.zeros_like(np.array(self.targets))
        for label in classes_list:
            mask = np.logical_or(mask, np.array(self.targets) == label)
        self.data = self.data[mask]
        self.targets = np.array(self.targets)[mask]
        self.classes = classes_list

    def apply_poisoning(self, classes: list):
        if len(set(self.classes).union(classes)) == 0:
            return
        self.poisoned_classes.extend(classes)

    def prepare_normal_data(self):
        pass


class SequentialCIFAR10(ContinualDataset):

    NAME = 'seq-cifar10'
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

    POISONING_TYPES = [
        DefocusBlur,
        GaussianNoise,
        ShotNoise,
        SpeckleNoise,
        RandomNoise,
        PixelPermutation,
        Identity,
    ]

    def get_dataset(self, train=True):
        """returns native version of represented dataset"""
        POISONING_SEVERITY = self.args.poisoning_severity
        POISONING = transforms.Compose([
            self.POISONING_TYPES[self.args.poisoning_type](POISONING_SEVERITY),
            transforms.ToPILImage()
        ])


        if train:
            return TrainCIFAR10(base_path() + 'CIFAR10',
                                transform=self.TRANSFORM, not_aug_transform=self.NO_AUG_TRANSFORM, poisoning_transform=POISONING)
        else:
            return TestCIFAR10(base_path() + 'CIFAR10',
                               transform=self.TEST_TRANSFORM, poisoning_transform=POISONING)

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(SequentialCIFAR10.N_CLASSES_PER_TASK
                        * SequentialCIFAR10.N_TASKS)

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
        return SequentialCIFAR10.get_batch_size()
