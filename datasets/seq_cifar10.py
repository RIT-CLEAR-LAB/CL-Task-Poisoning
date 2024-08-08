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
from datasets.transforms.driftTransforms import DefocusBlur, GaussianNoise, ShotNoise, SpeckleNoise, Identity
from datasets.mammoth_dataset import MammothDataset


class TrainCIFAR10(MammothDataset, CIFAR10):
    def __init__(self, root: str, train: bool = True, transform=None, target_transform=None,
                 not_aug_transform=None, drift_transform=None,
                 download: bool = False) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.not_aug_transform = not_aug_transform
        self.drift_transform = drift_transform
        assert transform is not None  # TODO fix parameter order
        assert not_aug_transform is not None
        assert drift_transform is not None
        self.classes = list(range(10))

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        if target in self.drifted_classes:
            img = self.drift_transform(img)
            not_aug_img = self.drift_transform(original_img)
        else:
            img = self.transform(img)
            not_aug_img = self.not_aug_transform(original_img)

        if self.target_transform is not None:
            target = self.target_transform(target)

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

    def apply_drift(self, classes: list):
        if len(set(self.classes).union(classes)) == 0:
            return
        self.drifted_classes.extend(classes)

        # TODO: figure out how to apply drift based on transform multiple times
        # maybe we should change transforms or change the drift severity?

    def prepare_normal_data(self):
        pass


class TestCIFAR10(MammothDataset, CIFAR10):
    """Workaround to avoid printing the already downloaded messages."""

    def __init__(self, root, train=True, transform=None, drift_transform=None, target_transform=None, download=False) -> None:
        self.root = root
        super().__init__(root, train, transform, target_transform, download=not self._check_integrity())
        self.drift_transform = drift_transform
        assert transform is not None  # TODO fix parameter order
        assert drift_transform is not None

        self.classes = list(range(10))

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if target in self.drifted_classes:
            img = self.drift_transform(img)
        else:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

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

    def apply_drift(self, classes: list):
        if len(set(self.classes).union(classes)) == 0:
            return
        self.drifted_classes.extend(classes)

    def prepare_normal_data(self):
        pass


class SequentialCIFAR10(ContinualDataset):

    NAME = 'seq-cifar10'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5

    DRIFT_TYPES = [
        DefocusBlur,
        GaussianNoise,
        ShotNoise,
        SpeckleNoise,
        Identity,
    ]

    def get_dataset(self, train=True):
        """returns native version of represented dataset"""
        DRIFT_SEVERITY = self.args.drift_severity
        TRAIN_DRIFT = self.DRIFT_TYPES[self.args.train_drift]
        DRIFT = self.DRIFT_TYPES[self.args.concept_drift]
        TRANSFORM = transforms.Compose([
            TRAIN_DRIFT(DRIFT_SEVERITY),
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        TEST_TRANSFORM = transforms.Compose([
            TRAIN_DRIFT(DRIFT_SEVERITY),
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        if train:
            DRIFT_TRANSFORM = transforms.Compose([
                DRIFT(DRIFT_SEVERITY),
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            return TrainCIFAR10(base_path() + 'CIFAR10', train=True, download=True, transform=TRANSFORM,
                                not_aug_transform=transforms.Compose([transforms.ToTensor()]), drift_transform=DRIFT_TRANSFORM)
        else:
            DRIFT_TRANSFORM = transforms.Compose([
                DRIFT(DRIFT_SEVERITY),
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ])
            return TestCIFAR10(base_path() + 'CIFAR10', train=False, download=True, transform=TEST_TRANSFORM,
                               drift_transform=DRIFT_TRANSFORM)

    def get_transform(self):
        DRIFT_SEVERITY = self.args.drift_severity
        DRIFT = self.DRIFT_TYPES[self.args.train_drift]
        TRANSFORM = transforms.Compose([
            DRIFT(DRIFT_SEVERITY),
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform = transforms.Compose(
            [transforms.ToPILImage(), TRANSFORM])
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
