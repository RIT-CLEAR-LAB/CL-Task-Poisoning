# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch.nn.functional as F
import torch.optim
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
from PIL import Image
from torchvision.datasets import CIFAR100
import numpy as np

from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import ContinualDataset
from utils.conf import base_path_dataset as base_path
from datasets.transforms.poisoningTransforms import DefocusBlur, GaussianNoise, ShotNoise, SpeckleNoise, PixelPermutation, Identity
from datasets.mammoth_dataset import MammothDataset


class TrainCIFAR100(MammothDataset, CIFAR100):
    def __init__(self, root: str, transform, not_aug_transform, poisoning_transform) -> None:
        self.root = root    # Workaround to avoid printing the already downloaded messages
        super().__init__(root, train=True, transform=transform, target_transform=None, download=not self._check_integrity())
        self.not_aug_transform = not_aug_transform
        self.poisoning_transform = poisoning_transform
        self.classes = list(range(100))

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


class TestCIFAR100(MammothDataset, CIFAR100):
    def __init__(self, root, transform, poisoning_transform) -> None:
        self.root = root    # Workaround to avoid printing the already downloaded messages
        super().__init__(root, train=False, transform=transform, target_transform=None, download=not self._check_integrity())
        self.poisoning_transform = poisoning_transform
        self.classes = list(range(100))

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:

        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img)

        if target in self.poisoned_classes:
            img = self.poisoning_transform(img)

        img = self.transform(img)

        return img, target

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
        pass

    def prepare_normal_data(self):
        pass


class SequentialCIFAR100(ContinualDataset):

    NAME = 'seq-cifar100'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 5
    N_TASKS = 20

    TRANSFORM = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    TEST_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    NO_AUG_TRANSFORM = transforms.Compose([transforms.ToTensor()])

    POISONING_TYPES = [
        DefocusBlur,
        GaussianNoise,
        ShotNoise,
        SpeckleNoise,
        PixelPermutation,
        Identity,
    ]

    def get_dataset(self, train=True):
        POISONING_SEVERITY = self.args.poisoning_severity
        POISONING = transforms.Compose(
            [
                self.POISONING_TYPES[
                    (
                        self.args.image_poisoning_type
                        if self.args.image_poisoning_type is not None
                        else -1
                    )
                ](POISONING_SEVERITY),
                transforms.ToPILImage(),
            ]
        )

        if train:
            return TrainCIFAR100(
                base_path() + "CIFAR100",
                transform=self.TRANSFORM,
                not_aug_transform=self.NO_AUG_TRANSFORM,
                poisoning_transform=POISONING,
            )
        else:
            return TestCIFAR100(
                base_path() + "CIFAR100",
                transform=self.TEST_TRANSFORM,
                poisoning_transform=POISONING,
            )

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(SequentialCIFAR100.N_CLASSES_PER_TASK
                        * SequentialCIFAR100.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.5071, 0.4867, 0.4408),
                                         (0.2675, 0.2565, 0.2761))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.5071, 0.4867, 0.4408),
                                (0.2675, 0.2565, 0.2761))
        return transform

    @staticmethod
    def get_scheduler(model, args) -> torch.optim.lr_scheduler:
        model.opt = torch.optim.SGD(model.net.parameters(), lr=args.lr, weight_decay=args.optim_wd, momentum=args.optim_mom)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [35, 45], gamma=0.1, verbose=False)
        return scheduler

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialCIFAR100.get_batch_size()
