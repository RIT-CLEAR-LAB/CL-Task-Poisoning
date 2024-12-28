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
from datasets.mammoth_dataset import MammothDataset


class TrainCIFAR100LabelPoisoning(MammothDataset, CIFAR100):
    def __init__(self, root: str, transform, not_aug_transform, poisoning_severity) -> None:
        self.root = root    # Workaround to avoid printing the already downloaded messages
        super().__init__(root, train=True, transform=transform, target_transform=None, download=not self._check_integrity())
        self.not_aug_transform = not_aug_transform
        self.poisoning_severity = poisoning_severity
        self.classes = list(range(100))

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:

        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
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

    def apply_poisoning(self, poisoned_classes: list, current_classes: list):
        if len(set(self.classes).union(poisoned_classes)) == 0:
            return
        self.poisoned_classes.extend(poisoned_classes)

        if len(poisoned_classes) < 2: 
            raise ValueError('At least 2 poisoned classes required to switch labels')

        switch_prob = [0.0, 0.25, 0.5, 0.75, 1.0][self.poisoning_severity - 1]

        for i in range(len(self.targets)):
            original_label = self.targets[i]
            if original_label in poisoned_classes and np.random.rand() < switch_prob:
                possible_labels = [cls for cls in current_classes]
                self.targets[i] = np.random.choice(possible_labels)

    def prepare_normal_data(self):
        pass


class TestCIFAR100LabelPoisoning(MammothDataset, CIFAR100):
    def __init__(self, root, transform, poisoning_severity) -> None:
        self.root = root    # Workaround to avoid printing the already downloaded messages
        super().__init__(root, train=False, transform=transform, target_transform=None, download=not self._check_integrity())
        self.poisoning_severity = poisoning_severity
        self.classes = list(range(100))

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:

        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img)
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


class SequentialCIFAR100LabelPoisoning(ContinualDataset):

    NAME = 'seq-cifar100-label-poisoning'
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

    def get_dataset(self, train=True):
        POISONING_SEVERITY = self.args.poisoning_severity

        if train:
            return TrainCIFAR100LabelPoisoning(
                base_path() + "CIFAR100",
                transform=self.TRANSFORM,
                not_aug_transform=self.NO_AUG_TRANSFORM,
                poisoning_severity=POISONING_SEVERITY,
            )
        else:
            return TestCIFAR100LabelPoisoning(
                base_path() + "CIFAR100",
                transform=self.TEST_TRANSFORM,
                poisoning_severity=POISONING_SEVERITY,
            )

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(SequentialCIFAR100LabelPoisoning.N_CLASSES_PER_TASK
                        * SequentialCIFAR100LabelPoisoning.N_TASKS)

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
        return SequentialCIFAR100LabelPoisoning.get_batch_size()
