# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Tuple
import torch.nn.functional as F
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
from PIL import Image
import numpy as np

from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import ContinualDataset
from utils.conf import base_path_dataset as base_path
from datasets.mammoth_dataset import MammothDataset


def download_tinyimagenet(root):
    if os.path.isdir(root) and len(os.listdir(root)) > 0:
        print('Download not needed, files already on disk.')
    else:
        from onedrivedownloader import download

        print('Downloading dataset')
        ln = "https://unimore365-my.sharepoint.com/:u:/g/personal/263133_unimore_it/EVKugslStrtNpyLGbgrhjaABqRHcE3PB_r2OEaV7Jy94oQ?e=9K29aD"
        download(ln, filename=os.path.join(root, 'tiny-imagenet-processed.zip'), unzip=True, unzip_path=root, clean=True)


def load_data(root, train=True):
    data = []
    for num in range(20):
        data.append(np.load(os.path.join(
            root, 'processed/x_%s_%02d.npy' %
                  ('train' if train else 'val', num + 1))))
    data = np.concatenate(np.array(data))

    targets = []
    for num in range(20):
        targets.append(np.load(os.path.join(
            root, 'processed/y_%s_%02d.npy' %
                  ('train' if train else 'val', num + 1))))
    targets = np.concatenate(np.array(targets))

    return data, targets


class TrainTinyImagenetLabelPoisoning(MammothDataset):
    def __init__(self, root: str, transform, not_aug_transform, poisoning_severity) -> None:
        super().__init__()
        self.transform = transform
        self.not_aug_transform = not_aug_transform
        self.poisoning_severity = poisoning_severity
        download_tinyimagenet(root)
        self.data, self.targets = load_data(root, train=True)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:

        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(np.uint8(255 * img))
        original_img = img.copy()
        img = self.transform(img)
        not_aug_img = self.not_aug_transform(original_img)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img

    def __len__(self):
        return len(self.data)

    def select_classes(self, current_classes: list[int]):
        if len(current_classes) == 0:
            self.data = np.array([])
            self.targets = np.array([])
            self.classes = []
            return

        mask = np.zeros_like(self.targets)
        for label in current_classes:
            mask = np.logical_or(mask, self.targets == label)
        self.data = self.data[mask]
        self.targets = self.targets[mask]
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


class TestTinyImagenetLabelPoisoning(MammothDataset):
    def __init__(self, root, transform, poisoning_severity) -> None:
        super().__init__()

        self.transform = transform
        self.poisoning_severity = poisoning_severity
        download_tinyimagenet(root)
        self.data, self.targets = load_data(root, train=False)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:

        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(np.uint8(255 * img))
        img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def select_classes(self, current_classes: list[int]):
        if len(current_classes) == 0:
            self.data = np.array([])
            self.targets = np.array([])
            self.classes = []
            return

        mask = np.zeros_like(self.targets)
        for label in current_classes:
            mask = np.logical_or(mask, self.targets == label)
        self.data = self.data[mask]
        self.targets = self.targets[mask]
        self.classes = current_classes

    def apply_poisoning(self, poisoned_classes: list):
        pass

    def prepare_normal_data(self):
        pass


class SequentialTinyImagenetLabelPoisoning(ContinualDataset):

    NAME = 'seq-tinyimg-label-poisoning'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 10

    TRANSFORM = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4480, 0.3975), (0.2770, 0.2691, 0.2821)),
    ])

    TEST_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4480, 0.3975), (0.2770, 0.2691, 0.2821)),
        ])

    NO_AUG_TRANSFORM = transforms.Compose([transforms.ToTensor()])

    def get_dataset(self, train=True):
        POISONING_SEVERITY = self.args.poisoning_severity

        if train:
            return TrainTinyImagenetLabelPoisoning(
                base_path() + "TINYIMG",
                transform=self.TRANSFORM,
                not_aug_transform=self.NO_AUG_TRANSFORM,
                poisoning_severity=POISONING_SEVERITY,
            )
        else:
            return TestTinyImagenetLabelPoisoning(
                base_path() + "TINYIMG",
                transform=self.TEST_TRANSFORM,
                poisoning_severity=POISONING_SEVERITY,
            )

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(SequentialTinyImagenetLabelPoisoning.N_CLASSES_PER_TASK
                        * SequentialTinyImagenetLabelPoisoning.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4802, 0.4480, 0.3975),
                                         (0.2770, 0.2691, 0.2821))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4802, 0.4480, 0.3975),
                                (0.2770, 0.2691, 0.2821))
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
        return SequentialTinyImagenetLabelPoisoning.get_batch_size()