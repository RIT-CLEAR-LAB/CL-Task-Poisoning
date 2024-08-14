
from typing import Tuple
from PIL import Image

import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
from torchvision.datasets import FashionMNIST
import numpy as np
import torch.optim
import torch.nn.functional as F

from datasets.utils.continual_dataset import ContinualDataset
from utils.conf import base_path_dataset as base_path
from datasets.mammoth_dataset import MammothDataset
from datasets.transforms.to_thre_channels import ToThreeChannels


class TrainFashionMNIST(MammothDataset, FashionMNIST):
    def __init__(self, root, transform, target_transform: dict, not_aug_transform, download=False):
        super().__init__(root, True, transform, target_transform, download)
        self.not_aug_transform = not_aug_transform
        self.classes = list(range(10))

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        original_img = self.not_aug_transform(img.copy())

        img = self.transform(img)
        target = self.target_transform[target]

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target, original_img

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

        # switch meta-class in target_transform dictionary
        for c in classes:
            self.target_transform[c] = - (self.target_transform[c] - 1)

    def prepare_normal_data(self):
        pass


class TestFashionMNIST(MammothDataset, FashionMNIST):
    def __init__(self, root, transform, target_transform: dict, download=False):
        super().__init__(root, False, transform, target_transform, download)
        self.classes = list(range(10))

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        img = self.transform(img)
        target = self.target_transform[target]

        if hasattr(self, 'logits'):
            return img, target, self.logits[index]

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

        # switch meta-class in target_transform dictionary
        for c in classes:
            self.target_transform[c] = - (self.target_transform[c] - 1)

    def prepare_normal_data(self):
        pass


class SequentialFashionMNIST(ContinualDataset):

    NAME = 'seq-fashionmnist'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5

    TRANSFORM = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ToThreeChannels(),
    ])

    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        ToThreeChannels(),
    ])

    NO_AUG = transforms.Compose([
        transforms.ToTensor(),
        ToThreeChannels(),
    ])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        metaclasses = [0] * 5 + [1] * 5
        random_state = np.random.RandomState(self.args.seed)
        random_state.shuffle(metaclasses)
        self.metaclass_mapping = {c: mc for c, mc in zip(range(10), metaclasses)}

    def get_dataset(self, train=True):
        if train:
            return TrainFashionMNIST(base_path() + 'FASHIONMNIST', transform=self.TRANSFORM, target_transform=self.metaclass_mapping,
                                     not_aug_transform=self.NO_AUG, download=True)
        else:
            return TestFashionMNIST(base_path() + 'FASHIONMNIST', transform=self.TEST_TRANSFORM, target_transform=self.metaclass_mapping,
                                    download=True)

    def get_transform(self):
        return transforms.Compose([transforms.ToPILImage(), self.TRANSFORM])

    @staticmethod
    def get_backbone():
        backbone = resnet18(nclasses=2)
        return backbone

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialFashionMNIST.get_batch_size()

    @staticmethod
    def get_scheduler(model, args) -> torch.optim.lr_scheduler:
        model.opt = torch.optim.SGD(model.net.parameters(), lr=args.lr, weight_decay=args.optim_wd, momentum=args.optim_mom)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [35, 45], gamma=0.1, verbose=False)
        return scheduler
