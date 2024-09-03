import copy
from typing import Tuple
import torch.nn.functional as F
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
from PIL import Image
from torchvision.datasets import CIFAR10
import numpy as np
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import ContinualDataset
from utils.conf import base_path_dataset as base_path
from datasets.mammoth_dataset import MammothDataset
from datasets.utils.cifar10_label_mapping import metaclass_target_mapping


class TrainCIFAR10LabelDrift(MammothDataset, CIFAR10):
    def __init__(self, root, transform, metaclass_mapping, not_aug_transform) -> None:
        self.root = root  # Workaround to avoid printing the already downloaded messages
        super().__init__(
            root,
            train=True,
            transform=transform,
            target_transform=None,
            download=not self._check_integrity(),
        )
        self.metaclass_mapping = metaclass_mapping
        self.not_aug_transform = not_aug_transform
        self.classes = list(range(10))

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode="RGB")
        original_image = img.copy()
        original_target = target.copy()
        not_aug_img = self.not_aug_transform(original_image)
        img = self.transform(img)
        target = self.metaclass_mapping[target]

        if hasattr(self, "logits"):
            return img, target, not_aug_img, original_target, self.logits[index]

        return img, target, not_aug_img, original_target

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

        # switch meta-class in target_transform dictionary
        for c in classes:
            self.metaclass_mapping[c] = 0 if self.metaclass_mapping[c] == 1 else 1

    def prepare_normal_data(self):
        pass


class TestCIFAR10LabelDrift(MammothDataset, CIFAR10):
    """Workaround to avoid printing the already downloaded messages."""

    def __init__(self, root, transform, metaclass_mapping) -> None:
        self.root = root
        super().__init__(
            root,
            train=False,
            transform=transform,
            target_transform=None,
            download=not self._check_integrity(),
        )
        self.metaclass_mapping = metaclass_mapping
        self.classes = list(range(10))

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        """
        Args: index (int): Index
        Returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        img = self.transform(img)
        original_target = target.copy()
        target = self.metaclass_mapping[target]

        return img, target, original_target

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

        # switch meta-class in target_transform dictionary
        for c in classes:
            self.metaclass_mapping[c] = 0 if self.metaclass_mapping[c] == 1 else 1

    def prepare_normal_data(self):
        pass


class SequentialCIFAR10LabelDrift(ContinualDataset):

    NAME = "seq-cifar10-label-drift"
    SETTING = "class-il"
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    HAS_LABEL_DRIFT = True

    TRANSFORM = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)),
        ]
    )

    TEST_TRANSFORM = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)),
        ]
    )

    NO_AUG_TRANSFORM = transforms.Compose([transforms.ToTensor()])

    def get_dataset(self, train=True):
        """returns native version of represented dataset"""

        if train:
            return TrainCIFAR10LabelDrift(
                base_path() + "CIFAR10",
                transform=self.TRANSFORM,
                metaclass_mapping=copy.deepcopy(metaclass_target_mapping),
                not_aug_transform=self.NO_AUG_TRANSFORM,
            )
        else:
            return TestCIFAR10LabelDrift(
                base_path() + "CIFAR10",
                transform=self.TEST_TRANSFORM,
                metaclass_mapping=copy.deepcopy(metaclass_target_mapping),
            )

    def get_transform(self):
        transform = transforms.Compose([transforms.ToPILImage(), self.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(nclasses=2)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)
        )
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615))
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
        return SequentialCIFAR10LabelDrift.get_batch_size()
