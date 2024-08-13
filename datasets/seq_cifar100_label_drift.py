from typing import Tuple
import torch.nn.functional as F
import torch.optim
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18, conv3x3
from PIL import Image
from torchvision.datasets import CIFAR100
import numpy as np
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import ContinualDataset
from utils.conf import base_path_dataset as base_path
from datasets.mammoth_dataset import MammothDataset
from datasets.utils.cifar100_label_mapping import (
    superclass_target_mapping,
    metaclass_target_mapping,
)


class TrainCIFAR100LabelDrift(MammothDataset, CIFAR100):
    def __init__(self, root, transform, target_transform, not_aug_transform):
        super().__init__(
            root,
            train=True,
            transform=transform,
            target_transform=target_transform,
            download=True,
        )
        self.superclass_mapping = target_transform.superclass_mapping
        self.metaclass_mapping = target_transform.metaclass_mapping
        self.not_aug_transform = not_aug_transform
        self.classes = list(range(100))

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
        not_aug_img = self.not_aug_transform(original_image)
        img = self.transform(img)
        target = self.superclass_mapping[target]
        target = self.metaclass_mapping[target]

        if hasattr(self, "logits"):
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

        # switch meta-class in target_transform dictionary
        for c in classes:
            s = self.superclass_mapping[c]
            self.metaclass_mapping[s] = 0 if self.metaclass_mapping[s] == 1 else 1

    def prepare_normal_data(self):
        pass


class TestCIFAR100LabelDrift(MammothDataset, CIFAR100):
    """Workaround to avoid printing the already downloaded messages."""

    def __init__(self, root, transform, target_transform):
        self.root = root
        super().__init__(
            root,
            train=False,
            transform=transform,
            target_transform=target_transform,
            download=not self._check_integrity(),
        )
        self.superclass_mapping = target_transform.superclass_mapping
        self.metaclass_mapping = target_transform.metaclass_mapping
        self.classes = list(range(100))

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
        target = self.superclass_mapping[target]
        target = self.metaclass_mapping[target]

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

        # switch meta-class in target_transform dictionary
        for c in classes:
            s = self.superclass_mapping[c]
            self.metaclass_mapping[s] = 0 if self.metaclass_mapping[s] == 1 else 1

    def prepare_normal_data(self):
        pass


class TargetTransform:
    def __init__(self, superclass_mapping, metaclass_mapping):
        self.superclass_mapping = superclass_mapping
        self.metaclass_mapping = metaclass_mapping


class SequentialCIFAR100LabelDrift(ContinualDataset):

    NAME = "seq-cifar100-label-drift"
    SETTING = "class-il"
    N_CLASSES_PER_TASK = 5
    N_TASKS = 20

    TRANSFORM = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    NO_AUG_TRANSFORM = transforms.Compose([transforms.ToTensor()])

    TARGET_TRANSFORM = TargetTransform(
        superclass_target_mapping, metaclass_target_mapping
    )

    def get_dataset(self, train=True):
        """returns native version of represented dataset"""

        if train:
            return TrainCIFAR100LabelDrift(
                base_path() + "CIFAR100LabelDrift",
                transform=self.TRANSFORM,
                target_transform=self.TARGET_TRANSFORM,
                not_aug_transform=self.NO_AUG_TRANSFORM,
            )
        else:
            return TestCIFAR100LabelDrift(
                base_path() + "CIFAR100LabelDrift",
                transform=self.NO_AUG_TRANSFORM,
                target_transform=self.TARGET_TRANSFORM,
            )

    def get_transform(self):
        transform = transforms.Compose([transforms.ToPILImage(), self.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        backbone = resnet18(nclasses=2)
        backbone.conv1 = conv3x3(3, backbone.nf)
        return backbone

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        )
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        return transform

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialCIFAR100LabelDrift.get_batch_size()

    @staticmethod
    def get_scheduler(model, args) -> torch.optim.lr_scheduler:
        model.opt = torch.optim.SGD(
            model.net.parameters(),
            lr=args.lr,
            weight_decay=args.optim_wd,
            momentum=args.optim_mom,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            model.opt, [35, 45], gamma=0.1, verbose=False
        )
        return scheduler
