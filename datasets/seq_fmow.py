from PIL import Image
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import pathlib
import collections

from torchvision.models import resnet18, ResNet18_Weights
from wilds import get_dataset

from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import ContinualDataset
from datasets.utils.wilds import WildsDatasetBase, get_valid_classes, get_class_details


class TrainFMoV(WildsDatasetBase):
    def __init__(self, transform, not_aug_transform, class_mapping, class_traps) -> None:
        self.subset_name = 'train'
        self.not_aug_transform = not_aug_transform
        super().__init__(transform, class_mapping, class_traps)

    def set_orignal_data(self):
        dataset = get_dataset(dataset="fmow", download=True).get_subset(self.subset_name)
        self.data_dir = pathlib.Path(dataset.dataset.data_dir)
        self.input_array = np.array([f'rgb_img_{i}.png' for i in dataset.indices])
        self.y_array = dataset.y_array
        self.metadata_array = dataset.metadata_array
        assert len(self.input_array) == len(self.y_array) == len(self.metadata_array)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        img_path = self.data_dir / 'images' / f'rgb_img_{index}.png'
        not_aug_img = Image.open(img_path).convert('RGB')
        label = self.y_array[index]

        img = self.transforms(not_aug_img)
        not_aug_img = self.not_aug_transform(not_aug_img)
        return img, label, not_aug_img


class TestFMoV(WildsDatasetBase):
    def __init__(self, transform, class_mapping, class_traps, use_validation=False) -> None:
        self.subset_name = 'val' if use_validation else 'test'
        super().__init__(transform, class_mapping, class_traps)

    def set_orignal_data(self):
        dataset = get_dataset(dataset="fmow", download=True).get_subset(self.subset_name)
        self.data_dir = pathlib.Path(dataset.dataset.data_dir)
        self.input_array = np.array([f'rgb_img_{i}.png' for i in dataset.indices])
        self.y_array = dataset.y_array
        self.metadata_array = dataset.metadata_array
        assert len(self.input_array) == len(self.y_array) == len(self.metadata_array)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        img_path = self.data_dir / 'images' / f'rgb_img_{index}.png'
        not_aug_img = Image.open(img_path).convert('RGB')
        label = self.y_array[index]

        img = self.transforms(not_aug_img)
        return img, label


class SequentialFMoV(ContinualDataset):
    NAME = 'seq-fmow'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 5
    N_TASKS = 12
    TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # TODO these are ImageNet norm stats
    ])
    NOT_AUG_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_groups = self.stream_spec.max_drifts_per_class + 1
        valid_classes = sorted(get_valid_classes('fmow', min_domains=n_groups))
        self.n_classes = len(valid_classes)
        self.class_mapping = {i: j for i, j in zip(valid_classes, range(len(valid_classes)))}

        n_classes = self.N_CLASSES_PER_TASK * self.N_TASKS
        if self.args.n_slots is not None:
            assert n_classes * n_groups >= self.N_TASKS * self.args.n_slots, f'not enough classes to fill all slots, n_groups = {n_groups}, n_slots={self.args.n_slots}'

        class_traps_ids, _ = get_class_details('fmow')

        self.class_traps = [collections.defaultdict(list) for _ in range(n_groups)]

        for label, trap_ids in class_traps_ids.items():
            if label not in valid_classes:
                continue
            for i in range(n_groups):
                group_size = len(trap_ids) // n_groups
                selected_traps = trap_ids[i*group_size:(i+1)*group_size]
                assert len(selected_traps) > 0
                new_label = self.class_mapping[label]
                self.class_traps[i][new_label].extend(selected_traps)

    def get_dataset(self, train=True):
        """returns native version of represented dataset"""
        if train:
            return TrainFMoV(self.TRANSFORM, self.NOT_AUG_TRANSFORM, self.class_mapping, self.class_traps)
        else:
            return TestFMoV(self.TEST_TRANSFORM, self.class_mapping, self.class_traps, use_validation=self.args.validation)

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialFMoV.TRANSFORM])
        return transform

    def get_backbone(self):
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        n_classes = SequentialFMoV.N_CLASSES_PER_TASK * SequentialFMoV.N_TASKS
        net.fc = nn.Linear(net.fc.weight.shape[1], self.n_classes)
        return net

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
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
        return SequentialFMoV.get_batch_size()
