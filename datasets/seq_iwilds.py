from PIL import Image
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torchvision.transforms as transforms

from wilds import get_dataset
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import ContinualDataset
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, ResNet18_Weights
from datasets.mammoth_dataset import MammothDataset

import copy
import collections
import functools


@functools.cache
def get_class_details() -> Tuple[dict[int, list], collections.Counter]:
    dataset = get_dataset(dataset="iwildcam", download=True)
    class_traps_ids = collections.defaultdict(set)
    class_counts = collections.Counter()

    for _, label, metadata in dataset:
        label = label.item()
        trap_id = metadata[0].item()
        class_traps_ids[label].add(trap_id)
        class_counts.update([label])

    for label, traps_ids in class_traps_ids.items():
        class_traps_ids[label] = list(traps_ids)

    return class_traps_ids, class_counts


def get_valid_classes(min_samples=500, min_domains=2):
    """return list of classes that have at last 500 samples and at least 2 domains"""
    class_traps_ids, class_counts = get_class_details()
    selected_classes = [label for label in class_traps_ids if len(class_traps_ids[label]) >= min_domains and class_counts[label] >= min_samples]
    return selected_classes


class WildsDatasetBase(MammothDataset):
    def __init__(self, transform, class_mapping, class_traps):
        super().__init__()
        self.transforms = transform
        self.class_mapping = class_mapping
        self.class_traps = class_traps

        n_classes = len(class_mapping)
        self.group_counter = [0 for _ in range(n_classes)]

        self.set_orignal_data()
        self.y_array = torch.Tensor([self.class_mapping[c.item()] if c.item() in self.class_mapping else -1 for c in self.y_array]).to(torch.long)
        valid_classes = list(class_mapping.values())
        self.select_classes(valid_classes)

    def set_orignal_data(self):
        dataset = get_dataset(dataset="iwildcam", download=True).get_subset(self.subset_name)
        self.data_dir = dataset.dataset.data_dir
        self.input_array = dataset.dataset._input_array[dataset.indices]
        self.y_array = dataset.y_array
        self.metadata_array = dataset.metadata_array
        assert len(self.input_array) == len(self.y_array) == len(self.metadata_array)

    def select_classes(self, classes_list: list[int]) -> None:
        if len(classes_list) == 0:
            self.input_array = []
            self.y_array = []
            self.metadata_array = []
            self.classes = []
            return

        idx = list()
        for c in classes_list:
            idx.extend(torch.argwhere(self.y_array == c).flatten().tolist())
        self.input_array = self.input_array[idx]
        self.y_array = self.y_array[idx]
        self.metadata_array = self.metadata_array[idx]

        self.classes = classes_list

    def apply_drift(self, classes: list):
        if len(set(self.classes).union(classes)) == 0:
            return
        self.drifted_classes.extend(classes)

        selected_traps = dict()
        for label in classes:
            self.group_counter[label] += 1
            group_idx = self.group_counter[label]
            selected_traps[label] = self.class_traps[group_idx][label]

        # we need to make sure that applying more than one drift works
        self.set_orignal_data()
        self.y_array = torch.Tensor([self.class_mapping[c.item()] if c.item() in self.class_mapping else -1 for c in self.y_array]).to(torch.long)
        self.select_classes(self.classes)
        self.select_domains(selected_traps)

    def prepare_normal_data(self):
        selected_traps = dict()
        for label in self.classes:
            allowed_traps = self.class_traps[0][label]
            selected_traps[label] = allowed_traps

        # we need to make sure that applying more than one drift works
        self.select_domains(selected_traps)

    def select_domains(self, class_traps: dict[list]) -> None:
        idx = list()
        for i, (label, meta) in enumerate(zip(self.y_array, self.metadata_array)):
            label = label.item()
            trap_id = meta[0].item()
            if label not in class_traps:
                idx.append(i)
                continue
            if trap_id in class_traps[label]:
                idx.append(i)

        self.input_array = self.input_array[idx]
        self.y_array = self.y_array[idx]
        self.metadata_array = self.metadata_array[idx]

    def __len__(self):
        return len(self.y_array)


class TrainIWilds(WildsDatasetBase):
    def __init__(self, transform, not_aug_transform, class_mapping, class_traps) -> None:
        self.subset_name = 'train'
        self.not_aug_transform = not_aug_transform
        super().__init__(transform, class_mapping, class_traps)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        img_path = self.data_dir / 'train' / self.input_array[index]
        not_aug_img = Image.open(img_path)
        label = self.y_array[index]

        img = self.transforms(not_aug_img)
        not_aug_img = self.not_aug_transform(not_aug_img)
        return img, label, not_aug_img


class TestIWilds(WildsDatasetBase):
    def __init__(self, transform, class_mapping, class_traps, use_validation=False) -> None:
        self.subset_name = 'val' if use_validation else 'test'
        super().__init__(transform, class_mapping, class_traps)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        assert len(self.input_array) == len(self.y_array) == len(self.metadata_array)
        img_path = self.data_dir / 'train' / self.input_array[index]
        not_aug_img = Image.open(img_path)
        label = self.y_array[index]

        img = self.transforms(not_aug_img)
        return img, label


class SequentialIWilds(ContinualDataset):
    NAME = 'seq-iwilds'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 5
    N_TASKS = 8
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
        valid_classes = sorted(get_valid_classes(min_domains=n_groups))
        self.n_classes = len(valid_classes)
        self.class_mapping = {i: j for i, j in zip(valid_classes, range(len(valid_classes)))}

        n_classes = self.N_CLASSES_PER_TASK * self.N_TASKS
        if self.args.n_slots is not None:
            assert n_classes * n_groups >= self.N_TASKS * self.args.n_slots, f'not enough classes to fill all slots, n_groups = {n_groups}, n_slots={self.args.n_slots}'

        class_traps_ids, _ = get_class_details()

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
            return TrainIWilds(self.TRANSFORM, self.NOT_AUG_TRANSFORM, self.class_mapping, self.class_traps)
        else:
            return TestIWilds(self.TEST_TRANSFORM, self.class_mapping, self.class_traps, use_validation=self.args.validation)

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialIWilds.TRANSFORM])
        return transform

    def get_backbone(self):
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        n_classes = SequentialIWilds.N_CLASSES_PER_TASK * SequentialIWilds.N_TASKS
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
        return SequentialIWilds.get_batch_size()
