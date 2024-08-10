import torch
import functools
import collections
import numpy as np
import pandas as pd

from typing import Tuple
from wilds import get_dataset

from datasets.mammoth_dataset import MammothDataset


@functools.cache
def get_class_details(dataset_name: str) -> Tuple[dict[int, list], collections.Counter]:
    dataset = get_dataset(dataset=dataset_name, download=True)
    class_domain_ids = collections.defaultdict(set)
    _, class_counts = np.unique(dataset.y_array, return_counts=True)

    dataframe = pd.DataFrame(zip(dataset.y_array.numpy(), dataset.metadata_array.numpy()[:, 0]), columns=['labels', 'domains'])
    grouped = dataframe.groupby('labels')
    class_domain_ids = dict()
    for label, grouped_domains in grouped:
        class_domain_ids[label] = list(np.unique(grouped_domains['domains']))

    return class_domain_ids, class_counts


def get_valid_classes(dataset_name: str, min_samples=500, min_domains=2):
    """return list of classes that have at last 500 samples and at least 2 domains"""
    class_domains_ids, class_counts = get_class_details(dataset_name)
    selected_classes = [label for label in class_domains_ids if len(class_domains_ids[label]) >= min_domains and class_counts[label] >= min_samples]
    return selected_classes


class WildsDatasetBase(MammothDataset):
    def __init__(self, transform, class_mapping, class_domains):
        super().__init__()
        self.transforms = transform
        self.class_mapping = class_mapping
        self.class_domains = class_domains

        n_classes = len(class_mapping)
        self.group_counter = [0 for _ in range(n_classes)]

        self.set_orignal_data()
        self.y_array = torch.Tensor([self.class_mapping[c.item()] if c.item() in self.class_mapping else -1 for c in self.y_array]).to(torch.long)
        valid_classes = list(class_mapping.values())
        self.select_classes(valid_classes)

    def set_orignal_data(self):
        raise NotImplementedError

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

        selected_domains = dict()
        for label in classes:
            self.group_counter[label] += 1
            group_idx = self.group_counter[label]
            selected_domains[label] = self.class_domains[group_idx][label]

        # we need to make sure that applying more than one drift works
        self.set_orignal_data()
        self.y_array = torch.Tensor([self.class_mapping[c.item()] if c.item() in self.class_mapping else -1 for c in self.y_array]).to(torch.long)
        self.select_classes(self.classes)
        self.select_domains(selected_domains)

    def prepare_normal_data(self):
        selected_domains = dict()
        for label in self.classes:
            allowed_domains = self.class_domains[0][label]
            selected_domains[label] = allowed_domains

        # we need to make sure that applying more than one drift works
        self.select_domains(selected_domains)

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
