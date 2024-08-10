import torch
import functools
import collections

from typing import Tuple
from wilds import get_dataset

from datasets.mammoth_dataset import MammothDataset


@functools.cache
def get_class_details(dataset_name: str) -> Tuple[dict[int, list], collections.Counter]:
    dataset = get_dataset(dataset=dataset_name, download=True)
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


def get_valid_classes(dataset_name: str, min_samples=500, min_domains=2):
    """return list of classes that have at last 500 samples and at least 2 domains"""
    class_traps_ids, class_counts = get_class_details(dataset_name)
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
