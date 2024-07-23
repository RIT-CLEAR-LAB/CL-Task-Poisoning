import pytest
import torch
import torch.utils.data
import numpy as np

from datasets import ContinualDataset, get_dataset  # nopep8


class args:
    dataset = 'seq-cifar10'
    concept_drift = 1
    n_slots = 4
    seed = 42
    n_drifts = None
    sequential_drifts = False
    drift_severity = 1
    batch_size = 32


def test_sample_tasks_equal_number_of_classes():
    dataset = get_dataset(args)

    for t in range(dataset.N_TASKS):
        train_loader, _ = dataset.get_drifted_data_loaders()
        train_dataset = train_loader.dataset
        if type(train_dataset) == torch.utils.data.ConcatDataset:
            classes = set()
            for _, label, _ in train_dataset:
                classes.add(label)
            classes = np.unique(list(classes))
        else:
            classes = np.unique(train_dataset.targets)
        print('train loader classes = ', classes)
        assert len(classes) == 4


def test_number_of_samples_in_each_task_is_correctl():
    dataset = get_dataset(args)

    for t in range(dataset.N_TASKS):
        train_loader, _ = dataset.get_drifted_data_loaders()
        train_dataset = train_loader.dataset
        assert len(train_dataset) == 20000
