import utils.buffer
import numpy as np
import pytest
import torch
import torchvision.transforms
from datasets.seq_tinyimagenet import base_path

from torchvision.datasets import CIFAR10


def get_sample_images(n):
    dataset = CIFAR10(base_path() + 'CIFAR10', download=True)
    images = list()
    to_tensor = torchvision.transforms.ToTensor()
    for i in range(n):
        img, _ = dataset[i]
        trans_img = to_tensor(img)
        images.append(trans_img)
    images = torch.stack(images)
    return images


def test_buffer_get_data_works():
    buffer = utils.buffer.Buffer(4, 'cpu')
    examples = get_sample_images(4)
    labels = torch.Tensor([1, 1, 2, 2])
    buffer.add_data(examples, labels)
    ret_examples, ret_labels = buffer.get_all_data()
    assert (examples == ret_examples).all()
    assert (ret_labels == labels).all()


def test_buffer_get_class_data_returns_all_data_from_single_class():
    buffer = utils.buffer.Buffer(4, 'cpu')
    examples = get_sample_images(4)
    labels = torch.Tensor([1, 1, 2, 2])
    buffer.add_data(examples, labels)

    samples1 = buffer.get_class_data(1)
    assert (samples1 == examples[:2]).all()
    samples2 = buffer.get_class_data(2)
    assert (samples2 == examples[2:]).all()


def test_buffer_get_class_data_raises_exception():
    buffer = utils.buffer.Buffer(4, 'cpu')
    examples = get_sample_images(4)
    labels = torch.Tensor([1, 1, 2, 2])
    buffer.add_data(examples, labels)

    with pytest.raises(ValueError):
        samples1 = buffer.get_class_data(3)


def test_buffer_flush_class_removes_samples():
    buffer = utils.buffer.Buffer(4, 'cpu')
    examples = get_sample_images(4)
    labels = torch.Tensor([1, 1, 2, 2])
    buffer.add_data(examples, labels)

    buffer.flush_class(1)
    assert len(buffer) == 2
    ret_examples, ret_labels = buffer.get_all_data()
    assert (examples[2:] == ret_examples[:buffer.num_seen_examples]).all()
    assert (labels[2:] == ret_labels[:buffer.num_seen_examples]).all()
    assert buffer.num_seen_examples == 2


def test_buffer_flush_class_removes_samples1():
    buffer = utils.buffer.Buffer(6, 'cpu')
    examples = get_sample_images(6)
    labels = torch.Tensor([1, 1, 2, 2, 3, 3])
    buffer.add_data(examples, labels)

    buffer.flush_class(1)
    assert len(buffer) == 4
    ret_examples, ret_labels = buffer.get_all_data()
    assert (examples[2:] == ret_examples[:buffer.num_seen_examples]).all()
    assert (labels[2:] == ret_labels[:buffer.num_seen_examples]).all()
    assert buffer.num_seen_examples == 4


def test_buffer_after_flush_class_add_data_works():
    buffer = utils.buffer.Buffer(6, 'cpu')
    all_examples = get_sample_images(7)
    examples = all_examples[:6]
    labels = torch.Tensor([1, 1, 2, 2, 3, 3])
    buffer.add_data(examples, labels)

    buffer.flush_class(1)
    new_example = all_examples[6:]
    new_label = torch.Tensor([4])
    buffer.add_data(new_example, new_label)

    assert len(buffer) == 5
    ret_examples, ret_labels = buffer.get_all_data()
    assert (all_examples[2:] == ret_examples[:buffer.num_seen_examples]).all()
    assert (labels[2:] == ret_labels[:4]).all()
    assert (new_label == ret_labels[4]).all()
    assert buffer.num_seen_examples == 5


def test_buffer_flush_class_removes_all_data():
    buffer = utils.buffer.Buffer(4, 'cpu')
    examples = get_sample_images(4)
    labels = torch.Tensor([1, 1, 2, 2])
    logits = torch.Tensor([0.9, 0.8, 0.2, 0.7])
    task_labels = torch.Tensor([1, 1, 1, 1])
    buffer.add_data(examples, labels, logits, task_labels)

    buffer.flush_class(1)
    assert len(buffer) == 2
    ret_examples, ret_labels, ret_logits, ret_task_labels = buffer.get_all_data()
    assert (examples[2:] == ret_examples[:buffer.num_seen_examples]).all()
    assert (labels[2:] == ret_labels[:buffer.num_seen_examples]).all()
    assert (logits[2:] == ret_logits[:buffer.num_seen_examples]).all()
    assert (task_labels[2:] == ret_task_labels[:buffer.num_seen_examples]).all()
    assert buffer.num_seen_examples == 2


def test_ballanced_buffer():
    buffer = utils.buffer.Buffer(6, 'cpu', mode='balanced')
    examples = get_sample_images(6)
    labels = torch.Tensor([1, 1, 2, 2, 2, 2])
    buffer.add_data(examples, labels)

    new_examples = get_sample_images(2)
    new_labels = torch.Tensor([3, 3])
    np.random.seed(0)
    buffer.add_data(new_examples, new_labels)

    _, updated_labels = buffer.get_all_data()
    buffer_classes, class_counts = torch.unique(updated_labels, return_counts=True)
    assert (buffer_classes == torch.Tensor([1, 2, 3])).all()
    assert (class_counts == torch.Tensor([2, 2, 2])).all()


def test_ballanced_buffer_other_class_counts():
    buffer = utils.buffer.Buffer(4, 'cpu', mode='balanced')
    examples = get_sample_images(4)
    labels = torch.Tensor([1, 1, 2, 2])
    buffer.add_data(examples, labels)

    new_examples = get_sample_images(2)
    new_labels = torch.Tensor([3, 3])
    np.random.seed(1)
    buffer.add_data(new_examples, new_labels)

    _, updated_labels = buffer.get_all_data()
    buffer_classes, class_counts = torch.unique(updated_labels, return_counts=True)
    assert (buffer_classes == torch.Tensor([1, 2, 3])).all()
    assert (class_counts == torch.Tensor([1, 1, 2])).all()
