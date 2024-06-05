import utils.buffer
import pytest
import torch
import torchvision.transforms
from datasets.seq_tinyimagenet import base_path

from torchvision.datasets import CIFAR10


def get_sample_images(n):
    dataset = CIFAR10(base_path() + 'CIFAR10')
    images = list()
    to_tensor = torchvision.transforms.ToTensor()
    for i in range(n):
        img, _ = dataset[i]
        trans_img = to_tensor(img)
        images.append(trans_img)
    images = torch.stack(images)
    return images


def test_buffer_get_data_works():
    buf = utils.buffer.Buffer(4, 'cpu')
    examples = get_sample_images(4)
    labels = torch.Tensor([1, 1, 2, 2])
    buf.add_data(examples, labels)
    ret_examples, ret_labels = buf.get_all_data()
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
