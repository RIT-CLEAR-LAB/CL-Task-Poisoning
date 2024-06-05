import utils.buffer
import pytest
import torch


def test_buffer_get_data_works():
    buf = utils.buffer.Buffer(4, 'cpu')
    examples = torch.Tensor([1, 2, 3, 4])
    labels = torch.Tensor([1, 1, 2, 2])
    buf.add_data(examples, labels)
    ret_examples, ret_labels = buf.get_all_data()
    assert (examples == ret_examples).all()
    assert (ret_labels == labels).all()
