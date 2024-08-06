import pytest
from datasets.stream_spec import *


def test_can_iterate():
    s = StreamSpecification(5, 10, 45, 2)
    n_tasks = 0
    for new_classes, old_classes in s:
        num_classes = len(new_classes) + len(old_classes)
        assert num_classes == 2
        n_tasks += 1
    assert n_tasks == 5


def test_drifted_classes_are_returned():
    s = StreamSpecification(5, 10, 45, 5)
    for i, _ in enumerate(s):
        if i > 1:
            assert len(s.all_drifted_classes) > 0
        else:
            assert len(s.all_drifted_classes) == 0
    assert len(s.all_drifted_classes) > 0


def test_drifted_classes():
    s = StreamSpecification(5, 10, 45, 5)
    last_class = 0
    for new_classes, old_classes in s:
        for c in s.drifted_classes:
            assert c in s.all_drifted_classes
            assert c <= last_class
        if len(new_classes) > 0:
            last_class = max(max(new_classes), last_class)
    assert len(s.all_drifted_classes) > 0


def test_cant_use_exclusive_args():
    with pytest.raises(AssertionError):
        s = StreamSpecification(5, 10, 45, 5, n_drifts=10, sequential_drifts=True)


def test_sequential_drifts_can_iterate():
    s = StreamSpecification(5, 10, 45, sequential_drifts=True)
    n_tasks = 0
    for i, (new_classes, old_classes) in enumerate(s):
        if i == 0:
            assert len(new_classes) == 2
            assert len(old_classes) == 0
        else:
            assert len(new_classes) == 2
            assert len(old_classes) == 2
        n_tasks += 1
    assert n_tasks == 5


def test_n_drifts_can_iterate():
    s = StreamSpecification(5, 10, 45, n_drifts=3)
    it = iter(s)
    assert next(it) == ([0, 1], [])
    assert next(it) == ([2, 3], [0, 1])
    assert next(it) == ([4, 5], [2, 3])
    assert next(it) == ([6, 7], [4, 5])
    assert next(it) == ([8, 9], [])


def test_n_drifts_can_iterate_ndrifts_1():
    s = StreamSpecification(5, 10, 45, n_drifts=1)
    it = iter(s)
    assert next(it) == ([0, 1], [])
    assert next(it) == ([2, 3], [])
    assert next(it) == ([4, 5], [0, 1, 2, 3])
    assert next(it) == ([6, 7], [])
    assert next(it) == ([8, 9], [])
