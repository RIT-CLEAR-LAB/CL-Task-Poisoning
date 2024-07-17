import pytest
from datasets.stream_spec import *


def test_can_iterate():
    s = StreamSpecification(5, 10, 45, 2)
    n_tasks = 0
    for task_classes in s:
        assert len(task_classes) == 2
        n_tasks += 1
    assert n_tasks == 5


def test_drifted_classes_are_returned():
    s = StreamSpecification(5, 10, 45, 5)
    for i, _ in enumerate(s):
        if i > 0:
            assert len(s.drifted_classes) > 0
        else:
            assert len(s.drifted_classes) == 0
    assert len(s.drifted_classes) > 0


def test_drifted_classes_last_task():
    s = StreamSpecification(5, 10, 45, 5)
    last_class = 0
    for tk in s:
        for c in s.drifted_classes_last_task:
            assert c in s.drifted_classes
            assert c <= last_class
        last_class = max(max(tk), last_class)
    assert len(s.drifted_classes) > 0


def test_cant_use_exclusive_args():
    with pytest.raises(AssertionError):
        s = StreamSpecification(5, 10, 45, 5, n_drifts=10, sequential_drifts=True)


def test_sequential_drifts_can_iterate():
    s = StreamSpecification(5, 10, 45, 2, sequential_drifts=True)
    n_tasks = 0
    for i, task_classes in enumerate(s):
        if i == 0:
            assert len(task_classes) == 2
        else:
            assert len(task_classes) == 4
        n_tasks += 1
    assert n_tasks == 5


def test_n_drifts_can_iterate():
    s = StreamSpecification(5, 10, 45, 2, n_drifts=3)
    it = iter(s)
    assert next(it) == [0, 1]
    assert next(it) == [0, 1, 2, 3]
    assert next(it) == [2, 3, 4, 5]
    assert next(it) == [4, 5, 6, 7]
    assert next(it) == [8, 9]


def test_n_drifts_can_iterate_ndrifts_1():
    s = StreamSpecification(5, 10, 45, 2, n_drifts=1)
    it = iter(s)
    assert next(it) == [0, 1]
    assert next(it) == [2, 3]
    assert next(it) == [0, 1, 2, 3, 4, 5]
    assert next(it) == [6, 7]
    assert next(it) == [8, 9]
