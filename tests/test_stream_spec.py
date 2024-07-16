import pytest
from datasets.stream_spec import *


def test_can_iterate():
    s = StreamSpecification(5, 2, 10, 45)
    n_tasks = 0
    for task_classes in s:
        assert len(task_classes) == 2
        n_tasks += 1
    assert n_tasks == 5


def test_drifted_classes_are_returned():
    s = StreamSpecification(5, 5, 10, 45)
    for i, _ in enumerate(s):
        if i > 0:
            assert len(s.drifted_classes) > 0
        else:
            assert len(s.drifted_classes) == 0
    assert len(s.drifted_classes) > 0


def test_drifted_classes_last_task():
    s = StreamSpecification(5, 5, 10, 45)
    last_class = 0
    for tk in s:
        for c in s.drifted_classes_last_task:
            assert c in s.drifted_classes
            assert c <= last_class
        last_class = max(max(tk), last_class)
    assert len(s.drifted_classes) > 0
