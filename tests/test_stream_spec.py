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


def test_n_drifts_5_tasks_1_drift():
    s = StreamSpecification(5, 10, 45, n_drifts=1)
    it = iter(s)
    assert next(it) == ([0, 1], [])
    assert next(it) == ([2, 3], [])
    assert next(it) == ([4, 5], [0, 1, 2, 3])
    assert next(it) == ([6, 7], [])
    assert next(it) == ([8, 9], [])
    with pytest.raises(StopIteration):
        next(it)


def test_n_drifts_5_tasks_3_drifts():
    s = StreamSpecification(5, 10, 45, n_drifts=3)
    it = iter(s)
    assert next(it) == ([0, 1], [])
    assert next(it) == ([2, 3], [0, 1])
    assert next(it) == ([4, 5], [2, 3])
    assert next(it) == ([6, 7], [4, 5])
    assert next(it) == ([8, 9], [])
    with pytest.raises(StopIteration):
        next(it)


def test_n_drifts_10_tasks_1_drift():
    s = StreamSpecification(10, 20, 45, n_drifts=1)
    it = iter(s)
    assert next(it) == ([0, 1], [])
    assert next(it) == ([2, 3], [])
    assert next(it) == ([4, 5], [])
    assert next(it) == ([6, 7], [])
    assert next(it) == ([8, 9], [])
    assert next(it) == ([10, 11], list(range(0, 10)))
    assert next(it) == ([12, 13], [])
    assert next(it) == ([14, 15], [])
    assert next(it) == ([16, 17], [])
    assert next(it) == ([18, 19], [])
    with pytest.raises(StopIteration):
        next(it)


def test_n_drifts_10_tasks_2_drifts():
    s = StreamSpecification(10, 20, 45, n_drifts=2)
    it = iter(s)
    assert next(it) == ([0, 1], [])
    assert next(it) == ([2, 3], [])
    assert next(it) == ([4, 5], [])
    assert next(it) == ([6, 7], list(range(0, 6)))
    assert next(it) == ([8, 9], [])
    assert next(it) == ([10, 11], [])
    assert next(it) == ([12, 13], list(range(6, 12)))
    assert next(it) == ([14, 15], [])
    assert next(it) == ([16, 17], [])
    assert next(it) == ([18, 19], [])
    with pytest.raises(StopIteration):
        next(it)


def test_n_drifts_10_tasks_3_drifts():
    s = StreamSpecification(10, 20, 45, n_drifts=3)
    it = iter(s)
    assert next(it) == ([0, 1], [])
    assert next(it) == ([2, 3], [])
    assert next(it) == ([4, 5], list(range(0, 4)))
    assert next(it) == ([6, 7], [])
    assert next(it) == ([8, 9], list(range(4, 8)))
    assert next(it) == ([10, 11], [])
    assert next(it) == ([12, 13], [])
    assert next(it) == ([14, 15], list(range(8, 14)))
    assert next(it) == ([16, 17], [])
    assert next(it) == ([18, 19], [])
    with pytest.raises(StopIteration):
        next(it)


def test_n_drifts_10_tasks_4_drifts():
    s = StreamSpecification(10, 20, 45, n_drifts=4)
    it = iter(s)
    assert next(it) == ([0, 1], [])
    assert next(it) == ([2, 3], [])
    assert next(it) == ([4, 5], list(range(0, 4)))
    assert next(it) == ([6, 7], [])
    assert next(it) == ([8, 9], list(range(4, 8)))
    assert next(it) == ([10, 11], [])
    assert next(it) == ([12, 13], list(range(8, 12)))
    assert next(it) == ([14, 15], [])
    assert next(it) == ([16, 17], list(range(12, 16)))
    assert next(it) == ([18, 19], [])
    with pytest.raises(StopIteration):
        next(it)


def test_n_drifts_10_tasks_9_drifts():
    s = StreamSpecification(10, 20, 45, n_drifts=9)
    it = iter(s)
    assert next(it) == ([0, 1], [])
    assert next(it) == ([2, 3], [0, 1])
    assert next(it) == ([4, 5], [2, 3])
    assert next(it) == ([6, 7], [4, 5])
    assert next(it) == ([8, 9], [6, 7])
    assert next(it) == ([10, 11], [8, 9])
    assert next(it) == ([12, 13], [10, 11])
    assert next(it) == ([14, 15], [12, 13])
    assert next(it) == ([16, 17], [14, 15])
    assert next(it) == ([18, 19], [16, 17])
    with pytest.raises(StopIteration):
        next(it)


def test_n_drifts_20_tasks_1_drift():
    s = StreamSpecification(20, 40, 45, n_drifts=1)
    it = iter(s)
    assert next(it) == ([0, 1], [])
    assert next(it) == ([2, 3], [])
    assert next(it) == ([4, 5], [])
    assert next(it) == ([6, 7], [])
    assert next(it) == ([8, 9], [])
    assert next(it) == ([10, 11], [])
    assert next(it) == ([12, 13], [])
    assert next(it) == ([14, 15], [])
    assert next(it) == ([16, 17], [])
    assert next(it) == ([18, 19], [])
    assert next(it) == ([20, 21], list(range(0, 20)))
    assert next(it) == ([22, 23], [])
    assert next(it) == ([24, 25], [])
    assert next(it) == ([26, 27], [])
    assert next(it) == ([28, 29], [])
    assert next(it) == ([30, 31], [])
    assert next(it) == ([32, 33], [])
    assert next(it) == ([34, 35], [])
    assert next(it) == ([36, 37], [])
    assert next(it) == ([38, 39], [])
    with pytest.raises(StopIteration):
        next(it)


def test_n_drifts_20_tasks_2_drifts():
    s = StreamSpecification(20, 40, 45, n_drifts=2)
    it = iter(s)
    assert next(it) == ([0, 1], [])
    assert next(it) == ([2, 3], [])
    assert next(it) == ([4, 5], [])
    assert next(it) == ([6, 7], [])
    assert next(it) == ([8, 9], [])
    assert next(it) == ([10, 11], [])
    assert next(it) == ([12, 13], list(range(0, 12)))
    assert next(it) == ([14, 15], [])
    assert next(it) == ([16, 17], [])
    assert next(it) == ([18, 19], [])
    assert next(it) == ([20, 21], [])
    assert next(it) == ([22, 23], [])
    assert next(it) == ([24, 25], [])
    assert next(it) == ([26, 27], list(range(12, 26)))
    assert next(it) == ([28, 29], [])
    assert next(it) == ([30, 31], [])
    assert next(it) == ([32, 33], [])
    assert next(it) == ([34, 35], [])
    assert next(it) == ([36, 37], [])
    assert next(it) == ([38, 39], [])
    with pytest.raises(StopIteration):
        next(it)


def test_n_drifts_20_tasks_3_drifts():
    s = StreamSpecification(20, 40, 45, n_drifts=3)
    it = iter(s)
    assert next(it) == ([0, 1], [])
    assert next(it) == ([2, 3], [])
    assert next(it) == ([4, 5], [])
    assert next(it) == ([6, 7], [])
    assert next(it) == ([8, 9], [])
    assert next(it) == ([10, 11], list(range(0, 10)))
    assert next(it) == ([12, 13], [])
    assert next(it) == ([14, 15], [])
    assert next(it) == ([16, 17], [])
    assert next(it) == ([18, 19], [])
    assert next(it) == ([20, 21], list(range(10, 20)))
    assert next(it) == ([22, 23], [])
    assert next(it) == ([24, 25], [])
    assert next(it) == ([26, 27], [])
    assert next(it) == ([28, 29], [])
    assert next(it) == ([30, 31], list(range(20, 30)))
    assert next(it) == ([32, 33], [])
    assert next(it) == ([34, 35], [])
    assert next(it) == ([36, 37], [])
    assert next(it) == ([38, 39], [])
    with pytest.raises(StopIteration):
        next(it)
