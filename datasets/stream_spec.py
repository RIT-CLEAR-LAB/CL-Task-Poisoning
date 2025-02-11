import numpy as np
import collections
from typing import Union


class StreamSpecification:
    def __init__(
        self,
        n_tasks: int,
        n_classes: int,
        random_seed: int = None,
        n_image_poisonings: Union[int, None] = None,
        n_label_flip_poisonings: Union[int, None] = None,
        n_backdoor_poisonings: Union[int, None] = None,
        classes_per_poisoning: int = 0,
    ) -> None:
        """
        Utility class that represents the class-task layout of the overall task stream.

        Args:
        n_tasks - number of tasks in the stream
        n_classes - number of all classes in dataset
        random_seed - random seed used for experiments
        n_image_poisonings - number of image poisonings created in whole task stream
        n_label_flip_poisonings - number of past label flip poisonings created in whole task stream
        n_backdoor_poisonings - number of backdoor poisonings created in whole task stream
        classes_per_poisoning - number of classes that poisoning is applied to
        """
        assert n_tasks >= 2, "need at least two tasks for continual learning"
        assert (
            n_image_poisonings is not None
            or n_label_flip_poisonings is not None
            or n_backdoor_poisonings is not None
        ), "must specify number of tasks to poison"

        if n_image_poisonings is not None:
            assert (
                n_label_flip_poisonings == None and n_backdoor_poisonings == None
            ), "you must select between n_image_poisonings or n_label_flip_poisonings or n_backdoor_poisonings argument (can not use them together)"
        if n_label_flip_poisonings is not None:
            assert (
                n_image_poisonings == None and n_backdoor_poisonings == None
            ), "you must select between n_image_poisonings or n_label_flip_poisonings or n_backdoor_poisonings argument (can not use them together)"
        if n_backdoor_poisonings is not None:
            assert (
                n_image_poisonings == None and n_label_flip_poisonings == None
            ), "you must select between n_image_poisonings or n_label_flip_poisonings or n_backdoor_poisonings argument (can not use them together)"

        self.n_tasks = n_tasks
        self.n_classes = n_classes
        self.random_seed = random_seed
        self.classes_per_poisoning = classes_per_poisoning

        self._new_classes: list[list[int]] = list()
        self._poisoned_classes: list[list[int]] = list()

        if n_image_poisonings is not None:
            self.create_n_poisonings(n_image_poisonings)
        elif n_label_flip_poisonings is not None:
            self.create_n_label_flip_poisonings(n_label_flip_poisonings)
        elif n_backdoor_poisonings is not None:
            self.create_n_poisonings(n_backdoor_poisonings)

        self.current_task = 0

    def create_n_poisonings(self, n_poisonings):
        assert n_poisonings < self.n_tasks, "# poisonings has to be less than # tasks"

        poisoning_interval = self.n_tasks // (n_poisonings + 1)
        poisoning_indices = np.linspace(poisoning_interval, self.n_tasks, num=n_poisonings, endpoint=False, dtype=int)
        poisoning_indices = list(poisoning_indices)
        print("Creating poisonings at tasks: ", poisoning_indices)

        classes_per_task = self.n_classes // self.n_tasks
        assert self.classes_per_poisoning <= classes_per_task, "Not enough classes to poison in current task"

        for t in range(self.n_tasks):
            new_task_classes = list(range(classes_per_task * t, classes_per_task * (t + 1)))
            if t in poisoning_indices:
                self._poisoned_classes.append(new_task_classes[-self.classes_per_poisoning:])
                self._new_classes.append(new_task_classes[:-self.classes_per_poisoning])
            else:
                self._new_classes.append(new_task_classes)
                self._poisoned_classes.append([])

    def create_n_label_flip_poisonings(self, n_poisonings):
        assert n_poisonings < self.n_tasks, "# poisonings has to be less than # tasks"

        poisoning_interval = self.n_tasks // (n_poisonings + 1)
        poisoning_indices = list(np.linspace(poisoning_interval, self.n_tasks, num=n_poisonings, endpoint=False, dtype=int))
        print('Creating poisonings at tasks: ', poisoning_indices)

        classes_per_task = self.n_classes // self.n_tasks
        poisoning_begin_idx = 0
        for t in range(self.n_tasks):
            new_classes = list(range(classes_per_task * t, classes_per_task * (t + 1)))
            self._new_classes.append(new_classes)
            if t in poisoning_indices:
                poisoned_classes = list(range(poisoning_begin_idx, classes_per_task * t))
                poisoned_classes = poisoned_classes[-min(self.classes_per_poisoning, len(poisoned_classes)):]
                self._poisoned_classes.append(poisoned_classes)
                poisoning_begin_idx = classes_per_task * t
            else:
                self._poisoned_classes.append([])

    def __next__(self):
        while self.current_task < self.n_tasks:
            yield self._new_classes[self.current_task], self._poisoned_classes[
                self.current_task
            ]
            self.current_task += 1

    def __iter__(self):
        return next(self)

    @property
    def all_poisoned_classes(self):
        """return all classes for which poisoning has occured so far"""
        all_poisoned = set()
        for task_poisoned in self._poisoned_classes[: self.current_task + 1]:
            for c in task_poisoned:
                all_poisoned.add(c)
        return list(all_poisoned)

    @property
    def poisoned_classes(self):
        """return classes from last task for which poisoning has occured"""
        return self._poisoned_classes[self.current_task]

    @property
    def new_classes(self):
        """return classes from current task"""
        return self._new_classes[self.current_task]

    @property
    def max_poisonings_per_class(self):
        if len(self._poisoned_classes) == 0:
            print("max_poisonings_per_class call: no poisonings in current setup")
            return 0
        counter = collections.Counter()
        for classes in self._poisoned_classes:
            counter.update(classes)
        max_occurence = counter.most_common(1)[0][1]
        return max_occurence


if __name__ == "__main__":
    ss = StreamSpecification(20, 100, 45, n_image_poisonings=10)
    ssi = iter(ss)
    for new, old in ssi:
        print(f"new = {new} old = {old}")
