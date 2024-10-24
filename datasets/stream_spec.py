import numpy as np
import collections
from typing import Union


class StreamSpecification:
    def __init__(
        self,
        n_tasks: int,
        n_classes: int,
        random_seed: int = None,
        n_poisonings: Union[int, None] = None,
        classes_per_poisoning: int = 0,
    ) -> None:
        """
        Utility class that represents the class-task layout of the overall task stream.

        Args:
        n_tasks - number of tasks in the stream
        n_classes - number of all classes in dataset
        random_seed - random seed used for experiments
        n_poisonings - number of poisonings created in whole task stream
        classes_per_poisoning - number of classes that poisoning is applied to with n_poisonings
        """
        assert n_tasks >= 2, "need at least two tasks for continual learning"
        assert n_poisonings is not None, "must specify number of tasks to poison"

        self.n_tasks = n_tasks
        self.n_classes = n_classes
        self.random_seed = random_seed
        self.n_poisonings = n_poisonings
        self.classes_per_poisoning = classes_per_poisoning

        self._new_classes: list[list[int]] = list()
        self._poisoned_classes: list[list[int]] = list()
        if n_poisonings is not None:
            self.create_n_poisonings()

        self.current_task = 0

    def create_n_poisonings(self):
        assert self.n_poisonings < self.n_tasks, "# poisonings has to be less than # tasks"

        poisoning_interval = self.n_tasks // (self.n_poisonings + 1)
        poisoning_indices = np.linspace(poisoning_interval, self.n_tasks, num=self.n_poisonings, endpoint=False, dtype=int)
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
    ss = StreamSpecification(20, 100, 45, n_poisonings=10)
    ssi = iter(ss)
    for new, old in ssi:
        print(f"new = {new} old = {old}")
