import numpy as np
import collections
from typing import Union


class StreamSpecification:
    def __init__(
        self,
        n_tasks: int,
        n_classes: int,
        random_seed: int = None,
        n_slots: Union[int, None] = None,
        n_poisonings: Union[int, None] = None,
        sequential_poisonings: bool = False,
        max_classes_per_poisoning: int = 5,
    ) -> None:
        """
        Utility class that represents the class-task layout of the overall task stream.

        Args:
        n_tasks - number of tasks in the stream
        n_classes - number of all classes in dataset
        random_seed - random seed used for experiments
        n_slots - number of classes per task, used for creating stream layout in the style of Class-Incremental learning with repetition
        n_poisonings - number of poisonings created in whole task stream
        sequential_poisonings - if true each task will consist of set of new classes and poisoned classes from previous task
        max_classes_per_poisoning - number of classes that poisoning is applied to. used only with n_poisonings
        """
        assert n_tasks >= 2, "need at least two tasks for continual learning"
        assert (
            n_slots is not None or n_poisonings is not None or sequential_poisonings
        ), "must specify poisoning type"

        if n_slots is not None:
            assert (
                n_poisonings == None and sequential_poisonings == False
            ), "you must use n_slots, n_poisonings or sequential_poisonings arguments (cant use them together)"
        if n_poisonings is not None:
            assert (
                n_slots == None and sequential_poisonings == False
            ), "you must use n_slots, n_poisonings or sequential_poisonings arguments (cant use them together)"
        if sequential_poisonings:
            assert (
                n_slots == None and n_poisonings is None
            ), "you must use n_slots, n_poisonings or sequential_poisonings arguments (cant use them together)"

        self.n_tasks = n_tasks
        self.n_slots = n_slots
        self.n_classes = n_classes
        self.random_seed = random_seed
        self.n_poisonings = n_poisonings
        self.sequential_poisonings = sequential_poisonings
        self.max_classes_per_poisoning = max_classes_per_poisoning

        self._new_classes: list[list[int]] = list()
        self._poisoned_classes: list[list[int]] = list()
        if n_poisonings is not None:
            self.create_n_poisonings()
        elif sequential_poisonings:
            self.create_sequential_poisonings()
        else:
            self.random_class_asigment()

        self.current_task = 0

    def create_n_poisonings(self):
        assert self.n_poisonings < self.n_tasks

        poisoning_duration = self.n_tasks // (self.n_poisonings + 1)
        poisoning_indexes = list(
            np.linspace(
                poisoning_duration,
                self.n_tasks,
                num=self.n_poisonings,
                endpoint=False,
                dtype=int,
            )
        )
        print("Creating poisonings at tasks: ", poisoning_indexes)

        classes_per_task = self.n_classes // self.n_tasks
        poisoning_begin_idx = 0
        for t in range(self.n_tasks):
            new_classes = list(range(classes_per_task * t, classes_per_task * (t + 1)))
            self._new_classes.append(new_classes)
            if t in poisoning_indexes:
                poisoned_classes = list(
                    range(poisoning_begin_idx, classes_per_task * t)
                )
                poisoned_classes = poisoned_classes[-self.max_classes_per_poisoning :]
                self._poisoned_classes.append(poisoned_classes)
                poisoning_begin_idx = classes_per_task * t
            else:
                self._poisoned_classes.append([])

    def create_sequential_poisonings(self):
        classes_per_task = self.n_classes // self.n_tasks
        assert (
            classes_per_task >= 2
        ), "At least two classes should be present in each new task"

        for t in range(self.n_tasks):
            self._new_classes.append(
                list(range(classes_per_task * t, classes_per_task * (t + 1)))
            )
            self._poisoned_classes.append(
                list(
                    range(
                        max(classes_per_task * t - classes_per_task, 0),
                        classes_per_task * t,
                    )
                )
            )

    def random_class_asigment(self):
        """Random class assigment to the tasks.
        Based on the notion of slot-based generator from https://arxiv.org/pdf/2301.11396
        Each task has n_slots slots. Each slot can be filled with exatly one class.
        When classes apears more than once in the stream we assume, that poisoning occured, and data of given class should change.
        When there are some classes left at the end, they are appended to the end of the stream.
        """
        classes_pool = list(range(self.n_classes))
        used_classes = set()
        random_state = np.random.RandomState(self.random_seed)

        for _ in range(self.n_tasks):
            if len(classes_pool) < self.n_slots:
                classes_pool.extend(range(self.n_classes))

            task_classes = [None]
            label = None
            for _ in range(self.n_slots):
                while label in task_classes:
                    label = random_state.choice(classes_pool, replace=False)
                task_classes.append(label)
                classes_pool.remove(label)
                if all(l in set(task_classes) for l in classes_pool):
                    break
            task_classes.remove(None)

            self._new_classes.append([])
            self._poisoned_classes.append([])
            for c in task_classes:
                if c not in used_classes:
                    self._new_classes[-1].append(c)
                    used_classes.add(c)
                else:
                    self._poisoned_classes[-1].append(c)

        class_mapping = dict()
        last_class = 0
        for task_class in self._new_classes:
            for c in task_class:
                if c not in class_mapping:
                    class_mapping[c] = last_class
                    last_class += 1

        self._new_classes = [
            sorted(class_mapping[c] for c in task_class)
            for task_class in self._new_classes
        ]
        self._poisoned_classes = [
            sorted(class_mapping[c] for c in task_class)
            for task_class in self._poisoned_classes
        ]

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
