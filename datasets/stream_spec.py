import numpy as np
import copy
import collections
from typing import Union


class StreamSpecification:
    def __init__(self, n_tasks: int, n_classes: int, random_seed: int = None,
                 n_slots: Union[int, None] = None, n_drifts: Union[int, None] = None, sequential_drifts: bool = False) -> None:
        """
        Utility class that represents the class-task layout of the overall task stream.

        Args:
        n_tasks - number of tasks in the stream
        n_slots - number of classes per task
        """
        assert n_tasks >= 2, 'need at least two tasks for continual learning'
        assert n_slots is not None or n_drifts is not None or sequential_drifts, 'must specify drift type'

        if n_slots is not None:
            assert n_drifts == None and sequential_drifts == False, 'you must use n_slots, n_drifts or sequential_drifts arguments (cant use them together)'
        if n_drifts is not None:
            assert n_slots == None and sequential_drifts == False, 'you must use n_slots, n_drifts or sequential_drifts arguments (cant use them together)'
        if sequential_drifts:
            assert n_slots == None and n_drifts is None, 'you must use n_slots, n_drifts or sequential_drifts arguments (cant use them together)'

        self.n_tasks = n_tasks
        self.n_slots = n_slots
        self.n_classes = n_classes
        self.random_seed = random_seed
        self.n_drifts = n_drifts
        self.sequential_drifts = sequential_drifts

        if n_drifts is not None:
            self.task_classes = self.create_n_drifts()
        elif sequential_drifts:
            self.task_classes = self.create_sequential_drifts()
        else:
            self.task_classes = self.random_class_asigment()

        self.current_task = 0

    def create_n_drifts(self):
        assert self.n_drifts < self.n_tasks

        drift_duration = self.n_tasks // (self.n_drifts + 1)
        drift_indexes = list()
        for i in range(drift_duration, self.n_tasks, drift_duration):
            if len(drift_indexes) == self.n_drifts:
                break
            drift_indexes.append(i)

        if len(drift_indexes) < self.n_drifts:
            drift_indexes.append(self.n_tasks-1)
        print('Creating drifts at tasks: ', drift_indexes)

        classes_per_task = self.n_classes // self.n_tasks
        last_class = 0
        last_drifted_class = -1
        task_classes = list()
        for t in range(self.n_tasks):
            new_classes = list(range(last_class, last_class + classes_per_task))
            if t in drift_indexes:
                min_label = min(new_classes)
                for c in range(last_drifted_class+1, min_label):
                    new_classes.append(c)
                last_drifted_class = min_label - 1
            task_classes.append(sorted(new_classes))
            last_class += classes_per_task

        return task_classes

    def create_sequential_drifts(self):
        """each task form 1 to n-1 contains set of new classes and drifted classes from previous task"""
        task_classes = list()

        classes_per_task = self.n_classes // self.n_tasks
        last_new_classes = []
        assert classes_per_task >= 2
        last_class = 0
        for t in range(self.n_tasks):
            new_classes = list(range(last_class, last_class + classes_per_task))
            classes_to_add = list()
            classes_to_add.extend(new_classes)
            classes_to_add.extend(last_new_classes)
            task_classes.append(sorted(classes_to_add))
            last_class += classes_per_task
            last_new_classes = copy.deepcopy(new_classes)
        return task_classes

    def random_class_asigment(self):
        """ Random class assigment to the tasks.
        Based on the notion of slot-based generator from https://arxiv.org/pdf/2301.11396
        Each task has n_slots slots. Each slot can be filled with exatly one class.
        When classes apears more than once in the stream we assume, that drift occured, and data of given class should change.
        When there are some classes left at the end, they are appended to the end of the stream.
        """
        classes_list = list(range(self.n_classes))
        random_state = np.random.RandomState(self.random_seed)

        task_classes = list()
        for _ in range(self.n_tasks):
            if len(classes_list) < self.n_slots:
                classes_list.extend(range(self.n_classes))

            selected_classes = [None]
            label = None
            for _ in range(self.n_slots):
                while label in selected_classes:
                    label = random_state.choice(classes_list, replace=False)
                selected_classes.append(label)

            selected_classes.remove(None)

            for c in selected_classes:
                classes_list.remove(c)
            task_classes.append(list(selected_classes))

        # assign remaining classes to the first task
        for c in classes_list:
            if c not in task_classes[0]:
                task_classes[0].append(c)

        class_mapping = dict()
        last_class = 0
        for task_class in task_classes:
            for c in task_class:
                if c not in class_mapping:
                    class_mapping[c] = last_class
                    last_class += 1

        new_task_classes = list()
        for task_class in task_classes:
            new_classes = sorted(class_mapping[c] for c in task_class)
            new_task_classes.append(new_classes)
        return new_task_classes

    def __next__(self):
        while self.current_task < self.n_tasks:
            yield self.task_classes[self.current_task]
            self.current_task += 1

    def __iter__(self):
        return next(self)

    @property
    def drifted_classes(self):
        """return all classes for which drift has occured so far"""
        seen_classes = set()
        drifted_cl = set()
        for i in range(min(self.current_task+1, self.n_tasks)):
            for c in self.task_classes[i]:
                if c in seen_classes:
                    drifted_cl.add(c)
                else:
                    seen_classes.add(c)
        return list(drifted_cl)

    @property
    def drifted_classes_last_task(self):
        """return classes from last task for which dirft has occured"""
        current_classes = self.task_classes[self.current_task]
        current_classes = set(current_classes)
        drifted_classes = set(self.drifted_classes)
        current_drifted = set.intersection(current_classes, drifted_classes)
        return list(current_drifted)

    @property
    def new_classes_last_task(self):
        """return classes from current task"""
        current_classes = self.task_classes[self.current_task]
        current_classes = set(current_classes)
        drifted_classes = set(self.drifted_classes)
        current_drifted = current_classes - drifted_classes
        return list(current_drifted)

    @property
    def max_drifts_per_class(self):
        counter = collections.Counter()
        for classes in self.task_classes:
            counter.update(classes)
        max_occurence = counter.most_common(1)[0][1]
        return max_occurence - 1


if __name__ == '__main__':
    s = StreamSpecification(5, 2, 10, 45, n_drifts=1)
    for task_classes in s:
        print('task_classes = ', task_classes)
        print('drifted_classes = ', s.drifted_classes)
        print('drifted_classes_last_task = ', s.drifted_classes_last_task)
        print()
