import numpy as np


class StreamSpecification:
    def __init__(self, n_tasks: int, n_slots: int, n_classes: int, random_seed: int = None) -> None:
        """
        Represents the class-task layout of the overall task stream.

        Args:
        n_tasks - number of tasks in the stream
        n_slots - number of classes per task
        """
        assert n_tasks >= 2, 'need at least two tasks for continual learning'
        assert n_slots >= 2, 'for classification task you need at least two classes in each task'
        self.n_tasks = n_tasks
        self.n_slots = n_slots
        self.n_classes = n_classes
        self.random_seed = random_seed

        self.task_classes = self.random_class_asigment()

        self.current_task = 0

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
            selected_classes = random_state.choice(classes_list, size=self.n_slots, replace=False)
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
        """return classes for which drift has occured"""
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
        current_classes = self.task_classes[self.current_task]
        current_classes = set(current_classes)
        drifted_classes = set(self.drifted_classes)
        current_drifted = set.intersection(current_classes, drifted_classes)
        return list(current_drifted)


if __name__ == '__main__':
    s = StreamSpecification(5, 5, 10, 45)
    for task_classes in s:
        print('task_classes = ', task_classes)
        print('drifted_classes = ', s.drifted_classes)
        print()
