import abc


class MammothDataset(abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classes = list()
        self.poisoned_classes = list()

    @abc.abstractmethod
    def select_classes(self, current_classes: list[int]):
        """changes dataset, so only classes specififed in current_classes remain"""
        raise NotImplementedError

    @abc.abstractmethod
    def apply_poisoning(self, poisoned_classes: list[int], current_classes: None | list[int] = None):
        """applies poisoning to selected classes"""
        raise NotImplementedError

    @abc.abstractmethod
    def prepare_normal_data(self):
        """prepares standard data, poisoning-free distribution"""
        raise NotImplementedError
