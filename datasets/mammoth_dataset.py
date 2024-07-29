import abc


class MammothDataset(abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classes = list()
        self.drifted_classes = list()

    @abc.abstractmethod
    def select_classes(self, classes_list: list[int]):
        """changes dataset, so only classes specififed in classes_list remain"""
        raise NotImplementedError

    @abc.abstractmethod
    def apply_drift(self, classes: list[int]):
        """applies drift to selected classes"""
        raise NotImplementedError

    @abc.abstractmethod
    def prepare_normal_data(self):
        """prepares standard data, drift-free distribution"""
        raise NotImplementedError
