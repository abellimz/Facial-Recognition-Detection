import abc

class ClassifierTrainer(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def train(self):
        """
        Trains classifier on given parameters
        """
        pass

    def save_classifier(self):
        """
        Saves classifier
        """
        pass