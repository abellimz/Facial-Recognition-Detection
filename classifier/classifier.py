import abc


class Classifier(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def new_model(self, class_labels, feature_shape):
        """
        Creates a new model with untrained weights.
        :param class_labels: List of classes.
        :param feature_shape: Shape of input feature.
        """
        pass

    @abc.abstractmethod
    def load_model(self, model_dir, model_basename):
        """
        Loads model into object
        :param model_dir Directory to be used for saving models.
        :param model_basename Base name of output model. File extension
            is appended to this basename in the overall filename.
        """
        pass

    @abc.abstractmethod
    def save_model(self, model_dir, model_basename):
        """
        Saves trained model.
        :param model_dir Directory to be used for saving models.
        :param model_basename Base name of output model. File extension
            is appended to this basename in the overall filename.
        """
        pass

    @abc.abstractmethod
    def train(self, features, labels):
        """
        Trains model in object with given array of training features.
        Labels should also be supplied such that they correspond to their
        respective training feature..
        :param train_images_paths: array of training features
        :param labels: list of labels for each training features.
        """
        pass

    @abc.abstractmethod
    def infer(self, images_path):
        """
        Performs inference using model on given list of features.
        :param features: list of features in numpy
        :return: list of list of confidence values in shape (n_examples, n_classes)
            For each confidence value list, every element at ith index corresponds to
            the confidence value for the ith class.
        """
        pass

    @abc.abstractmethod
    def get_classes(self):
        """
        Returns the list of classes used by this classifier.
        The order of classes corresponds to that in confidence scores from
        inference.
        """
        pass