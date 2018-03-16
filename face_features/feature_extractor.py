import abc

class FeatureExtractor(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def load_model(self, model):
        """
        Loads model into object
        :param model: file name for model to be loaded.
        """
        pass

    @abc.abstractmethod
    def extract_features(self, images_paths):
        """
        Extracts features for images corresponding to given list of image
        paths.
        :param images_paths: List of paths to images to be processed.
        :return: A list of features list corresponding to images in given order
        """
        pass