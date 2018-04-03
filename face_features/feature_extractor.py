import abc

class FeatureExtractor(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def extract_features(self, images_paths):
        """
        Extracts features for images corresponding to given list of image
        paths.
        :param images_paths: List of paths to images to be processed.
        :return: A list of features list corresponding to images in given order
        """
        pass