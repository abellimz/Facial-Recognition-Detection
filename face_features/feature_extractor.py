import abc
import numpy as np
from skimage.io import imread
from skimage.transform import resize


IMAGE_SIZE_DEFAULT = 224


def parse_image_file(filename, resize_height, resize_width):
  image = imread(filename)
  image_resized = resize(image, (resize_height, resize_width))
  return image_resized


def load_data(image_paths,
              resize_height=IMAGE_SIZE_DEFAULT, resize_width=IMAGE_SIZE_DEFAULT):
    num_samples = len(image_paths)
    images = np.zeros((num_samples, resize_height, resize_width, 3))
    for i in range(num_samples):
        img = parse_image_file(image_paths[i], resize_height, resize_width)
        images[i, :, :, :] = img
    return images


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