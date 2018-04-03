import math
import os

import numpy as np
import tensorflow as tf

from skimage.io import imread
from skimage.transform import resize
from tensorflow.python.platform import gfile

from common import config
from face_features.feature_extractor import FeatureExtractor

IMAGE_SIZE = 160
TENSOR_NAME_INPUT = "input:0"
TENSOR_NAME_EMBEDDINGS = "embeddings:0"
TENSOR_NAME_PHASE_TRAIN = "phase_train:0"


def parse_image_file(filename):
  image = imread(filename)
  image_resized = resize(image, (IMAGE_SIZE, IMAGE_SIZE))
  return image_resized


def load_data(image_paths):
    num_samples = len(image_paths)
    images = np.zeros((num_samples, IMAGE_SIZE, IMAGE_SIZE, 3))
    for i in range(num_samples):
        img = parse_image_file(image_paths[i])
        images[i, :, :, :] = img
    return images


class FaceNet(FeatureExtractor):
    def __init__(self):
        self.graph = None
        self.input_tensor = None
        self.embedding_tensor = None
        self.phase_train_tensor = None
        self._load_model()

    def _load_model(self):
        """
        Loads model from a protobuf file with a frozen graph
        :param model_filename: file name for model to be loaded.
        """
        model_filename = os.path.dirname(os.path.dirname(__file__)) + \
                         "/model/facenet_inception_resnet_MS_celeb.pb"
        self.graph = tf.Graph()
        with self.graph.as_default():
            with gfile.FastGFile(model_filename, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                self.input_tensor, self.embedding_tensor, self.phase_train_tensor = \
                    tf.import_graph_def(
                        graph_def, name='',
                        return_elements=
                        [TENSOR_NAME_INPUT, TENSOR_NAME_EMBEDDINGS,
                        TENSOR_NAME_PHASE_TRAIN])

    def extract_features(self, images_paths):
        """
        Extracts features for images corresponding to given list of image
        paths.
        :param images_paths: List of paths to images to be processed.
        :return: A list of features list corresponding to images in given order
        """
        if self.graph is None:
            raise Exception("Model needs to be loaded first")
        sess = tf.Session(graph=self.graph)
        sess.as_default()

        embedding_size = self.embedding_tensor.get_shape()[1]

        # Run forward pass to calculate embeddings
        print('Running feature extraction on images')
        batch_size = config.FACENET_BATCH_SIZE
        num_images = len(images_paths)
        num_batches = int(math.ceil(1.0 * num_images / batch_size))
        emb_array = np.zeros((num_images, embedding_size))
        for i in range(num_batches):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, num_images)
            paths_batch = images_paths[start_index:end_index]
            images = load_data(paths_batch)
            feed_dict = {self.input_tensor: images, self.phase_train_tensor: False}
            emb_array[start_index:end_index, :] = \
                sess.run(self.embedding_tensor, feed_dict=feed_dict)
            print("Computed features for %d batches of %d images" %
                  ((i+1), batch_size))
        sess.close()
        return emb_array.tolist()


class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)

