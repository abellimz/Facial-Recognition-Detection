import math
import os

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import gfile

from common import config
from common.config import FACENET_TRAIN_BATCH, FACENET_IMAGE_SIZE
from face_features.feature_extractor import FeatureExtractor, load_data

TENSOR_NAME_INPUT = "input:0"
TENSOR_NAME_EMBEDDINGS = "embeddings:0"
TENSOR_NAME_PHASE_TRAIN = "phase_train:0"


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

    def extract_features(self, image_paths):
        """
        Extracts features for images corresponding to given list of image
        paths.
        :param image_paths: List of paths to images to be processed.
        :return: A list of features list corresponding to images in given order
        """
        if self.graph is None:
            raise Exception("Model needs to be loaded first")
        sess = tf.Session(graph=self.graph)
        sess.as_default()

        # Run forward pass to calculate embeddings
        batched_image_paths = np.array_split(
            image_paths, len(image_paths) / FACENET_TRAIN_BATCH + 1)
        results = []
        for idx, batch in enumerate(batched_image_paths):
            print("Extracting features: batch %d of %d"
                  % (idx + 1, len(batched_image_paths)))
            images = load_data(batch, FACENET_IMAGE_SIZE, FACENET_IMAGE_SIZE)
            feed_dict = {self.input_tensor: images, self.phase_train_tensor: False}
            result = sess.run(self.embedding_tensor, feed_dict=feed_dict)
            results.extend(result.tolist())
        sess.close()
        return results

