import os

import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.utils import CustomObjectScope

from common.config import FACENET_TRAIN_BATCH, FACENET_IMAGE_SIZE
from face_features.feature_extractor import FeatureExtractor, load_data

class KerasFaceNet(FeatureExtractor):

    model = None

    def __init__(self):
        self._load_model()

    def _load_model(self):
        """
        Loads model from a protobuf file with a frozen graph
        :param model_filename: file name for model to be loaded.
        """
        model_filename = os.path.dirname(os.path.dirname(__file__)) + \
                         "/model/facenet_keras.h5"
        with CustomObjectScope({'tf': tf}):
            self.model = load_model(model_filename)

    def extract_features(self, image_paths):
        """
        Extracts features for images corresponding to given list of image
        paths.
        :param image_paths: List of paths to images to be processed.
        :return: A list of features list corresponding to images in given order
        """
        if self.model is None:
            raise Exception("Model needs to be loaded first")

        # Run forward pass to calculate embeddings
        batched_image_paths = np.array_split(
            image_paths, len(image_paths) / FACENET_TRAIN_BATCH + 1)
        results = []
        for idx, batch in enumerate(batched_image_paths):
            print("Extracting features: batch %d of %d"
                  % (idx + 1, len(batched_image_paths)))
            images = load_data(batch, FACENET_IMAGE_SIZE, FACENET_IMAGE_SIZE)
            result = self.model.predict(images)
            results.extend(result.tolist())
        return results

