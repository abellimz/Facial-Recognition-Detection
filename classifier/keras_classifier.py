import os
import abc

from keras.models import load_model

from common import constants, utility
from classifier.classifier import Classifier
from common.config import MLP_KERAS_INPUT_NAME, MLP_KERAS_OUTPUT_NAME
from common.utility import save_coreml_model


class KerasClassifier(Classifier, metaclass=abc.ABCMeta):
    def __init__(self):
        self.labels2Idx = None
        self.labels = None
        self.model = None

    def new_model(self, class_labels, feature_shape):
        self.labels2Idx = utility.list_to_dict(class_labels)
        self.labels = class_labels

    def save_model(self, model_dir, model_basename):
        """
        Saves trained sklearn model.
        Saves in pickle and coreml formats.
        :param model_dir Directory to be used for saving models.
        :param model_basename Base name of output model. File extension
            is appended to this basename in the overall filename.
        """
        if self.model is None:
            raise Exception("Classifier model is not created yet")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Save labels file
        labels_path = os.path.join(model_dir, constants.FILENAME_LABELS)
        self._save_labels_file(labels_path)

        # Save to pickle model file
        keras_model_path = os.path.join(
            model_dir, model_basename + constants.EXTENSION_H5)
        self.model.save(keras_model_path)

        # Save to coreml model file
        coreml_model_path = os.path.join(
            model_dir, model_basename + constants.EXTENSION_COREML)
        save_coreml_model(self.model, coreml_model_path, labels_path,
                          MLP_KERAS_INPUT_NAME, MLP_KERAS_OUTPUT_NAME)

    def load_model(self, model_dir, model_basename):
        """
        Loads model into object
        :param model_path: file name for model to be loaded.
        """
        labels_path = os.path.join(model_dir, constants.FILENAME_LABELS)
        self._load_labels_file(labels_path)

        keras_model_path = os.path.join(
            model_dir, model_basename + constants.EXTENSION_H5)
        self.model = load_model(keras_model_path)

    def _save_labels_file(self, filename):
        with open(filename, "w") as f:
            for label in self.labels:
                f.write(label + '\n')

    def _load_labels_file(self, filename):
        if not os.path.exists(filename):
            raise Exception("No labels file found in models directory")

        with open(filename, "r") as f:
            self.labels = [x.strip("\n") for x in f.readlines()]
            self.labels2Idx = utility.list_to_dict(self.labels)
