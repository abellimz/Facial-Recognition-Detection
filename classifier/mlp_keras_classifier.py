import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam

from common.config import (
    MLP_CLASSIFIER_NUM_EPOCHS,
    MLP_CLASSIFIER_BATCH_SIZE,
    MLP_CLASSIFIER_HIDDEN_SIZE
)
from classifier.keras_classifier import KerasClassifier

class MLPKerasClassifier(KerasClassifier):

    def new_model(self, class_labels, feature_shape):
        super(MLPKerasClassifier, self).new_model(class_labels, feature_shape)
        self.model = Sequential()
        self.model.add(
            Dense(MLP_CLASSIFIER_HIDDEN_SIZE,
                  activation="relu",
                  input_shape=feature_shape,
                  name="hidden_layer"))
        self.model.add(
            Dense(len(self.labels),
                  activation="softmax",
                  name="output_layer"))
        self.model.compile(optimizer=adam(lr=0.003),
                           loss="sparse_categorical_crossentropy",
                           metrics=["accuracy"])

    def train(self, features, labels):
        """
        Trains NN model with given list of training features.
        Labels should also be supplied such that they correspond to their
        respective training feature..
        :param features: list of training features where each feature is a list
        :param labels: list of labels corresponding to each numpy training features.
        """
        if self.model is None:
            raise Exception("No model created or loaded yet")

        encoded_labels = list(map(lambda x: self.labels2Idx[x], labels))
        encoded_labels = np.asarray(encoded_labels)
        features = np.asarray([np.asarray(feature) for feature in features])
        zipped_data = zip(features, encoded_labels)
        np.random.shuffle(zipped_data)
        features, labels = (np.array(x) for x in zip(*zipped_data))
        self.model.fit(features, labels,
                       batch_size = MLP_CLASSIFIER_BATCH_SIZE,
                       epochs = MLP_CLASSIFIER_NUM_EPOCHS,
                       # validation_split=0.1,
                       shuffle=True)

    def infer(self, features):
        """
        Performs inference using model on given list of features.
        :param features: list of features in numpy
        :return: list of list of confidence values in shape (n_examples, n_classes)
            For each confidence value list, every element at ith index corresponds to
            the confidence value for the ith class in the classes property.
        """
        if self.model is None:
            raise Exception("No model created or loaded yet")

        features = np.asarray(features)
        return self.model.predict(features, MLP_CLASSIFIER_BATCH_SIZE).tolist()

    def get_classes(self):
        """
        Returns the list of classes used by this classifier.
        The order of classes corresponds to that in confidence scores from
        inference. This list is also in the same order as class_labels if new_model()
        method is used to generate the model. Otherwise, it is in the same order as
        defined in loaded model.
        """
        if self.labels is None:
            raise Exception("Classes are only available after model creation/load")
        return self.labels