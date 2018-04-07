import numpy as np
from keras.applications.mobilenet import MobileNet as MobileNetKeras


from common.config import IMAGE_SIZE_MOBILENET, MOBILENET_TRAIN_BATCH
from face_features.feature_extractor import FeatureExtractor, load_data


class MobileNet(FeatureExtractor):

    graph = None
    input_tensor = None
    embedding_tensor = None
    phase_train_tensor = None
    model = None

    def __init__(self):
        self.graph = None
        self.input_tensor = None
        self.embedding_tensor = None
        self.phase_train_tensor = None
        self._load_model()

    def _load_model(self):
        """
        Loads keras pre-trained mobilenet model
        """
        self.model = MobileNetKeras(
            input_shape=(IMAGE_SIZE_MOBILENET, IMAGE_SIZE_MOBILENET, 3),
            include_top=False,
            weights='imagenet', pooling='max')

    def extract_features(self, image_paths):
        """
        Extracts features for images corresponding to given list of image
        paths.
        :param image_paths: List of paths to images to be processed.
        :return: A list of features list corresponding to images in given order
        """
        if self.model is None:
            raise Exception("Model needs to be loaded first")

        batched_image_paths = np.array_split(image_paths,
                                             MOBILENET_TRAIN_BATCH)
        results = []
        for idx, batch in enumerate(batched_image_paths):
            print("Extracting features: batch %d of %d"
                  % (idx + 1, len(batched_image_paths)))
            image_data = load_data(batch,
                                   IMAGE_SIZE_MOBILENET, IMAGE_SIZE_MOBILENET)
            result = self.model.predict(image_data).tolist()
            results.extend(result)
        return results
