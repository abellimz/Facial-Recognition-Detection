import cv2
import os

import tensorflow as tf
import numpy as np

from detection.face_detector import FaceDetector
from common.config import PRED_THRESHOLD_SSD


class SsdFaceDetector(FaceDetector):
    detection_graph = None
    sess = None

    def __init__(self):
        model_filename = os.path.dirname(os.path.dirname(__file__)) + "/model/ssd_widerface.pb"
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_filename, 'rb') as f:
                serialized_graph = f.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with self.detection_graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

        self.sess = tf.Session(graph=self.detection_graph)

    def detectFaces(self, image_path):
        """
        Detects faces in given image and returns a list of 4-tuples of faces.
        Each 4-tuple is made up of (originX, originY, width, height) coordinates
        """
        image = cv2.imread(image_path)
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        # Actual detection.
        ssd_outputs = self.sess.run([boxes, scores, classes],
                                    feed_dict={image_tensor: image_np_expanded})
        boxes, scores, classes = [np.squeeze(result) for result in ssd_outputs]
        result_boxes = []
        for idx, box in enumerate(boxes):
            if classes[idx] != 1 or scores[idx] < PRED_THRESHOLD_SSD:
                continue
            result_boxes.append(box.tolist())
        image_shape = image_np.shape
        image_height = image_shape[0]
        image_width = image_shape[1]

        for idx, (topY, topX, botY, botX) in enumerate(result_boxes):
            newTopX = int(topX * image_width)
            newTopY = int(topY * image_height)
            widthX = int((botX - topX) * image_width)
            widthY = int((botY - topY) * image_height)
            result_boxes[idx] = [newTopX, newTopY, widthX, widthY]
        return result_boxes

