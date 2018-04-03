import unittest

import os
import numpy as np

from detection.opencv_face_detector import OpenCVFaceDetector

class FaceDetectTest(unittest.TestCase):
    def test_opencv_detect(self):
        face_detector = OpenCVFaceDetector()
        expected = [[524, 262, 348, 348], [559, 816, 318, 318]]
        boxes = face_detector.detectFaces(os.path.dirname(__file__)
                                          + "/data/test_check_in.jpg")
        self.assertTrue(np.array_equal(expected, boxes))


if __name__ == '__main__':
    unittest.main()
