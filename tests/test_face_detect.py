import unittest

import os
import numpy as np

from detection.haar_face_detector import HaarFaceDetector
from detection.ssd_face_detector import SsdFaceDetector


class FaceDetectTest(unittest.TestCase):
    def test_haar_detect(self):
        face_detector = HaarFaceDetector()
        expected = [[524, 262, 348, 348], [559, 816, 318, 318]]
        boxes = face_detector.detectFaces("./data/test_check_in.jpg")
        self.assertEqual(expected, boxes)

    def test_ssd_detect(self):
        face_detector = SsdFaceDetector()
        expected = [[586, 832, 244, 263], [535, 260, 288, 280]]
        boxes = face_detector.detectFaces("./data/test_check_in.jpg")
        self.assertEqual(expected, boxes)


if __name__ == '__main__':
    unittest.main()
