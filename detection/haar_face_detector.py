from detection.face_detector import FaceDetector
import cv2

class HaarFaceDetector(FaceDetector):
    def detectFaces(self, image_path):
        """
        Detects faces in given image and returns a list of 4-tuples of faces.
        Each 4-tuple is made up of (originX, originY, width, height) coordinates
        """
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                             "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces.tolist()
