from detection.face_detector import FaceDetector
import cv2

class OpenCVFaceDetector(FaceDetector):
    def detectFaces(self, image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                             "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        cv2.imwrite('/tmp/img.jpg', img[faces[0][1]: faces[0][1] + faces[0][3],
                                    faces[0][0]: faces[0][0] + faces[0][2]])
        return faces
