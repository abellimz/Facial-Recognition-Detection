import abc

class FaceDetector(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def detectFaces(self, image_path):
        """
        Detects faces in given image and returns a list of 4-tuples of faces.
        Each 4-tuple is made up of (originX, originY, width, height) coordinates
        """
        pass
