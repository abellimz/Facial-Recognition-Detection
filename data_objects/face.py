from collections import OrderedDict


class Face:
    rect = None
    photo = None
    facenet_feature = None

    def __init__(self, json):
        self.rect = json["rect"]
        self.photo = json["photo"]
        self.facenet_feature = json["facenet_feature"]

    def toOrderedDict(self):
        return OrderedDict([
            ("rect", self.rect),
            ("photo", self.photo),
            ("facenet_feature", self.facenet_feature)
        ])
