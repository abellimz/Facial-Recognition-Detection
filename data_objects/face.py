from collections import OrderedDict


class Face:
    rect = None
    photo = None
    facenet_feature = None

    def __init__(self, rect, photo, facenet_feature):
        self.rect = rect
        self.photo = photo
        self.facenet_feature = facenet_feature \
            if "facenet_feature" in facenet_feature \
            else []

    @classmethod
    def fromJson(cls, json):
        return cls(
            json["rect"],
            json["photo"],
            json["facenet_feature"] \
                if "facenet_feature" in json["facenet_feature"] else [])

    def toOrderedDict(self):
        return OrderedDict([
            ("rect", self.rect),
            ("photo", self.photo),
            ("facenet_feature", self.facenet_feature)
        ])
