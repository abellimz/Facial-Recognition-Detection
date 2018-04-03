from collections import OrderedDict


class Face:
    rect = None
    photo = None

    def __init__(self, rect, photo):
        self.rect = rect
        self.photo = photo

    @classmethod
    def fromJson(cls, json):
        return cls(
            json["rect"],
            json["photo"])

    def toOrderedDict(self):
        return OrderedDict([
            ("rect", self.rect),
            ("photo", self.photo)
        ])
