from collections import OrderedDict


class Feature:
    extractor = None
    values = None

    def __init__(self, extractor, values):
        self.extractor = extractor
        self.values = values

    @classmethod
    def fromJson(cls, json):
        return cls(
            json["extractor"],
            json["values"])

    def toOrderedDict(self):
        return OrderedDict([
            ("extractor", self.extractor),
            ("values", self.values)
        ])
