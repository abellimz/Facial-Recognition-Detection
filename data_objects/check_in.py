from collections import OrderedDict

from data_objects.face import Face


class CheckIn:
    id = None
    class_name = None
    class_organisation_id = None
    attendance_date = None
    a_rec_time = None
    thumbnail = None
    photo = None

    def __init__(self, json):
        self.id = json["checkin_id"]
        self.class_name = json["class_name"]
        self.class_organisation_id = json["class_organisation_id"]
        self.attendance_date = json["attendance_date"]
        self.a_rec_time = json["a_rec_time"]
        self.thumbnail = json["thumbnail"]
        self.photo = json["photo"]
        self.faces = [Face(face_json) for face_json in json["faces"]] \
            if "faces" in json else []

    def toOrderedDict(self):
        return OrderedDict([
            ("checkin_id", self.id),
            ("class_name", self.class_name),
            ("class_organisation_id", self.class_organisation_id),
            ("attendance_date", self.attendance_date),
            ("a_rec_time", self.a_rec_time),
            ("thumbnail", self.thumbnail),
            ("photo", self.photo),
            ("faces", [face.toOrderedDict() for face in self.faces])
        ])
