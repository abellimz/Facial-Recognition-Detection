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
    faces = None
    student_face = None

    def __init__(self, id, class_name, class_organisation_id,
                 attendance_date, a_rec_time, thumbnail,
                 photo, faces, student_face):
        self.id = id
        self.class_name = class_name
        self.class_organisation_id = class_organisation_id
        self.attendance_date = attendance_date
        self.a_rec_time = a_rec_time
        self.thumbnail = thumbnail
        self.photo = photo
        self.faces = faces
        self.student_face = student_face

    @classmethod
    def fromJson(cls, json):
        return cls(
            json["checkin_id"],
            json["class_name"],
            json["class_organisation_id"],
            json["attendance_date"],
            json["a_rec_time"],
            json["thumbnail"],
            json["photo"],
            [Face.fromJson(face_json) for face_json in json["faces"]] \
                if "faces" in json else [],
            json["student_face"] if "student_face" in json else -1)

    def toOrderedDict(self):
        return OrderedDict([
            ("checkin_id", self.id),
            ("class_name", self.class_name),
            ("class_organisation_id", self.class_organisation_id),
            ("attendance_date", self.attendance_date),
            ("a_rec_time", self.a_rec_time),
            ("thumbnail", self.thumbnail),
            ("photo", self.photo),
            ("faces", [face.toOrderedDict() for face in self.faces]),
            ("student_face", self.student_face)
        ])
