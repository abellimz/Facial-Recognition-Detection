from collections import OrderedDict

from data_objects.check_in import CheckIn


class Student:
    name = None
    username = None
    gender = None
    dob = None
    school_name = None
    school_organisation_id = None
    check_ins = None

    def __init__(self, name, username, gender, dob, school_name,
                 school_organisation_id, check_ins):
        self.name = name
        self.username = username
        self.gender = gender
        self.dob = dob
        self.school_name = school_name
        self.school_organisation_id = school_organisation_id
        self.check_ins = check_ins

    @classmethod
    def fromJson(cls, json):
        return cls(
            json["name"],
            json["username"],
            json["gender"],
            json["dob"],
            json["school_name"],
            json["school_organisation_id"],
            [CheckIn(check_in_json) for check_in_json in json["check_ins"]])

    def toOrderedDict(self):
        return OrderedDict([
            ("name", self.name),
            ("username", self.username),
            ("gender", self.gender),
            ("dob", self.dob),
            ("school_name", self.school_name),
            ("school_organisation_id", self.school_organisation_id),
            ("check_ins", [x.toOrderedDict() for x in self.check_ins])
        ])