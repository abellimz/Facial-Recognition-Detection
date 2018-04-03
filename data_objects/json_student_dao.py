import json

import os

from data_objects.student_dao import StudentDAO
from data_objects.student import Student


class JsonStudentDAO(StudentDAO):
    json_path = None
    students = None

    def __init__(self, json_path):
        self.json_path = json_path
        with open(json_path) as f:
            json_dict = json.load(f)
            self.students = [Student.fromJson(student_json) for student_json in json_dict]

    def getAllStudents(self):
        """ Returns a list of students with respective check_ins"""
        return self.students

    def saveStudents(self, students, out_path):
        """ Saves a list of students to out_path"""
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))

        with open(out_path, "w") as f:
            ordered_check_ins = [x.toOrderedDict() for x in students]
            json.dump(ordered_check_ins, f, indent=2)
