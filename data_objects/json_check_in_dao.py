import glob
import json

import os

from data_objects.check_in_dao import CheckInDAO
from data_objects.check_in import CheckIn

class JsonCheckInDao(CheckInDAO):

    checkIns = None

    def __init__(self, json_dir):
        self.checkIns = {}
        json_basenames = glob.glob1(json_dir, "*CheckIn*.json")
        for json_basename in json_basenames:
            json_path = os.path.join(json_dir, json_basename)
            data = json.load(open(json_path))
            for check_in_dict in data:
                check_in = CheckIn(check_in_dict)
                if check_in.child_name not in self.checkIns:
                    self.checkIns[check_in.child_name] = []
                self.checkIns[check_in.child_name].append(check_in)

    def getAllCheckIns(self):
        """ Returns a dictionary of studentName: [CheckIn]"""
        return self.checkIns