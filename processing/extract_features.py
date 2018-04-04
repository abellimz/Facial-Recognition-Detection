import argparse

import os
import sys

from data_objects.feature import Feature
from data_objects.json_student_dao import JsonStudentDAO
from face_features.mobilenet import MobileNet


class FeatureHelper():
    checkin_image_dir = None
    crop_image_dir = None
    feature_extractor = None

    def __init__(self):
        self.feature_extractor = MobileNet()

    def process_students(self, students):
        student_idxs = []
        check_in_idxs = []
        image_paths = []
        # Get all images first
        for sidx, student in enumerate(students):
            for cidx, check_in in enumerate(student.check_ins):
                student_face_idx = check_in.student_face
                if student_face_idx < 0:
                    continue
                student_face = check_in.faces[student_face_idx]
                image_paths.append(student_face.photo)
                student_idxs.append(sidx)
                check_in_idxs.append(cidx)

        features = self.feature_extractor.extract_features(image_paths)

        # Save features to original dictionary
        for idx, feature in enumerate(features):
            student_idx = student_idxs[idx]
            check_in_idx = check_in_idxs[idx]
            check_in = students[student_idx].check_ins[check_in_idx]
            check_in.feature = Feature("mobilenet", feature)


def main(args):
    if not os.path.exists(args.data_file):
        print("Could not find data file in %s" % args.data_file)

    student_dao = JsonStudentDAO(args.data_file)
    students = student_dao.getAllStudents()

    feature_helper = FeatureHelper()
    feature_helper.process_students(students)
    student_dao.saveStudents(students, args.out_file)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    # Data files args
    parser.add_argument('--data_file', type=str,
                        help='Json file containing data')
    parser.add_argument('--out_file', type=str,
                        help='Json file to output data')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))