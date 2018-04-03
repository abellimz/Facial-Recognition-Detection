import argparse
import os
import sys

from data_objects.json_student_dao import JsonStudentDAO


def main(args):
    if not os.path.exists(args.data_file):
        print("Could not find data file in %s" % args.data_file)

    student_dao = JsonStudentDAO(args.data_file)
    students = student_dao.getAllStudents()
    for idx, student in enumerate(students):
        print("Labelling student %d/%d: %s" %
              (idx, len(students), student.name))
        for check_in in student.check_ins:
            student_face = -1
            for idx, face in enumerate(check_in.faces):
                # Using this as indicator that this is the correct face
                if os.path.exists(face.photo):
                    student_face = idx
                    break
            check_in.student_face = student_face
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