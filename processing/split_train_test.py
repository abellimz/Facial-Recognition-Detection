import argparse
import sys

import os

from data_objects.json_student_dao import JsonStudentDAO
from data_objects.student import Student


def main(args):
    if not os.path.exists(args.data_file):
        print("Could not find data file in %s" % args.data_file)

    student_dao = JsonStudentDAO(args.data_file)
    students = student_dao.getAllStudents()
    train_students = []
    test_students = []
    for student in students:
        train_student = Student(
            student.name, student.username, student.gender,
            student.dob, student.school_name, student.school_organisation_id,
            student.check_ins[args.train_start - 1 : args.train_end])
        test_student = Student(
            student.name, student.username, student.gender,
            student.dob, student.school_name, student.school_organisation_id,
            student.check_ins[args.test_start - 1 : args.test_end])
        train_students.append(train_student)
        test_students.append(test_student)

    student_dao.saveStudents(train_students, args.train_out_file)
    student_dao.saveStudents(test_students, args.test_out_file)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    # Data files args
    parser.add_argument('--data_file', type=str,
                        help='Json file containing data')
    parser.add_argument('--train_out_file', type=str,
                        help='Json output file containing train data')
    parser.add_argument('--test_out_file', type=str,
                        help='Json output file containing test data')
    # Partitioning args
    parser.add_argument('--train_start', type=int,
                        help='Start index of training set')
    parser.add_argument('--train_end', type=int,
                        help='End index of training set')
    parser.add_argument('--test_start', type=int,
                        help='End index of test set')
    parser.add_argument('--test_end', type=int,
                        help='End index of test set')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))