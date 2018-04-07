import argparse
import os
import sys

from classifier.keras_classifier import KerasClassifier
from classifier.mlp_keras_classifier import MLPKerasClassifier
from common.constants import FEATURE_DIMEN_MOBILENET, FEATURE_DIMEN_FACENET
from data_objects.json_student_dao import JsonStudentDAO

class TrainHelper():

    clf = None
    feature_dim = None

    def __init__(self, model_type):
        self.clf = MLPKerasClassifier()
        if model_type == "facenet":
            self.feature_dim = FEATURE_DIMEN_FACENET
        elif model_type == "mobilenet":
            self.feature_dim = FEATURE_DIMEN_MOBILENET

    def train(self, students):
        """
        Trains classifier on given data set
        """
        features = []
        labels = []
        for student in students:
            for check_in in student.check_ins:
                feature = check_in.feature
                if feature.values is None \
                        or len(feature.values) != self.feature_dim:
                    continue
                features.append(feature.values)
                # username as a temporary label
                labels.append(student.username)

        self.clf.new_model(list(set(labels)), (self.feature_dim,))
        self.clf.train(features, labels)

    def save_classifier(self, model_dir, model_basename):
        """
        Saves classifier to model_dir with model_base_name(before extension)
        Multiple model files are created with different extensions but same
        base names.
        """
        self.clf.save_model(model_dir, model_basename)

def main(args):
    if not os.path.exists(args.data_file):
        print("Could not find data file in %s" % args.data_file)

    student_dao = JsonStudentDAO(args.data_file)
    students = student_dao.getAllStudents()

    train_helper = TrainHelper(args.feature_type)
    train_helper.train(students)
    train_helper.save_classifier(args.model_dir, args.model_basename)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    # Data files args
    parser.add_argument('--data_file', type=str,
                        help='Json file containing data')
    parser.add_argument('--model_dir', type=str,
                        help='Directory to save model in')
    parser.add_argument('--model_basename', type=str,
                        help='Base name of the multiple models, without extension')
    parser.add_argument('--feature_type', type=str,
                        default="facenet",
                        help='Type of feature: facenet or mobilenet')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))