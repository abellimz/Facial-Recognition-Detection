import argparse

import os
import sys

from common.utility import photo_url_to_image_path, make_cropped_images
from data_objects.face import Face
from data_objects.json_student_dao import JsonStudentDAO
from detection.ssd_face_detector import SsdFaceDetector


class DetectionHelper():
    checkin_image_dir = None
    crop_image_dir = None
    save_crops = None
    face_detector = None

    def __init__(self, image_dir, crop_image_dir, save_crops):
        self.checkin_image_dir = image_dir
        self.crop_image_dir = crop_image_dir
        self.save_crops = save_crops
        self.face_detector = SsdFaceDetector()

    def process_check_in(self, check_in, crop_label):
        """ Processes a given check in object by running detection
            and then using that information to make cropped images.

            crop_label is a label for the given crop and is used as a name
            for the subfolder in the crop_image_dir for this particular
            check in.

            Final saved paths of cropped images and detected boxes
            information are saved into the check_in object.
        """
        image_path = photo_url_to_image_path(self.checkin_image_dir, check_in.photo)

        face_rects = self.face_detector.detectFaces(image_path)
        if face_rects is None:
            raise ValueError("Face detector returned None")

        faces = []
        cropped_image_paths = make_cropped_images(
            image_path,
            os.path.join(self.crop_image_dir, crop_label),
            check_in.id, face_rects, self.save_crops)

        for idx, face_rect in enumerate(face_rects):
            photo = cropped_image_paths[idx]
            face = Face(face_rect, photo)
            faces.append(face)

        check_in.faces = faces

def main(args):
    if not os.path.exists(args.data_file):
        print("Could not find data file in %s" % args.data_file)

    student_dao = JsonStudentDAO(args.data_file)
    students = student_dao.getAllStudents()
    detection_helper = DetectionHelper(args.checkin_image_dir,
                                       args.crop_image_dir,
                                       args.save_crops)
    for idx, student in enumerate(students):
        print("Processing student %d/%d: %s" %
              (idx + 1, len(students), student.name))
        for check_in in student.check_ins:
            detection_helper.process_check_in(check_in, student.username)

    student_dao.saveStudents(students, args.out_file)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    # Data files args
    parser.add_argument('--data_file', type=str,
                        help='Json file containing data')
    parser.add_argument('--out_file', type=str,
                        help='Json file to output data')
    # Image output args
    parser.add_argument('--checkin_image_dir', type=str,
                        help='Base directory of ')
    parser.add_argument('--save_crops', type=bool,
                        help='Whether to crop and save images')
    parser.add_argument('--crop_image_dir', type=str,
                        help='Base directory for cropped images')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))