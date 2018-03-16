import json
import sys
import argparse

import os
import urllib.request as request

from common.constants import JSON_KEY_LABEL, JSON_KEY_EMBEDDINGS
from face_features.facenet import FaceNet
from data.json_check_in_dao import JsonCheckInDao
from common import utility


def photo_url_to_filename(image_dir, photo_url):
    bucket_id = utility.get_photo_bucket_id(photo_url)
    bucket_path = os.path.join(image_dir, bucket_id)
    photo_basename = os.path.basename(photo_url)
    filename = os.path.join(bucket_path, photo_basename)
    if not os.path.exists(bucket_path):
        os.makedirs(bucket_path)
    if not os.path.exists(filename):
        response = request.urlopen(photo_url)
        data = response.read()
        with open(filename, "wb") as f:
            f.write(data)
    return filename

def main(args):
    json_check_in_dao = JsonCheckInDao(args.check_in_dir)
    all_check_ins = json_check_in_dao.getAllCheckIns()
    image_paths = []
    child_names = []
    for child_name, child_check_ins in all_check_ins.items():
        for child_check_in in child_check_ins:
            image_path = photo_url_to_filename(
                args.image_dir, child_check_in.photo_url)
            image_paths.append(image_path)
            child_names.append(child_name)
    facenet = FaceNet()
    facenet.load_model(args.model)
    embeddings = facenet.extract_features(image_paths)
    data = []
    for idx, child_name in enumerate(child_names):
        data_point = {}
        data_point[JSON_KEY_LABEL] = child_name
        data_point[JSON_KEY_EMBEDDINGS] = embeddings[idx]
        data.append(data_point)
    with open(args.features_file, 'w') as outfile:
        json.dump(data, outfile, indent=2)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--check_in_dir', type=str,
                        help='Directory containing check-in information in json files')
    parser.add_argument('--image_dir', type=str,
                        help='Directory for loading/saving all images')
    parser.add_argument('--features_file', type=str,
                        help='Path to feature file')
    parser.add_argument('--model', type=str,
                        help='Path to model files')


    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))