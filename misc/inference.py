import argparse
import os
import sys

import numpy as np

from classifier.mlp_keras_classifier import MLPKerasClassifier
from common.utility import make_cropped_images
from detection.ssd_face_detector import SsdFaceDetector
from face_features.facenet_keras import KerasFaceNet
from face_features.mobilenet import MobileNet


def main(args):
    if not os.path.exists(args.image):
        print("Could not find image file: %s" % args.image)

    face_detector = SsdFaceDetector()
    face_rects = face_detector.detectFaces(args.image)
    face_image_paths = make_cropped_images(
        args.image, "/tmp", os.path.basename(args.image).split(".")[0],
        face_rects, args.save_crops)

    feature_extractor = KerasFaceNet()
    all_features = feature_extractor.extract_features(face_image_paths)

    classifier = MLPKerasClassifier()
    classifier.load_model(args.model_dir, args.model_basename)
    all_probs = classifier.infer(all_features)
    classes = classifier.get_classes()
    prediction_idxs = [np.argmax(probs) for probs in all_probs]
    predictions = [classes[idx] for idx in prediction_idxs]
    scores = [all_probs[idx][pred_idx] for idx, pred_idx in enumerate(prediction_idxs)]
    print("predictions: %s" % str(predictions))
    print("scores: %s" % str(scores))

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    # Data files args
    parser.add_argument('--image', type=str,
                        help='File name of image to run inference on')
    parser.add_argument('--model_dir', type=str,
                        help='Directory to model')
    parser.add_argument('--model_basename', type=str,
                        help='Base name of model, without extension')
    parser.add_argument('--save_crops', type=bool,
                        help='Whether to save crops')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))