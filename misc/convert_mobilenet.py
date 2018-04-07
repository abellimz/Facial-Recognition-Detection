import argparse
import os
import sys

from common.config import MOBILENET_OUTPUT_NAME, MOBILENET_INPUT_NAME, MOBILENET_SCALE, MOBILENET_BLUE_BIAS, \
    MOBILENET_GREEN_BIAS, MOBILENET_RED_BIAS
from common.utility import save_coreml_keras
from face_features.mobilenet import MobileNet


def main(args):
    if not os.path.exists(os.path.dirname(args.model_file)):
        os.makedirs(os.path.dirname(args.model_file))

    mobilenet = MobileNet()
    print("Converting & Saving model to %s" % args.model_file)
    save_coreml_keras(mobilenet.model, args.model_file,
                      input_names=MOBILENET_INPUT_NAME,
                      image_input_names=MOBILENET_INPUT_NAME,
                      output_names=MOBILENET_OUTPUT_NAME,
                      image_scale=MOBILENET_SCALE,
                      red_bias=MOBILENET_RED_BIAS,
                      green_bias=MOBILENET_GREEN_BIAS,
                      blue_bias=MOBILENET_BLUE_BIAS)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str,
                        help='File name of output model')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))