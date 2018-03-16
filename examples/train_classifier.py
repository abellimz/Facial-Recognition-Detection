import argparse
import sys
import os

from training.classifier_json_trainer import ClassifierJsonTrainer

def main(args):
    if not os.path.exists(args.features_dir):
        raise Exception("Features directory not found")

    trainer = ClassifierJsonTrainer(
        args.features_dir, args.model_dir, args.model_basename)
    trainer.train()
    trainer.save_classifier()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--features_dir', type=str,
                        help='Path to feature file')
    parser.add_argument('--model_dir', type=str,
                        help='Path to model files')
    parser.add_argument('--model_basename', type=str,
                        help='Path to model files')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))