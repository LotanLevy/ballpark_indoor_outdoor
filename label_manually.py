

import argparse
import os

def get_args_parser():
    parser = argparse.ArgumentParser(description='Process constraint collector args.')
    parser.add_argument('--images_path', '-p',  type=str, required=True)
    parser.add_argument('--output_path', type=str, default=os.getcwd())
    return parser


def main():
    args = get_args_parser().parse_args()
    print(vars(args))
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)


if __name__ == "__main__":
    main()