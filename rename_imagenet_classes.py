

import argparse
import ast
import os

def get_args_parser():
    parser = argparse.ArgumentParser(description='Process constraint collector args.')
    parser.add_argument('--imagenet_path', "-s", type=str, required=True)
    parser.add_argument('--imagenet_labels_map_path', "-m", type=str, required=True)
    return parser

def get_class_label(number_str_label, labels_map):
    label_int = int(number_str_label)
    label = labels_map[label_int]
    if "," in label:
        label = label.split(",")[0]
    full_label = number_str_label+"_"+label.replace(" ", "_")
    return full_label

def parse_imagenet_map(map_path):
    with open(map_path) as f:
        data = ast.literal_eval(f.read())
        print(data)
    return data

def rename_directories(dir_path, labels_map):
    for dir in os.listdir(dir_path):
        label = get_class_label(dir, labels_map)
        os.rename(os.path.join(dir_path, dir), os.path.join(dir_path, label))
        print(os.path.join(dir_path, dir), os.path.join(dir_path, label))


def main():
    args = get_args_parser().parse_args()
    labels_map = parse_imagenet_map(args.imagenet_labels_map_path)
    rename_directories(args.imagenet_path, labels_map)




main()