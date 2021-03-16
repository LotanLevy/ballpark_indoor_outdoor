

import argparse
import csv
import os
from shutil import copyfile


def get_args_parser():
    parser = argparse.ArgumentParser(description='Process constraint collector args.')
    parser.add_argument('--root_path', '-r',  type=str, required=True)
    parser.add_argument('--bags_map_file', '-m',  type=str, required=True)

    parser.add_argument('--output_path', '-o',  type=str, required=True)

    return parser


def parse_bags_map(bags_map_file):
    word2classes = dict()
    with open(bags_map_file) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            word2classes[row[0].replace("'", "").strip()] = [item.strip() for item in row[1].replace("'", "").split(",")]
    print(word2classes)
    return word2classes

def copy_bag_2_dir(bag_dir, output_dir):
    for file_name in os.listdir(bag_dir):
        src_path = os.path.join(bag_dir, file_name)
        dst_path = os.path.join(output_dir, file_name)
        copyfile(src_path, dst_path)


def merge_and_copy_sub_bags(bags_map_dict, data_root_path, output_path):
    data_classes = os.listdir(data_root_path)
    for bag in bags_map_dict:
        subbags = bags_map_dict[bag]
        bag_output_path = os.path.join(output_path, bag)
        if not os.path.exists(bag_output_path):
            os.makedirs(bag_output_path)
        for subbag in subbags:
            if subbag not in data_classes:
                print("{} not in {}".format(subbag, data_root_path))
            else:
                subbag_path = os.path.join(data_root_path, subbag)
                copy_bag_2_dir(subbag_path, bag_output_path)





if __name__ == "__main__":
    args = get_args_parser().parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    bags_map_dict = parse_bags_map(args.bags_map_file)
    merge_and_copy_sub_bags(bags_map_dict, args.root_path, args.output_path)

