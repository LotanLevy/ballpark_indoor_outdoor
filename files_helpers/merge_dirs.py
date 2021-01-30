

import argparse
import os
from shutil import copyfile


def get_args_parser():
    parser = argparse.ArgumentParser(description='Process constraint collector args.')
    parser.add_argument('--root_path', '-r',  type=str, required=True)
    parser.add_argument('--output_path', '-o',  type=str, required=True)

    return parser




def create_dirs_dict(root):
    dirs_dict = dict()
    for dir in os.listdir(root):
        dir_path = os.path.join(root, dir)
        for sub_dir in os.listdir(dir_path):
            sub_dir_path = os.path.join(dir_path, sub_dir)
            if sub_dir not in dirs_dict:
                dirs_dict[sub_dir] = []
            dirs_dict[sub_dir].append(sub_dir_path)
    return dirs_dict


def copy_dir(src, dst):
    for item in os.listdir(src):
        full_src_path = os.path.join(src, item)
        new_path = os.path.join(dst, item)
        copyfile(full_src_path, new_path)



def merge_into_output(dirs_dict, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for dir in dirs_dict:
        dir_path = os.path.join(output_path, dir)
        os.makedirs(dir_path)
        for path in dirs_dict[dir]:
            copy_dir(path, dir_path)

if __name__ == "__main__":
    args = get_args_parser().parse_args()

    dirs_dict = create_dirs_dict(args.root_path)
    merge_into_output(dirs_dict, args.output_path)


