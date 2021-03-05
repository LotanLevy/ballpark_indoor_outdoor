
from files_helpers.classes_dir_builder import level_helper
import os
from shutil import copyfile
import random
import argparse



def get_args_parser():
    parser = argparse.ArgumentParser(description='Process constraint collector args.')
    parser.add_argument('--orig_dir', '-o',  type=str, required=True)
    parser.add_argument('--dest_dir', '-d',  type=str, required=True)
    parser.add_argument('--size_factor', '-s',  type=float, default=0.4)

    return parser
#
# dst_cls_path = "C:\\Users\\lotan\\Documents\\studies\\phoenix\\datasets\\ballpark_datasets\\small_stab\\train"
# src_cls_path = "C:\\Users\\lotan\\Documents\\studies\\phoenix\\datasets\\ballpark_datasets\\stab\\train"
SUFFIXES = [".jpg"]


# REDUCE_FACTOR = 15


def copy_src(src_dir, dst_dir, size_factor=0.5):
    for cls in os.listdir(src_dir):
        cls_dir = os.path.join(src_dir, cls)
        new_cls_dir = os.path.join(dst_dir, cls)
        if not os.path.exists(new_cls_dir):
            os.makedirs(new_cls_dir)
        for sub_cls in os.listdir(cls_dir):
            sub_cls_dir = os.path.join(cls_dir, sub_cls)
            new_sub_cls_dir = os.path.join(new_cls_dir, sub_cls)
            if not os.path.exists(new_sub_cls_dir):
                os.makedirs(new_sub_cls_dir)
            size = int(len(os.listdir(sub_cls_dir)) * size_factor)
            selected_paths = random.sample(os.listdir(sub_cls_dir), size)
            for file in selected_paths:
                file_path = os.path.join(sub_cls_dir, file)
                new_file_path = os.path.join(new_sub_cls_dir, file)
                copyfile(file_path, new_file_path)


def main():
    args = get_args_parser().parse_args()
    print(vars(args))
    if not os.path.exists(args.dest_dir):
        os.makedirs(args.dest_dir)

    copy_src(args.orig_dir, args.dest_dir, args.size_factor)


if __name__ == "__main__":
    main()
