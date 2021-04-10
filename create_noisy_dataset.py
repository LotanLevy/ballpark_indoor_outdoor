
from files_helpers.classes_dir_builder import level_helper
import argparse
import os
from shutil import copyfile
import random
from affordance_tools.ConstraintsParser import ConstraintsParser
import random
from shutil import copyfile


# noisy_images_src = "C:\\Users\\lotan\\Documents\\studies\\phoenix\\datasets\\ADE20K_2016_07_26\\images\\training"
# dst_cls_path = "C:\\Users\\lotan\\Documents\\studies\\phoenix\\datasets\\ballpark_datasets\\noisy_io\\noisy_train"
# src_cls_path = "C:\\Users\\lotan\\Documents\\studies\\phoenix\\datasets\\ballpark_datasets\\io\\train"
SUFFIX = ".jpg"

def get_args_parser():
    parser = argparse.ArgumentParser(description='Process constraint collector args.')
    parser.add_argument('--src_path', '-s', type=str, required=True)
    parser.add_argument('--noise_path', '-n', type=str, required=True)
    parser.add_argument('--output_path',  '-o', type=str, required=True)
    parser.add_argument('--constraints_file',  '-c', type=str, required=True)
    parser.add_argument('--noise_percent', '-np', type=int)
    return parser


noisy_images_src = "/datasets/ballpark_datasets/dine/train"
dst_cls_path = "C:\\Users\\lotan\\Documents\\studies\\phoenix\\datasets\\ballpark_datasets\\noisy_dine\\noisy_train"
src_cls_path = "/datasets/ballpark_datasets/dine/train"

DIRS_LEVELS = 2

NOISE = 30

def read_all_images_from_root(dir_path, images_list):
    for file_name in os.listdir(dir_path):
        cur_path = os.path.join(dir_path, file_name)
        if os.path.isdir(cur_path):
            images_list += read_all_images_from_root(cur_path, [])
        elif cur_path.endswith(SUFFIX):
            images_list.append(cur_path)
    return images_list

def get_relevant_bags(constraints_path):
    constraints_parser = ConstraintsParser(constraints_path)
    return constraints_parser.all_classes




def copy_src(src_dir, dst_dir):
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
            for file in os.listdir(sub_cls_dir):
                file_path = os.path.join(sub_cls_dir, file)
                new_file_path = os.path.join(new_sub_cls_dir, file)
                copyfile(file_path, new_file_path)


#
# def create_noisy_classes(src_dir, noisy_src, dst_dir, noise_size_for_cls = 0.1):
#     copy_src(src_dir, dst_dir)
#     files_paths = level_helper(noisy_src, DIRS_LEVELS, [], SUFFIXES)
#     for cls in os.listdir(dst_dir):
#         for sub_cls in os.listdir(os.path.join(dst_dir, cls)):
#             dst_sub_cls_dir = os.path.join(os.path.join(dst_dir, cls), sub_cls)
#             size = int(len(os.listdir(dst_sub_cls_dir)) * noise_size_for_cls)
#             print(dst_sub_cls_dir)
#             print(size)
#             noise_paths = random.sample(files_paths, size)
#             for path in noise_paths:
#                 file_name = os.path.basename(path)
#                 copyfile(path, os.path.join(dst_sub_cls_dir, file_name))


# create_noisy_classes(src_cls_path, noisy_images_src, dst_cls_path+"_"+str(NOISE), noise_size_for_cls=NOISE/100)



def copy_files_int_output(paths, dest_path):
    for path in paths:
        new_path = os.path.join(dest_path, os.path.basename(os.path.dirname(path)) + os.path.basename(path))
        copyfile(path, new_path)


def copy_bag_with_noise(bag_src, noise_paths, dest_path, noise_size):
    bag_paths = read_all_images_from_root(bag_src, [])
    bag_noise_paths = random.sample(noise_paths, int((noise_size / 100) * len(bag_paths)))

    new_bag_paths = bag_paths + bag_noise_paths
    copy_files_int_output(new_bag_paths, dest_path)
    print("bag {} got {} files and {} noise".format(os.path.basename(bag_src), len(bag_paths), len(bag_noise_paths)))




def main():
    args = get_args_parser().parse_args()
    print(vars(args))
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    noise_paths = read_all_images_from_root(args.noise_path, [])
    bags = get_relevant_bags(args.constraints_file)
    for bag in bags:
        bag_src = os.path.join(args.src_path, bag)
        bag_dest = os.path.join(args.output_path, bag)
        if not os.path.exists(bag_dest):
            os.makedirs(bag_dest)
        copy_bag_with_noise(bag_src, noise_paths, bag_dest, args.noise_percent)




if __name__ == "__main__":
    main()