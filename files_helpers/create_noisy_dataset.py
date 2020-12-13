
from files_helpers.classes_dir_builder import level_helper
import os
from shutil import copyfile
import random

noisy_images_src = "C:\\Users\\lotan\\Documents\\studies\\phoenix\\datasets\\ADE20K_2016_07_26\\images\\training"
dst_cls_path = "C:\\Users\\lotan\\Documents\\studies\\phoenix\\datasets\\ballpark_datasets\\dine\\noisy_train_5"
src_cls_path = "C:\\Users\\lotan\\Documents\\studies\\phoenix\\datasets\\ballpark_datasets\\dine\\train"
SUFFIXES = [".jpg"]

DIRS_LEVELS = 3

NOISE = 5/100


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



def create_noisy_classes(src_dir, noisy_src, dst_dir, noise_size_for_cls = 0.1):
    copy_src(src_dir, dst_dir)
    files_paths = level_helper(noisy_src, DIRS_LEVELS, [], SUFFIXES)
    for cls in os.listdir(dst_dir):
        for sub_cls in os.listdir(os.path.join(dst_dir, cls)):
            dst_sub_cls_dir = os.path.join(os.path.join(dst_dir, cls), sub_cls)
            size = int(len(os.listdir(dst_sub_cls_dir)) * noise_size_for_cls)
            print(dst_sub_cls_dir)
            print(size)
            noise_paths = random.sample(files_paths, size)
            for path in noise_paths:
                file_name = os.path.basename(path)
                copyfile(path, os.path.join(dst_sub_cls_dir, file_name))


create_noisy_classes(src_cls_path, noisy_images_src, dst_cls_path, noise_size_for_cls=NOISE)