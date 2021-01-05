
from files_helpers.classes_dir_builder import level_helper
import os
from shutil import copyfile
import random

dst_cls_path = "C:\\Users\\lotan\\Documents\\studies\\phoenix\\datasets\\ballpark_datasets\\small_stab\\train"
src_cls_path = "C:\\Users\\lotan\\Documents\\studies\\phoenix\\datasets\\ballpark_datasets\\stab\\train"
SUFFIXES = [".jpg"]


REDUCE_FACTOR = 15


def copy_src(src_dir, dst_dir, reduce_factor=0.25):
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
            size = int(len(os.listdir(sub_cls_dir)) * (1-reduce_factor))
            selected_paths = random.sample(os.listdir(sub_cls_dir), size)
            for file in selected_paths:
                file_path = os.path.join(sub_cls_dir, file)
                new_file_path = os.path.join(new_sub_cls_dir, file)
                copyfile(file_path, new_file_path)



copy_src(src_cls_path, dst_cls_path+str(REDUCE_FACTOR), reduce_factor=(REDUCE_FACTOR/100))


