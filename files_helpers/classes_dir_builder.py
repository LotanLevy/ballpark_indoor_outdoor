import os
from shutil import copyfile


SRC_ROOT = "C:\\Users\\lotan\\Documents\\studies\\Affordances\\datasets\\ADE20K_2016_07_26\\ADE20K_2016_07_26\\images\\training"
DST_ROOT = "C:\\Users\\lotan\\Documents\\studies\\Affordances\\datasets"

ROOT_DIR_NAME = "ade20k_io_training"

KEYS = {"indoor": ["indoor"],"outdoor":["outdoor"]}
SUFFIXES = [".jpg"]

DIRS_LEVELS = 3

def level_helper(cur_dir, cur_level, cur_paths, relevant_paths_suffix):
    for file in os.listdir(cur_dir):
        file_path = os.path.join(cur_dir, file)
        if os.path.isdir(file_path):
            if cur_level == 0:
                continue
            cur_paths += level_helper(file_path, cur_level - 1, [], relevant_paths_suffix)
        else:
            for suffix in relevant_paths_suffix:
                if file.endswith(suffix):
                    cur_paths.append(file_path)
    return cur_paths


def get_relevant_paths(src, search_level, keys, suffixes):
    path2cls = dict()
    files_paths = level_helper(src, search_level, [], suffixes)
    for path in files_paths:
        sub_path = os.path.relpath(path, src)
        for key in keys:
            for sub_key in keys[key]:
                if sub_key in sub_path:
                    if key not in path2cls:
                        path2cls[key] = []
                    path2cls[key].append(path)
    return path2cls

def write_into_dest(paths2cls, dest):
    for cls in paths2cls:
        cls_path = os.path.join(dest, cls)
        if not os.path.exists(cls_path):
            os.makedirs(cls_path)
        for path in paths2cls[cls]:
            file_name = os.path.basename(path)
            copyfile(path, os.path.join(cls_path, file_name))


if __name__ == "__main__":

    output_dir = os.path.join(DST_ROOT, ROOT_DIR_NAME)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    relevant_paths = get_relevant_paths(SRC_ROOT, DIRS_LEVELS, KEYS, SUFFIXES)
    write_into_dest(relevant_paths, output_dir)

