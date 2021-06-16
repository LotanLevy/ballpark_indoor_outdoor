import os
import random
from shutil import copyfile


ROOT_PATH = "C:\\Users\\lotan\\Documents\\studies\\Affordances\\datasets\\256_ObjectCategories"

OUTPUT_PATH = "C:\\Users\\lotan\\Documents\\studies\\phoenix\\datasets\\ballpark_datasets\\new\\contain\\labeled_test"

TRAIN_CLASSES = []

def get_relevant_dirs(root, excluded_classes):
    relevant_dirs = []
    for cls in os.listdir(root):
        if cls not in excluded_classes:
            relevant_dirs.append(os.path.join(root, cls))
    return relevant_dirs

def copy_paths_to_dir(paths, cls, dir_path):
    for path in paths:
        new_path = os.path.join(dir_path, cls+"_"+os.path.basename(path))
        copyfile(path, new_path)


def subsample_from_dirs(relevant_dirs, output_path, max_num_from_dir):
    for path in relevant_dirs:
        dir_files = os.listdir(path)
        relevant_indices = random.sample(range(0, len(dir_files)), min(len(dir_files), max_num_from_dir))
        relevant_paths = [os.path.join(path, dir_files[i]) for i in relevant_indices]
        copy_paths_to_dir(relevant_paths, os.path.basename(path), output_path)

def main():
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    relevant_dirs = get_relevant_dirs(ROOT_PATH, TRAIN_CLASSES)
    subsample_from_dirs(relevant_dirs, OUTPUT_PATH, 10)


if __name__=="__main__":
    main()