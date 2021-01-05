
import os
import numpy as np
from shutil import copyfile


MAX_CLASSES_NUM = 100

CLS_NOT_INCLUDED = ["abbey", "apse", "arch", "atrium", "attic", "basilica", "bridge",
                    "building_facade", "castle", "cathedral", "chapel", "church", "city", "cloister", "corridor", "courthouse", "courtyard", "house", "cottage", "diner", "fortress", "great_hall", "house", "motel", "office_building"]

OUTPUT_PATH = "C:\\Users\\lotan\\Documents\\studies\\phoenix\\datasets\\ballpark_datasets\\tourist_place\\classes_exploration"
ROOT_PATH = "C:\\Users\\lotan\\Documents\\studies\\phoenix\\datasets\\ADE20K_2016_07_26\\images\\training"


def count_images_in_dir(src_path):
    new_counter = 0
    for file in os.listdir(src_path):
        file_path = os.path.join(src_path, file)
        if os.path.isdir(file_path):
            new_counter += count_images_in_dir(os.path.join(src_path, file))
        elif file.endswith("jpg"):
            new_counter += 1
    return new_counter


classes_paths = []
classes = []
for letter in os.listdir(ROOT_PATH):
    if letter in ["misc", "outliers"]:
        continue
    letter_path = os.path.join(ROOT_PATH, letter)
    for cls_name in os.listdir(letter_path):
        classes.append(cls_name)
        if cls_name not in CLS_NOT_INCLUDED:
            cls_path = os.path.join(letter_path, cls_name)
            images_in_num = count_images_in_dir(cls_path)
            if images_in_num < 3:
                continue
            print(cls_name, images_in_num)
            classes_paths.append(cls_path)

with open(os.path.join(os.getcwd(), "classes.txt"), 'w') as file:
    for cls in classes:
        file.writelines(cls +"\n")





relevant_paths = np.random.choice(classes_paths, size=MAX_CLASSES_NUM, replace=False)
print(classes_paths)

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)





def copy_dir(src_path, dest_path):
    for file in os.listdir(src_path):
        file_path = os.path.join(src_path, file)
        if os.path.isdir(file_path):
            copy_dir(os.path.join(src_path, file), dest_path)
        elif file.endswith("jpg"):
            new_path = os.path.join(dest_path, file)
            print( file + " copied")
            copyfile(file_path, new_path)




for path in relevant_paths:
    class_name = os.path.basename(path)
    new_cls_path = os.path.join(OUTPUT_PATH, class_name)
    if not os.path.exists(new_cls_path):
        os.makedirs(new_cls_path)
    copy_dir(path, new_cls_path)










