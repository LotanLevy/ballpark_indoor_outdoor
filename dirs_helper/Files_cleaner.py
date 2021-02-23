import os
from shutil import copyfile

ROOT_PATH = "C:\\Users\\lotan\\Documents\\studies\\Affordances\\datasets\\ADE20K_2016_07_26\\ADE20K_2016_07_26\\images\\validation"
DST_PATH = "C:\\Users\\lotan\\Documents\\studies\\Affordances\\datasets\\ADE20K_cleaned\\validation"




classes_paths = []
classes = []
for letter in os.listdir(ROOT_PATH):
    if letter in ["misc", "outliers"]:
        continue
    letter_path = os.path.join(ROOT_PATH, letter)
    for cls_name in os.listdir(letter_path):
        classes.append(cls_name)
        cls_path = os.path.join(letter_path, cls_name)
        classes_paths.append(cls_path)



def copy_dir(src_path, dest_path, file_name_addition):
    for file in os.listdir(src_path):
        file_path = os.path.join(src_path, file)
        if os.path.isdir(file_path):
            copy_dir(os.path.join(src_path, file), dest_path, file_name_addition + "_" + file)
        elif file.endswith("jpg"):
            new_path = os.path.join(dest_path, file_name_addition+"_"+file)
            print(file + " copied")
            copyfile(file_path, new_path)

for path in classes_paths:
    class_name = os.path.basename(path)
    new_cls_path = os.path.join(DST_PATH, class_name)
    if not os.path.exists(new_cls_path):
        os.makedirs(new_cls_path)
    copy_dir(path, new_cls_path, "")