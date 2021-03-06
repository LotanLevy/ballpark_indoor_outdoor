import os
from shutil import copy
##### train
output_path = "C:\\Users\\lotan\\Documents\\studies\\phoenix\\datasets\\ballpark_datasets\\teaching"

src_root = "C:\\Users\\lotan\\Documents\\studies\\phoenix\\datasets\\ballpark_datasets\\teaching\\train"

negative_classes=['parlor', 'park', 'office', 'kiosk', 'hotel_room','food_court', 'dining_room', 'casino', 'basement', 'barbershop', 'bar']
positive_classes=['courtroom', 'amphitheater', 'lecture_room', 'classroom', 'auditorium']

dest_dir_name = "train_polar"
if not os.path.exists(output_path):
    os.makedirs(output_path)

dest_dir = os.path.join(output_path, dest_dir_name)
negatives_dir = os.path.join(dest_dir, "0")
positives_dir = os.path.join(dest_dir, "1")

with open(os.path.join(output_path, dest_dir_name + "_classes_split.txt"), 'w') as f:
    f.write("negative_classes={}".format(negative_classes))
    f.write("\npositive_classes={}".format(positive_classes))



os.makedirs(os.path.join(output_path, dest_dir_name))
os.makedirs(os.path.join(output_path, negatives_dir))
os.makedirs(os.path.join(output_path, positives_dir))


def copy_dir_items(src_path, dest_path, cls_name):
    for item in os.listdir(src_path):
        item_path = os.path.join(src_path, item)
        if os.path.isdir(item_path):
            for sub_item in os.listdir(item_path):
                sub_item_path = os.path.join(item_path, sub_item)
                dest_item_path = os.path.join(dest_path, cls_name + "_" + item + "_" + sub_item)
                copy(sub_item_path, dest_item_path)
        else:
            dest_item_path = os.path.join(dest_path, cls_name+"_"+item)
            copy(item_path, dest_item_path)


for dir in os.listdir(src_root):
    if dir in positive_classes:
        copy_dir_items(os.path.join(src_root, dir), positives_dir, dir)
    elif dir in negative_classes:
        copy_dir_items(os.path.join(src_root, dir), negatives_dir, dir)
    else:
        print("dir {} doesn't belong to any list".format(dir))

#
# def parse_labels_map(label_map_path):
#     paths2labels_dict = dict()
#     labels = []
#     with open(label_map_path, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             path2label = line.strip().split("$")
#             paths2labels_dict[path2label[0]] = int(path2label[1])
#     return paths2labels_dict
#
# # validation
# output_path = "C:\\Users\\lotan\\Documents\\studies\\phoenix\\datasets"
# path2label_map_file = "C:\\Users\\lotan\\Documents\\studies\\affordances\\deep_one_class\\af_ballpark\\files_paths2labels.txt"
# dest_dir_name = "binary_val_indoor_outdoor"
#
# path2label = parse_labels_map(path2label_map_file)
# i=0
# for path, label in path2label.items():
#     dest_item_path = os.path.join(os.path.join(output_path, dest_dir_name), str(label))
#     if not os.path.exists(dest_item_path):
#         os.makedirs(dest_item_path)
#     copy(path, os.path.join(dest_item_path, str(i)+".jpg"))
#     i += 1
