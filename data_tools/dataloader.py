import tensorflow as tf
import numpy as np
from affordance_tools.Bag import Bag
import os



class Dataloader:
    def __init__(self, root_dir, batch_size, input_size, split_val, shuffle=False,
                                  preprocess_func=lambda x: x, use_aug=False, labels_map_path=None):
        self.root_dir = root_dir
        self.input_size = input_size
        self.labels = "inferred"
        self.paths2labels_dict = None if labels_map_path is None else self._parse_label_map(labels_map_path)

        self.train_iter, self.val_iter = self._get_iterators_by_root_dir(root_dir, batch_size, (input_size, input_size),
                                                                         split_val,
                                                                         shuffle=shuffle,
                                                                         preprocess_func=preprocess_func,
                                                                         use_aug=use_aug)

    def _parse_label_map(self, map_path):
        paths2labels_dict = dict()
        labels = []
        with open(map_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                path2label = line.strip().split("$")
                paths2labels_dict[path2label[0]] = float(path2label[1])
        return paths2labels_dict


    def _get_iterators_by_root_dir(self, root_dir, batch_size, input_size, split_val, shuffle=False,
                                  preprocess_func=lambda x: x, use_aug=False):

        data_gen = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=use_aug, vertical_flip=False,
                                                                   preprocessing_function=preprocess_func,
                                                                   validation_split=split_val)
        train_iter = data_gen.flow_from_directory(root_dir, target_size=input_size,
                                                  batch_size=batch_size, class_mode='categorical', shuffle=shuffle,
                                                  subset="training")

        val_iter = data_gen.flow_from_directory(root_dir, target_size=input_size,
                                                batch_size=batch_size, class_mode='categorical', shuffle=shuffle,
                                                subset="validation")
        return train_iter, val_iter

    def write_filespaths_into_file(self, output_path):
        with open(os.path.join(output_path, "files_paths.txt"), 'w') as f:
            for path in self.train_iter.filepaths:
                f.write(path + "\n")



    def split_into_bags(self, train=True):
        data_iter = self.train_iter if train else self.val_iter
        bags = dict()
        for cls_name, label in data_iter.class_indices.items():
            items_indices = np.where(data_iter.labels == label)[0]
            bag = Bag(cls_name, data_iter, items_indices, self.input_size, self.paths2labels_dict)
            bags[cls_name] = bag
            # bag.display_bag()
        print("Bags names {}".format(bags.keys()))
        return bags

# PATH = "C:\\Users\\lotan\\Documents\\studies\\phoenix\\datasets\\ballpark_outdoors_indoors"
# OUTPUT_PATH = "C:\\Users\\lotan\\Documents\\studies\\affordances\\deep_one_class\\af_ballpark"
# LABELS_PATH = "C:\\Users\\lotan\\Documents\\studies\\affordances\\deep_one_class\\af_ballpark\\files_paths2labels.txt"
# a = Dataloader(PATH, 1, 224, 0, labels_map_path=LABELS_PATH)
# a.split_into_bags()
# # a.write_filespaths_into_file(OUTPUT_PATH)