

import numpy as np
import tensorflow as tf
from PIL import Image
import os
import re
from sklearn.model_selection import train_test_split


class Dataloader:
    def __init__(self, paths, labels, input_size, cls2label, shuffle=False, preprocess_func=lambda x: x):
        assert (len(paths) == len(labels))
        self.labels = np.array(labels)
        self.preprocess_func = preprocess_func
        self.paths = paths
        self.classes_num = len(cls2label)
        self.cls2label = cls2label
        self.input_size = input_size
        self.shuffle = shuffle

        # iterator settings
        self.indices = np.arange(len(self.paths)).astype(np.int)
        self.reset()

    def __len__(self):
        return len(self.paths)

    def next(self, batch_size):
        # takes the batch indices
        relevant_indices = self.indices[self.cur_idx: self.cur_idx + batch_size]
        self.cur_idx += batch_size
        # reads the images and their labels of the batch
        relevant_paths = [self.paths[i] for i in relevant_indices]
        labels = self.labels[relevant_indices]
        labels = tf.keras.utils.to_categorical(labels, num_classes=self.classes_num)

        images = np.concatenate([self.load_img(path) for path in relevant_paths])

        return self.preprocess_func(images), labels, relevant_paths

    def get_data_by_label(self, label, size=None, shuffle=False):
        size = len(self.paths) if size is None else size
        relevant_indices = np.where(self.labels == label)[0][:size]
        if shuffle:
            np.random.shuffle(relevant_indices)
        relevant_paths = [self.paths[i] for i in relevant_indices]
        labels = self.labels[relevant_indices]
        labels = tf.keras.utils.to_categorical(labels, num_classes=self.classes_num)
        images = np.concatenate([self.load_img(path) for path in relevant_paths])
        return self.preprocess_func(images), labels, relevant_paths

    def reset(self):
        if self.shuffle and len(self.indices) > 0:
            np.random.shuffle(self.indices)
        self.cur_idx = 0

    def load_img(self, image_path):
        image = Image.open(image_path, 'r')
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize(self.input_size, Image.NEAREST)
        image = np.array(image).astype(np.float32)
        return np.expand_dims(image, axis=0)


    def has_next(self, batch_size):
        return self.cur_idx + self.batch_size < len(self.indices)


    def write_data(self, output_path, loader_name):
        with open(os.path.join(output_path, loader_name + "_paths.txt"), 'w') as f:
            for p in self.paths:
                f.write(p + "\n")
        with open(os.path.join(output_path, loader_name + "_labels.txt"), 'w') as f:
            for l in self.labels:
                f.write(str(l) + "\n")

def pad_dirs_with_zeros(root_dir):
    dirs = os.listdir(root_dir)
    longest_name = 0

    for dir in dirs:
        path = os.path.join(root_dir, dir)
        if os.path.isdir(path):
            longest_name = max(longest_name, len(dir))

    for dir in dirs:
        if len(dir) < longest_name:
            zeros = "0" * (longest_name - len(dir))
            new_name = zeros + dir
            os.rename(os.path.join(root_dir, dir), os.path.join(root_dir, new_name))
            print("rename dir {}, with {}".format(dir, new_name))

def convert_subdir2cls(subdir_name):
    if re.match("0*$", subdir_name):
        return "0"
    reg = "[0]*((.(?![0])).*)$"
    m = re.match(reg, subdir_name)
    if m:
        return m.group(1)
    else:
        return subdir_name

def map_clsname2label(root_dir):
    cls2label = dict()
    label_idx = 0
    for sub_dir in sorted(os.listdir(root_dir)):
        cls2label[convert_subdir2cls(sub_dir)] = label_idx
        label_idx += 1
    return cls2label


def get_directory_iterators(root_dir, batch_size=2, input_size=(224,224), split_val=0, shuffle=False, preprocess_func=lambda x: x):
    cls2label = map_clsname2label(root_dir)
    print(cls2label)

    paths = []
    labels = []
    for sub_dir in sorted(os.listdir(root_dir)):
        full_path = os.path.join(root_dir, sub_dir)
        if os.path.isdir(full_path):
            for file in os.listdir(full_path):
                paths.append(os.path.join(full_path, file))
                labels.append(cls2label[convert_subdir2cls(sub_dir)])

    if split_val > 0:
        X_train, X_test, y_train, y_test = train_test_split(paths, labels, test_size=split_val, shuffle=shuffle)
    else:
        X_train, X_test, y_train, y_test = paths, [], labels, []

    train_iter = Dataloader(paths=X_train, labels=y_train, input_size=input_size,
                            cls2label=cls2label, shuffle=True, preprocess_func=preprocess_func)
    val_iter = Dataloader(paths=X_test, labels=y_test, input_size=input_size,
                          cls2label=cls2label, shuffle=True, preprocess_func=preprocess_func)
    return train_iter, val_iter

