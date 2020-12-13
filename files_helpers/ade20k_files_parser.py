
import numpy as np
import h5py
import scipy.io as sio
import os
from shutil import copyfile
from PIL import Image

PATH = "C:\\Users\\lotan\\Downloads\\ADE20K_2016_07_26\\ADE20K_2016_07_26\\index_ade20k.mat"


# test = sio.loadmat(PATH)
# print(test['index'])




def build_sub_classes_from_root(root_path, dest, sub_cls2label):
    if not os.path.exists(dest):
        os.makedirs(dest)
    for cls in os.listdir(root_path):
        cls_path = os.path.join(root_path, cls)
        for sub_cls in os.listdir(cls_path):
            if sub_cls not in sub_cls2label:
                print("problem in " + cls)
            sub_cls_path = os.path.join(cls_path, sub_cls)
            sub_cls_dest = os.path.join(dest, sub_cls2label[sub_cls])
            if not os.path.exists(sub_cls_dest):
                os.makedirs(sub_cls_dest)
            for im_name in os.listdir(sub_cls_path):
                im_path = os.path.join(sub_cls_path, im_name)
                im_dest = os.path.join(sub_cls_dest, im_name)
                copyfile(im_path, im_dest)
                # image = Image.open(im_path, 'r')
                # image.save(os.path.join(sub_cls_dest, im_name))

##### train

root = "C:\\Users\\lotan\\Documents\\studies\\phoenix\\datasets\\ballpark_datasets\\dine\\noisy_train_10"
dest = "C:\\Users\\lotan\\Documents\\studies\\phoenix\\datasets\\ballpark_datasets\\dine\\noisy_train_10_binary"
sub_cls2label = {"cant": "0", "can": "1"}
build_sub_classes_from_root(root, dest, sub_cls2label)



