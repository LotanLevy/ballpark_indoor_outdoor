
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from data_tools.dataloader import Dataloader
from tensorflow.keras.applications import vgg16
from PIL import Image
from files_helpers.best_classes_visualization import zipped_images, plot_image_without_axes, stack_with_spaces
from cv2 import cv2
from features_visualization_helper import gradCAM_by_feature
from args_helper import get_features_model, get_preprocessing_func_by_name







def get_args_parser():
    parser = argparse.ArgumentParser(description='Process constraint collector args.')

    parser.add_argument('--svm_ckpt_path',  type=str, default=None)
    parser.add_argument('--ballpark_ckpt_path', type=str)

    parser.add_argument('--input_size',  type=int, default=224)
    parser.add_argument('--training_data_path',  type=str, default=None)
    parser.add_argument('--features_level',  type=int, default=-2)
    parser.add_argument('--image_size',  type=int, default=448)


    parser.add_argument('--output_path', type=str, default=None)

    return parser


def get_prediction_func(model_type, ckpt_path):
    if model_type == "svm":
        w = np.load(os.path.join(ckpt_path, "svm_weights.npy"))
        b = np.load(os.path.join(ckpt_path, "svm_bias.npy"))
        return lambda features: np.dot(w, features.T) + b
    else:
        w = np.load(os.path.join(ckpt_path, "ballpark_weights.npy"))
        b = 0
        if w.shape[0] > 4096:
            b = w[0]
            w = w[1:]
        return w, b

def plot_weigts(weights):
    plt.figure()
    x = None
    width = 0.3
    cur = None
    for name, w in weights.items():
        if x is None:
            x = np.arange(len(w))
            cur = x - width

        plt.bar(cur, height=w, label=name)
        cur += width
    plt.legend()
    plt.show()


def display_max_features(dataloader, w, features_num, images_size, output_path):


    features_index = np.argsort(w)[-1*features_num:]
    all_features, paths = dataloader.get_all_features()
    features2paths = dict()
    features2pos = dict()

    for pos, i in enumerate(features_index):
        place_in_order = len(features_index)-pos
        feature_values = all_features[:, i]
        max_images_indices = np.argsort(feature_values, axis=None)[-25:]
        repre_paths = [paths[j] for j in max_images_indices] # from the smallest scored to the highest
        result = zipped_images(repre_paths, [None]*len(repre_paths), images_size, vertical=False, title_pos="bottom", max_in_line=5)
        plot_image_without_axes(result, "{}_place_in_order_feature_{}".format(place_in_order, i), output_path)
        features2paths[i] = repre_paths
        features2pos[i] = place_in_order # the position of the score relating to all the paths
    return features2paths, features2pos


def explore_images_patches_by_feature_index(paths, model, preprocess_func, feature_index, cols_num=5):

    rows_num = np.floor(len(paths)/cols_num).astype(np.int)
    rows = []

    for r in range(rows_num):
        relevant_paths = paths[r*cols_num:r*cols_num+cols_num]
        row_items = []
        for path in relevant_paths:
            heatmap_img = gradCAM_by_feature(path, preprocess_func, model, feature_index, conv_layer="block5_conv3")
            row_items.append(cv2.resize(heatmap_img, (448,448)))
        rows.append(stack_with_spaces(row_items, vertical=False))
    result = stack_with_spaces(rows, vertical=True)

    return result

def main():
    args = get_args_parser().parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    ballpark_paths = [str(item) for item in args.ballpark_ckpt_path.split(',')]

    weights = dict()


    for ckpt_path in ballpark_paths:
        name, path = ckpt_path.split("$")
        w, b = get_prediction_func("ballpark", path)
        weights[name] = w
    plot_weigts(weights)


    if args.training_data_path is not None:
        for name in weights:
            features_path = os.path.join(os.path.join(args.output_path, "features_repr"), name)
            if not os.path.exists(features_path):
                os.makedirs(features_path)
            print("Initialize dataloader")
            w=weights[name]
            # preprocessing_func = lambda input_data: vgg16.preprocess_input(np.copy(input_data.astype('float32')))
            preprocessing_func = get_preprocessing_func_by_name(args.nn_model)

            nn_model = get_features_model(args.nn_model, args.input_size, features_level=args.features_level)

            train_dataloader = Dataloader(nn_model, args.training_data_path, 1, args.input_size, 0, features_level=args.features_level,
                                          preprocess_func=preprocessing_func)
            features2paths, features2pos = display_max_features(train_dataloader, w, features_num=5, images_size=args.image_size, output_path=features_path)

            model = train_dataloader.get_features_model(args.input_size, features_level=args.features_level)
            print(model.summary())

            for i in features2paths:
                max_image = np.array(Image.open(features2paths[i][-1], 'r').convert('RGB'))

                heatmaps_for_features = explore_images_patches_by_feature_index(features2paths[i], model, train_dataloader.preprocess_func, i)

                plot_image_without_axes(heatmaps_for_features, "{}_place_in_order_feature_{}".format(features2pos[i], i), features_path)




if __name__=="__main__":
    main()