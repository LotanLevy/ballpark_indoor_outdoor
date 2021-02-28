

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import os



def get_preprocessing_func_by_name(model_name):
    if model_name == "vgg16":
        return lambda input_data: tf.keras.applications.vgg16.preprocess_input(np.copy(input_data.astype('float32')))
    elif model_name == "resnet50":
        return lambda input_data: tf.keras.applications.resnet50.preprocess_input(np.copy(input_data.astype('float32')))
    else:
        print("Unrecognized nn name")
        return

def get_features_model(model_name, input_size, features_level=-2):
    if model_name == "vgg16":
        model = tf.keras.applications.VGG16(include_top=True, input_shape=(input_size, input_size, 3),
                                                 weights='imagenet')
    elif model_name == "resnet50":
        model = tf.keras.applications.ResNet50(include_top=True, input_shape=(input_size, input_size, 3),
                                                 weights='imagenet')
    else:
        print("Unrecognized nn name")
        return None
    return Model(inputs=model.input, outputs=model.layers[features_level].output)



def get_prediction_func(weights_ckpt_path, bias_ckpt_path):
    w = np.load(weights_ckpt_path)
    if bias_ckpt_path != '':
        b = np.load(bias_ckpt_path)
    else:
        b = w[0]
        w = w[1:]

    return lambda X: get_pred(w, b, X)


def get_pred(w, b, X):
    if X.shape[1] - w.shape[0] == 1:
        new_w = np.concatenate((np.array([b]),w))
        return np.dot(new_w, X.T)
    return np.dot(w, X.T) + b



def parse_weights_paths(models_args):
    ballpark_paths = [str(item) for item in models_args.split(',')]
    weights = dict()

    for ckpt_path in ballpark_paths:
        args = ckpt_path.split("$")
        name, path = args[0], args[1]
        if "ballpark" in name:
            weights_path = os.path.join(path, "ballpark_weights.npy")
            bias_path = ""
            pred_func = get_prediction_func(weights_path, bias_path)
            weights[name] = pred_func
        elif "svm" in name:
            for subdir in os.listdir(path):
                weights_path = os.path.join(os.path.join(path, subdir), "svm_weights.npy")
                bias_path = os.path.join(os.path.join(path, subdir), "svm_bias.npy")
                pred_func = get_prediction_func(weights_path, bias_path)
                weights[name + "_{}".format(subdir)] = pred_func
        else:
            print("name {} is not legal - pass".format(name))
            continue
    return weights

# def get_classification_by_values(positive_val, negative_val, preds, threshold):
#     results = np.zeros(preds.shape)
#     results[np.where(preds < threshold)[0]] = negative_val
#     results[np.where(preds >= threshold)[0]] = positive_val
#     return results.astype(np.int)


def build_data_by_classes(bags, classes):
    features, paths = None, []
    for cls in classes:
        bag_features, bag_paths = bags[cls].get_features()
        if features is None:
            features = bag_features
        else:
            features = np.concatenate((features, bag_features))
        paths += bag_paths
    return features, paths





def prepare_svm_data(data_bags, negative_classes, positive_classes, positive_label=1, negative_label=0):
    positive_features, positive_paths = build_data_by_classes(data_bags, positive_classes)
    negative_features, negative_paths = build_data_by_classes(data_bags, negative_classes)

    X = np.concatenate((positive_features, negative_features))
    y_positive = positive_label * np.ones(positive_features.shape[0])
    y_negative = negative_label * np.ones(negative_features.shape[0])
    y = np.concatenate((y_positive, y_negative)).astype(np.int)
    paths = positive_paths + negative_paths
    return X, y, paths




