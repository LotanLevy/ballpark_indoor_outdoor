

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import metrics
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import math
from skimage.transform import resize
import itertools
import random





def get_data_size(bags_dict):
    length = 0
    bag2indices_range = dict()
    for bag_name, bag in bags_dict.items():
        bag2indices_range[bag_name] = range(length, length + len(bag))
        length += len(bag)
    return length, bag2indices_range


def make_predictions_for_bag(bag, weights, bias):
    bag_features, paths = bag.get_features()
    pred = bag_features.dot(weights)
    if bias is not None:
        pred += bias
    return pred, paths

def display_roc_graph(output_path, title, preds, labels):
    fpr, tpr, _ = metrics.roc_curve(labels, preds)

    # AUC
    auc = metrics.auc(fpr, tpr)

    # Plot the ROC curve
    fig = plt.figure()
    plt.plot(fpr, tpr, label='DeepOneClassification(AUC = %.2f)' % auc)
    plt.legend()
    plt.title(title + 'ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.savefig(os.path.join(output_path, title + "roc_graph.png"))
    plt.show()
    plt.close(fig)
    plt.clf()
    plt.cla()

def get_classifications_by_roc(scores, labels):
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return (scores >= optimal_threshold).astype(int)

def display_random_images(output_path, paths, size, title):
    random.shuffle(paths)
    fig = plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.tight_layout()
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    columns = 4
    rows = math.ceil(size / columns)
    for i in range(1, min(columns * rows + 1, len(paths) + 1)):
        path = paths[i - 1]
        image = Image.open(path)
        fig.add_subplot(rows, columns, i)
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.imshow(image)
    # plt.show()
    plt.savefig(os.path.join(output_path, title.replace(" ", "_")))
    plt.close(fig)


def display_paths_according_to_the_confusion_matrix(output_path, paths, pred_labels, labels):
    tn_paths, fp_paths, fn_paths, tp_paths = [], [], [], []
    for pred, label, path in zip(pred_labels, labels, paths):
        if pred == 0 and label == 0:
            tn_paths.append(path)
        elif pred == 1 and label == 0:
            fp_paths.append(path)
        elif pred == 0 and label == 1:
            fn_paths.append(path)
        else:
            tp_paths.append(path)

    display_random_images(output_path, tn_paths, 20, "true negatives- total_size {}".format(len(tn_paths)))
    display_random_images(output_path, fp_paths, 20, "false positives- total_size {}".format(len(fp_paths)))
    display_random_images(output_path, fn_paths, 20, "false negatives- total_size {}".format(len(fn_paths)))
    display_random_images(output_path, tp_paths, 20, "true positives- total_size {}".format(len(tp_paths)))






def display_predictions_for_bag(output_path, bag, preds, paths):
    fig = plt.figure(figsize=(8, 8))
    plt.title(bag.cls_name)
    plt.tight_layout()
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    columns = 4
    rows = 5
    for i in range(1, min(columns * rows + 1, len(paths)+1)):
        path = paths[i-1]
        image = Image.open(path)
        # image = bag.data_iterator[im_idx][0][0].astype(np.int)
        # path = bag.data_iterator.filepaths[im_idx]
        label = bag.path2label_dict[path]
        pred = preds[i-1]
        fig.add_subplot(rows, columns, i)
        plt.title("true {},\n pred {}".format(label, "{:10.2f}".format(pred)))
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.imshow(image)
    # plt.show()
    plt.savefig(os.path.join(output_path, "results_for_cls_{}".format(bag.cls_name)))
    plt.close(fig)
    plt.clf()
    plt.cla()

def getImage(path, zoom):
    image = plt.imread(path)
    image = resize(image, (224, 224,3))
    return OffsetImage(image, zoom=zoom)

def create_images_graph(output_path, paths, scores, zoom=0.08, columns=20, max_objects=None):
    # scores_graph_path = os.path.join(output_path, "score_graphs")
    # if not os.path.exists(scores_graph_path):
    #     os.makedirs(scores_graph_path)
    if max_objects is None:
        max_objects = len(scores)
    # paths2cls = map_path_to_class(paths)
    indices = np.argsort(scores)[:max_objects]
    scores = scores[indices]

    step = 10

    x = list(range(columns)) * math.ceil(len(indices) / float(columns))

    x = [step * i for i in x]
    x = x[:len(scores)]
    fig, ax = plt.subplots()
    # ax.scatter(x, scores[indices])
    for i in range(max_objects):
        idx = indices[i]
        ab = AnnotationBbox(getImage(paths[idx], zoom), (x[i], scores[i]), frameon=False)
        ax.scatter(x[i], scores[i])
        ax.add_artist(ab)
    ax.update_datalim(np.column_stack([x, scores]))
    ax.autoscale(-1 * max(scores), max(scores) * 1.1)
    ax.set_xlim(-1, max(x) * 1.1)
    plt.ylabel("classifier score")
    plt.xlabel("axis without meaning")
    # plt.title(name)
    plt.savefig(os.path.join(output_path, "scores_visualization_indoor_outdoor.png"), dpi=500)
    plt.show()
