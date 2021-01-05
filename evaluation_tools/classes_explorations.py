

from data_tools.dataloader import Dataloader
import argparse
import os
from PIL import Image
from evaluation_tools.evaluate import make_predictions_for_bag
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg16
import matplotlib.gridspec as gridspec
import scipy.stats
from shutil import copyfile


class ModelPreds:
    def __init__(self, name, type, weights_path):
        self.pref_func = get_prediction_func(type, weights_path)
        self.name = name
        self.preds = dict()
        self.paths = dict()
        self.means = dict()



    def set_features(self, bag_name, features, paths):
        self.preds[bag_name] = self.pref_func(features)
        self.paths[bag_name] = paths
        self.means[bag_name] = np.average(self.preds[bag_name])

    def get_row_bag(self, bag_name, image_size, output_path):
        return create_orderd_images(self.paths, self.preds, image_size, self.name + bag_name.replace(".", "_"), output_path)
    def get_row_with_repre_for_bag(self, idx_score, image_size, relevant_bags):
        paths = []
        scores = []
        for bag in relevant_bags:
            relevant_idx = np.argsort(self.preds[bag])[-1 * idx_score]
            scores.append(self.preds[bag][relevant_idx])
            paths.append(self.paths[bag][relevant_idx])
        return create_row_images(paths, image_size), scores

    def write_bags_rows(self, image_size, size, output_path):
        model_dir = os.path.join(output_path, "model_{}_begs".format(self.name))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        for bag_name in self.preds:
            if len(self.preds[bag_name]) <= 1:
                continue
            self.write_bag(bag_name, image_size, size, model_dir)

    def write_bag(self, bag_name, image_size, size, output_path):
        rand_idx = np.random.choice(len(self.preds[bag_name]), size=min(len(self.preds[bag_name]), size), replace=False)
        scores = []
        for i in rand_idx:
            scores.append(self.preds[bag_name][i])
        sorted_idx = np.argsort(np.array(scores))
        paths = []
        for i in sorted_idx:
            paths.append(self.paths[bag_name][i])

        row = create_row_images(paths, image_size)
        titles = ["%.2f" % scores[i] for i in sorted_idx]
        plot_row_and_scores(titles,
                            row, image_size, "{}_{}".format(self.name, bag_name.replace(".", "")), output_path)

    def models_results(self, output_path, max_files):
        model_dir = os.path.join(output_path, "model_{}_scores".format(self.name))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        for bag in self.preds:
            bag_dir = os.path.join(model_dir, bag)
            if not os.path.exists(bag_dir):
                os.makedirs(bag_dir)
            relevant_indices = np.argsort(self.preds[bag])
            if len(relevant_indices) > max_files:
                relevant_indices = relevant_indices[-1*max_files:]
            for i in relevant_indices:
                abs_name = os.path.basename(self.paths[bag][i])
                score_string = "%.2f" % self.preds[bag][i]
                copyfile(self.paths[bag][i], os.path.join(bag_dir, score_string.replace(".", "_") + "_"+abs_name))







def calculate_ci(data, confidence=0.95):
    se = scipy.stats.sem(data)
    return se * scipy.stats.t.ppf((1 + confidence) / 2., len(data) - 1)


def plot_scores_for_model(models, relevant_classes, output_path):
    barWidth = 0.4

    plt.figure()
    ticks_pos = np.arange(len(relevant_classes))
    colors = ["blue", "red", "green"]

    dist_from_prev_bar = 0

    for i, model in enumerate(models):
        relevant_avg = [model.means[bag] for bag in relevant_classes]
        ci = [model.means[bag] - calculate_ci(model.preds[bag]) for bag in relevant_classes]
        ri = [x + dist_from_prev_bar for x in ticks_pos]
        dist_from_prev_bar += barWidth
        plt.bar(ri, relevant_avg, width=barWidth, color=colors[i], edgecolor='black', yerr=ci, label=model.name)

    plt.xticks([r + dist_from_prev_bar/2 for r in ticks_pos], relevant_classes, fontsize=8, rotation=30)
    plt.ylabel('average')
    plt.xlabel('classes')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_path, "models_stat"), bbox_inches='tight')
    plt.show()




def get_args_parser():
    parser = argparse.ArgumentParser(description='Process constraint collector args.')
    parser.add_argument('--data_path',  type=str, required=True)
    parser.add_argument('--ssvm_ckpt_path',  type=str, default=None)
    parser.add_argument('--psvm_ckpt_path',  type=str, default=None)
    parser.add_argument('--ballpark_ckpt_path',  type=str, default=None)

    parser.add_argument('--model_type',  choices=['ballpark', 'svm'], default='ballpark')
    parser.add_argument('--input_size',  type=int, default=224)
    parser.add_argument('--image_size',  type=int, default=448)

    parser.add_argument('--max_files',  type=int, default=10)



    parser.add_argument('--output_path', type=str, default=os.getcwd())

    return parser

def create_orderd_images(paths, scores, images_size, name, output_path):
    ordered_indices = np.argsort(scores)
    titles = []
    row = None
    for i in ordered_indices:
        image = np.array(Image.open(paths[i], 'r').convert('RGB').resize((images_size, images_size)))
        if row is None:
            row = image/ 255
        else:
            row = np.hstack((row, image / 255))
        titles.append("%.2f" % scores[i])

    return titles, row

def create_row_images(paths, images_size):
    row = None
    for path in paths:
        image = np.array(Image.open(path, 'r').convert('RGB').resize((images_size, images_size)))
        if row is None:
            row = image/ 255
        else:
            row = np.hstack((row, image / 255))
    return row




def plot_row_and_scores(titles, row, images_size, name, output_path):
    plt.gca().set_axis_off()
    plt.figure(figsize = (6,3))
    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        labelleft=False)  # labels along the bottom edge are off

    plt.imshow(row)
    ticks_vals = np.arange(images_size // 2, (len(titles)) * images_size, step=images_size)
    plt.xticks(ticks_vals, titles, fontsize=8, rotation=30)
    plt.tight_layout()
    plt.margins(0, 0)
    plt.rc('font', size=8)
    plt.savefig(os.path.join(output_path, name), dpi=300, bbox_inches='tight')
    plt.show()



def get_prediction_func(model_type, ckpt_path):
    if model_type == "svm":
        w = np.load(os.path.join(ckpt_path, "svm_weights.npy"))
        b = np.load(os.path.join(ckpt_path, "svm_bias.npy"))
        return lambda features: np.dot(w, features.T) + b
    else:
        w = np.load(os.path.join(ckpt_path, "ballpark_weights.npy"))
        return lambda features: features.dot(w)

def main():
    args = get_args_parser().parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    models = []

    if args.ssvm_ckpt_path is not None:
        models.append(ModelPreds("supervised_svm", "svm", args.ssvm_ckpt_path))
    if args.psvm_ckpt_path is not None:
        models.append(ModelPreds("polar_svm", "svm", args.psvm_ckpt_path))
    if args.ballpark_ckpt_path is not None:
        models.append(ModelPreds("ballpark", "ballpark", args.ballpark_ckpt_path))

    preprocessing_func = lambda input_data: vgg16.preprocess_input(np.copy(input_data.astype('float32')))

    counter = 0
    # for bag_name in os.listdir(args.data_path):
    #     # if bag_name == "library":continue
    #     # if counter > 20:
    #     #     break
    #     # counter += 1
    #     bag_path = os.path.join(args.data_path, bag_name)
    #
    #     val_dataloader = Dataloader(bag_path, 1, args.input_size, 0, preprocess_func=preprocessing_func,)
    #     val_bags = val_dataloader.split_into_bags(train=True)
    #
    #     bag_features, bag_paths = None, []
    #
    #     for label_name, label_bag in val_bags.items():
    #         sub_bag_features, sub_bag_paths = label_bag.get_features()
    #         if sub_bag_features is not None and sub_bag_features.shape[0] != 0:
    #             if bag_features is None:
    #                 bag_features = np.array(sub_bag_features)
    #             else:
    #                 bag_features = np.concatenate((bag_features, np.array(sub_bag_features)))
    #             bag_paths += sub_bag_paths
    #         else:
    #             print("{},{} empty".format(bag_name, label_name))
    #             continue
    #     if bag_features is None:
    #         print(bag_name)
    #         continue
    #     for model in models:
    #         model.set_features(bag_name, bag_features, bag_paths)

    val_dataloader = Dataloader(args.data_path, 1, args.input_size, 0, preprocess_func=preprocessing_func, )
    val_bags = val_dataloader.split_into_bags(train=True)
    for bag_name, bag in val_bags.items():
        bag_features, bag_paths = None, []
        sub_bag_features, sub_bag_paths = bag.get_features()
        if sub_bag_features is not None and sub_bag_features.shape[0] != 0:
            if bag_features is None:
                bag_features = np.array(sub_bag_features)
            else:
                bag_features = np.concatenate((bag_features, np.array(sub_bag_features)))
            bag_paths += sub_bag_paths
        else:
            print("{} empty".format(bag_name))
            continue

        for model in models:
            model.set_features(bag_name, bag_features, bag_paths)
    for model in models:
        model.models_results(args.output_path, 10)

    with open( os.path.join(args.output_path,"ranked_classes.txt"), 'w') as f:
        for model in models:
            f.write("model {}: classes sorted by mean\n".format(model.name))
            f.write(str([(k,"%.2f" % v) for k, v in sorted(model.means.items(), key=lambda item: item[1])]))
            f.write("\n______________________")


    relevant_classes_dict = dict()

    for model in models:
        for bag in model.means:
            if bag not in relevant_classes_dict:
                relevant_classes_dict[bag] = model.means[bag]
            else:
                relevant_classes_dict[bag] += model.means[bag]

    # best_keys = [k for k, v in sorted(relevant_classes_dict.items(), key=lambda item: item[1])][-10:]
    best_keys = [k for k, v in sorted(relevant_classes_dict.items(), key=lambda item: item[1])][-10:]

    plot_scores_for_model(models, best_keys, args.output_path)

    for model in models:
        titles = []
        for bag in best_keys:
            titles.append("{}".format(bag))
        row, scores = model.get_row_with_repre_for_bag(1, args.image_size, best_keys)
        plot_row_and_scores(titles,
                            row, args.image_size, model.name, args.output_path)
        model.write_bags_rows(args.image_size,10, args.output_path)







main()