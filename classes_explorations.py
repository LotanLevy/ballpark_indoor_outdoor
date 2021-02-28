

from data_tools.dataloader import Dataloader
import matplotlib.font_manager as fm

import argparse
import os
from PIL import Image, ImageFont, ImageDraw
from evaluation_tools.evaluate import make_predictions_for_bag
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg16
import matplotlib.gridspec as gridspec
import scipy.stats
from shutil import copyfile
from files_helpers.best_classes_visualization import zipped_images, plot_image_without_axes
from args_helper import get_features_model, get_preprocessing_func_by_name



class ModelPreds:
    def __init__(self, name, type, weights_path):
        self.pref_func = get_prediction_func(type, weights_path)
        self.name = name
        self.preds = dict()
        self.paths = dict()
        self.means = dict()

    def get_save_max_for_relevant_classes(self, output_path, low_percent=40, high_percent=100):
        dir_path = os.path.join(output_path, "{}_from_{}_to_{}".format(self.name, low_percent, high_percent))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        sorted_keys = [k for k, v in sorted(self.means.items(), key=lambda item: item[1])]
        small_idx = np.floor(len(sorted_keys) * (low_percent/100)).astype(np.int)
        max_idx = np.ceil(len(sorted_keys) * (high_percent/100)).astype(np.int)
        print("saving from {} to {}".format(small_idx, max_idx))
        cls2path = dict()
        cls2name = dict()
        for i, cls in zip(range(small_idx, max_idx+1), sorted_keys[small_idx: max_idx+1]):
            max_idx = np.argmax(self.preds[cls])
            cls2path[cls] = self.paths[cls][max_idx]
            name = "{}_{}_{}".format(i, cls, "%.2f" % self.means[cls])
            copyfile(cls2path[cls], os.path.join(dir_path, name.replace(".", "_")+".jpg"))

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
        # rand_idx = np.random.choice(len(self.preds[bag_name]), size=min(len(self.preds[bag_name]), size), replace=False)
        rand_idx = np.arange(min(len(self.preds[bag_name]), size))

        scores = []
        paths = []

        for i in np.arange(len(self.preds[bag_name])):
            scores.append(self.preds[bag_name][i])
            paths.append(self.paths[bag_name][i])
        sorted_idx = np.argsort(np.array(scores))
        small_sorted = np.argsort(np.array(scores))[-1*(min(len(self.preds[bag_name]), size)):]
        sorted_paths = [paths[i] for i in small_sorted]
        # paths = []
        # for i in sorted_idx:
        #     paths.append(self.paths[bag_name][rand_idx[i]])

        row = create_row_images(sorted_paths, image_size)
        titles = ["%.2f" % scores[i] for i in small_sorted]
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

    def plot_best_bags(self, max_num, images_size, output_path, repr_image_pos=50, type="mean"):
        sort_by_dict = self.means
        if type == "max":
            sort_by_dict = {cls: np.max(self.preds[cls]) for cls in self.preds}
        sorted_keys = [k for k, v in sorted(sort_by_dict.items(), key=lambda item: item[1])]
        if max_num > len(sorted_keys):
            max_num = 0
        relevant_classes = sorted_keys[-1*max_num:]
        relevant_means = [sort_by_dict[k] for k in relevant_classes]
        repr_images_paths = []
        titles = []
        for cls in relevant_classes:
            sorted_indices = np.argsort(self.preds[cls])
            repr_idx = sorted_indices[max(0, min(len(sorted_indices)-1, np.ceil((repr_image_pos/100)*len(sorted_indices)).astype(np.int) - 1))]
            repr_images_paths.append(self.paths[cls][repr_idx])
            titles.append("%.2f" % sort_by_dict[cls] + ", " + cls)
        result = zipped_images(repr_images_paths, titles, images_size, vertical=False, title_pos="bottom")
        plot_image_without_axes(result, "best_classes_{}_im_pos_{}_type_{}".format(self.name, repr_image_pos, type), output_path)


    def plot_best_scored_images(self, output_path, image_size, col_num=10, row_num=2, low_percent=45, high_percent=75, relevant_classes=None):
        dir_path = os.path.join(output_path, "sorted_images_{}_from_{}_to_{}".format(self.name, low_percent, high_percent))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if relevant_classes is None:
            sorted_keys = [k for k, v in sorted(self.means.items(), key=lambda item: item[1])]
            # small_idx = np.floor(len(sorted_keys) * (low_percent / 100)).astype(np.int)
            small_idx = 0
            max_idx = np.ceil(len(sorted_keys) * (high_percent / 100)).astype(np.int)
            print("saving images from classes {}".format(sorted_keys))
            relevant_classes = sorted_keys[small_idx: max_idx+1]
            positions = range(small_idx, max_idx+1)
        else:
            positions = range(len(relevant_classes))


        all_scores = np.array([])
        classes = []

        all_paths = []

        for i, cls in zip(positions, relevant_classes):
            all_scores = np.concatenate((all_scores, self.preds[cls]))
            all_paths += self.paths[cls]
            classes += [cls] * len(self.paths[cls])


        # create_image from high_percent to low_percent
        relevant_indices = np.argsort(all_scores)[-1 * col_num:]
        # relevant_scores = [all_scores[i] for i in relevant_indices]
        relevant_paths = [all_paths[i] for i in relevant_indices]
        relevant_classes = [classes[i] for i in relevant_indices]
        # relevant_classes = [classes[i] for i in relevant_indices]
        row = create_row_images(relevant_paths, image_size, titles=relevant_classes)
        titles = ["{}_{}".format(classes[i], "%.2f" % all_scores[i]) for i in relevant_indices]
        plot_row_and_scores(titles,
                            row, image_size, "{}_best_scored_images_from_{}_to_{}".format(self.name, low_percent, high_percent), output_path)
        #create image of the first col_num * row_num best images selected by the method
        relevant_indices = np.argsort(all_scores)
        cur_idx = len(relevant_indices)
        rows = []
        titles = []

        mid_idx = int(len(relevant_indices)/2)
        low_idx = col_num + 1
        titles = ["Best", "Median"]
        for j in [cur_idx, mid_idx]:
            row_indices = relevant_indices[(j - col_num):j]
            row_paths = [all_paths[i] for i in row_indices]
            row_classes = [classes[i] for i in row_indices]
            rows.append(create_row_images(row_paths, image_size, titles=row_classes))


        image = stack_with_spaces(rows, vertical=True)

        plt.gca().set_axis_off()
        plt.figure()
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off

        plt.imshow(image)
        ticks_vals = np.arange(0, (len(titles)) * image_size, step=image_size)
        plt.yticks(ticks_vals, titles, fontsize=8, rotation=90)
        plt.ylabel("Position in the ranking")
        plt.tight_layout()
        plt.margins(0, 0)
        plt.rc('font', size=8)
        plt.savefig(os.path.join(output_path, "{}_{}_best_scored_images.jpg".format(self.name, col_num * row_num)), dpi=300, bbox_inches='tight')
        plt.close("all")





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
    plt.close("all")


def get_args_parser():
    parser = argparse.ArgumentParser(description='Process constraint collector args.')
    parser.add_argument('--data_path',  type=str, required=True)
    parser.add_argument('--excluded_bags',  type=str, required=True)

    parser.add_argument('--ssvm_ckpt_path',  type=str, default=None)
    parser.add_argument('--psvm_ckpt_path',  type=str, default=None)
    parser.add_argument('--ballpark_ckpt_path',  type=str, default=None)

    parser.add_argument('--max_class_rank',  type=int, default=75)
    parser.add_argument('--input_size',  type=int, default=224)
    parser.add_argument('--image_size',  type=int, default=448)

    parser.add_argument('--max_files',  type=int, default=10)
    parser.add_argument('--best_num',  type=int, default=10)
    parser.add_argument('--best_type',  type=str, default="mean", choices=["mean", "max"])
    parser.add_argument('--nn_model', type=str, default="vgg16", choices=["vgg16, resnet"])
    parser.add_argument('--features_level',  type=int, default=-2)

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

def create_row_images(paths, images_size, titles = None):
    images = []
    for i, path in enumerate(paths):
        image = np.array(Image.open(path, 'r').convert('RGB').resize((images_size, images_size)))
        if titles is not None:

            text = Image.new('RGB', (image.shape[1], np.floor(image.shape[0]/5).astype(np.int)), color=(255, 255, 255))
            d = ImageDraw.Draw(text)
            font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')), size=60)
            d.text((0, 0), titles[i], fill=(0, 0, 0), font=font)
            image = np.vstack([image, text])
        # image = np.array(image.resize((images_size, images_size)))
        images.append(image/ 255)
        # if row is None:
        #     row = image/ 255
        # else:
        #     row = np.hstack((row, image / 255))
    return stack_with_spaces(images, vertical=False)




def stack_with_spaces(images, vertical=False, space=10):
    get_space = lambda : np.ones((space, images[0].shape[1],3)) if vertical else np.ones((images[0].shape[0], space, 3))
    to_stack = []
    text = ["ttt"] * len(images)
    for image in images:
        if len(to_stack) != 0:
            to_stack.append(get_space())
        to_stack.append(image)
    stack_func = np.vstack if vertical else np.hstack
    return stack_func(to_stack)





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
    plt.close("all")

def get_bag_features_with_bias(bag_features):
    bias_row = np.ones((bag_features.shape[0], 1))
    bag_features = np.concatenate((bias_row, bag_features), axis=1)
    return bag_features

def get_prediction_func(model_type, ckpt_path):
    if model_type == "svm":
        w = np.load(os.path.join(ckpt_path, "svm_weights.npy"))
        b = np.load(os.path.join(ckpt_path, "svm_bias.npy"))
        return lambda features: np.dot(w, features.T) + b
    else:
        w = np.load(os.path.join(ckpt_path, "ballpark_weights.npy"))
        return lambda features: features.dot(w) if features.shape[0] == w.shape[0] else get_bag_features_with_bias(features).dot(w)



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

    # preprocessing_func = lambda input_data: vgg16.preprocess_input(np.copy(input_data.astype('float32')))

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

    with open(args.excluded_bags, 'r') as file:
        content = ""
        for line in file.readlines():
            content += line
        excluded_bags = content.split(",")
        excluded_bags_clean = [bag.strip().replace("'", "") for bag in excluded_bags]
        print(excluded_bags_clean)
    nn_model = get_features_model(args.nn_model, args.input_size, features_level=args.features_level)
    preprocessing_func = get_preprocessing_func_by_name(args.nn_model)
    val_dataloader = Dataloader(nn_model, args.data_path, 1, args.input_size, 0, preprocess_func=preprocessing_func, )
    val_bags = val_dataloader.split_into_bags(train=True)
    for bag_name, bag in val_bags.items():
        if bag_name in excluded_bags_clean:
            print("{} ignored".format(bag_name))
            continue
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
        # model.models_results(args.output_path, 10)
        model.plot_best_bags(args.best_num, args.image_size, args.output_path, repr_image_pos=50, type=args.best_type)
        model.plot_best_bags(args.best_num, args.image_size, args.output_path, repr_image_pos=100, type=args.best_type)

        model.get_save_max_for_relevant_classes(args.output_path, low_percent=40, high_percent=100)
        # model.plot_best_scored_images(args.output_path,args.image_size)

    relevant_classes_dict = dict()
    for model in models:
        for bag in model.means:
            if bag not in relevant_classes_dict:
                relevant_classes_dict[bag] = model.means[bag]
            else:
                relevant_classes_dict[bag] += model.means[bag]

    sorted_keys = [k for k, v in sorted(relevant_classes_dict.items(), key=lambda item: item[1])]
    low_percent = 45
    high_percent = 75
    small_idx = np.floor(len(sorted_keys) * (low_percent / 100)).astype(np.int)
    max_idx = np.ceil(len(sorted_keys) * (high_percent / 100)).astype(np.int)

    for model in models:
        model.plot_best_scored_images(args.output_path,args.image_size, high_percent=args.max_class_rank, relevant_classes=None)
        model.write_bags_rows(args.image_size, 10, args.output_path)








main()