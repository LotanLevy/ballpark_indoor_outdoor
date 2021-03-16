
import argparse
import os
import args_helper
import numpy as np
import json
from data_tools.dataloader import Dataloader
import visualizations.visualization_helper as vh
from shutil import copyfile



def get_args_parser():
    parser = argparse.ArgumentParser(description='Process constraint collector args.')
    parser.add_argument('--data_path',  type=str, required=True)
    parser.add_argument('--excluded_bags',  type=str, default=None)
    parser.add_argument('--output_path', type=str, default=os.getcwd())


    parser.add_argument('--models_paths',  type=str, default=None)

    parser.add_argument('--input_size',  type=int, default=224)
    parser.add_argument('--image_size',  type=int, default=448)
    parser.add_argument('--cols_num',  type=int, default=10)

    parser.add_argument('--nn_model', type=str, default="vgg16", choices=["vgg16, resnet"])
    parser.add_argument('--features_level',  type=int, default=-2)

    parser.add_argument('--display_best_imgs',  action="store_true")
    parser.add_argument('--display_best_imgs_for_relevant_cls',  action="store_true")

    parser.add_argument('--display_imgs_for_class', action="store_true")


    parser.add_argument('--max_imgs_in_image',  type=int, default=30)
    parser.add_argument('--max_classes',  type=float, default=0.4)
    parser.add_argument('--best_type_for_cls',  type=str, default="max", choices=["mean", "max"])


    parser.add_argument('--peeks',  type=str, default="1,0.75")



    parser.add_argument('--best_type_for_imgs',  type=str, default="max", choices=["mean", "max"])
    parser.add_argument('--mean_pos',  type=int, default=0.75)


    return parser


class ModelPreds:
    def __init__(self, name, pred_func, images_pos, classes_pos, max_images_num, max_classes_num, mean_pos):
        self.pref_func = pred_func
        self.name = name
        self.preds = dict()
        self.paths = dict()
        self.means = dict()
        self.images_pos = images_pos
        self.classes_pos = classes_pos
        self.max_images_num = max_images_num
        self.max_classes_num = max_classes_num
        self.mean_pos = mean_pos

    def set_features(self, bag_name, features, paths):
        self.preds[bag_name] = self.pref_func(features)
        self.paths[bag_name] = paths
        self.means[bag_name] = np.average(self.preds[bag_name])

    def get_relevant_classes(self):
        sorted_keys = [k for k, v in sorted(self.means.items(), key=lambda item: item[1])]
        classes_num = int(np.floor(self.max_classes_num * len(sorted_keys)))
        max_position = len(sorted_keys) - 1

        print("cls_num", classes_num, len(sorted_keys), max(0, max_position - classes_num), max_position+1)

        if self.classes_pos == "mean":
            max_position = np.int(np.floor(max_position * self.mean_pos))
        return sorted_keys[max(0, max_position - classes_num + 1) : max_position+1]

    def get_relevant_images(self, all_scores, classes, all_paths, peek_pos):
        max_position = np.int(np.floor((len(all_scores) - 1) * peek_pos))
        print(peek_pos, max_position, len(all_scores) - 1)

        relevant_indices = np.argsort(all_scores)[max_position - self.max_images_num + 1: max_position + 1]
        relevant_paths = [all_paths[i] for i in relevant_indices]
        if classes is None:
            titles = ["{}".format("%.2f" % all_scores[i]) for i in relevant_indices]
        else:
            titles = ["{}_{}".format(classes[i][:min(len(classes[i]), 8)], "%.2f" % all_scores[i]) for i in relevant_indices]
        return relevant_paths, titles

    def get_repre_images_for_cls(self, cls, peek_pos):
        return self.get_relevant_images(self.preds[cls], None, self.paths[cls], peek_pos)

    def get_repre_images_for_models(self, peek_pos, relevant_classes=None):
        all_scores = np.array([])
        classes = []
        all_paths = []

        if relevant_classes is None:
            relevant_classes = self.means.keys()

        for cls in relevant_classes:
            all_scores = np.concatenate((all_scores, self.preds[cls]))
            all_paths += self.paths[cls]
            classes += [cls] * len(self.paths[cls])
        return self.get_relevant_images(all_scores, classes, all_paths, peek_pos)


    def save_max_for_classes(self, output_path, low_percent=40, high_percent=100):
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



def build_models(models, bags, excluded_bags):

    for bag_name, bag in bags.items():
        if bag_name in excluded_bags:
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

        for name, model in models.items():
            model.set_features(bag_name, bag_features, bag_paths)



def parse_excluded_file(excluded_bags_file):
    if excluded_bags_file is None:
        return []
    with open(excluded_bags_file, 'r') as file:
        content = ""
        for line in file.readlines():
            content += line
        excluded_bags = content.split(",")
        excluded_bags_clean = [bag.strip().replace("'", "") for bag in excluded_bags]
        print("excluded_bags", excluded_bags_clean)
        return excluded_bags_clean

def display_best_imgs(model, cols_num, image_size, output_path, peek_pos):
    relevant_paths, titles = model.get_repre_images_for_models(peek_pos)
    image = vh.build_image(relevant_paths, titles, cols_num, image_size)
    if image is None:
        return
    title = "{}_best_images_{}_images_for_peek_{}".format(model.name, len(relevant_paths), str(peek_pos).replace(".", "_"))
    vh.display_images_of_image(image, 448, 4, title, output_path)

def display_best_imgs_for_relevant_cls(model, cols_num, image_size, output_path, peek_pos):
    relevant_classes = model.get_relevant_classes()
    relevant_paths, titles = model.get_repre_images_for_models(peek_pos, relevant_classes=relevant_classes)
    image = vh.build_image(relevant_paths, titles, cols_num, image_size)
    if image is None:
        return
    title = "{}_best_images_for_best_cls_for_peek_{}".format(model.name, str(peek_pos).replace(".", "_"))
    vh.display_images_of_image(image, 448, 4, title, output_path)

def save_best_classes(output_path, model):
    with open(os.path.join(output_path, "{}_best_classes.txt".format(model.name)), 'w') as file:
        for k, v in sorted(model.means.items(), key=lambda item: item[1]):
            file.write(str(k) + ", " + str(v) + "\n")
        # file.write(str([(k,model.means[k]) for k, v in sorted(model.means.items(), key=lambda item: item[1])]))

def display_imgs_for_class(model, cols_num, image_size, output_path, peek_pos):
    new_output_path = os.path.join(output_path, "class_representations")
    if not os.path.exists(new_output_path):
        os.makedirs(new_output_path)
    relevant_classes = model.get_relevant_classes()
    for cls in relevant_classes:
        relevant_paths, titles = model.get_repre_images_for_cls(cls=cls, peek_pos=peek_pos)
        print(cls, len(relevant_paths))
        image = vh.build_image(relevant_paths, titles, cols_num, image_size)
        if image is None:
            return
        title = "{}_best_images_for_cls_{}[{}]".format(model.name, cls, ("%.2f" % model.means[cls]).replace(".", "_"))
        vh.display_images_of_image(image, 448, 4, title, new_output_path)




def main():
    args = get_args_parser().parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    with open(os.path.join(args.output_path, "experiment_params.txt"), 'w') as f:
        json.dump(vars(args), f, indent=4)

    peeks_list = [float(item.strip()) for item in args.peeks.split(",")]
    print(peeks_list)

    models2pred_func = args_helper.parse_weights_paths(args.models_paths)
    excluded_bags = parse_excluded_file(args.excluded_bags)

    models = dict()
    for model_name in models2pred_func:
        models[model_name] = ModelPreds(model_name, models2pred_func[model_name], args.best_type_for_imgs,
                                        args.best_type_for_cls, args.max_imgs_in_image, args.max_classes, args.mean_pos)

    # create_preds
    print("builds models with predictions")

    nn_model = args_helper.get_features_model(args.nn_model, args.input_size, features_level=args.features_level)
    preprocessing_func = args_helper.get_preprocessing_func_by_name(args.nn_model)
    val_dataloader = Dataloader(nn_model, args.data_path, 1, args.input_size, 0, preprocess_func=preprocessing_func, )
    val_bags = val_dataloader.split_into_bags(train=True)
    build_models(models, val_bags, excluded_bags)

    print("create outputs")

    for name, model in models.items():
        # if args.display_best_imgs:
        for peek_pos in peeks_list:
            display_best_imgs(model, args.cols_num, args.image_size, args.output_path, peek_pos)
        # if args.display_best_imgs_for_relevant_cls:
            display_best_imgs_for_relevant_cls(model, args.cols_num, args.image_size, args.output_path, peek_pos)
        # if args.display_imgs_for_class:
        # display_imgs_for_class(model, args.cols_num, args.image_size, args.output_path)
        save_best_classes(args.output_path, model)
        model.save_max_for_classes(args.output_path)

if __name__=="__main__":
    main()