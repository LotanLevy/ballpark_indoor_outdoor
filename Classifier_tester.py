



import os
import numpy as np
import argparse
from data_tools.dataloader import Dataloader
from args_helper import get_features_model, get_preprocessing_func_by_name, parse_weights_paths, get_prediction_func\
    ,prepare_svm_data
from evaluation_tools.evaluate import *
from affordance_tools.ContraintsParser import ConstraintsParser
import visualizations.visualization_helper as vh

from sklearn.metrics import confusion_matrix




def get_args_parser():
    parser = argparse.ArgumentParser(description='Process constraint collector args.')
    parser.add_argument('--test_data_path',  type=str, required=True)
    parser.add_argument('--classify_mode',  action="store_true")


    parser.add_argument('--input_size',  type=int, default=224)
    parser.add_argument('--weights_paths',  type=str, default=None, help="paths arguments split by with comma, "
                                                                         "each argument phrase should contain 3 "
                                                                         "arguments: the model name, the full weights "
                                                                         "path and the full bias path, "
                                                                         "the arguments should be separated by $")
    parser.add_argument('--positive_label',  type=int, default=1)
    parser.add_argument('--negative_label',  type=int, default=0)
    parser.add_argument('--ballpark_threshold',  type=float, default=0.5)

    parser.add_argument('--output_path', type=str, default=os.getcwd())
    parser.add_argument('--nn_model', type=str, default="vgg16", choices=["vgg16", "resnet50"])
    parser.add_argument('--features_level',  type=int, default=-2)
    parser.add_argument('--clear_test',  action="store_true")
    parser.add_argument('--constraints_file',  type=str, default=None)


    return parser






def get_binary_labels_and_data(dataloader, positive_label, negative_label):
    val_bags = dataloader.split_into_bags(train=True)
    X, y, paths = prepare_svm_data(val_bags, negative_classes=["0"], positive_classes=["1"], positive_label=positive_label, negative_label=negative_label)
    return X, y, paths


def get_classification_threshold(self, classify_mode, model_type, scores, labels):
    if not classify_mode:
        return get_roc_threshold(scores, labels)
    elif model_type == "ballpark":
        return (self.positive_val - self.negative_val) / 2.0
    elif self.model_type == "svm":
        return 0.0
    else:
        print("illegal model type")
        return None

def clear_data_by_constraints_bags(constraints_file, X, y, paths):
    print("clean test paths")
    constraints_parser = ConstraintsParser(constraints_file)
    constrainted_classes = constraints_parser.all_classes
    legal_indices = []
    for i,path in enumerate(paths):
        paths_cls = os.path.basename(path).split("__")[0]
        if paths_cls not in constrainted_classes:
            legal_indices.append(i)
        else:
            print(path +" removed")
    clean_paths = [paths[j] for j in legal_indices]
    return X[np.array(legal_indices)], y[np.array(legal_indices)], clean_paths


def parse_classes_from_paths(paths):
    classes = []
    for path in paths:
        if "__" in path:
            classes.append(os.path.basename(path).split("__")[0])
        else:
            classes.append("")
    return classes



def main():
    args = get_args_parser().parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    models = parse_weights_paths(args.weights_paths)
    preprocessing_func = get_preprocessing_func_by_name(args.nn_model)
    nn_model = get_features_model(args.nn_model, args.input_size, args.features_level)

    dataloader = Dataloader(nn_model, args.test_data_path, 1, args.input_size, 0, preprocess_func=preprocessing_func)
    X, y, paths = get_binary_labels_and_data(dataloader, args.positive_label, args.negative_label)
    if args.clear_test and args.constraints_file is not None:
        X, y, paths = clear_data_by_constraints_bags(args.constraints_file, X, y, paths)

    models_evaluations = dict()
    evaluator = Evaluator(args.positive_label, args.negative_label, classify_mode=args.classify_mode)
    columns_names = evaluator.get_titles()

    for name in models:
        pred_func = models[name]
        scores = pred_func(X)

        if not args.classify_mode:
            threshold = get_roc_threshold(scores, y)
        elif "ballpark" in name:
            threshold = args.ballpark_threshold
        elif "svm" in name:
            threshold = 0.0
        else:
            print("illegal model type - {}".format(name))
            threshold = None


        models_evaluations[name] = evaluator.evaluate(scores, y, threshold)

        tn_paths, fp_paths, fn_paths, tp_paths = get_paths_of_confusion_matrix(paths, scores, y, evaluator)
        fn_maximal_paths, fn_scores = select_paths_by_number(fn_paths, 30, position="max")# creative thinking
        fp_maximal_paths, fp_scores = select_paths_by_number(fp_paths, 30, position="max")# wrong classifications
        fn_image = vh.build_image(list(fn_maximal_paths), ["{} {}".format("%.2f" % score_cls[0], score_cls[1]) for score_cls in list(zip(fn_scores, parse_classes_from_paths(fn_maximal_paths)))], 10, 448)
        fp_image = vh.build_image(list(fp_maximal_paths), ["{} {}".format("%.2f" % score_cls[0], score_cls[1]) for score_cls in list(zip(fp_scores, parse_classes_from_paths(fp_maximal_paths)))], 10, 448)

        vh.display_images_of_image(fn_image, 448, 4, "{}_fn_with_maximal_scores".format(name), args.output_path)
        vh.display_images_of_image(fp_image, 448, 4, "{}_fp_with_maximal_scores".format(name), args.output_path)



    save_evaluations(models_evaluations, args.output_path)




if __name__=="__main__":
    main()
