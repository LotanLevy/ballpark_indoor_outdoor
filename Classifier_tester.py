



import os
import numpy as np
import argparse
from data_tools.dataloader import Dataloader
from args_helper import get_features_model, get_preprocessing_func_by_name, parse_weights_paths, get_prediction_func, \
    get_classification_by_values,prepare_svm_data
from evaluation_tools.evaluate import *
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
    parser.add_argument('--output_path', type=str, default=os.getcwd())
    parser.add_argument('--nn_model', type=str, default="vgg16", choices=["vgg16", "resnet50"])
    parser.add_argument('--features_level',  type=int, default=-2)

    return parser






def get_binary_labels_and_data(dataloader, positive_label, negative_label):
    val_bags = dataloader.split_into_bags(train=True)
    X, y, paths = prepare_svm_data(val_bags, negative_classes=["0"], positive_classes=["1"], positive_label=positive_label, negative_label=negative_label)
    return X, y, paths





def main():
    args = get_args_parser().parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    models = parse_weights_paths(args.weights_paths)
    preprocessing_func = get_preprocessing_func_by_name(args.nn_model)
    nn_model = get_features_model(args.nn_model, args.input_size, args.features_level)

    dataloader = Dataloader(nn_model, args.test_data_path, 1, args.input_size, 0, preprocess_func=preprocessing_func)
    X, y, paths = get_binary_labels_and_data(dataloader, args.positive_label, args.negative_label)

    models_evaluations = dict()
    evaluator = Evaluator()
    columns_names = evaluator.get_titles()

    for name in models:
        pred_func = models[name]
        scores = pred_func(X)
        if args.classify_mode:
            all_preds = get_classification_by_values(args.positive_label, args.negative_label, scores)
        else:
            all_preds = get_classifications_by_roc(scores, y)
        models_evaluations[name] = evaluator.get_evaluation_titles(y, all_preds)

    save_evaluations(columns_names, models_evaluations, args.output_path)




if __name__=="__main__":
    main()
