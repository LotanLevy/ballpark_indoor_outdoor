
import argparse
import os
import json
from affordance_tools.ContraintsParser import ConstraintsParser
from affordance_tools.RegressionModel import RegressionModel
from affordance_tools.EntropyClassifier import EntropyClassifier
from affordance_tools.BallparkClassifier_2 import BallparkClassifier2
from affordance_tools.BallparkClassifier import BallparkClassifier
# from affordance_tools.RegressionWithEntropy import RegressionWithEntropy

from tensorflow.keras.applications import vgg16
from data_tools.dataloader import Dataloader
from sklearn.metrics import confusion_matrix
import numpy as np
from evaluation_tools.evaluate import *
import matplotlib.pyplot as plt
from shutil import copyfile
from args_helper import get_features_model, get_preprocessing_func_by_name, prepare_svm_data
from sklearn import svm





def get_args_parser():
    parser = argparse.ArgumentParser(description='Process constraint collector args.')
    parser.add_argument('--constraints_file', '-cf',  type=str, required=True)
    parser.add_argument('--train_root_path',  type=str, required=True)
    parser.add_argument('--val_root_path',  type=str, required=True)
    parser.add_argument('--cls_method',  type=str, default="regress", choices=['test', 'class', 'regress', 'regwithentropy', 'clswithentropy'])
    parser.add_argument('--features_level',  type=int, default=-2)

    parser.add_argument('--input_size',  type=int, default=224)
    parser.add_argument('--split_val',  type=float, default=0.2)
    parser.add_argument('--reg_val',  type=float, default=10**-1)
    parser.add_argument('--reg_type',  type=str, default="l2", choices=["l2", "entropy"])
    parser.add_argument('--loss_type',  type=str, default="l2", choices=["l2", "entropy"])
    parser.add_argument('--nn_model', type=str, default="vgg16", choices=["vgg16", "resnet50"])

    parser.add_argument('--polar_svm_param', type=float, default=0.3, help="constraints bound for the polar classes")

    parser.add_argument('--test_type',  type=str, choices=['explore', 'valid', 'pass'], default='valid')

    parser.add_argument('--labels_map_path',  type=str, default=None)
    parser.add_argument('--output_path', type=str, default=os.getcwd())

    parser.add_argument('--no_ballpark',  action="store_true")
    parser.add_argument('--no_svm',  action="store_true")


    return parser


def run_svm(constraints_parser, train_bags, polar_bound, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # prepare data for svm
    negative_classes, positive_classes = constraints_parser.get_negative_and_positive_classes_by_bound(polar_bound)
    if len(negative_classes) == 0:
        print("There is no polar negative classes")
    elif len(positive_classes) == 0:
        print("There is no polar positive classes")
    X, y, paths = prepare_svm_data(train_bags, negative_classes, positive_classes)
    clf = svm.SVC(kernel='linear')

    clf.fit(X, y)

    w = clf.coef_.flatten()
    b = clf.intercept_

    print("X shape {} \ny shape {} \nw shape {} \nb shape {}".format(X.shape, y.shape, w.shape, b.shape))
    np.save(os.path.join(output_path, "svm_weights"), w)
    np.save(os.path.join(output_path, "svm_bias"), b)
    return w, b

def get_ballpark_model(ballpark_type):
    if ballpark_type == "class":
        ballpark_object = BallparkClassifier
    elif ballpark_type == "regress":
        ballpark_object = RegressionModel
    elif ballpark_type == "clswithentropy":
        ballpark_object = EntropyClassifier
    else:
        print("wrong ballpark model type")
        return None
    return ballpark_object

def run_ballpark_model(ballpark_type, constraints, train_bags, reg_val, reg_type, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print("Initialize ballpark model")
    ballpark_object = get_ballpark_model(ballpark_type)
    ballpark_model = ballpark_object(constraints, train_bags)
    print("Start ballpark learning")
    w_t, y_t, _ = ballpark_model.solve_w_y(reg_val=reg_val, weights_path=os.path.join(output_path, "ballpark_weights"), reg_type=reg_type, output_path=output_path)
    np.save(os.path.join(output_path, "ballpark_weights"), w_t)
    return w_t





def main():
    args = get_args_parser().parse_args()
    print(vars(args))
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    with open(os.path.join(args.output_path, "experiment_params.txt"), 'w') as f:
        json.dump(vars(args), f, indent=4)
    print("Reads constraints file")
    constraints = ConstraintsParser(args.constraints_file)

    print("Initialize dataloader")
    preprocessing_func = get_preprocessing_func_by_name(args.nn_model)
    nn_model = get_features_model(args.nn_model, args.input_size, features_level=args.features_level)
    train_dataloader = Dataloader(nn_model, args.train_root_path, 1, args.input_size, 0, features_level=args.features_level, preprocess_func=preprocessing_func)
    print("Split data into bags")
    train_bags = train_dataloader.split_into_bags(train=True)
    print("Run SVM model")
    if not args.no_svm:
        svm_output_path = os.path.join(args.output_path, os.path.join("svm_model", "{}".format(args.polar_svm_param.replace(".", ""))))
        svm_w, svm_b = run_svm(constraints, train_bags, args.polar_svm_param, svm_output_path)
    if not args.no_ballpark:
        ballpark_output_path = os.path.join(args.output_path, "ballpark_model")
        ballpark_w = run_ballpark_model(args.cls_method, constraints, train_bags, args.reg_val, args.reg_type, ballpark_output_path)


    # # writes ballpark constraints into file
    # with open(os.path.join(args.output_path, "constraints.txt"), 'w') as wr:
    #     with open(args.constraints_file, 'r') as rf:
    #         content = rf.read()
    #         wr.write(content)
    #
    # print("Initialize ballpark model")
    # if args.cls_method == "class":
    #     ballpark_object = BallparkClassifier
    # elif args.cls_method == "regress":
    #     ballpark_object = RegressionModel
    # elif args.cls_method == "clswithentropy":
    #     ballpark_object = EntropyClassifier
    # else:
    #     print("wrong ballpark model type")
    #     return None
    # b = ballpark_object(constraints, train_bags)
    # print("Start ballpark learning")
    # w_t, y_t, _ = b.solve_w_y(reg_val=args.reg_val, weights_path=os.path.join(args.output_path, "ballpark_weights"), reg_type=args.reg_type, output_path=args.output_path)
    # np.save(os.path.join(args.output_path, "ballpark_weights"), w_t)
    #
    # if args.test_type == "pass":
    #     return
    #
    # val_dataloader = Dataloader(nn_model, args.val_root_path, 1, args.input_size, 0, features_level=args.features_level, preprocess_func=preprocessing_func, labels_map_path=args.labels_map_path)
    # val_bags = val_dataloader.split_into_bags(train=True)
    # w_t = np.load(os.path.join(args.output_path, "ballpark_weights.npy"))
    #
    # all_labels = np.array([])
    # all_preds = np.array([])
    # all_paths = []
    # for bag_name, bag in val_bags.items():
    #     bag_preds, paths = make_predictions_for_bag(bag, w_t)
    #     if bag.path2label_dict is None:
    #         bag_labels = np.ones(len(paths)) * bag.bag_label
    #     else:
    #         bag_labels = np.array([bag.path2label_dict[path] for path in paths])
    #
    #     # bag_labels = np.array([bag.path2label_dict[path] for path in paths])
    #     all_labels = np.concatenate((all_labels, bag_labels))
    #     # if args.cls_method == "test" or args.cls_method == "class":
    #     #     bag_preds = np.maximum(np.zeros(bag_preds.shape), np.sign(bag_preds))
    #     all_preds = np.concatenate((all_preds, bag_preds))
    #     all_paths += paths
    #     # display_predictions_for_bag(args.output_path, bag, bag_preds, paths)
    #
    # # for classifying
    # if args.test_type == 'valid':
    #     display_roc_graph(args.output_path, "indoor_outdoor", all_preds, all_labels)
    #     pred_classifications = get_classifications_by_roc(all_preds, all_labels)
    #     tn, fp, fn, tp = confusion_matrix(all_labels, pred_classifications).ravel()
    #     print(tn, fp, fn, tp)
    #     display_paths_according_to_the_confusion_matrix(args.output_path, all_paths, pred_classifications, all_labels)
    #     indices = np.random.choice(len(all_paths), 40, replace=False)
    #     paths_to_display = [all_paths[i] for i in indices]
    #     create_images_graph(args.output_path, paths_to_display, all_preds[indices])
    #
    # # for exploration
    # if args.test_type == 'explore':
    #     images_path = os.path.join(args.output_path, "scored_test")
    #     os.makedirs(images_path, exist_ok=True)
    #     for path, pred in list(zip(all_paths, all_preds)):
    #         file_name = os.path.basename(path)
    #         copyfile(path, os.path.join(images_path, "{}_{}".format(format(pred, '.2f').replace(".", "_"),file_name)))
    #
    #     # if not args.cls_method:
    #     #     display_roc_graph(args.output_path, "indoor_outdoor", all_preds, all_labels)
    #     #     final_classification = get_classifications_by_roc(all_preds, all_labels)
    #     # else:
    #     #     final_classification = all_preds
    #     # images_path = os.path.join(args.output_path, "splitted_test")
    #     # os.makedirs(images_path, exist_ok=True)
    #     # labels = np.unique(final_classification)
    #     # for label in labels:
    #     #     label_name = 0 if label <= 0 else 1
    #     #     label_path = os.path.join(images_path, str(label_name))
    #     #     if not os.path.exists(label_path):
    #     #         os.makedirs(label_path)
    #     #     labels_indices = np.where(final_classification == label)[0]
    #     #     for i in labels_indices:
    #     #         file_path = all_paths[i]
    #     #         file_name = os.path.basename(file_path)
    #     #         copyfile(file_path, os.path.join(label_path, file_name))
    #     #
    #     #
    #     #
    #
    #
    #


if __name__ == "__main__":
    main()
