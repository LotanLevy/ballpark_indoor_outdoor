
import argparse
import os
from affordance_tools.ContraintsParser import ConstraintsParser
from affordance_tools.BallparkModels import BallparkModels
from affordance_tools.BallparkClassifier import BallparkClassifier

from tensorflow.keras.applications import vgg16
from data_tools.dataloader import Dataloader
from sklearn.metrics import confusion_matrix
import numpy as np
from evaluation_tools.evaluate import *
import matplotlib.pyplot as plt

def get_args_parser():
    parser = argparse.ArgumentParser(description='Process constraint collector args.')
    parser.add_argument('--constraints_file', '-cf',  type=str, required=True)
    parser.add_argument('--train_root_path',  type=str, required=True)
    parser.add_argument('--val_root_path',  type=str, required=True)
    parser.add_argument('--cls_method',  action="store_true")


    parser.add_argument('--input_size',  type=int, default=224)
    parser.add_argument('--split_val',  type=int, default=0.2)


    parser.add_argument('--labels_map_path',  type=str, default=None)
    parser.add_argument('--output_path', type=str, default=os.getcwd())
    return parser


def main():
    args = get_args_parser().parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    print("Reads constraints file")
    constraints = ConstraintsParser(args.constraints_file)
    preprocessing_func = lambda input_data: vgg16.preprocess_input(np.copy(input_data.astype('float32')))
    with open(os.path.join(args.output_path, "constraints.txt"), 'w') as wr:
        with open(args.constraints_file, 'r') as rf:
            content = rf.read()
            wr.write(content)
    print("Initialize dataloader")
    train_dataloader = Dataloader(args.train_root_path, 1, args.input_size, 0, preprocess_func=preprocessing_func)
    val_dataloader = Dataloader(args.val_root_path, 1, args.input_size, 0, preprocess_func=preprocessing_func, labels_map_path=args.labels_map_path)

    print("Split data into bags")
    train_bags = train_dataloader.split_into_bags(train=True)
    val_bags = val_dataloader.split_into_bags(train=True)

    print("Initialize ballpark model")
    ballpark_object = BallparkClassifier if args.cls_method else BallparkModels
    b = ballpark_object(constraints, train_bags)
    print("Start ballpark learning")
    w_t, y_t, prob_value = b.solve_w_y(weights_path=os.path.join(args.output_path, "ballpark_weights"))
    np.save(os.path.join(args.output_path, "ballpark_weights"), w_t)
    w_t = np.load(os.path.join(args.output_path, "ballpark_weights.npy"))
    all_labels = np.array([])
    all_preds = np.array([])
    all_paths = []
    for bag_name, bag in val_bags.items():
        bag_preds, paths = make_predictions_for_bag(bag, w_t)
        if bag.path2label_dict is None:
            bag_labels = np.ones(len(paths)) * bag.bag_label
        else:
            bag_labels = np.array([bag.path2label_dict[path] for path in paths])

        # bag_labels = np.array([bag.path2label_dict[path] for path in paths])
        all_labels = np.concatenate((all_labels, bag_labels))
        all_preds = np.concatenate((all_preds, np.sign(bag_preds)))
        all_paths += paths
        # display_predictions_for_bag(args.output_path, bag, bag_preds, paths)
    display_roc_graph(args.output_path, "indoor_outdoor", all_preds, all_labels)
    pred_classifications = get_classifications_by_roc(all_preds, all_labels)
    tn, fp, fn, tp = confusion_matrix(all_labels, pred_classifications).ravel()
    display_paths_according_to_the_confusion_matrix(args.output_path, all_paths, pred_classifications, all_labels)
    print(tn, fp, fn, tp)

    # sorted_indices = np.argsort(all_preds)
    # fig = plt.figure()
    # incdices = np.random.randint(0, len(sorted_indices), 40)
    indices = np.random.choice(len(all_paths), 40, replace=False)
    paths_to_display = [all_paths[i] for i in indices]
    create_images_graph(args.output_path, paths_to_display, all_preds[indices])




if __name__ == "__main__":
    main()
