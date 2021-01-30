
from data_tools.dataloader import Dataloader
import argparse
import os
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import numpy as np
from tensorflow.keras.applications import vgg16
from evaluation_tools.evaluate import *
from sklearn.metrics import confusion_matrix
from sklearn import svm
from shutil import copyfile



NEGATIVE_VAL=0



def svm_prediction(w, b, x):
    return np.dot(w, x.T) +b

def svm_solve(X, y, C=10):
    # Initializing values and computing H. Note the 1. to force to float type
    m, n = X.shape
    y = y.reshape(-1, 1) * 1.
    X_dash = y * X
    H = np.dot(X_dash, X_dash.T) * 1.

    # Converting into cvxopt format - as previously
    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((m, 1)))
    G = cvxopt_matrix(np.vstack((np.eye(m) * -1, np.eye(m))))
    h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))

    # Run solver
    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])

    # ==================Computing and printing parameters===============================#
    w = ((y * alphas).T @ X).reshape(-1, 1)
    S = (alphas > 1e-4).flatten()
    b = y[S] - np.dot(X[S], w)

    # Display results
    print('Alphas = ', alphas[alphas > 1e-4])
    print('w = ', w.flatten().shape)
    print('b = ', b[0])
    return w.flatten(), b[0]





def get_args_parser():
    parser = argparse.ArgumentParser(description='Process constraint collector args.')
    parser.add_argument('--train_root_path',  type=str, required=True)
    parser.add_argument('--val_root_path',  type=str, required=True)

    parser.add_argument('--input_size',  type=int, default=224)
    parser.add_argument('--split_val',  type=int, default=0.2)
    parser.add_argument('--max_files',  type=int, default=None)
    parser.add_argument('--max_experiments',  type=int, default=1)
    parser.add_argument('--noise',  type=int, default=None)





    parser.add_argument('--labels_map_path',  type=str, default=None)
    parser.add_argument('--output_path', type=str, default=os.getcwd())
    parser.add_argument('--test_type',  type=str, choices=['explore', 'valid'], default='valid')

    return parser

def sample_noise(dataloader, size):
    paths = dataloader.train_iter.filepaths
    relevant_indices = np.random.choice(range(len(paths)), size)
    features_model = dataloader.get_features_model(dataloader.input_size, dataloader.features_level)

    features_mat = None
    paths = []

    print(dataloader.train_iter.labels[relevant_indices])

    for i in relevant_indices:
        try:
            preprocessed_image = dataloader.train_iter[i][0]
        except:
            print("Problem in loading " + dataloader.train_iter.filepaths[i])
            continue
        paths.append(dataloader.train_iter.filepaths[i])

        features_vec = features_model(preprocessed_image)
        features_mat = features_vec if features_mat is None else np.concatenate((features_mat, features_vec))
    return features_mat, paths


def main():
    args = get_args_parser().parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    print("Reads constraints file")
    preprocessing_func = lambda input_data: vgg16.preprocess_input(np.copy(input_data.astype('float32')))
    print("Initialize dataloader")
    train_dataloader = Dataloader(args.train_root_path, 1, args.input_size, 0, preprocess_func=preprocessing_func)
    val_dataloader = Dataloader(args.val_root_path, 1, args.input_size, 0, preprocess_func=preprocessing_func,
                                labels_map_path=args.labels_map_path)

    print("Split data into bags")
    train_bags = train_dataloader.split_into_bags(train=True)
    val_bags = val_dataloader.split_into_bags(train=True)

    results_dict = {"eer": 0 , "auc": 0, "f_score":0}

    for i in range(args.max_experiments):

        exp_output_path = os.path.join(args.output_path, str(i))
        if not os.path.exists(exp_output_path):
            os.makedirs(exp_output_path)

        if not os.path.exists(os.path.join(exp_output_path, "svm_weights.npy")):
            print(train_bags.keys())
            positive_features, _ = train_bags["1"].get_features(max_size=args.max_files)
            negative_features, _ = train_bags["0"].get_features(max_size=args.max_files)
            if args.noise:
                p = args.noise/100
                positive_amount = np.ceil((p)*positive_features.shape[0]).astype(np.int)
                negative_amount = np.ceil((p)*negative_features.shape[0]).astype(np.int)

                np.concatenate((positive_features, sample_noise(train_dataloader, positive_amount)[0]))
                np.concatenate((negative_features, sample_noise(train_dataloader, negative_amount)[0]))
                print("noise added with sizes {}, {}".format(positive_amount, negative_amount))

            y_positive = np.ones(positive_features.shape[0])
            y_negative = NEGATIVE_VAL * np.ones(negative_features.shape[0])

            X = np.concatenate((positive_features, negative_features))
            y = np.concatenate((y_positive, y_negative))
            print("start fitting")


            clf = svm.SVR(kernel='linear', C=1.0, epsilon=0.2)

            clf.fit(X, y)

            w = clf.coef_.flatten()
            b = clf.intercept_

            print("X shape {} \ny shape {} \nw shape {} \nb shape {}".format(X.shape, y.shape, w.shape, b.shape))
            np.save(os.path.join(exp_output_path, "svm_weights"), w)
            np.save(os.path.join(exp_output_path, "svm_bias"), b)
        else:
            print("loads existing weights")
            w = np.load(os.path.join(exp_output_path, "svm_weights.npy"))
            b = np.load(os.path.join(exp_output_path, "svm_bias.npy"))



        if args.test_type == "valid":
            positive_features, positive_paths = val_bags["1"].get_features()
            negative_features, negative_paths = val_bags["0"].get_features()
            X = np.concatenate((positive_features, negative_features))
            y_positive = np.ones(positive_features.shape[0])
            y_negative = NEGATIVE_VAL * np.ones(negative_features.shape[0])
            all_labels = np.concatenate((y_positive, y_negative))

            all_preds = svm_prediction(w, b, X)
            all_paths = positive_paths + negative_paths

            binary_labels = (all_labels == 1).astype(np.int)
            binary_preds = (all_preds == 1).astype(np.int)


            eer, auc = display_roc_graph(exp_output_path, "indoor_outdoor", all_preds, all_labels)
            pred_classifications = get_classifications_by_roc(all_preds, all_labels)
            tn, fp, fn, tp = confusion_matrix(binary_labels, pred_classifications).ravel()
            display_paths_according_to_the_confusion_matrix(exp_output_path, all_paths, pred_classifications, binary_labels)
            sum_data = fp + fn + tp + tn

            accuracy = (tp + tn) / sum_data

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f_score = 2 * (precision * recall) / (precision + recall)
            print(eer, auc, f_score)
            results_dict["eer"] += eer/args.max_experiments
            results_dict["auc"] += auc/args.max_experiments
            results_dict["f_score"] += f_score/args.max_experiments



            indices = np.random.choice(len(all_paths), 40, replace=False)
            paths_to_display = [all_paths[i] for i in indices]
            create_images_graph(exp_output_path, paths_to_display, all_preds[indices])

        # for exploration
        if args.test_type == 'explore':
            all_preds = np.array([])
            all_paths = []
            for bag_name, bag in val_bags.items():
                bag_features, bag_paths = bag.get_features()

                bag_preds = svm_prediction(w, b, bag_features)
                all_preds = np.concatenate((all_preds, bag_preds))
                all_paths += bag_paths


            images_path = os.path.join(exp_output_path, "scored_test")
            os.makedirs(images_path, exist_ok=True)
            for path, pred in list(zip(all_paths, all_preds)):
                file_name = os.path.basename(path)
                copyfile(path, os.path.join(images_path, "{}_{}".format(format(pred, '.2f').replace(".", "_"),file_name)))

    print(results_dict)


    with open(os.path.join(args.output_path, "avg_results.txt"), "w") as f:
        f.write(str(results_dict))




if __name__ == '__main__':
    main()


