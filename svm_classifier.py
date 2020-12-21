
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




def svm_prediction(w, b, x):
    return np.sign(np.dot(w, x.T) +b)

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



    parser.add_argument('--labels_map_path',  type=str, default=None)
    parser.add_argument('--output_path', type=str, default=os.getcwd())
    return parser


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

    positive_features, _ = train_bags["1"].get_features()
    negative_features, _ = train_bags["0"].get_features()
    if args.max_files is not None:
        pos_idx = np.random.choice(positive_features.shape[0], args.max_files)
        neg_idx = np.random.choice(negative_features.shape[0], args.max_files)

        positive_features = positive_features[pos_idx]
        negative_features = negative_features[neg_idx]


    y_positive = np.ones(positive_features.shape[0])
    y_negative = -1 * np.ones(negative_features.shape[0])


    X = np.concatenate((positive_features, negative_features))
    y = np.concatenate((y_positive, y_negative))

    # w, b = svm_solve(X, y)

    clf = svm.SVC(kernel='linear')


    clf.fit(X, y)

    w = clf.coef_.flatten()
    b = clf.intercept_

    print("X shape {} \ny shape {} \nw shape {} \nb shape {}".format(X.shape, y.shape, w.shape, b.shape))





    positive_features, positive_paths = val_bags["1"].get_features()
    negative_features, negative_paths = val_bags["0"].get_features()
    X = np.concatenate((positive_features, negative_features))
    y_positive = np.ones(positive_features.shape[0])
    y_negative = -1 * np.ones(negative_features.shape[0])
    all_labels = np.concatenate((y_positive, y_negative))

    all_preds = svm_prediction(w, b, X)
    all_paths = positive_paths + negative_paths

    binary_labels = (all_labels == 1).astype(np.int)
    binary_preds = (all_preds == 1).astype(np.int)

    np.save(os.path.join(args.output_path, "svm_weights"), w)
    np.save(os.path.join(args.output_path, "svm_bias"), b)

    display_roc_graph(args.output_path, "indoor_outdoor", binary_preds, binary_labels)
    pred_classifications = get_classifications_by_roc(binary_preds, binary_labels)
    tn, fp, fn, tp = confusion_matrix(binary_labels, pred_classifications).ravel()
    display_paths_according_to_the_confusion_matrix(args.output_path, all_paths, pred_classifications, binary_labels)
    print(tn, fp, fn, tp)

    indices = np.random.choice(len(all_paths), 40, replace=False)
    paths_to_display = [all_paths[i] for i in indices]
    create_images_graph(args.output_path, paths_to_display, all_preds[indices])







if __name__ == '__main__':
    main()


