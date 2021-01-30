
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt



def get_args_parser():
    parser = argparse.ArgumentParser(description='Process constraint collector args.')

    parser.add_argument('--svm_ckpt_path',  type=str, default=None)
    parser.add_argument('--ballpark_ckpt_path', type=str)

    parser.add_argument('--input_size',  type=int, default=224)


    parser.add_argument('--output_path', type=str, default=os.getcwd())

    return parser


def get_prediction_func(model_type, ckpt_path):
    if model_type == "svm":
        w = np.load(os.path.join(ckpt_path, "svm_weights.npy"))
        b = np.load(os.path.join(ckpt_path, "svm_bias.npy"))
        return lambda features: np.dot(w, features.T) + b
    else:
        w = np.load(os.path.join(ckpt_path, "ballpark_weights.npy"))
        b = 0
        return w, b

def plot_weigts(weights):
    plt.figure()
    x = None
    width = 0.3
    cur = None
    for name, w in weights.items():
        if x is None:
            x = np.arange(len(w))
            cur = x - width

        plt.bar(cur, height=w, label=name)
        cur += width
    plt.legend()
    plt.show()

def main():
    args = get_args_parser().parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    ballpark_paths = [str(item) for item in args.ballpark_ckpt_path.split(',')]

    weights = dict()


    if args.svm_ckpt_path is not None:
        w, b = get_prediction_func("svm", args.svm_ckpt_path)
        weights.append(w)
    for ckpt_path in ballpark_paths:
        name, path = ckpt_path.split("$")
        w, b = get_prediction_func("ballpark", path)
        weights[name] = w
    plot_weigts(weights)



if __name__=="__main__":
    main()