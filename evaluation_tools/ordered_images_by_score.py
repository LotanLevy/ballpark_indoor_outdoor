

from data_tools.dataloader import Dataloader
import argparse
import os
from PIL import Image
from evaluation_tools.evaluate import make_predictions_for_bag
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg16
import matplotlib.gridspec as gridspec




def get_args_parser():
    parser = argparse.ArgumentParser(description='Process constraint collector args.')
    parser.add_argument('--data_path',  type=str, required=True)
    parser.add_argument('--svm_ckpt_path',  type=str, required=True)
    parser.add_argument('--ballpark_ckpt_path',  type=str, required=True)

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
    plt.xticks(ticks_vals, titles)
    plt.tight_layout()
    plt.margins(0, 0)
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

    svm_prediction_func = get_prediction_func("svm", args.svm_ckpt_path)
    ballpark_prediction_func = get_prediction_func("ballpark", args.ballpark_ckpt_path)

    preprocessing_func = lambda input_data: vgg16.preprocess_input(np.copy(input_data.astype('float32')))


    val_dataloader = Dataloader(args.data_path, 1, args.input_size, 0, preprocess_func=preprocessing_func,)
    val_bags = val_dataloader.split_into_bags(train=True)

    svm_all_titles = []
    ballpark_all_title = []

    svm_all_preds = []
    ballpark_all_preds = []

    svm_paths = []
    ballpark_paths = []

    for bag_name, bag in val_bags.items():
        bag_features, bag_paths = bag.get_features()
        indices = np.random.choice(bag_features.shape[0], min(args.max_files, len(bag_paths)), replace=False)
        bag_features = bag_features[indices]
        bag_paths = [bag_paths[i] for i in indices]

        svm_preds = svm_prediction_func(bag_features)
        svm_paths.append(bag_paths[np.argmax(svm_preds)])
        svm_all_preds.append(np.max(svm_preds))

        ballpark_preds = ballpark_prediction_func(bag_features)
        ballpark_paths.append(bag_paths[np.argmax(ballpark_preds)])
        ballpark_all_preds.append(np.max(ballpark_preds))


        svm_ordered_titles, svm_row_image = create_orderd_images(bag_paths, svm_preds, args.image_size, "svm_" + bag_name.replace(".", "_"), args.output_path)
        svm_all_titles.append(svm_ordered_titles[-1] + "\n" + bag_name.split(".")[1][:min(len(bag_name), 8)])

        ballpark_ordered_titles, ballpark_row_image = create_orderd_images(bag_paths, ballpark_preds, args.image_size, "ballpark_" + bag_name.replace(".", "_"), args.output_path)
        ballpark_all_title.append(ballpark_ordered_titles[-1]+ "\n" + bag_name.split(".")[1][:min(len(bag_name), 8)])


        plot_row_and_scores(svm_ordered_titles, svm_row_image, args.image_size, "svm_"+bag_name.replace(".", "_"), args.output_path)
        plot_row_and_scores(ballpark_ordered_titles, ballpark_row_image, args.image_size, "ballpark_"+bag_name.replace(".", "_"), args.output_path)


        # plt.savefig(os.path.join(args.output_path,  bag_name.replace(".", "_")), dpi=300)
    svm_row = create_row_images(svm_paths, args.image_size)
    ballpark_row = create_row_images(ballpark_paths, args.image_size)

    plot_row_and_scores(svm_all_titles, svm_row, args.image_size, "svm_all" , args.output_path)
    plot_row_and_scores(ballpark_all_title, ballpark_row, args.image_size, "ballpark_all" ,
                              args.output_path)

main()