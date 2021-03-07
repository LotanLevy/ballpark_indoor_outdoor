
import os
from shutil import copyfile
import argparse
from affordance_tools.ContraintsParser import ConstraintsParser
import visualizations.visualization_helper as vh


import matplotlib
matplotlib.use('Agg')

def get_args_parser():
    parser = argparse.ArgumentParser(description='Process constraint collector args.')
    parser.add_argument('--dir_path', '-p',  type=str, required=True)
    parser.add_argument('--max_files',  type=int, default=10)
    parser.add_argument('--output_path', type=str, default=os.getcwd())
    parser.add_argument('--image_size',  type=int, default=448)


    return parser

def parse_dir(dir_path):
    model_name = os.path.basename(dir_path).split("_")[0]
    print(model_name)
    cls2path = dict()
    cls2score = dict()
    for image_name in os.listdir(dir_path):
        image_path = os.path.join(dir_path, image_name)
        image_params = image_name.split(".")[0].split("_")
        cls_name, cls_score = '_'.join(image_params[1:-2]), float(image_params[-2]+"."+image_params[-1])
        cls2path[cls_name] = image_path
        cls2score[cls_name] = cls_score
    return model_name, cls2path, cls2score

def display_image(model_name, cls2path, cls2score, max_classes,image_size, output_path):
    sorted_classes = [k for k, v in sorted(cls2score.items(), key=lambda item: item[1])][-1 *max_classes: ]
    assert len(sorted_classes) == min(max_classes, len(cls2score.keys()))
    paths = [cls2path[cls] for cls in sorted_classes]
    titles = ["{}_{}".format(cls, cls2score[cls]) for cls in sorted_classes]
    image = vh.build_image(paths, titles, max_classes, image_size)
    vh.display_images_of_image(image, 448, 4, model_name, output_path)


def main():
    args = get_args_parser().parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)


    model_name, cls2path, cls2score = parse_dir(args.dir_path)

    display_image(model_name, cls2path, cls2score, args.max_files, args.image_size, args.output_path)


if __name__=="__main__":
    main()
