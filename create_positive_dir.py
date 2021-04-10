
import os
from shutil import copyfile
import argparse
from affordance_tools.ContraintsParser import ConstraintsParser



def get_args_parser():
    parser = argparse.ArgumentParser(description='Process constraint collector args.')
    parser.add_argument('--constraints_file', '-cf',  type=str, required=True)
    parser.add_argument('--root_path',  type=str, required=True)

    parser.add_argument('--polar_param', type=str, default="0.3,0.3", help="constraints bound for the polar classes")

    parser.add_argument('--output_path', type=str, default=os.getcwd())

    return parser

def copy_cls2dir(cls_path, dir_path):
    for filename in os.listdir(cls_path):
        file = os.path.join(cls_path, filename)
        if os.path.isdir(file):
            for inner_filename in os.listdir(file):
                inner_file = os.path.join(file, inner_filename)
                new_file_path = os.path.join(dir_path, inner_filename)
                copyfile(inner_file, new_file_path)
        else:
            new_file_path = os.path.join(dir_path, filename)
            copyfile(file, new_file_path)


def main():
    args = get_args_parser().parse_args()
    print(vars(args))
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        for sub_dir in ["0", "1"]:
            os.makedirs(os.path.join(args.output_path, sub_dir))

    constraints_parser = ConstraintsParser(args.constraints_file)
    polar_params = [float(num) for num in args.polar_param.split(",")]
    negative_classes, positive_classes = constraints_parser.get_negative_and_positive_classes_by_bound(polar_params)

    print(negative_classes)
    print(positive_classes)

    for cls in os.listdir(args.root_path):
        cls_path = os.path.join(args.root_path, cls)
        if cls in positive_classes:
            positive_path = os.path.join(args.output_path, "1")
            print("copy cls {} to pos".format(cls))
            copy_cls2dir(cls_path, positive_path)
        elif cls in negative_classes:
            print("copy cls {} to neg".format(cls))
            positive_path = os.path.join(args.output_path, "0")
            copy_cls2dir(cls_path, positive_path)



if __name__ == "__main__":
    main()