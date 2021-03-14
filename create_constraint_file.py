import os
import numpy as np
labels_map = {"1": 1, "0": 0}
from affordance_tools.ContraintsParser import ConstraintsParser
import argparse

root = "C:\\Users\\lotan\\Documents\\studies\\\Affordances\\datasets\\ballpark_datasets\\desert\\train"
dest = "C:\\Users\\lotan\\Documents\\studies\\phoenix\\ballpark_indoor_outdoor\\explore_constraints\\desert_true_constraints_eps_01.txt"
EPS = 0.1

def get_indoor_size(root_path, sub_cls2label):
    parts = dict()
    for cls in os.listdir(root_path):
        cls_path = os.path.join(root_path, cls)
        total_size = 0
        positive_size = 0
        for sub_cls in os.listdir(cls_path):
            if sub_cls not in sub_cls2label:
                print("problem in " + cls)
            sub_cls_path = os.path.join(cls_path, sub_cls)
            total_size += len(os.listdir(sub_cls_path))
            if sub_cls2label[sub_cls] == 1:
                positive_size = len(os.listdir(sub_cls_path))
        if total_size == 0:
            print(cls + " is empty")
        parts[cls] = positive_size/total_size
    return parts

def create_lower_and_upper_bounds(true_percent, epsilon):
    parts = dict()
    for cls, percent in true_percent.items():
        lower, upper = max(0, percent - epsilon), min(1, percent + epsilon)
        parts[cls] = (lower, upper)
    return parts

def create_mutual_bounds(true_percent, epsilon):
    percents = np.array(list(true_percent.values()))
    classes = np.array(list(true_percent.keys()))
    sorted_idx = np.argsort(percents)
    prv_i = 0
    bounds = dict()
    for i in range(sorted_idx.shape[0]):
        if percents[sorted_idx[i]] == percents[sorted_idx[prv_i]]:
            i += 1
        elif percents[sorted_idx[i]] > percents[sorted_idx[prv_i]]:
            k = i
            while(k+1<sorted_idx.shape[0] and  percents[sorted_idx[i]] == percents[sorted_idx[i+1]]):
                k += 1
            for j in range(prv_i, i):
                for t in range(i, k+1):
                    diff = percents[sorted_idx[t]] - percents[sorted_idx[j]]
                    lower, upper = max(0, diff - epsilon), min(1, diff + epsilon)
                    bounds["{} - {}".format(classes[sorted_idx[t]], classes[sorted_idx[j]])] = (lower, upper)
            prv_i = k
    return bounds


def write_bounds(parts, dest):
    with open(dest, 'w') as f:
        for cls, bounds in parts.items():
            f.write("{} < {} < {}\n".format(format(bounds[0], '.2f'), cls, format(bounds[1], '.2f')))

# creates constraints from splitted dir
# true_indoor_percent = get_indoor_size(root, labels_map)
# bounds = create_lower_and_upper_bounds(true_indoor_percent, EPS)
# diff_bounds = create_mutual_bounds(true_indoor_percent, EPS)
# all_bounds = {**bounds, **diff_bounds}
# write_bounds(all_bounds, dest)

# creates from constraints file
def build_diff_constraints_from_lower_and_upper(constraints_parser):
    diff_constraints_bounds = dict()

    lowers = constraints_parser.lower_bounds
    uppers = constraints_parser.upper_bounds
    for cls1, lower in lowers.items():
        for cls2, upper in uppers.items():
            if lower >= upper:
                c1_upper = uppers[cls1] if cls1 in uppers else 1
                c2_lower = lowers[cls2] if cls2 in lowers else 0
                diff_lower = lower - upper
                diff_upper = c1_upper - c2_lower
                diff_constraints_bounds["{} - {}".format(cls1, cls2)] = (diff_lower, diff_upper)
    return diff_constraints_bounds

def get_all_bounds(constraints_path):
    constraints_parser = ConstraintsParser(constraints_path)
    constraints_bounds = build_diff_constraints_from_lower_and_upper(constraints_parser)
    for cls, lower in constraints_parser.lower_bounds.items():
        constraints_bounds[cls] = (lower, None)
    for cls, upper in constraints_parser.upper_bounds.items():
        if cls in constraints_bounds:
            lower = constraints_bounds[cls][0]
            constraints_bounds[cls] = (lower, upper)
        else:
            constraints_bounds[cls] = (None, upper)
    return constraints_bounds

def write_bounds_into_file(all_bounds_dict, dest_full_path):
    first_line=True
    with open(dest_full_path, "w") as f:
        for name, bounds in all_bounds_dict.items():
            if first_line:
                first_line = False
            else:
                f.write("\n")
            if bounds[0] is not None and bounds[1] is not None:
                f.write("{} < {} < {}".format(format(bounds[0], '.2f'), name, format(bounds[1], '.2f')))
            elif bounds[0] is not None:
                f.write("{} > {}".format(name, format(bounds[0], '.2f')))
            else:
                f.write("{} < {}".format(name, format(bounds[1], '.2f')))


def get_args_parser():
    parser = argparse.ArgumentParser(description='Process constraint collector args.')
    parser.add_argument('--src_path', '-s', type=str, required=True)
    parser.add_argument('--dest_path',  '-d', type=str, required=True)
    parser.add_argument('--expand_constraints',  '-expand', action="store_true")
    parser.add_argument('--create_auto_constraints', '-auto', action="store_true")
    parser.add_argument('--auto_eps', '-e', type=float, default=0.1)
    parser.add_argument('--neg_label', type=str, default="0")
    parser.add_argument('--pos_label', type=str, default="1")
    return parser

def main():
    args = get_args_parser().parse_args()
    print(vars(args))
    if args.expand_constraints:
        if not os.path.exists(args.dest_path):
            os.makedirs(args.dest_path)
        dest_file = os.path.join(args.dest_path, "{}_full.txt".format(os.path.abspath(args.src_path).split(".")[0]))
        all_bounds = get_all_bounds(args.src_path)
        write_bounds_into_file(all_bounds, dest_file)
    else:
        # creates constraints from split dir
        labels_map = {args.pos_label: 1, args.neg_label: 0}
        true_indoor_percent = get_indoor_size(args.src_path, labels_map)
        bounds = create_lower_and_upper_bounds(true_indoor_percent, args.auto_eps)
        diff_bounds = create_mutual_bounds(true_indoor_percent, args.auto_eps)
        all_bounds = {**bounds, **diff_bounds}
        write_bounds(all_bounds, args.dest_path)



if __name__ == "__main__":
    main()
# PATH = "C:\\Users\\lotan\\Documents\\studies\\phoenix\\projects\\ballpark_indoor_outdoor\\explore_constraints\\teaching_constraints.txt"
# DEST = "C:\\Users\\lotan\\Documents\\studies\\phoenix\\projects\\ballpark_indoor_outdoor\\explore_constraints\\teaching_constraints_full.txt"
# all_bounds = get_all_bounds(PATH)
# write_bounds_into_file(all_bounds, DEST)




