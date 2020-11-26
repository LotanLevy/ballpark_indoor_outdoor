import os
import numpy as np
labels_map = {"swimmable": 1, "non_swimmable": 0}

root = "C:\\Users\\lotan\\Documents\\studies\\phoenix\\datasets\\swim_ade20k\\train"
dest = "C:\\Users\\lotan\\Documents\\studies\\phoenix\\projects\\ballpark_indoor_outdoor\\constraints\\swim_ade20k_true_constraints_eps_03.txt"
EPS = 0.3

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


true_indoor_percent = get_indoor_size(root, labels_map)
bounds = create_lower_and_upper_bounds(true_indoor_percent, EPS)
diff_bounds = create_mutual_bounds(true_indoor_percent, EPS)
all_bounds = {**bounds, **diff_bounds}
write_bounds(all_bounds, dest)



