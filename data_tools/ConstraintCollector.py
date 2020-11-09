
import argparse
import os
from data_tools.Dataloader1 import get_directory_iterators
from data_tools.CompareWindow import WindowComparison

def get_args_parser():
    parser = argparse.ArgumentParser(description='Process constraint collector args.')
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--output_path', type=str, default=os.getcwd())
    return parser


def get_cls_images(root_dir):
    cls2images = dict()
    all_data_iter, _ = get_directory_iterators(root_dir)
    for cls, label in all_data_iter.cls2label.items():
        images, label, paths = all_data_iter.get_data_by_label(label, size=1)
        cls2images[cls] = paths
    return cls2images

class binarySearchConstraints:
    def __init__(self, cls2images):
        self.cls2images = cls2images

    def ask_user(self, cls1, cls2):
        path1 = self.cls2images[cls1][0]
        path2 = self.cls2images[cls2][0]
        w = WindowComparison()
        w.open("which is more likely to stab?", path1, path2, cls1, cls2)
        val = w.compare_val
        if val is not None:
            return int(val)
        return val

    def binary_search_helper(self, cls, current_sorted, low_idx, high_idx):
        if len(current_sorted) == 0:
            return (0, False) if self.ask_user(cls, cls) is not None else (None, None)
        if high_idx <= low_idx:
            user_answer = self.ask_user(current_sorted[low_idx][0], cls)
            if user_answer is None:
                return None, None
            if user_answer >= 0:
                return low_idx, user_answer == 0
            else:
                return low_idx + 1, False
            # return low_idx if self.ask_user(current_sorted[low_idx], cls) >= 0 else low_idx + 1
        mid = (high_idx + low_idx) // 2
        if high_idx > low_idx:
            mid_cls_compare_cls = self.ask_user(current_sorted[mid][0], cls)
            if mid_cls_compare_cls is None:
                return None, None
            if mid_cls_compare_cls == 0: # current_sorted[mid] == cls
                return mid, True
            elif mid_cls_compare_cls == 1: # current_sorted[mid] > cls
                return self.binary_search_helper(cls, current_sorted, low_idx, mid-1)
            else: # current_sorted[mid] < cls
                return self.binary_search_helper(cls, current_sorted, mid+1, high_idx)


    def sort_by_user(self, to_sort):
        targets = []
        current_sorted = []
        for cls in to_sort:
            print(current_sorted)
            index, equal = self.binary_search_helper(cls, current_sorted, 0, len(current_sorted)-1)
            if index is None:
                targets.append(cls)
            elif equal:
                current_sorted[index].append(cls)
            else:
                current_sorted.insert(index, [cls])

        return current_sorted

def save_sorting_result_into_file(output_path, sort_result):
    with open(os.path.join(output_path, 'constraints.txt'), 'w') as f:
        for listitem in sort_result:
            f.write('%s\n' % listitem)


def main():
    args = get_args_parser().parse_args()
    cls2images = get_cls_images(args.root_dir)

    c_helper = binarySearchConstraints(cls2images)
    sorted_classes = c_helper.sort_by_user(cls2images)
    save_sorting_result_into_file(args.output_path, sorted_classes)





if __name__=="__main__":
    main()
