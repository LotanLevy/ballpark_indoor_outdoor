
import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

SRC_PATH = "C:\\Users\\lotan\\Documents\\studies\\Affordances\\datasets\\ADE20K_cleaned\\training"
OUTPUT_PATH = "C:\\Users\\lotan\\Documents\\studies\\Affordances\\experiments\\ballpark\\exploration"

CLASSES = ["gatehouse", "palace", "labyrinth", "imaret", "sacristy", "quadrangle", "legislative_chamber", "escarpment", "oast_house", "turkish_bath", "root_cellar", "igloo", "road_cut", "gorge", "gasworks", "spillway", "baseball_field"]
MAX_NUM = 2
IMAGE_SIZE = 224

def create_row_images(paths, images_size):
    row = None
    for path in paths:
        image = np.array(Image.open(path, 'r').convert('RGB').resize((images_size, images_size)))
        if row is None:
            row = image/ 255
        else:
            row = np.vstack((row, image / 255))
    return row


def create_cls_row(cls_name, root, max_num, images_size):
    cls_path = os.path.join(root, cls_name)
    images_in_class = [os.path.join(cls_path, file) for file in os.listdir(cls_path)]
    random_paths = random.sample(images_in_class, k=min(len(images_in_class), max_num))
    col = create_row_images(random_paths, images_size)
    empty_images = np.ones((images_size * max(0, max_num - len(images_in_class)),images_size,  3))
    return np.concatenate((col, empty_images), axis=0)



def plot_row_and_scores(titles, rows, images_size, name, output_path):
    plt.gca().set_axis_off()
    plt.figure(figsize = (6,3))
    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        labelleft=False)  # labels along the bottom edge are off
    plt.imshow(np.hstack(rows))
    ticks_vals = np.arange(images_size // 2, (len(titles)) * images_size, step=images_size)
    plt.xticks(ticks_vals, titles, fontsize=6, rotation=90)
    plt.tight_layout()
    plt.margins(0, 0)
    plt.rc('font', size=8)
    plt.savefig(os.path.join(output_path, name), dpi=300, bbox_inches='tight')
    plt.show()

def create_rows_for_classes(root, max_num, images_size):
    rows = []
    titles = []
    for cls in CLASSES:
        row = create_cls_row(cls, root, max_num, images_size)
        rows.append(row)
        titles.append(cls)
    plot_row_and_scores(titles, rows, images_size, "classes_repre_tourist", OUTPUT_PATH)


create_rows_for_classes(SRC_PATH, MAX_NUM, IMAGE_SIZE)
