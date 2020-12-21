
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
PATH = "C:\\Users\\lotan\\Documents\\studies\\phoenix\\ballpark_indoor_outdoor\\scored_test"
REGEX = "([-]*[\d]+_[\d]+)_.*"
IMAGE_SIZE = 244

def get_scores2paths(root_path):
    scores = []
    paths = []
    for image_name in os.listdir(root_path):
        m_range = re.match(REGEX, image_name)
        score = float(m_range.group(1).replace("_", "."))
        scores.append(score)
        paths.append(os.path.join(root_path, image_name))

    return paths, scores


def get_random_percentile_images(ranges, num_for_range, paths, scores):
    sort_indices = np.argsort(np.array(scores))
    images_for_range = {}
    for range_ in ranges:
        min_idx = np.floor((range_[0]/100) * len(sort_indices)).astype(np.int)
        max_idx = np.ceil((range_[1] / 100) * len(sort_indices)).astype(np.int)

        relevant_indices = sort_indices[min_idx: max_idx]

        random_indices = np.random.choice(relevant_indices, min(num_for_range, len(relevant_indices)), replace=False)
        percentile_range = "({}%,{}%)".format(range_[0], range_[1])
        images_paths =[paths[idx] for idx in np.sort(random_indices)]

        images_for_range[percentile_range] = images_paths
    return images_for_range

def create_percentile_image(images_for_range, num_for_range):
    fig = plt.figure()
    n = 1

    rows = []

    for percentile_title, images_paths in images_for_range.items():
        missing_images_num = num_for_range - len(images_paths)
        empty_images = np.ones((IMAGE_SIZE, missing_images_num * IMAGE_SIZE, 3))

        for image_path in images_paths:
            image = np.array(Image.open(image_path, 'r').convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE)))
            empty_images = np.hstack((empty_images, image/255))
        # ax = fig.add_subplot(len(images_for_range), 1 ,n)
        rows.append(empty_images)

    plt.gca().invert_yaxis()

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off

    result = np.vstack(list(reversed(rows)))
    plt.imshow(result)
    ticks_vals = np.arange(IMAGE_SIZE//2, (len(images_for_range))*IMAGE_SIZE, step=IMAGE_SIZE)
    plt.yticks(ticks_vals, list(reversed(list(images_for_range.keys()))))
    plt.xlabel("scores values ->")
    plt.show()



paths, scores = get_scores2paths(PATH)
images_for_range = get_random_percentile_images([(0,50), (50,100)], 5, paths, scores)
create_percentile_image(images_for_range, 5)
print(images_for_range)