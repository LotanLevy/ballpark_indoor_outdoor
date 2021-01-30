import argparse
import os
from shutil import copyfile
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import re




def get_args_parser():
    parser = argparse.ArgumentParser(description='Process constraint collector args.')
    parser.add_argument('--root_path', '-r',  type=str, required=True)
    parser.add_argument('--number2visualize', '-n',  type=int, default=35)
    parser.add_argument('--image_size',  type=int, default=448)

    parser.add_argument('--output_path', '-o',  type=str, required=True)
    return parser

def get_sorted_paths(root):
    paths_dict = dict()
    for file in os.listdir(root):
        pos = int(file.split("_")[0])
        paths_dict[pos] = os.path.join(root, file)
    sorted_paths = [v for k, v in sorted(paths_dict.items(), key=lambda item: item[0])]
    return sorted_paths



def plot_image_without_axes(image, title, output_path):
    plt.gca().set_axis_off()
    plt.figure()

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off

    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        right=False,  # ticks along the bottom edge are off
        left=False,  # ticks along the top edge are off
        labelleft=False)  # labels along the bottom edge are off


    plt.imshow(image)

    plt.tight_layout()
    plt.margins(0, 0)
    plt.savefig(os.path.join(output_path, title +".jpg"), dpi=300, bbox_inches='tight')
    plt.close("all")


def place_title_with_image(image, title, title_pos="bottom"):
    horizontal_title = True if title_pos in ["left", "right"] else False
    stack_func = np.hstack if horizontal_title else np.vstack
    if title is not None:
        if horizontal_title:
            text = Image.new('RGB', (np.floor(image.shape[1] / 5, image.shape[0]).astype(np.int)), color=(255, 255, 255))
        else:
            text = Image.new('RGB', (image.shape[1], np.floor(image.shape[0]/5).astype(np.int)), color=(255, 255, 255))

        d = ImageDraw.Draw(text)
        font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')), size=60)
        d.text((0, 0), title, fill=(0, 0, 0), font=font)
        if title_pos in ["bottom", "right"]:
            image = stack_func([image, text])
            return image
        elif title_pos in ["top", "left"]:
            image = stack_func([text, image])
            return image
        else:
            print("unrecognized direction")
    return None

def zipped_images(paths, title, images_size, vertical=False, title_pos="bottom", max_in_line=10):
    titled_images = []
    for path, title in list(zip(paths, title)):
        image = np.array(Image.open(path, 'r').convert('RGB').resize((images_size, images_size)))
        image = image/255
        titled_image = place_title_with_image(image, title, title_pos=title_pos)
        titled_images.append(titled_image)

    if len(titled_images) % max_in_line > 0:
        white_images_num = max_in_line - len(titled_images) % max_in_line
        titled_images += [np.ones((titled_images[0].shape[0], titled_images[0].shape[1], 3)) for _ in range(white_images_num)]

    grouped_images = []
    print(len(titled_images))
    for i in range(np.floor(len(titled_images)/max_in_line).astype(np.int)):
        print(i*max_in_line,i*max_in_line+max_in_line)
        relevant_images = titled_images[i*max_in_line:i*max_in_line+max_in_line]
        grouped_images.append(stack_with_spaces(relevant_images, vertical=vertical))

    return stack_with_spaces(grouped_images, vertical=(not vertical))

def stack_with_spaces(images, vertical=False, space=10):
    get_space = lambda : np.ones((space, images[0].shape[1],3)) if vertical else np.ones((images[0].shape[0], space, 3))
    to_stack = []
    text = ["ttt"] * len(images)
    for image in images:
        if len(to_stack) != 0:
            to_stack.append(get_space())
        to_stack.append(image)
    stack_func = np.vstack if vertical else np.hstack
    return stack_func(to_stack)


def create_image_from_paths(sorted_paths, max_num, images_size):
    paths = []
    titles = []
    if max_num > len(sorted_paths):
        max_num = 0
    for path in sorted_paths[-1*max_num:]:
        file_name = os.path.basename(path).split(".")[0]
        m=re.match("([\d]*)_([\w]*)_([\d]*)_([\d]*)", file_name)
        values = file_name.split("_")
        pos, cls, score = m.group(1), m.group(2), float(m.group(3) + "." + m.group(4))
        title = " " + str(score) + ", " + cls + " "
        # image = np.array(Image.open(path, 'r').convert('RGB').resize((images_size, images_size)))
        # images.append(image/255)
        titles.append(title)
        paths.append(path)
    result = zipped_images(paths, titles, images_size, vertical=False, title_pos="bottom")
    return result


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    sorted_paths = get_sorted_paths(args.root_path)
    result = create_image_from_paths(sorted_paths, args.number2visualize, args.image_size)
    plot_image_without_axes(result, "best_classes", args.output_path)

