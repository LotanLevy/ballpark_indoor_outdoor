

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import os



def stack_with_spaces(images, vertical=False, space=10):
    get_space = lambda : np.ones((space, images[0].shape[1],3)) if vertical else np.ones((images[0].shape[0], space, 3))
    to_stack = []
    for image in images:
        if len(to_stack) != 0:
            to_stack.append(get_space())
        to_stack.append(image)
    stack_func = np.vstack if vertical else np.hstack
    return stack_func(to_stack)


def build_image(images_paths, titles, cols, image_size):
    last_idx = len(images_paths)-1
    rows = []
    for i in range(last_idx, 0, -cols):
        start = max(0, i - cols + 1)
        row_paths = [images_paths[i] for i in range(start, i+1)]
        row_titles = [titles[i] for i in range(start, i+1)]
        rows.append(create_row_image(row_paths, image_size, titles=row_titles, cols_num=cols))
    if len(rows) == 0:
        return None
    image = stack_with_spaces(rows, vertical=True)
    return image


def display_images_of_image(image, image_size, cols_num, title, output_path):
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
        left=False,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        labelleft=False)  # labels along the bottom edge are off

    plt.imshow(image)
    ticks_vals = np.arange(0, (cols_num) * image_size, step=image_size)
    plt.yticks(ticks_vals, fontsize=8, rotation=90)
    plt.ylabel("higher is better")
    plt.xlabel("right is better")
    plt.tight_layout()
    plt.margins(0, 0)
    plt.rc('font', size=8)
    plt.savefig(os.path.join(output_path, title), dpi=300, bbox_inches='tight')
    plt.close("all")




def create_row_image(paths, images_size, titles = None, cols_num=None):
    images = []
    for i, path in enumerate(paths):
        image = np.array(Image.open(path, 'r').convert('RGB').resize((images_size, images_size)))

        if titles is not None:
            text = Image.new('RGB', (image.shape[1], np.floor(image.shape[0]/5).astype(np.int)), color=(255, 255, 255))
            d = ImageDraw.Draw(text)
            font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')), size=60)
            d.text((0, 0), titles[i], fill=(0, 0, 0), font=font)
            image = np.vstack([image, text])
        images.append(image/ 255)
    if cols_num is not None and len(images) < cols_num:
        images += [np.ones(images[0].shape) for _ in range(cols_num - len(images))]
    return stack_with_spaces(images, vertical=False)


if __name__=="__main__":
    ROOT = "C:\\Users\\lotan\\Documents\\studies\\phoenix\\datasets\\test\\airport_ticket_counter"
    import os
    paths = [os.path.join(ROOT, path) for path in os.listdir(ROOT)]
    titles = [path for path in os.listdir(ROOT)]
    image = build_image(paths, titles, 4, 448)
    display_images_of_image(image, 448, 4, None)


